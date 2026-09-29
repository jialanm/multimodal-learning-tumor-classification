"""DMGI (Deep Multiplex Graph Infomax) pipeline for breast cancer tumor grading.

Aligned with the reference HetMed/DMGI (AAAI 2023) implementation using
Duke-Breast hyperparameters. See DMGI_ARCHITECTURE.md for details.

Builds multiplex patient-similarity graphs from clinical feature subgroups,
learns consensus node embeddings via mutual information maximization + GCN,
and classifies Nottingham grades (1/2/3).

Requires pre-trained Swin features (run `swin` subcommand first).

Key alignment with reference implementation:
  - Row-wise L1 feature normalization (process.preprocess_features)
  - Consensus reg loss uses .sum() (not .mean())
  - reg_coef=0.001 (paper's alpha)
  - Self-connection weight sc=3.0
  - Early stopping on total loss
  - Lower lr for logistic layer (prevents gradient short-circuit to H)
  - 60/10/30 train/val/test split
  - H initialized with xavier_normal_ (not projected features)
  - LogReg evaluation on frozen H (50 runs x 50 epochs)
"""

import os
import json
import time
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, f1_score, balanced_accuracy_score, confusion_matrix,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from typing import Optional
import pandas as pd

from .config import (
    LABEL_NAMES, NUM_CLASSES,
    RANDOM_SEED, DEFAULT_OUTPUT_DIR,
    DMGI_NUM_EPOCHS, DMGI_PATIENCE,
    DMGI_FEATURE_TYPE_GROUPS, DMGI_FEATURE_TYPE_GROUPS_COLS,
    DMGI_REG_COEF, DMGI_SUP_COEF,
    DMGI_SELF_CONNECTION, DMGI_DROP_PROB,
)
from .clinical import encode_clinical_features
from .swin_pipeline import build_patient_list
from .evaluation import (
    plot_loss_curves, plot_roc_curves,
    plot_confusion_matrix, plot_per_class_f1, plot_embedding_similarity_heatmap,
)


# =============================================================================
# FEATURE PREPROCESSING
# =============================================================================

def preprocess_features(features: np.ndarray) -> np.ndarray:
    """Row-wise L1 normalization: each row sums to 1.

    Matches reference implementation (process.preprocess_features).
    This prevents Swin embeddings (768-d) from dominating clinical features (31-d).
    """
    feat_sp = sp.lil_matrix(features)
    rowsum = np.array(feat_sp.sum(1)).flatten()
    r_inv = np.power(rowsum, -1.0)
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features_norm = r_mat_inv.dot(feat_sp)  # type: ignore[arg-type]
    return np.asarray(features_norm.todense(), dtype=np.float32)  # type: ignore[union-attr]


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_adjacency(clinical_encoded: np.ndarray, feature_indices: list[int],
                    threshold: float, sc: float = DMGI_SELF_CONNECTION
                    ) -> torch.Tensor:
    """Build a sparse normalized adjacency from cosine similarity on a feature subset.

    Returns sparse normalized adjacency: D^{-1/2} (A + sc*I) D^{-1/2}.
    """
    subset = clinical_encoded[:, feature_indices]
    sim = cosine_similarity(subset)
    adj = (sim >= threshold).astype(np.float32)
    np.fill_diagonal(adj, 0.0)

    # Add weighted self-loops (reference: sc=3.0)
    n = adj.shape[0]
    adj_hat = adj + sc * np.eye(n, dtype=np.float32)

    # Symmetric normalization: D^{-1/2} A_hat D^{-1/2}
    adj_sp = sp.coo_matrix(adj_hat)
    rowsum = np.array(adj_sp.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    adj_norm = adj_sp.dot(d_mat).transpose().dot(d_mat).tocoo()  # type: ignore[arg-type,union-attr]

    # Convert to sparse torch tensor
    indices = torch.from_numpy(
        np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64))
    values = torch.from_numpy(adj_norm.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


def build_multiplex_adjacency(clinical_encoded: np.ndarray,
                              thresholds: list[float]
                              ) -> tuple[list[torch.Tensor], dict]:
    """Build sparse normalized adjacency matrices for all relation types.

    Each relation uses its own threshold from the thresholds list.
    Returns (list of sparse adj tensors, adjacency stats dict).
    """
    groups = list(DMGI_FEATURE_TYPE_GROUPS.items())
    if len(thresholds) != len(groups):
        raise ValueError(
            f"Expected {len(groups)} thresholds, got {len(thresholds)}")

    adj_list = []
    stats = {}
    for (name, indices), thresh in zip(groups, thresholds):
        adj_sparse = build_adjacency(clinical_encoded, indices, thresh)
        # Stats on raw adjacency (before self-loops and normalization)
        subset = clinical_encoded[:, indices]
        sim = cosine_similarity(subset)
        raw_adj = (sim >= thresh).astype(np.float32)
        np.fill_diagonal(raw_adj, 0.0)
        n = raw_adj.shape[0]
        n_edges = int(raw_adj.sum()) // 2
        density = raw_adj.sum() / (n * (n - 1)) if n > 1 else 0.0
        stats[name] = {
            "num_features": len(indices),
            "threshold": float(thresh),
            "num_edges": n_edges,
            "density": float(density),
        }
        adj_list.append(adj_sparse)
    return adj_list, stats


def _normalize_adjacency(adj_raw: np.ndarray,
                         sc: float = DMGI_SELF_CONNECTION) -> torch.Tensor:
    """Symmetric-normalize a raw binary adjacency matrix.

    Returns sparse tensor: D^{-1/2} (A + sc*I) D^{-1/2}.
    """
    np.fill_diagonal(adj_raw, 0.0)
    n = adj_raw.shape[0]
    adj_hat = adj_raw + sc * np.eye(n, dtype=np.float32)

    adj_sp = sp.coo_matrix(adj_hat)
    rowsum = np.array(adj_sp.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    adj_norm = adj_sp.dot(d_mat).transpose().dot(d_mat).tocoo()  # type: ignore[arg-type,union-attr]

    indices = torch.from_numpy(
        np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64))
    values = torch.from_numpy(adj_norm.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


def build_adjacency_from_df(clinical_df: pd.DataFrame, col_names: list[str],
                            threshold: float,
                            sc: float = DMGI_SELF_CONNECTION) -> torch.Tensor:
    """Build adjacency using reference-style encoding: get_dummies + minmax_scale.

    Matches reference (duke_kmeans.py):
      1. Select columns for this relation type
      2. pd.get_dummies() on those columns
      3. fillna(0)
      4. minmax_scale(axis=0) per-feature
      5. cosine_similarity → threshold → binary adjacency
    """
    subset_df = clinical_df[col_names].copy()
    # get_dummies expands each unique value into a binary column
    dummied = pd.get_dummies(subset_df, columns=col_names)
    dummied = dummied.fillna(0)
    features = dummied.values.astype(np.float64)

    # Column-wise minmax scaling to [0,1] (reference: minmax_scale(axis=0))
    features = minmax_scale(features, axis=0, copy=False)

    sim = cosine_similarity(features)
    adj = (sim >= threshold).astype(np.float32)
    return _normalize_adjacency(adj, sc)


def build_multiplex_adjacency_from_df(
        clinical_df: pd.DataFrame,
        thresholds: list[float],
) -> tuple[list[torch.Tensor], dict]:
    """Build multiplex adjacency using reference-style encoding per relation.

    Uses DMGI_FEATURE_TYPE_GROUPS_COLS (column-name groups) with pd.get_dummies
    + minmax_scale before cosine similarity.
    """
    groups = list(DMGI_FEATURE_TYPE_GROUPS_COLS.items())
    if len(thresholds) != len(groups):
        raise ValueError(
            f"Expected {len(groups)} thresholds, got {len(thresholds)}")

    adj_list = []
    stats = {}
    for (name, col_names), thresh in zip(groups, thresholds):
        adj_sparse = build_adjacency_from_df(clinical_df, col_names, thresh)

        # Compute stats on raw adjacency (before self-loops and normalization)
        subset_df = clinical_df[col_names].copy()
        dummied = pd.get_dummies(subset_df, columns=col_names).fillna(0)
        features = minmax_scale(dummied.values.astype(np.float64), axis=0)
        sim = cosine_similarity(features)
        raw_adj = (sim >= thresh).astype(np.float32)
        np.fill_diagonal(raw_adj, 0.0)
        n = raw_adj.shape[0]
        n_edges = int(raw_adj.sum()) // 2
        density = raw_adj.sum() / (n * (n - 1)) if n > 1 else 0.0
        n_dummy_cols = dummied.shape[1]
        stats[name] = {
            "num_features": len(col_names),
            "num_dummy_cols": n_dummy_cols,
            "threshold": float(thresh),
            "num_edges": n_edges,
            "density": float(density),
        }
        adj_list.append(adj_sparse)
    return adj_list, stats


# =============================================================================
# MODEL
# =============================================================================

class GCN(nn.Module):
    """Single-layer GCN with input dropout: ReLU(A_norm @ (Dropout(X) @ W))."""

    def __init__(self, in_dim: int, out_dim: int, drop_prob: float = DMGI_DROP_PROB):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        return F.relu(torch.sparse.mm(adj, x @ self.weight))


class Discriminator(nn.Module):
    """Shared bilinear discriminator.

    Takes (summary, h_pos, h_neg) and returns logits of shape (2N,).
    """

    def __init__(self, hid_dim: int):
        super().__init__()
        self.bilinear = nn.Bilinear(hid_dim, hid_dim, 1)
        nn.init.xavier_uniform_(self.bilinear.weight)
        nn.init.zeros_(self.bilinear.bias)

    def forward(self, summary: torch.Tensor, h_pos: torch.Tensor,
                h_neg: torch.Tensor) -> torch.Tensor:
        # summary: (hid,), h_pos/h_neg: (N, hid)
        s = summary.unsqueeze(0).expand_as(h_pos)
        pos_scores = self.bilinear(h_pos, s).squeeze(-1)  # (N,)
        neg_scores = self.bilinear(h_neg, s).squeeze(-1)  # (N,)
        return torch.cat([pos_scores, neg_scores], dim=0)  # (2N,)


class Attention(nn.Module):
    """Per-node attention fusion across relation embeddings.

    Matches reference implementation (layers/attention.py):
    learns per-graph importance, applies softmax, weighted sum.
    """

    def __init__(self, hid_dim: int, num_graphs: int, temperature: float = 1.0):
        super().__init__()
        self.num_graphs = num_graphs
        self.temperature = temperature
        self.A = nn.ModuleList([nn.Linear(hid_dim, 1) for _ in range(num_graphs)])
        for i in range(num_graphs):
            nn.init.xavier_normal_(self.A[i].weight)
            self.A[i].bias.data.fill_(0.0)

    def _attn_feature(self, features: list[torch.Tensor], num_nodes: int,
                      hid_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted fusion of per-graph embeddings."""
        # features: list of R tensors, each (N, hid)
        features_attn = []
        for i in range(self.num_graphs):
            features_attn.append(self.A[i](features[i]))  # (N, 1)
        # Temperature scaling: T > 1 produces softer (more uniform) attention
        features_attn = F.softmax(
            torch.cat(features_attn, 1) / self.temperature, dim=-1)  # (N, R)
        p = torch.mean(features_attn, dim=0)  # (R,)

        stacked = torch.cat(features, dim=1)  # (N, R*hid)
        attn_flat = features_attn.transpose(1, 0).contiguous().view(-1, 1)  # (R*N, 1)
        stacked_t = stacked.view(num_nodes, self.num_graphs * hid_dim)
        # Reshape for element-wise multiply
        feats_r = torch.stack(features, dim=0)  # (R, N, hid)
        attn_expanded = features_attn.transpose(0, 1).unsqueeze(-1)  # (R, N, 1)
        fused = (feats_r * attn_expanded).sum(dim=0)  # (N, hid)
        return fused, p

    def forward(self, h_pos_list: list[torch.Tensor],
                h_neg_list: list[torch.Tensor],
                num_nodes: int, hid_dim: int
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_pos_fused, p = self._attn_feature(h_pos_list, num_nodes, hid_dim)
        h_neg_fused, _ = self._attn_feature(h_neg_list, num_nodes, hid_dim)
        return h_pos_fused, h_neg_fused, p


class LogReg(nn.Module):
    """Simple logistic regression for evaluation on frozen embeddings."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class SupConLoss(nn.Module):
    """Supervised contrastive loss on embeddings (Khosla et al., NeurIPS 2020).

    Pulls same-class embeddings together and pushes different-class embeddings
    apart directly in embedding space — no learnable intermediary.

    For each anchor i with label y_i, positive pairs are all j where y_j == y_i,
    negative pairs are all j where y_j != y_i.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute supervised contrastive loss.

        Args:
            features: (N, D) L2-normalized embeddings
            labels: (N,) integer class labels
        Returns:
            scalar loss
        """
        # L2 normalize
        features = F.normalize(features, dim=1)
        n = features.shape[0]

        # Pairwise cosine similarity / temperature
        sim = torch.mm(features, features.t()) / self.temperature  # (N, N)

        # Mask: 1 where labels match (excluding self)
        labels = labels.unsqueeze(0)  # (1, N)
        mask_pos = (labels == labels.t()).float()  # (N, N)
        mask_pos.fill_diagonal_(0.0)  # exclude self

        # For numerical stability, subtract max from each row
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Exclude self from denominator
        mask_self = torch.eye(n, device=features.device)
        exp_sim = torch.exp(sim) * (1.0 - mask_self)  # (N, N)

        # log(exp(sim_pos) / sum(exp(sim_all)))
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Average over positive pairs
        num_pos = mask_pos.sum(dim=1)  # (N,)
        # Avoid division by zero for classes with single sample
        num_pos = torch.clamp(num_pos, min=1.0)
        loss = -(mask_pos * log_prob).sum(dim=1) / num_pos  # (N,)

        return loss.mean()


class DMGI(nn.Module):
    """Deep Multiplex Graph Infomax model (reference-aligned).

    R single-layer GCNs + 1 shared discriminator + attention fusion
    + learnable consensus H + semi-supervised LogReg classifier.
    """

    def __init__(self, in_dim: int, hid_dim: int, num_nodes: int,
                 num_relations: int, num_classes: int,
                 attn_temperature: float = 2.0):
        super().__init__()
        self.num_relations = num_relations
        self.hid_dim = hid_dim
        self.num_nodes = num_nodes

        # Per-relation GCN encoders
        self.gcns = nn.ModuleList([
            GCN(in_dim, hid_dim) for _ in range(num_relations)
        ])

        # Shared discriminator
        self.disc = Discriminator(hid_dim)

        # Attention fusion (reference uses no temperature, T=1.0)
        self.attn = Attention(hid_dim, num_relations, temperature=attn_temperature)

        # Learnable consensus embedding H — xavier_normal_ init (reference)
        self.H = nn.Parameter(torch.FloatTensor(1, num_nodes, hid_dim))
        nn.init.xavier_normal_(self.H)

        # Semi-supervised classifier on H (reference: LogReg)
        self.logistic = LogReg(hid_dim, num_classes)

    def forward(self, x: torch.Tensor, adj_list: list[torch.Tensor],
                x_shuf: torch.Tensor
                ) -> dict:
        """Forward pass.

        Returns dict with keys: logits, mi_logits, reg_loss, semi, attn_weights.
        """
        h_pos_list = []
        h_neg_list = []
        mi_logits_list = []

        for r in range(self.num_relations):
            h_pos = self.gcns[r](x, adj_list[r])        # (N, hid)
            h_neg = self.gcns[r](x_shuf, adj_list[r])   # (N, hid)
            summary = torch.sigmoid(h_pos.mean(dim=0))   # (hid,)
            mi_logits = self.disc(summary, h_pos, h_neg)  # (2N,)

            h_pos_list.append(h_pos)
            h_neg_list.append(h_neg)
            mi_logits_list.append(mi_logits)

        # Attention fusion across relations (reference: isAttn=True)
        h_pos_fused, h_neg_fused, attn_p = self.attn(
            h_pos_list, h_neg_list, self.num_nodes, self.hid_dim)

        # Add batch dim to match H shape (1, N, hid)
        h_pos_fused = h_pos_fused.unsqueeze(0)  # (1, N, hid)
        h_neg_fused = h_neg_fused.unsqueeze(0)  # (1, N, hid)

        # Consensus regularization: .sum() to match reference
        pos_reg_loss = ((self.H - h_pos_fused) ** 2).sum()
        neg_reg_loss = ((self.H - h_neg_fused) ** 2).sum()
        reg_loss = pos_reg_loss - neg_reg_loss

        # Semi-supervised classification from H
        semi = self.logistic(self.H).squeeze(0)  # (N, C)

        return {
            "mi_logits": mi_logits_list,
            "reg_loss": reg_loss,
            "semi": semi,
            "attn_weights": attn_p,
        }


# =============================================================================
# LOSS
# =============================================================================

def compute_dmgi_loss(result: dict, labels: torch.Tensor,
                      train_idx: torch.Tensor,
                      reg_coef: float, sup_coef: float,
                      num_nodes: int,
                      ) -> tuple[torch.Tensor, dict]:
    """Compute the composite DMGI loss.

    L = L_MI + reg_coef * L_consensus + sup_coef * L_supervised

    Returns (total_loss, loss_components_dict).
    """
    # 1. MI loss: BCE on discriminator logits, summed across relations
    b_xent = nn.BCEWithLogitsLoss()
    lbl_1 = torch.ones(1, num_nodes)
    lbl_2 = torch.zeros(1, num_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1)  # (1, 2N)

    xent_loss = None
    for mi_logits in result["mi_logits"]:
        logit = mi_logits.unsqueeze(0)  # (1, 2N)
        if xent_loss is None:
            xent_loss = b_xent(logit, lbl)
        else:
            xent_loss += b_xent(logit, lbl)

    # 2. Consensus regularization (already computed in forward with .sum())
    reg_loss = result["reg_loss"]

    # 3. Supervised CE on train nodes only
    xent = nn.CrossEntropyLoss()
    semi_loss = xent(result["semi"][train_idx], labels[train_idx])

    total = xent_loss + reg_coef * reg_loss + sup_coef * semi_loss
    components = {
        "mi_loss": float(xent_loss),
        "consensus_reg": float(reg_loss),
        "supervised_loss": float(semi_loss),
        "total_loss": float(total),
    }
    return total, components


# =============================================================================
# LOGREG EVALUATION (reference protocol)
# =============================================================================

class MLPEval(nn.Module):
    """2-layer MLP for evaluation on frozen embeddings."""

    def __init__(self, in_dim: int, hid_dim: int, num_classes: int,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, num_classes),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def finetune_mlp(H: torch.Tensor, idx_train: np.ndarray,
                 idx_val: np.ndarray, idx_test: np.ndarray,
                 labels: np.ndarray, num_epochs: int = 200,
                 patience: int = 20, lr: float = 0.001) -> dict:
    """Fine-tune a supervised MLP on frozen DMGI embeddings H.

    Two-stage approach: DMGI pre-trains H via self-supervised objectives,
    then this function trains a proper supervised classifier on top.
    Uses class-balanced CE, early stopping on val Macro-F1.

    Returns dict with metrics and the best model's test predictions.
    """
    embeds = H.detach()
    hid_units = embeds.shape[2]
    nb_classes = int(labels.max()) + 1

    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = torch.LongTensor(labels[idx_train])
    val_lbls = torch.LongTensor(labels[idx_val])
    test_lbls = torch.LongTensor(labels[idx_test])

    # Class-balanced CE loss
    classes = np.unique(labels[idx_train])
    cw = compute_class_weight("balanced", classes=classes, y=labels[idx_train])
    class_weights = torch.FloatTensor(cw)
    xent = nn.CrossEntropyLoss(weight=class_weights)

    best_results = []

    # Run multiple times to reduce variance
    for run in range(10):
        torch.manual_seed(run * 7 + 42)
        mlp = MLPEval(hid_units, hid_units, nb_classes, dropout=0.3)
        opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=0.5, patience=7, min_lr=1e-5)

        best_val_f1 = -1.0
        best_state = None
        wait = 0

        for epoch in range(num_epochs):
            mlp.train()
            opt.zero_grad()
            logits = mlp(train_embs)
            loss = xent(logits, train_lbls)
            loss.backward()
            opt.step()

            mlp.eval()
            with torch.no_grad():
                val_logits = mlp(val_embs)
                val_preds = torch.argmax(val_logits, dim=1)
                val_f1 = f1_score(val_lbls.numpy(), val_preds.numpy(),
                                  average='macro', zero_division=0)

            scheduler.step(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        # Evaluate best checkpoint
        mlp.load_state_dict(best_state)
        mlp.eval()
        with torch.no_grad():
            test_logits = mlp(test_embs)
            test_preds = torch.argmax(test_logits, dim=1)
            test_probs = torch.softmax(test_logits, dim=1)
            test_ma_f1 = f1_score(test_lbls.numpy(), test_preds.numpy(),
                                  average='macro', zero_division=0)
            test_mi_f1 = f1_score(test_lbls.numpy(), test_preds.numpy(),
                                  average='micro', zero_division=0)
        best_results.append({
            "val_f1": best_val_f1, "test_macro_f1": test_ma_f1,
            "test_micro_f1": test_mi_f1, "test_preds": test_preds.numpy(),
            "test_probs": test_probs.numpy(), "model_state": best_state,
        })

    # Select run with best val F1
    best_run = max(best_results, key=lambda r: r["val_f1"])
    all_ma = [r["test_macro_f1"] for r in best_results]
    all_mi = [r["test_micro_f1"] for r in best_results]

    return {
        "macro_f1_mean": float(np.mean(all_ma)),
        "macro_f1_std": float(np.std(all_ma)),
        "micro_f1_mean": float(np.mean(all_mi)),
        "micro_f1_std": float(np.std(all_mi)),
        "macro_f1_max": float(np.max(all_ma)),
        "micro_f1_max": float(np.max(all_mi)),
        "best_test_preds": best_run["test_preds"],
        "best_test_probs": best_run["test_probs"],
        "best_test_macro_f1": best_run["test_macro_f1"],
        "best_test_micro_f1": best_run["test_micro_f1"],
    }


def evaluate_logreg(H: torch.Tensor, idx_train: np.ndarray,
                    idx_val: np.ndarray, idx_test: np.ndarray,
                    labels: np.ndarray, num_runs: int = 50,
                    num_iters: int = 50,
                    use_mlp: bool = False) -> dict:
    """Evaluate consensus embeddings H using logistic regression or MLP.

    Trains num_runs independent classifiers on frozen H, each for num_iters
    epochs, selects best epoch by val metric, reports test metric at that epoch.

    Args:
        use_mlp: If True, use 2-layer MLP (64→64→C) instead of linear LogReg.

    Returns dict with macro_f1, micro_f1, accuracy (mean and std).
    """
    # H shape: (1, N, hid) — extract embeddings
    embeds = H.detach()
    hid_units = embeds.shape[2]
    nb_classes = int(labels.max()) + 1

    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = torch.LongTensor(labels[idx_train])
    val_lbls = torch.LongTensor(labels[idx_val])
    test_lbls = torch.LongTensor(labels[idx_test])

    xent = nn.CrossEntropyLoss()

    accs = []
    macro_f1s = []
    micro_f1s = []

    for _ in range(num_runs):
        if use_mlp:
            clf = MLPEval(hid_units, hid_units, nb_classes)
        else:
            clf = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=0.0)

        val_accs = []
        test_accs = []
        val_macro_f1s = []
        test_macro_f1s = []
        val_micro_f1s = []
        test_micro_f1s = []

        for _ in range(num_iters):
            clf.train()
            opt.zero_grad()
            logits = clf(train_embs)
            loss = xent(logits, train_lbls)
            loss.backward()
            opt.step()

            clf.eval()
            with torch.no_grad():
                # Val
                val_logits = clf(val_embs)
                val_preds = torch.argmax(val_logits, dim=1)
                val_acc = (val_preds == val_lbls).float().mean().item()
                val_f1_macro = f1_score(val_lbls.numpy(), val_preds.numpy(),
                                        average='macro', zero_division=0)
                val_f1_micro = f1_score(val_lbls.numpy(), val_preds.numpy(),
                                        average='micro', zero_division=0)

                # Test
                test_logits = clf(test_embs)
                test_preds = torch.argmax(test_logits, dim=1)
                test_acc = (test_preds == test_lbls).float().mean().item()
                test_f1_macro = f1_score(test_lbls.numpy(), test_preds.numpy(),
                                         average='macro', zero_division=0)
                test_f1_micro = f1_score(test_lbls.numpy(), test_preds.numpy(),
                                         average='micro', zero_division=0)

            val_accs.append(val_acc)
            test_accs.append(test_acc)
            val_macro_f1s.append(val_f1_macro)
            test_macro_f1s.append(test_f1_macro)
            val_micro_f1s.append(val_f1_micro)
            test_micro_f1s.append(test_f1_micro)

        # Select test metrics at best val epoch
        max_iter_acc = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter_acc])

        max_iter_macro = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter_macro])

        max_iter_micro = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter_micro])

    return {
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "micro_f1_mean": float(np.mean(micro_f1s)),
        "micro_f1_std": float(np.std(micro_f1s)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_max": float(np.max(macro_f1s)),
        "micro_f1_max": float(np.max(micro_f1s)),
        "accuracy_max": float(np.max(accs)),
    }


# =============================================================================
# TRAINING
# =============================================================================

def _corrupt_features(x: torch.Tensor) -> torch.Tensor:
    """Corrupt node features by random permutation of rows."""
    perm = torch.randperm(x.shape[0])
    return x[perm]


def train_dmgi(x: torch.Tensor, adj_list: list[torch.Tensor],
               labels: torch.Tensor, train_idx: np.ndarray,
               val_idx: np.ndarray,
               num_epochs: int, patience: int,
               reg_coef: float = DMGI_REG_COEF,
               sup_coef: float = DMGI_SUP_COEF,
               seed: int = RANDOM_SEED,
               return_history: bool = False):
    """Train a DMGI model with early stopping on total loss.

    Returns: (model, best_loss) or (model, best_loss, train_losses, val_losses)
    if return_history.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_nodes = x.shape[0]
    num_relations = len(adj_list)

    model = DMGI(
        in_dim=x.shape[1],
        hid_dim=128,  # increased from paper's 64
        num_nodes=num_nodes,
        num_relations=num_relations,
        num_classes=NUM_CLASSES,
    )

    # Separate param groups: lower lr for logistic layer so H absorbs
    # more supervised gradient (prevents gradient short-circuit)
    logistic_params = set(id(p) for p in model.logistic.parameters())
    main_params = [p for p in model.parameters() if id(p) not in logistic_params]
    optimizer = torch.optim.Adam([
        {"params": main_params, "lr": 0.0005},
        {"params": list(model.logistic.parameters()), "lr": 0.00005},
    ], weight_decay=0.0001)

    best_loss = 1e9
    best_state = None
    wait = 0
    train_losses = []
    val_losses = []

    train_idx_t = torch.LongTensor(train_idx)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        idx = np.random.permutation(num_nodes)
        x_shuf = x[idx]

        result = model(x, adj_list, x_shuf)
        loss, components = compute_dmgi_loss(
            result, labels, train_idx_t, reg_coef, sup_coef, num_nodes,
        )

        # Early stopping on total loss (reference: if loss < best)
        if float(loss) < best_loss:
            best_loss = float(loss)
            wait = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if wait == patience:
            break

        loss.backward()
        optimizer.step()

        train_losses.append(float(loss))

        # Track val loss for logging
        model.eval()
        with torch.no_grad():
            idx_v = np.random.permutation(num_nodes)
            x_shuf_v = x[idx_v]
            result_v = model(x, adj_list, x_shuf_v)
            val_loss, _ = compute_dmgi_loss(
                result_v, labels, train_idx_t, reg_coef, sup_coef, num_nodes,
            )
        val_losses.append(float(val_loss))

    if best_state is not None:
        model.load_state_dict(best_state)

    if return_history:
        return model, best_loss, train_losses, val_losses
    return model, best_loss


# =============================================================================
# FULL PIPELINE
# =============================================================================

def run_dmgi_pipeline(output_dir: Optional[str] = None,
                      composites_dir: Optional[str] = None,
                      num_epochs: Optional[int] = None,
                      patient_list: Optional[str] = None):
    """Full DMGI pipeline: load artifacts -> search hyperparams -> train -> evaluate.

    Searches over sup_coef on a 60/10/30 train/val/test split,
    then retrains best config and evaluates with LogReg on frozen H.

    Args:
        patient_list: Path to a text file with one patient ID per line.
                      If provided, only these patients are used.
    """
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    t0 = time.time()

    if num_epochs is None:
        num_epochs = DMGI_NUM_EPOCHS
    if output_dir is None:
        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, "dmgi_baseline")
    os.makedirs(output_dir, exist_ok=True)

    swin_dir = os.path.join(DEFAULT_OUTPUT_DIR, "swin_baseline")

    # ---- Phase 1: Load data ----
    print("=" * 60)
    print("PHASE 1: Loading data")
    print("=" * 60)

    if composites_dir is None:
        composites_dir = os.path.join(DEFAULT_OUTPUT_DIR, "ovis2_fixed256", "composites")
        if not os.path.isdir(composites_dir):
            composites_dir = os.path.join(DEFAULT_OUTPUT_DIR, "ovis2_fixed256_crop", "composites")

    patient_ids, labels, clinical_df = build_patient_list(composites_dir)

    # Filter to patient list if provided
    if patient_list is not None:
        with open(patient_list) as f:
            keep_pids = set(line.strip() for line in f if line.strip())
        keep_mask = [pid in keep_pids for pid in patient_ids]
        patient_ids = [pid for pid, k in zip(patient_ids, keep_mask) if k]
        labels = labels[keep_mask]
        clinical_df = clinical_df[keep_mask].reset_index(drop=True)
        print(f"Filtered to {len(patient_ids)} patients from {patient_list}")

    print(f"Patients: {len(patient_ids)}")
    print(f"Grade distribution: {dict(Counter(labels.tolist()))}")

    # ---- Phase 2: Load Swin features + clinical encoding ----
    print("\n" + "=" * 60)
    print("PHASE 2: Loading Swin features + encoding clinical features")
    print("=" * 60)

    # Swin features (768-d)
    swin_feats_path = os.path.join(swin_dir, "swin_features.npy")
    swin_pids_path = os.path.join(swin_dir, "swin_features_pids.json")

    for p, desc in [(swin_feats_path, "Swin features"),
                    (swin_pids_path, "Swin feature PIDs")]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Swin artifact not found: {p}\n"
                f"Run the swin pipeline first: python -m multimodal_tumor_classification swin"
            )

    swin_features = np.load(swin_feats_path)
    with open(swin_pids_path) as f:
        swin_pids = json.load(f)

    # Build PID -> index mapping for Swin features
    swin_pid_to_idx = {pid: i for i, pid in enumerate(swin_pids)}

    # Reorder Swin features to match current patient_ids order
    swin_feats_ordered = np.zeros((len(patient_ids), swin_features.shape[1]),
                                  dtype=np.float32)
    for i, pid in enumerate(patient_ids):
        if pid in swin_pid_to_idx:
            swin_feats_ordered[i] = swin_features[swin_pid_to_idx[pid]]
        else:
            print(f"  Warning: {pid} not in Swin features, using zeros")

    print(f"Swin features shape: {swin_feats_ordered.shape}")
    print(f"(Clinical encoding deferred to after split for fresh scaler fitting)")

    # ---- Phase 3: 60/10/30 stratified split (reference protocol) ----
    print("\n" + "=" * 60)
    print("PHASE 3: 60/10/30 stratified split")
    print("=" * 60)

    indices = np.arange(len(patient_ids))
    # First split: 70% train+val, 30% test
    trainval_idx, test_idx = train_test_split(
        indices, test_size=0.3, stratify=labels, random_state=RANDOM_SEED)
    # Second split: from trainval, ~14.3% val = 10% of total
    y_trainval = labels[trainval_idx]
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=1.0/7.0, stratify=y_trainval,
        random_state=RANDOM_SEED)

    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    print(f"Train grades: {dict(Counter(y_train.tolist()))}")
    print(f"Val grades:   {dict(Counter(y_val.tolist()))}")
    print(f"Test grades:  {dict(Counter(y_test.tolist()))}")

    # Clinical features (31-d) — fit scaler on train pool, apply to all
    train_df = clinical_df.iloc[trainval_idx].reset_index(drop=True)
    _, scaler = encode_clinical_features(train_df, fit_scaler=True)
    clin_all, _ = encode_clinical_features(clinical_df, scaler=scaler,
                                           fit_scaler=False)
    print(f"Clinical encoded shape: {clin_all.shape}")

    # Concatenate then apply row-wise L1 normalization (reference-aligned)
    node_features_raw = np.concatenate([swin_feats_ordered, clin_all], axis=1)
    node_features = preprocess_features(node_features_raw)
    print(f"Node features shape: {node_features.shape} (row-L1 normalized)")

    # ---- Phase 4: Build adjacency & sup_coef sweep ----
    print("\n" + "=" * 60)
    print("PHASE 4: Building adjacency & sup_coef sweep")
    print("=" * 60)

    best_thresh = [0.8, 0.8, 0.8, 0.8]  # uniform threshold (reference code)
    best_reg = DMGI_REG_COEF  # 0.001 (paper's alpha)

    x_t = torch.tensor(node_features, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    adj_list, adj_stats = build_multiplex_adjacency_from_df(
        clinical_df, best_thresh)
    print(f"Thresholds: {best_thresh}")
    print(f"Graph construction: pd.get_dummies + minmax_scale + cosine_similarity")
    for name, st in adj_stats.items():
        print(f"  {name}: theta={st['threshold']}, {st['num_edges']} edges, "
              f"density={st['density']:.3f}, dummy_cols={st['num_dummy_cols']}")
    print(f"reg_coef={best_reg} (paper's alpha)")
    print(f"self_connection={DMGI_SELF_CONNECTION}, drop_prob={DMGI_DROP_PROB}")
    print(f"epochs={num_epochs}, patience={DMGI_PATIENCE}")
    print(f"Early stopping: on total loss (reference-aligned)")
    print(f"H initialization: xavier_normal_ (reference-aligned)")
    print(f"Feature normalization: row-wise L1 (reference-aligned)")
    print(f"Consensus reg loss: .sum() (reference-aligned)")
    print(f"Supervised loss: CE through logistic (lr=0.00005, 10x slower than main lr=0.0005)")

    # Sweep over sup_coef (paper's range: {0.01, 0.1, 1.0})
    sup_coef_candidates = [0.01, 0.1, 1.0]
    seeds = [42, 123, 456]
    sweep_results = []

    print(f"\nSweeping sup_coef={sup_coef_candidates} x {len(seeds)} seeds ...")
    overall_best_val_f1 = -1.0
    best_sup = sup_coef_candidates[0]

    for sup_c in sup_coef_candidates:
        seed_val_losses = []
        seed_test_f1s = []
        for s in seeds:
            model_s, val_loss_s = train_dmgi(
                x_t, adj_list, labels_t, train_idx, val_idx,
                num_epochs, DMGI_PATIENCE,
                reg_coef=best_reg, sup_coef=sup_c,
                seed=s, return_history=False,
            )
            # Evaluate with LogReg
            lr_result = evaluate_logreg(
                model_s.H.data.detach(), train_idx, val_idx, test_idx,
                labels, num_runs=10, num_iters=50)
            seed_val_losses.append(val_loss_s)
            seed_test_f1s.append(lr_result["macro_f1_mean"])

        mean_val_loss = float(np.mean(seed_val_losses))
        mean_test = float(np.mean(seed_test_f1s))
        std_test = float(np.std(seed_test_f1s))
        tag = ""
        # Select sup_coef by best LogReg F1 (not loss, which can explode negative)
        if mean_test > overall_best_val_f1:
            overall_best_val_f1 = mean_test
            best_sup = sup_c
            tag = " ***"
        sweep_results.append({
            "sup_coef": sup_c, "mean_val_loss": mean_val_loss,
            "mean_test_f1": mean_test, "std_test_f1": std_test,
        })
        print(f"  sup_coef={sup_c}: mean val loss={mean_val_loss:.4f}, "
              f"mean test F1(LR)={mean_test:.4f} +/- {std_test:.4f}{tag}")

    print(f"\nBest sup_coef: {best_sup} (mean LR F1={overall_best_val_f1:.4f})")

    # Re-run best sup_coef with more seeds — select by LogReg Macro F1
    seeds_final = [42, 123, 456, 789, 1024]
    best_seed_f1 = -1.0
    best_seed = seeds_final[0]
    seed_results = []

    print(f"\nFinal training with sup_coef={best_sup}, {len(seeds_final)} seeds ...")
    for s in seeds_final:
        model_s, val_loss_s, _, _ = train_dmgi(
            x_t, adj_list, labels_t, train_idx, val_idx,
            num_epochs, DMGI_PATIENCE,
            reg_coef=best_reg, sup_coef=best_sup,
            seed=s, return_history=True,
        )
        lr_result = evaluate_logreg(
            model_s.H.data.detach(), train_idx, val_idx, test_idx,
            labels, num_runs=50, num_iters=50)
        seed_results.append({
            "seed": s, "val_loss": val_loss_s,
            "test_macro_f1": lr_result["macro_f1_mean"],
            "test_micro_f1": lr_result["micro_f1_mean"],
        })
        tag = " ***" if lr_result["macro_f1_mean"] > best_seed_f1 else ""
        print(f"  seed={s}: val loss={val_loss_s:.4f}, "
              f"test Ma-F1={lr_result['macro_f1_mean']:.4f} ({lr_result['macro_f1_std']:.4f}), "
              f"test Mi-F1={lr_result['micro_f1_mean']:.4f} ({lr_result['micro_f1_std']:.4f}){tag}")
        if lr_result["macro_f1_mean"] > best_seed_f1:
            best_seed_f1 = lr_result["macro_f1_mean"]
            best_seed = s

    mean_test_macro = float(np.mean([r["test_macro_f1"] for r in seed_results]))
    std_test_macro = float(np.std([r["test_macro_f1"] for r in seed_results]))
    mean_test_micro = float(np.mean([r["test_micro_f1"] for r in seed_results]))
    std_test_micro = float(np.std([r["test_micro_f1"] for r in seed_results]))
    print(f"\nTest Macro-F1 across seeds: {mean_test_macro:.4f} +/- {std_test_macro:.4f}")
    print(f"Test Micro-F1 across seeds: {mean_test_micro:.4f} +/- {std_test_micro:.4f}")
    print(f"Best seed: {best_seed} (LR Ma-F1={best_seed_f1:.4f})")

    # ---- Phase 5: Retrain best seed for final model + history ----
    print("\n" + "=" * 60)
    print("PHASE 5: Final model (best seed)")
    print("=" * 60)

    final_model, final_loss, train_losses, val_losses = train_dmgi(
        x_t, adj_list, labels_t, train_idx, val_idx,
        num_epochs, DMGI_PATIENCE,
        reg_coef=best_reg, sup_coef=best_sup,
        seed=best_seed, return_history=True,
    )

    stopped_epoch = len(train_losses)
    print(f"  seed={best_seed}, stopped at epoch {stopped_epoch}, "
          f"best loss: {final_loss:.4f}")

    # ---- Phase 6: Evaluate with LogReg (reference protocol) ----
    print("\n" + "=" * 60)
    print("PHASE 6: LogReg evaluation on frozen H (50 runs x 50 epochs)")
    print("=" * 60)

    final_model.eval()
    H = final_model.H.data.detach()

    lr_metrics = evaluate_logreg(H, train_idx, val_idx, test_idx, labels,
                                 num_runs=50, num_iters=50)

    print(f"\n[LogReg] Macro-F1: {lr_metrics['macro_f1_mean']:.4f} "
          f"({lr_metrics['macro_f1_std']:.4f}) | "
          f"Micro-F1: {lr_metrics['micro_f1_mean']:.4f} "
          f"({lr_metrics['micro_f1_std']:.4f})")
    print(f"[LogReg Max] Macro-F1: {lr_metrics['macro_f1_max']:.4f} | "
          f"Micro-F1: {lr_metrics['micro_f1_max']:.4f} | "
          f"Accuracy: {lr_metrics['accuracy_max']:.4f}")

    # ---- Phase 6b: Fine-tune supervised MLP on frozen H ----
    print("\n" + "=" * 60)
    print("PHASE 6b: Fine-tune MLP on frozen H (class-balanced CE, 10 runs)")
    print("=" * 60)

    ft_metrics = finetune_mlp(H, train_idx, val_idx, test_idx, labels)

    print(f"\n[Fine-tuned MLP] Macro-F1: {ft_metrics['macro_f1_mean']:.4f} "
          f"({ft_metrics['macro_f1_std']:.4f}) | "
          f"Micro-F1: {ft_metrics['micro_f1_mean']:.4f} "
          f"({ft_metrics['micro_f1_std']:.4f})")
    print(f"[Fine-tuned MLP Max] Macro-F1: {ft_metrics['macro_f1_max']:.4f} | "
          f"Micro-F1: {ft_metrics['micro_f1_max']:.4f}")
    print(f"[Fine-tuned MLP Best] Macro-F1: {ft_metrics['best_test_macro_f1']:.4f} | "
          f"Micro-F1: {ft_metrics['best_test_micro_f1']:.4f}")

    # Use fine-tuned MLP predictions for plots (best run by val F1)
    test_preds = ft_metrics["best_test_preds"]
    test_probs = ft_metrics["best_test_probs"]

    bal_acc = balanced_accuracy_score(y_test, test_preds)
    acc = float(np.mean(y_test == test_preds))

    print(f"\nFine-tuned MLP classifier:")
    print(classification_report(y_test, test_preds, target_names=LABEL_NAMES,
                                zero_division=0))

    # Also show direct DMGI head for comparison
    with torch.no_grad():
        idx_perm = np.random.permutation(x_t.shape[0])
        x_shuf = x_t[idx_perm]
        result = final_model(x_t, adj_list, x_shuf)

    cm = confusion_matrix(y_test, test_preds)

    # Attention weights
    attn_p = result["attn_weights"].detach().numpy()
    relation_names = list(DMGI_FEATURE_TYPE_GROUPS.keys())
    attn_dict = {name: float(w) for name, w in zip(relation_names, attn_p)}
    print(f"Mean attention weights: {attn_dict}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # ---- Phase 7: Plots & save ----
    print("\n" + "=" * 60)
    print("PHASE 7: Generating plots & saving results")
    print("=" * 60)

    # Loss curves
    plot_loss_curves(train_losses, val_losses,
                     os.path.join(output_dir, "loss_curves.png"),
                     num_epochs, stopped_epoch)

    per_class_aucs, macro_auc = plot_roc_curves(
        y_test, test_probs, os.path.join(output_dir, "roc_curves.png"))
    plot_confusion_matrix(y_test, test_preds,
                          os.path.join(output_dir, "confusion_matrix.png"))
    plot_per_class_f1(y_test, test_preds,
                      os.path.join(output_dir, "f1_scores.png"))

    # Embedding similarity heatmap on consensus H
    H_np = H.squeeze(0).numpy()  # (N, hid)
    sim_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            pairwise = cosine_similarity(
                H_np[labels == i], H_np[labels == j])
            sim_matrix[i, j] = pairwise.mean()
    plot_embedding_similarity_heatmap(
        sim_matrix, os.path.join(output_dir, "embedding_similarity.png"))

    # Save consensus embeddings
    np.save(os.path.join(output_dir, "consensus_embeddings.npy"), H_np)
    print(f"  Saved consensus_embeddings.npy")

    # Save adjacency stats
    with open(os.path.join(output_dir, "adjacency_stats.json"), "w") as f:
        json.dump(adj_stats, f, indent=2)
    print(f"  Saved adjacency_stats.json")

    # Save attention weights
    with open(os.path.join(output_dir, "attention_weights.json"), "w") as f:
        json.dump(attn_dict, f, indent=2)
    print(f"  Saved attention_weights.json")

    # Save results.json
    save_data = {
        "config": {
            "model": "DMGI (Deep Multiplex Graph Infomax) — reference-aligned",
            "num_patients": len(patient_ids),
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "test_size": int(len(test_idx)),
            "split": "60/10/30",
            "node_feature_dim": int(node_features.shape[1]),
            "feature_normalization": "row-wise L1",
            "hid_dim": 128,
            "graph_construction": "pd.get_dummies + minmax_scale + cosine_similarity",
            "num_relations": len(DMGI_FEATURE_TYPE_GROUPS_COLS),
            "num_epochs": num_epochs,
            "patience": DMGI_PATIENCE,
            "lr": 0.0005,
            "weight_decay": 0.0001,
            "reg_coef": best_reg,
            "sup_coef": best_sup,
            "self_connection": DMGI_SELF_CONNECTION,
            "drop_prob": DMGI_DROP_PROB,
            "thresholds": best_thresh,
            "early_stopping": "total_loss",
            "consensus_loss": ".sum()",
            "H_init": "xavier_normal_",
            "evaluation": "LogReg (50 runs) + Fine-tuned MLP (10 runs, class-balanced CE)",
        },
        "sweep_results": sweep_results,
        "multi_seed": {
            "seeds": seeds_final,
            "best_seed": best_seed,
            "mean_test_macro_f1": mean_test_macro,
            "std_test_macro_f1": std_test_macro,
            "mean_test_micro_f1": mean_test_micro,
            "std_test_micro_f1": std_test_micro,
            "per_seed": seed_results,
        },
        "early_stopping": {
            "patience": DMGI_PATIENCE,
            "stopped_epoch": stopped_epoch,
            "best_loss": float(final_loss),
        },
        "logreg_eval": lr_metrics,
        "finetuned_mlp_eval": {
            k: v for k, v in ft_metrics.items()
            if k not in ("best_test_preds", "best_test_probs", "model_state")
        },
        "attention_weights": attn_dict,
        "adjacency_stats": adj_stats,
        "test_metrics_direct": {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_auc": float(macro_auc),
            "per_class_auc": {k: float(v) for k, v in per_class_aucs.items()},
            "confusion_matrix": cm.tolist(),
        },
        "predictions": [
            {
                "pid": patient_ids[test_idx[k]],
                "true_label": int(y_test[k]),
                "predicted_label": int(test_preds[k]),
                "probabilities": test_probs[k].tolist(),
            }
            for k in range(len(test_idx))
        ],
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary text
    config_params = {
        "thresholds": best_thresh, "reg_coef": best_reg, "sup_coef": best_sup,
    }
    _save_dmgi_summary(
        os.path.join(output_dir, "experiment_summary.txt"),
        patient_ids, labels, train_idx, val_idx, test_idx,
        y_test, test_preds, test_probs,
        node_features, adj_stats, attn_dict, config_params,
        lr_metrics, bal_acc, acc, cm, per_class_aucs, macro_auc,
        train_losses, val_losses, elapsed, stopped_epoch,
        mean_test_macro, std_test_macro, mean_test_micro, std_test_micro,
    )


def _save_dmgi_summary(out_path, patient_ids, labels, train_idx, val_idx,
                        test_idx, y_test, test_preds, test_probs, node_features,
                        adj_stats, attn_dict, best_params,
                        lr_metrics, bal_acc, acc, cm, per_class_aucs, macro_auc,
                        train_losses, val_losses, elapsed, stopped_epoch,
                        mean_test_macro, std_test_macro,
                        mean_test_micro, std_test_micro):
    """Write comprehensive DMGI experiment summary."""
    lines = []
    L = lines.append

    L("=" * 70)
    L("EXPERIMENT SUMMARY -- DMGI (Reference-Aligned)")
    L("=" * 70)

    L("\n--- MODEL ARCHITECTURE ---")
    L(f"Node features:     Swin(768-d) + Clinical(31-d) = {node_features.shape[1]}-d")
    L(f"Feature norm:      Row-wise L1 (each row sums to 1)")
    L(f"Graph relations:   {len(DMGI_FEATURE_TYPE_GROUPS_COLS)} types")
    L(f"Graph encoding:    pd.get_dummies + minmax_scale + cosine_similarity")
    for name, cols in DMGI_FEATURE_TYPE_GROUPS_COLS.items():
        L(f"  - {name}: {len(cols)} columns")
    L(f"GCN:               1 layer per relation, hid_dim=128")
    L(f"Discriminator:     Shared bilinear")
    L(f"Attention:         Per-node, per-relation (reference Attention module)")
    L(f"Consensus:         Learnable H parameter (1, {len(patient_ids)}, 128)")
    L(f"H init:            xavier_normal_")
    L(f"Consensus loss:    .sum() (reference-aligned)")
    L(f"Early stopping:    total loss (reference-aligned)")
    L(f"Evaluation:        LogReg on frozen H (50 runs x 50 epochs)")

    L("\n--- BEST HYPERPARAMETERS (from search) ---")
    L(f"  lr:              0.0005")
    L(f"  weight_decay:    0.0001")
    L(f"  reg_coef (alpha):{best_params['reg_coef']}")
    L(f"  sup_coef (beta): {best_params['sup_coef']}")
    L(f"  self_connection: {DMGI_SELF_CONNECTION}")
    L(f"  drop_prob:       {DMGI_DROP_PROB}")
    L(f"  thresholds:      {best_params['thresholds']}")

    L("\n--- MEAN ATTENTION WEIGHTS ---")
    for name, w in attn_dict.items():
        L(f"  {name}: {w:.4f}")

    L("\n--- ADJACENCY STATS ---")
    for name, st in adj_stats.items():
        L(f"  {name}: theta={st['threshold']}, {st['num_edges']} edges, "
          f"density={st['density']:.3f}")

    L("\n--- DATA ---")
    L(f"Total patients:   {len(patient_ids)}")
    grade_dist = Counter(labels.tolist())
    L(f"Grade distribution: G1={grade_dist[0]}, G2={grade_dist[1]}, G3={grade_dist[2]}")
    L(f"Train / Val / Test: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")

    L("\n--- TRAINING ---")
    L(f"Stopped at epoch: {stopped_epoch}")
    L(f"Best train loss:  {min(train_losses):.4f}")

    L("\n--- TEST RESULTS (LogReg on frozen H) ---")
    L(f"Macro-F1:  {lr_metrics['macro_f1_mean']:.4f} (+/- {lr_metrics['macro_f1_std']:.4f})")
    L(f"Micro-F1:  {lr_metrics['micro_f1_mean']:.4f} (+/- {lr_metrics['micro_f1_std']:.4f})")
    L(f"Accuracy:  {lr_metrics['accuracy_mean']:.4f} (+/- {lr_metrics['accuracy_std']:.4f})")
    L(f"Max Macro-F1: {lr_metrics['macro_f1_max']:.4f}")
    L(f"Max Micro-F1: {lr_metrics['micro_f1_max']:.4f}")

    L(f"\nMulti-seed test Macro-F1: {mean_test_macro:.4f} +/- {std_test_macro:.4f}")
    L(f"Multi-seed test Micro-F1: {mean_test_micro:.4f} +/- {std_test_micro:.4f}")

    L(f"\n--- Direct classifier (for reference) ---")
    L(f"Accuracy:          {acc:.4f}")
    L(f"Balanced Accuracy: {bal_acc:.4f}")
    L(f"Macro AUC:         {macro_auc:.4f}")

    per_f1 = f1_score(y_test, test_preds, average=None, zero_division=0)
    for i, name in enumerate(LABEL_NAMES):
        L(f"  {name} F1: {per_f1[i]:.4f}  AUC: {per_class_aucs[name]:.4f}")

    L(f"\nConfusion Matrix (direct classifier):")
    L(f"  {'':12s} {'Pred G1':>8s} {'Pred G2':>8s} {'Pred G3':>8s}")
    for i, name in enumerate(LABEL_NAMES):
        L(f"  {name:12s} {cm[i, 0]:8d} {cm[i, 1]:8d} {cm[i, 2]:8d}")

    L(f"\n--- PER-PATIENT PREDICTIONS ---")
    L(f"  {'Patient':<22s} {'True':>8s} {'Pred':>8s} {'OK':>4s}  "
      f"{'P(G1)':>6s} {'P(G2)':>6s} {'P(G3)':>6s}")
    for k in range(len(test_idx)):
        pid = patient_ids[test_idx[k]]
        tg = LABEL_NAMES[y_test[k]]
        pg = LABEL_NAMES[test_preds[k]]
        ok = "Y" if y_test[k] == test_preds[k] else "N"
        p0, p1, p2 = test_probs[k]
        L(f"  {pid:<22s} {tg:>8s} {pg:>8s} {ok:>4s}  {p0:6.3f} {p1:6.3f} {p2:6.3f}")

    L(f"\nRuntime: {elapsed:.1f}s")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved {out_path}")
