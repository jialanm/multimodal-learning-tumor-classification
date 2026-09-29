"""Graph-Augmented Supervised Learning Pipeline.

Combines end-to-end supervised training (Swin+MLP) with DMGI's multiplex
patient-similarity graphs as a smoothness regularizer.

Loss: L = CE(logits[train_idx], labels[train_idx]) + lambda * L_smooth(Z, adj_list)
Where L_smooth = (1/R) * sum_r ||Z - A_r @ Z||^2 / N_valid

Training is full-batch (N=899, feasible since Swin features are pre-extracted).
CE on train nodes only; smoothness on all nodes (transductive — no label leakage
since graphs are built from clinical features).
"""

import os
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, f1_score, balanced_accuracy_score, confusion_matrix,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from itertools import product
from typing import Optional

from .config import (
    LABEL_COLUMN, LABEL_MAP, LABEL_NAMES, NUM_CLASSES,
    RANDOM_SEED, DEVICE, DEFAULT_OUTPUT_DIR,
    GRAPH_AUG_NUM_EPOCHS, GRAPH_AUG_PATIENCE,
    GRAPH_AUG_THRESHOLDS,
    DMGI_FEATURE_TYPE_GROUPS_COLS,
    SWIN_NUM_CV_FOLDS, SWIN_PATIENCE,
)
from .clinical import load_clinical_dataframe, encode_clinical_features
from .swin_pipeline import (
    MultimodalClassifier, build_patient_list, extract_swin_features,
    _make_loader,
)
from .evaluation import (
    plot_loss_curves, plot_roc_curves,
    plot_confusion_matrix, plot_per_class_f1, plot_embedding_similarity_heatmap,
)


# =============================================================================
# GRAPH CONSTRUCTION (smoothness variant — no self-loops, row-normalized)
# =============================================================================

def build_smoothness_adjacency(clinical_df: pd.DataFrame,
                               col_names: list[str],
                               threshold: float) -> torch.Tensor:
    """Build row-normalized adjacency for graph smoothness regularization.

    Same graph construction as DMGI (get_dummies + minmax_scale + cosine_similarity
    + threshold) but:
      - No self-loops (self-loops would make smoothness trivially small)
      - Row-normalized: A_smooth = D^{-1} A (each row sums to 1)

    Returns dense float32 tensor of shape (N, N).
    """
    subset_df = clinical_df[col_names].copy()
    dummied = pd.get_dummies(subset_df, columns=col_names).fillna(0)
    features = dummied.values.astype(np.float64)
    features = minmax_scale(features, axis=0, copy=False)

    sim = cosine_similarity(features)
    adj = (sim >= threshold).astype(np.float32)
    np.fill_diagonal(adj, 0.0)  # no self-loops

    # Row-normalize: D^{-1} A
    rowsum = adj.sum(axis=1)
    with np.errstate(divide="ignore"):
        d_inv = np.where(rowsum > 0, 1.0 / rowsum, 0.0)
    adj_norm = adj * d_inv[:, np.newaxis]

    return torch.tensor(adj_norm, dtype=torch.float32)


def build_multiplex_smoothness_adjacency(
        clinical_df: pd.DataFrame,
        thresholds: list[float],
) -> tuple[list[torch.Tensor], dict]:
    """Build smoothness adjacency matrices for all R=4 clinical relation types.

    Returns (list of dense adj tensors, stats dict).
    """
    groups = list(DMGI_FEATURE_TYPE_GROUPS_COLS.items())
    if len(thresholds) != len(groups):
        raise ValueError(
            f"Expected {len(groups)} thresholds, got {len(thresholds)}")

    adj_list = []
    stats = {}
    for (name, col_names), thresh in zip(groups, thresholds):
        adj = build_smoothness_adjacency(clinical_df, col_names, thresh)
        n = adj.shape[0]

        # Stats on raw adjacency (before normalization)
        subset_df = clinical_df[col_names].copy()
        dummied = pd.get_dummies(subset_df, columns=col_names).fillna(0)
        feats = minmax_scale(dummied.values.astype(np.float64), axis=0)
        sim = cosine_similarity(feats)
        raw_adj = (sim >= thresh).astype(np.float32)
        np.fill_diagonal(raw_adj, 0.0)
        n_edges = int(raw_adj.sum()) // 2
        density = raw_adj.sum() / (n * (n - 1)) if n > 1 else 0.0
        n_isolated = int((raw_adj.sum(axis=1) == 0).sum())

        stats[name] = {
            "num_features": len(col_names),
            "num_dummy_cols": dummied.shape[1],
            "threshold": float(thresh),
            "num_edges": n_edges,
            "density": float(density),
            "num_isolated": n_isolated,
        }
        adj_list.append(adj)

    return adj_list, stats


# =============================================================================
# GRAPH SMOOTHNESS LOSS
# =============================================================================

def compute_graph_smoothness_loss(Z: torch.Tensor,
                                  adj_list: list[torch.Tensor],
                                  normalize: bool = False) -> torch.Tensor:
    """Graph smoothness loss over multiplex adjacency.

    L = (1/R) * sum_r ||Z - A_r @ Z||^2 / N_valid_r

    Masks out isolated nodes (no neighbors in relation r) to avoid
    penalizing embedding magnitude for disconnected nodes.

    Args:
        Z: (N, D) fused embeddings
        adj_list: list of R dense (N, N) row-normalized adjacency matrices
        normalize: if True, L2-normalize Z before computing loss
    Returns:
        Scalar loss tensor
    """
    if normalize:
        Z = torch.nn.functional.normalize(Z, p=2, dim=-1)

    R = len(adj_list)
    loss = torch.tensor(0.0, device=Z.device)

    for A_r in adj_list:
        A_r = A_r.to(Z.device)
        neighbor_mean = A_r @ Z  # (N, D) — weighted average of neighbors
        diff = Z - neighbor_mean  # (N, D)

        # Mask: only nodes that have at least one neighbor
        has_neighbor = (A_r.sum(dim=1) > 0).float()  # (N,)
        n_valid = has_neighbor.sum()

        if n_valid > 0:
            per_node_sq = (diff ** 2).sum(dim=1)  # (N,)
            loss = loss + (per_node_sq * has_neighbor).sum() / n_valid

    return loss / R


# =============================================================================
# TRAINING (full-batch, graph-augmented)
# =============================================================================

def train_graph_augmented(img_feats: np.ndarray, clin_feats: np.ndarray,
                          labels: np.ndarray,
                          train_idx: np.ndarray, val_idx: np.ndarray,
                          adj_list: list[torch.Tensor],
                          hparams: dict,
                          lam: float,
                          num_epochs: int,
                          patience: int = GRAPH_AUG_PATIENCE,
                          device: str = "cpu",
                          return_history: bool = False):
    """Train MultimodalClassifier with graph smoothness regularization.

    Full-batch forward on all N nodes:
      - CE loss on train_idx only
      - Smoothness loss on all nodes (transductive)

    Args:
        lam: lambda coefficient for smoothness loss (0.0 = pure Swin+MLP)
        return_history: if True, also return (train_losses, val_losses, stopped_epoch)

    Returns:
        (val_macro_f1, model, val_preds) or
        (val_macro_f1, model, val_preds, train_losses, val_losses, stopped_epoch)
    """
    torch.manual_seed(RANDOM_SEED)

    N = img_feats.shape[0]
    img_t = torch.tensor(img_feats, dtype=torch.float32).to(device)
    clin_t = torch.tensor(clin_feats, dtype=torch.float32).to(device)
    labels_t = torch.tensor(labels, dtype=torch.long).to(device)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long).to(device)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long).to(device)

    model = MultimodalClassifier(
        img_dim=img_feats.shape[1], clinical_dim=clin_feats.shape[1],
        proj_dim=hparams["proj_dim"], hidden_dim=hparams["hidden_dim"],
        dropout_rate=hparams["dropout"],
    ).to(device)

    if hparams["class_weight"] == "balanced":
        cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES),
                                  y=labels[train_idx])
        weight = torch.tensor(cw, dtype=torch.float32).to(device)
    else:
        weight = None
    ce_criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"],
                                 weight_decay=hparams["weight_decay"])

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    train_losses = []
    val_losses = []
    stopped_epoch = num_epochs

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Full-batch forward
        logits = model(img_t, clin_t)  # (N, C)

        # CE on train nodes
        ce_loss = ce_criterion(logits[train_idx_t], labels_t[train_idx_t])

        # Smoothness on fused embeddings (all nodes)
        if lam > 0:
            Z = model.get_fused_embedding(img_t, clin_t)  # (N, 2*proj_dim)
            smooth_loss = compute_graph_smoothness_loss(Z, adj_list)
            total_loss = ce_loss + lam * smooth_loss
        else:
            total_loss = ce_loss

        total_loss.backward()
        optimizer.step()
        train_losses.append(float(total_loss))

        # Validation loss (CE only, for early stopping)
        model.eval()
        with torch.no_grad():
            logits_eval = model(img_t, clin_t)
            val_ce = ce_criterion(logits_eval[val_idx_t], labels_t[val_idx_t])
        val_losses.append(float(val_ce))

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                stopped_epoch = epoch + 1
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on val set
    model.eval()
    with torch.no_grad():
        logits_final = model(img_t, clin_t)
        val_preds = logits_final[val_idx_t].argmax(dim=1).cpu().numpy()

    val_f1 = f1_score(labels[val_idx], val_preds, average="macro", zero_division=0)

    if return_history:
        return val_f1, model, val_preds, train_losses, val_losses, stopped_epoch
    return val_f1, model, val_preds


# =============================================================================
# CROSS-VALIDATION (graph-augmented)
# =============================================================================

def cross_validate_graph_augmented(img_feats: np.ndarray, clin_feats: np.ndarray,
                                   labels: np.ndarray,
                                   pool_idx: np.ndarray,
                                   adj_list: list[torch.Tensor],
                                   hparams: dict, lam: float,
                                   num_epochs: int, patience: int,
                                   device: str = "cpu",
                                   n_splits: int = SWIN_NUM_CV_FOLDS):
    """K-fold stratified CV for graph-augmented pipeline.

    Transductive: all N patients are forwarded through the model (and used for
    smoothness), but CE is computed only on fold-train indices within pool_idx.

    Args:
        img_feats, clin_feats, labels: full dataset (N patients)
        pool_idx: indices into the full dataset to split into K folds

    Returns (mean_f1, std_f1).
    """
    pool_labels = labels[pool_idx]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_f1s = []
    for tr_local, va_local in skf.split(pool_idx, pool_labels):
        # Map local fold indices back to global indices
        tr_global = pool_idx[tr_local]
        va_global = pool_idx[va_local]
        f1, _, _ = train_graph_augmented(
            img_feats, clin_feats, labels,
            tr_global, va_global, adj_list, hparams, lam,
            num_epochs, patience, device,
        )
        fold_f1s.append(f1)
    return float(np.mean(fold_f1s)), float(np.std(fold_f1s))


# =============================================================================
# TRAINING (hybrid: mini-batch CE + periodic graph smoothness)
# =============================================================================

def train_hybrid(img_feats: np.ndarray, clin_feats: np.ndarray,
                 labels: np.ndarray,
                 train_idx: np.ndarray, val_idx: np.ndarray,
                 adj_list: list[torch.Tensor],
                 hparams: dict,
                 lam: float,
                 num_epochs: int,
                 patience: int = GRAPH_AUG_PATIENCE,
                 device: str = "cpu",
                 return_history: bool = False):
    """Train MultimodalClassifier with mini-batch CE + full-batch graph smoothness.

    Each epoch has two phases:
      Phase A: Mini-batch CE with WeightedRandomSampler (identical to Swin)
      Phase B: Full-batch graph smoothness forward (if lam > 0)

    Same optimizer for both phases. Early stopping on val CE loss.

    Returns same signature as train_graph_augmented.
    """
    torch.manual_seed(RANDOM_SEED)

    N = img_feats.shape[0]
    # Full tensors for smoothness phase and val evaluation
    img_all = torch.tensor(img_feats, dtype=torch.float32).to(device)
    clin_all = torch.tensor(clin_feats, dtype=torch.float32).to(device)
    labels_t = torch.tensor(labels, dtype=torch.long).to(device)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long).to(device)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long).to(device)

    model = MultimodalClassifier(
        img_dim=img_feats.shape[1], clinical_dim=clin_feats.shape[1],
        proj_dim=hparams["proj_dim"], hidden_dim=hparams["hidden_dim"],
        dropout_rate=hparams["dropout"],
    ).to(device)

    if hparams["class_weight"] == "balanced":
        cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES),
                                  y=labels[train_idx])
        weight = torch.tensor(cw, dtype=torch.float32).to(device)
    else:
        weight = None
    ce_criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"],
                                 weight_decay=hparams["weight_decay"])

    # Mini-batch DataLoader for train (with WeightedRandomSampler)
    train_loader = _make_loader(
        img_feats[train_idx], clin_feats[train_idx], labels[train_idx],
        hparams["batch_size"], shuffle=True,
    )

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    train_losses = []
    val_losses = []
    stopped_epoch = num_epochs

    for epoch in range(num_epochs):
        # === Phase A: Mini-batch CE ===
        model.train()
        epoch_ce_loss, n_samples = 0.0, 0
        for img_b, clin_b, y_b in train_loader:
            img_b, clin_b, y_b = img_b.to(device), clin_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = ce_criterion(model(img_b, clin_b), y_b)
            loss.backward()
            optimizer.step()
            epoch_ce_loss += loss.item() * len(y_b)
            n_samples += len(y_b)

        # === Phase B: Combined full-batch CE + graph smoothness ===
        # One optimizer step with both CE and smoothness gradients to avoid
        # overwriting Phase A progress with smoothness-only gradients.
        if lam > 0:
            model.train()
            optimizer.zero_grad()
            logits_all = model(img_all, clin_all)
            ce_full = ce_criterion(logits_all[train_idx_t], labels_t[train_idx_t])
            Z = model.get_fused_embedding(img_all, clin_all)
            smooth_loss = lam * compute_graph_smoothness_loss(
                Z, adj_list, normalize=True)
            (ce_full + smooth_loss).backward()
            optimizer.step()
            train_losses.append(epoch_ce_loss / n_samples + smooth_loss.item())
        else:
            train_losses.append(epoch_ce_loss / n_samples)

        # === Validation CE loss (for early stopping) ===
        model.eval()
        with torch.no_grad():
            logits_eval = model(img_all, clin_all)
            val_ce = ce_criterion(logits_eval[val_idx_t], labels_t[val_idx_t])
        val_losses.append(float(val_ce))

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                stopped_epoch = epoch + 1
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on val set
    model.eval()
    with torch.no_grad():
        logits_final = model(img_all, clin_all)
        val_preds = logits_final[val_idx_t].argmax(dim=1).cpu().numpy()

    val_f1 = f1_score(labels[val_idx], val_preds, average="macro", zero_division=0)

    if return_history:
        return val_f1, model, val_preds, train_losses, val_losses, stopped_epoch
    return val_f1, model, val_preds


# =============================================================================
# CROSS-VALIDATION (hybrid)
# =============================================================================

def cross_validate_hybrid(img_feats: np.ndarray, clin_feats: np.ndarray,
                          labels: np.ndarray,
                          pool_idx: np.ndarray,
                          adj_list: list[torch.Tensor],
                          hparams: dict, lam: float,
                          num_epochs: int, patience: int,
                          device: str = "cpu",
                          n_splits: int = SWIN_NUM_CV_FOLDS):
    """K-fold stratified CV for hybrid training.

    Transductive: all N patients forwarded (smoothness on all), CE on fold-train only.
    """
    pool_labels = labels[pool_idx]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_f1s = []
    for tr_local, va_local in skf.split(pool_idx, pool_labels):
        tr_global = pool_idx[tr_local]
        va_global = pool_idx[va_local]
        f1, _, _ = train_hybrid(
            img_feats, clin_feats, labels,
            tr_global, va_global, adj_list, hparams, lam,
            num_epochs, patience, device,
        )
        fold_f1s.append(f1)
    return float(np.mean(fold_f1s)), float(np.std(fold_f1s))


# =============================================================================
# FULL PIPELINE
# =============================================================================

def run_graph_augmented_pipeline(output_dir: Optional[str] = None,
                                 composites_dir: Optional[str] = None,
                                 num_epochs: Optional[int] = None,
                                 patient_list: Optional[str] = None,
                                 split: str = "60-10-30",
                                 mode: str = "hybrid"):
    """Full graph-augmented pipeline.

    Phase 1-3: Load data, Swin features, split
    Phase 4: Build smoothness adjacency
    Phase 5: Two-stage grid search (MLP hparams then lambda sweep)
    Phase 6: Train final model, evaluate on test
    Phase 7: Plots + save results.json

    Args:
        split: "60-10-30" (matching DMGI) or "80-20" (matching Swin).
               With 80-20, CV is on the 80% train pool and the final model
               early-stops on the 20% test set (matching Swin protocol).
        mode: "full-batch" (original) or "hybrid" (mini-batch CE + graph smoothness).
    """
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    t0 = time.time()

    if num_epochs is None:
        num_epochs = GRAPH_AUG_NUM_EPOCHS
    if output_dir is None:
        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, "graph_augmented")
    os.makedirs(output_dir, exist_ok=True)

    swin_dir = os.path.join(DEFAULT_OUTPUT_DIR, "swin_baseline")
    device = "cpu"  # MLP training on CPU is fast enough

    is_hybrid = (mode == "hybrid")
    cv_func = cross_validate_hybrid if is_hybrid else cross_validate_graph_augmented
    train_func = train_hybrid if is_hybrid else train_graph_augmented
    patience = SWIN_PATIENCE if is_hybrid else GRAPH_AUG_PATIENCE

    # ================================================================
    # Phase 1: Load data
    # ================================================================
    print("=" * 60)
    print(f"PHASE 1: Loading data  [mode={mode}]")
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

    # ================================================================
    # Phase 2: Load Swin features + clinical encoding
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Loading Swin features + encoding clinical features")
    print("=" * 60)

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

    # Reorder Swin features to match current patient_ids order
    swin_pid_to_idx = {pid: i for i, pid in enumerate(swin_pids)}
    img_feats = np.zeros((len(patient_ids), swin_features.shape[1]),
                         dtype=np.float32)
    for i, pid in enumerate(patient_ids):
        if pid in swin_pid_to_idx:
            img_feats[i] = swin_features[swin_pid_to_idx[pid]]
        else:
            print(f"  Warning: {pid} not in Swin features, using zeros")

    print(f"Swin features shape: {img_feats.shape}")
    print(f"(Clinical encoding deferred to after split for fresh scaler fitting)")

    # ================================================================
    # Phase 3: Data split
    # ================================================================
    indices = np.arange(len(patient_ids))

    if split == "80-20":
        split_label = "80/20"
        print("\n" + "=" * 60)
        print("PHASE 3: 80/20 stratified split (matching Swin)")
        print("=" * 60)
        trainval_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=RANDOM_SEED)
        # No separate val — final model early-stops on test (matching Swin)
        train_idx = trainval_idx
        val_idx = test_idx
    elif split == "70-10-20":
        split_label = "70/10/20"
        print("\n" + "=" * 60)
        print("PHASE 3: 70/10/20 stratified split")
        print("=" * 60)
        trainval_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=RANDOM_SEED)
        y_trainval = labels[trainval_idx]
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=1.0/8.0, stratify=y_trainval,
            random_state=RANDOM_SEED)
    else:
        split_label = "60/10/30"
        print("\n" + "=" * 60)
        print("PHASE 3: 60/10/30 stratified split (matching DMGI)")
        print("=" * 60)
        trainval_idx, test_idx = train_test_split(
            indices, test_size=0.3, stratify=labels, random_state=RANDOM_SEED)
        y_trainval = labels[trainval_idx]
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=1.0/7.0, stratify=y_trainval,
            random_state=RANDOM_SEED)

    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    print(f"Train grades: {dict(Counter(y_train.tolist()))}")
    if split not in ("80-20",):
        print(f"Val grades:   {dict(Counter(y_val.tolist()))}")
    print(f"Test grades:  {dict(Counter(y_test.tolist()))}")

    # Clinical features (31-d) — fit scaler on train pool, apply to all
    train_df = clinical_df.iloc[trainval_idx].reset_index(drop=True)
    _, scaler = encode_clinical_features(train_df, fit_scaler=True)
    clin_feats, _ = encode_clinical_features(clinical_df, scaler=scaler,
                                             fit_scaler=False)
    print(f"Clinical encoded shape: {clin_feats.shape}")
    print(f"Image dim: {img_feats.shape[1]}  Clinical dim: {clin_feats.shape[1]}")

    # ================================================================
    # Phase 4: Build smoothness adjacency
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: Building smoothness adjacency (multiplex)")
    print("=" * 60)

    adj_list, adj_stats = build_multiplex_smoothness_adjacency(
        clinical_df, GRAPH_AUG_THRESHOLDS)

    print(f"Thresholds: {GRAPH_AUG_THRESHOLDS}")
    print(f"Normalization: row-normalized D^{{-1}}A (no self-loops)")
    for name, st in adj_stats.items():
        print(f"  {name}: theta={st['threshold']}, {st['num_edges']} edges, "
              f"density={st['density']:.3f}, isolated={st['num_isolated']}")

    # ================================================================
    # Phase 5: Two-stage grid search
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 5: Two-stage grid search")
    print("=" * 60)

    # --- Stage 1: MLP hparams with lambda=0 ---
    print("\n--- Stage 1: MLP hyperparameters (lambda=0) ---")

    param_grid = {
        "lr": [1e-3, 5e-4, 1e-4],
        "class_weight": [None, "balanced"],
        "dropout": [0.3, 0.5, 0.7],
        "hidden_dim": [64, 128],
        "weight_decay": [1e-4],
        "proj_dim": [64],
        "batch_size": [16],
    }

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    print(f"Searching {len(combos)} combinations x {SWIN_NUM_CV_FOLDS} folds ...")

    stage1_results = []
    best_f1 = -1.0
    best_params = None

    for i, combo in enumerate(combos):
        hp = dict(zip(keys, combo))
        mean_f1, std_f1 = cv_func(
            img_feats, clin_feats, labels,
            trainval_idx, adj_list, hp, lam=0.0,
            num_epochs=num_epochs, patience=patience,
            device=device,
        )
        stage1_results.append({**hp, "mean_f1": mean_f1, "std_f1": std_f1})
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_params = hp.copy()
        if (i + 1) % 12 == 0 or i + 1 == len(combos):
            elapsed = time.time() - t0
            print(f"  [{i + 1:3d}/{len(combos)}] best CV F1 = {best_f1:.4f}  ({elapsed:.0f}s)")

    stage1_results.sort(key=lambda x: x["mean_f1"], reverse=True)
    print(f"\nBest MLP hparams: {best_params}")
    print(f"Best CV Macro-F1 (lambda=0): {best_f1:.4f}")

    # --- Stage 2: Lambda sweep with best MLP hparams ---
    print("\n--- Stage 2: Lambda sweep ---")

    lambda_candidates = [0.0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    stage2_results = []
    best_lambda_f1 = -1.0
    best_lambda = 0.0

    for lam_c in lambda_candidates:
        mean_f1, std_f1 = cv_func(
            img_feats, clin_feats, labels,
            trainval_idx, adj_list, best_params, lam=lam_c,
            num_epochs=num_epochs, patience=patience,
            device=device,
        )
        tag = ""
        if mean_f1 > best_lambda_f1:
            best_lambda_f1 = mean_f1
            best_lambda = lam_c
            tag = " ***"
        stage2_results.append({
            "lambda": lam_c, "mean_f1": mean_f1, "std_f1": std_f1,
        })
        elapsed = time.time() - t0
        print(f"  lambda={lam_c:<6.3f}  CV F1={mean_f1:.4f} +/- {std_f1:.4f}  ({elapsed:.0f}s){tag}")

    print(f"\nBest lambda: {best_lambda} (CV F1={best_lambda_f1:.4f})")

    # ================================================================
    # Phase 6: Train final model & evaluate on test
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 6: Train final model & evaluate on test set")
    print("=" * 60)

    print(f"Hparams: {best_params}")
    print(f"Lambda: {best_lambda}")

    # Train on train_idx, early-stop on val_idx, evaluate on test_idx
    val_f1, final_model, _, train_losses, val_losses, stopped_epoch = \
        train_func(
            img_feats, clin_feats, labels,
            train_idx, val_idx, adj_list, best_params,
            lam=best_lambda, num_epochs=num_epochs,
            patience=patience, device=device,
            return_history=True,
        )

    # Get test predictions
    final_model.eval()
    img_t = torch.tensor(img_feats, dtype=torch.float32).to(device)
    clin_t = torch.tensor(clin_feats, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits_all = final_model(img_t, clin_t)
        test_logits = logits_all[test_idx]
        test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
        test_preds = test_logits.argmax(dim=1).cpu().numpy()

    print(f"\nVal Macro-F1: {val_f1:.4f}")
    print(f"Stopped at epoch: {stopped_epoch}")
    print("\nClassification Report (test set):")
    print(classification_report(y_test, test_preds, target_names=LABEL_NAMES,
                                zero_division=0))

    macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(y_test, test_preds, average="micro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, test_preds)
    acc = float(np.mean(y_test == test_preds))

    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Micro F1:          {micro_f1:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Accuracy:          {acc:.4f}")

    cm = confusion_matrix(y_test, test_preds)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # ================================================================
    # Phase 7: Plots & save results
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 7: Generating plots & saving results")
    print("=" * 60)

    plot_loss_curves(train_losses, val_losses,
                     os.path.join(output_dir, "loss_curves.png"),
                     num_epochs, stopped_epoch)
    per_class_aucs, macro_auc = plot_roc_curves(
        y_test, test_probs, os.path.join(output_dir, "roc_curves.png"))
    plot_confusion_matrix(y_test, test_preds,
                          os.path.join(output_dir, "confusion_matrix.png"))
    plot_per_class_f1(y_test, test_preds,
                      os.path.join(output_dir, "f1_scores.png"))

    # Embedding similarity heatmap
    final_model.eval()
    with torch.no_grad():
        embeddings = final_model.get_fused_embedding(img_t, clin_t).cpu().numpy()

    sim_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            pairwise = cosine_similarity(embeddings[labels == i],
                                         embeddings[labels == j])
            sim_matrix[i, j] = pairwise.mean()

    plot_embedding_similarity_heatmap(
        sim_matrix, os.path.join(output_dir, "embedding_similarity.png"))

    # Save adjacency stats
    with open(os.path.join(output_dir, "adjacency_stats.json"), "w") as f:
        json.dump(adj_stats, f, indent=2)
    print(f"  Saved adjacency_stats.json")

    # Save results.json
    save_data = {
        "config": {
            "model": f"Graph-Augmented Swin+MLP ({mode})",
            "training_mode": mode,
            "num_patients": len(patient_ids),
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "test_size": int(len(test_idx)),
            "split": split_label,
            "clinical_dim": int(clin_feats.shape[1]),
            "swin_dim": int(img_feats.shape[1]),
            "num_epochs": num_epochs,
            "patience": patience,
            "cv_folds": SWIN_NUM_CV_FOLDS,
            "graph_construction": "pd.get_dummies + minmax_scale + cosine_similarity",
            "graph_normalization": "row-normalized D^{-1}A (no self-loops)",
            "num_relations": len(DMGI_FEATURE_TYPE_GROUPS_COLS),
            "thresholds": GRAPH_AUG_THRESHOLDS,
        },
        "best_hyperparams": {
            k: (v if v is not None else "none")
            for k, v in (best_params or {}).items()
        },
        "best_lambda": best_lambda,
        "stage1_cv_best_f1": float(stage1_results[0]["mean_f1"]),
        "stage2_results": stage2_results,
        "early_stopping": {
            "patience": patience,
            "stopped_epoch": stopped_epoch,
            "best_epoch": int(np.argmin(val_losses)) + 1,
        },
        "test_metrics": {
            "macro_f1": float(macro_f1),
            "micro_f1": float(micro_f1),
            "balanced_accuracy": float(bal_acc),
            "accuracy": float(acc),
            "macro_auc": float(macro_auc),
            "per_class_auc": {k: float(v) for k, v in per_class_aucs.items()},
            "confusion_matrix": cm.tolist(),
        },
        "adjacency_stats": adj_stats,
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

    # Save artifacts
    np.save(os.path.join(output_dir, "fused_embeddings.npy"), embeddings)
    torch.save(final_model.state_dict(), os.path.join(output_dir, "model.pt"))
    with open(os.path.join(output_dir, "best_hyperparams.json"), "w") as f:
        json.dump({
            **(best_params or {}),
            "lambda": best_lambda,
        }, f, indent=2, default=str)
    print(f"Saved artifacts to {output_dir}")
