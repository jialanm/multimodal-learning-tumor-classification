"""Swin-Tiny + clinical MLP pipeline: feature extraction, model training, grid search."""

import os
import json
import glob
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from PIL import Image
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.class_weight import compute_class_weight
from typing import Optional
from itertools import product
from collections import Counter

from .config import (
    LABEL_COLUMN, LABEL_MAP, LABEL_NAMES, NUM_CLASSES,
    RANDOM_SEED, DEVICE, DEFAULT_OUTPUT_DIR,
    SWIN_NUM_EPOCHS, SWIN_NUM_CV_FOLDS, SWIN_PATIENCE,
    BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
)
from .clinical import load_clinical_dataframe, encode_clinical_features
from .evaluation import (
    evaluate_predictions, plot_loss_curves, plot_roc_curves,
    plot_confusion_matrix, plot_per_class_f1, plot_embedding_similarity_heatmap,
    save_summary_txt,
)

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# =============================================================================
# DATA
# =============================================================================

def build_patient_list(composites_dir: str):
    """Match patients with both composites and valid grade labels."""
    df = load_clinical_dataframe()
    composite_pids = set(os.listdir(composites_dir))
    df = df[df["Patient ID"].isin(composite_pids)].reset_index(drop=True)
    patient_ids = df["Patient ID"].tolist()
    labels = np.array([LABEL_MAP[int(row[LABEL_COLUMN])] for _, row in df.iterrows()])
    return patient_ids, labels, df


def extract_swin_features(patient_ids: list, composites_dir: str,
                          output_dir: str, device: str = DEVICE) -> np.ndarray:
    """
    Extract Swin-Tiny 768-d embeddings per patient (avg-pooled over slices).
    Results are cached to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    cache_feats = os.path.join(output_dir, "swin_features.npy")
    cache_pids = os.path.join(output_dir, "swin_features_pids.json")

    if os.path.exists(cache_feats) and os.path.exists(cache_pids):
        with open(cache_pids) as f:
            cached_pids = json.load(f)
        if cached_pids == patient_ids:
            print("Loading cached Swin-Tiny features...")
            return np.load(cache_feats)

    print(f"Extracting Swin-Tiny features on {device}...")
    swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
    swin.head = nn.Identity()  # type: ignore[assignment]
    swin = swin.to(device).eval()

    all_embeddings = []
    for i, pid in enumerate(patient_ids):
        patient_dir = os.path.join(composites_dir, pid)
        slice_paths = sorted(glob.glob(os.path.join(patient_dir, "*.png")))

        if not slice_paths:
            print(f"  Warning: no composites for {pid}, using zeros")
            all_embeddings.append(np.zeros(768, dtype=np.float32))
            continue

        tensors: list[torch.Tensor] = [IMG_TRANSFORM(Image.open(sp).convert("RGB")) for sp in slice_paths]  # pyright: ignore[reportAssignmentType]
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            embs = swin(batch).cpu().numpy()
        all_embeddings.append(embs.mean(axis=0))

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i + 1}/{len(patient_ids)}] {pid}")

    features = np.array(all_embeddings, dtype=np.float32)
    print(f"  Done: {features.shape[0]} patients x {features.shape[1]}-d embeddings")

    np.save(cache_feats, features)
    with open(cache_pids, "w") as f:
        json.dump(patient_ids, f)
    return features


# =============================================================================
# MODEL
# =============================================================================

class MultimodalClassifier(nn.Module):
    """Two-branch fusion: Swin image features + clinical features -> grade prediction."""

    def __init__(self, img_dim, clinical_dim, proj_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.img_proj = nn.Sequential(nn.Linear(img_dim, proj_dim), nn.ReLU())
        self.clin_proj = nn.Sequential(nn.Linear(clinical_dim, proj_dim), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(2 * proj_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, NUM_CLASSES),
        )

    def get_fused_embedding(self, img_feats, clin_feats):
        """Return the fused embedding (2*proj_dim) before the classifier head."""
        return torch.cat([self.img_proj(img_feats), self.clin_proj(clin_feats)], dim=1)

    def forward(self, img_feats, clin_feats):
        return self.classifier(self.get_fused_embedding(img_feats, clin_feats))


# =============================================================================
# TRAINING
# =============================================================================

def _make_loader(img, clin, labels, batch_size, shuffle):
    ds = TensorDataset(
        torch.tensor(img, dtype=torch.float32),
        torch.tensor(clin, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    if shuffle:
        class_counts = np.bincount(labels, minlength=NUM_CLASSES)
        sample_weights = 1.0 / class_counts[labels]
        sampler = WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=len(ds),
                                        replacement=True, generator=g)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def train_and_evaluate(img_tr, clin_tr, y_tr, img_val, clin_val, y_val,
                       hparams, device, num_epochs, patience=SWIN_PATIENCE):
    """Train with early stopping. Returns (val_macro_f1, model, preds)."""
    torch.manual_seed(RANDOM_SEED)

    model = MultimodalClassifier(
        img_dim=img_tr.shape[1], clinical_dim=clin_tr.shape[1],
        proj_dim=hparams["proj_dim"], hidden_dim=hparams["hidden_dim"],
        dropout_rate=hparams["dropout"],
    ).to(device)

    if hparams["class_weight"] == "balanced":
        cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_tr)
        weight = torch.tensor(cw, dtype=torch.float32).to(device)
    else:
        weight = None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"],  # pyright: ignore[reportPrivateImportUsage]
                                 weight_decay=hparams["weight_decay"])

    train_loader = _make_loader(img_tr, clin_tr, y_tr, hparams["batch_size"], shuffle=True)
    val_loader = _make_loader(img_val, clin_val, y_val, hparams["batch_size"], shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for _ in range(num_epochs):
        model.train()
        for img_b, clin_b, y_b in train_loader:
            img_b, clin_b, y_b = img_b.to(device), clin_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(img_b, clin_b), y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        v_loss, v_n = 0.0, 0
        with torch.no_grad():
            for img_b, clin_b, y_b in val_loader:
                img_b, clin_b, y_b = img_b.to(device), clin_b.to(device), y_b.to(device)
                v_loss += criterion(model(img_b, clin_b), y_b).item() * len(y_b)
                v_n += len(y_b)

        if v_loss / v_n < best_val_loss:
            best_val_loss = v_loss / v_n
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    with torch.no_grad():
        for img_b, clin_b, _ in val_loader:
            logits = model(img_b.to(device), clin_b.to(device))
            preds.extend(logits.argmax(dim=1).cpu().tolist())

    macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
    return macro_f1, model, np.array(preds)


def cross_validate(img_feats, clin_feats, labels, hparams, device, num_epochs):
    """5-fold stratified CV. Returns (mean_f1, std_f1)."""
    skf = StratifiedKFold(n_splits=SWIN_NUM_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_f1s = []
    for train_idx, val_idx in skf.split(img_feats, labels):
        f1, _, _ = train_and_evaluate(
            img_feats[train_idx], clin_feats[train_idx], labels[train_idx],
            img_feats[val_idx], clin_feats[val_idx], labels[val_idx],
            hparams, device, num_epochs,
        )
        fold_f1s.append(f1)
    return np.mean(fold_f1s), np.std(fold_f1s)


def train_final_with_history(img_tr, clin_tr, y_tr, img_val, clin_val, y_val,
                             hparams, device, num_epochs, patience=SWIN_PATIENCE):
    """
    Train final model recording loss history.
    Returns: (model, preds, probs, train_losses, val_losses, stopped_epoch)
    """
    torch.manual_seed(RANDOM_SEED)

    model = MultimodalClassifier(
        img_dim=img_tr.shape[1], clinical_dim=clin_tr.shape[1],
        proj_dim=hparams["proj_dim"], hidden_dim=hparams["hidden_dim"],
        dropout_rate=hparams["dropout"],
    ).to(device)

    if hparams["class_weight"] == "balanced":
        cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_tr)
        weight = torch.tensor(cw, dtype=torch.float32).to(device)
    else:
        weight = None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"],  # pyright: ignore[reportPrivateImportUsage]
                                 weight_decay=hparams["weight_decay"])

    train_loader = _make_loader(img_tr, clin_tr, y_tr, hparams["batch_size"], shuffle=True)
    val_loader = _make_loader(img_val, clin_val, y_val, hparams["batch_size"], shuffle=False)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    stopped_epoch = num_epochs

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, n = 0.0, 0
        for img_b, clin_b, y_b in train_loader:
            img_b, clin_b, y_b = img_b.to(device), clin_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(img_b, clin_b), y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y_b)
            n += len(y_b)
        train_losses.append(epoch_loss / n)

        model.eval()
        v_loss, v_n = 0.0, 0
        with torch.no_grad():
            for img_b, clin_b, y_b in val_loader:
                img_b, clin_b, y_b = img_b.to(device), clin_b.to(device), y_b.to(device)
                v_loss += criterion(model(img_b, clin_b), y_b).item() * len(y_b)
                v_n += len(y_b)
        val_losses.append(v_loss / v_n)

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                stopped_epoch = epoch + 1
                print(f"  Early stopping at epoch {stopped_epoch} "
                      f"(best val loss {best_val_loss:.4f} at epoch {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_logits = []
    with torch.no_grad():
        for img_b, clin_b, _ in val_loader:
            all_logits.append(model(img_b.to(device), clin_b.to(device)))
    logits_cat = torch.cat(all_logits)
    probs = torch.softmax(logits_cat, dim=1).cpu().numpy()
    preds = logits_cat.argmax(dim=1).cpu().numpy()

    return model, preds, probs, train_losses, val_losses, stopped_epoch


# =============================================================================
# FULL PIPELINE
# =============================================================================

def run_swin_pipeline(output_dir: Optional[str] = None, composites_dir: Optional[str] = None,
                      num_epochs: Optional[int] = None, patient_list: Optional[str] = None):
    """Full Swin pipeline: extract features -> grid search CV -> train final -> evaluate."""
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    t0 = time.time()

    if num_epochs is None:
        num_epochs = SWIN_NUM_EPOCHS
    if output_dir is None:
        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, "swin_baseline")
    if composites_dir is None:
        composites_dir = os.path.join(DEFAULT_OUTPUT_DIR, "ovis2_fixed256", "composites")
        # Fall back to ovis2_fixed256_crop if the new path doesn't exist
        if not os.path.isdir(composites_dir):
            composites_dir = os.path.join(DEFAULT_OUTPUT_DIR, "ovis2_fixed256_crop", "composites")
    os.makedirs(output_dir, exist_ok=True)

    device_swin = DEVICE
    device_mlp = "cpu"

    # Phase 1: Load data
    print("=" * 60)
    print("PHASE 1: Loading data")
    print("=" * 60)
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

    # Phase 2: Extract features
    print("\n" + "=" * 60)
    print("PHASE 2: Swin-Tiny feature extraction")
    print("=" * 60)
    img_feats = extract_swin_features(patient_ids, composites_dir, output_dir, device_swin)

    # Phase 3: Split
    print("\n" + "=" * 60)
    print("PHASE 3: 80/20 stratified split")
    print("=" * 60)
    indices = np.arange(len(patient_ids))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=RANDOM_SEED)
    img_train, img_test = img_feats[train_idx], img_feats[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    train_df = clinical_df.iloc[train_idx].reset_index(drop=True)
    test_df = clinical_df.iloc[test_idx].reset_index(drop=True)
    clin_train, scaler = encode_clinical_features(train_df, fit_scaler=True)
    clin_test, _ = encode_clinical_features(test_df, scaler=scaler, fit_scaler=False)

    print(f"Train: {len(train_idx)}  Test: {len(test_idx)}")
    print(f"Train grades: {dict(Counter(y_train.tolist()))}")
    print(f"Test grades:  {dict(Counter(y_test.tolist()))}")
    print(f"Clinical dim: {clin_train.shape[1]}  Image dim: {img_train.shape[1]}")

    # Phase 4: Grid search
    print("\n" + "=" * 60)
    print("PHASE 4: Grid search with 5-fold CV")
    print("=" * 60)

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

    search_results = []
    best_f1 = -1.0
    best_params = None

    for i, combo in enumerate(combos):
        hp = dict(zip(keys, combo))
        mean_f1, std_f1 = cross_validate(img_train, clin_train, y_train,
                                         hp, device_mlp, num_epochs)
        search_results.append({**hp, "mean_f1": mean_f1, "std_f1": std_f1})
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_params = hp.copy()
        if (i + 1) % 24 == 0 or i + 1 == len(combos):
            elapsed = time.time() - t0
            print(f"  [{i + 1:3d}/{len(combos)}] best CV F1 = {best_f1:.4f}  ({elapsed:.0f}s)")

    search_results.sort(key=lambda x: x["mean_f1"], reverse=True)
    print(f"\nBest: {best_params}")

    # Phase 5: Final model
    print("\n" + "=" * 60)
    print("PHASE 5: Train final model & evaluate on test set")
    print("=" * 60)

    final_model, test_preds, test_probs, train_losses, val_losses, stopped_epoch = \
        train_final_with_history(img_train, clin_train, y_train,
                                 img_test, clin_test, y_test,
                                 best_params, device_mlp, num_epochs)

    print("\nClassification Report:")
    print(classification_report(y_test, test_preds, target_names=LABEL_NAMES, zero_division=0))

    macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(y_test, test_preds, average="micro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, test_preds)
    acc = float(np.mean(y_test == test_preds))

    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Accuracy:          {acc:.4f}")

    cm = confusion_matrix(y_test, test_preds)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # Phase 6: Plots & save
    print("\n" + "=" * 60)
    print("PHASE 6: Generating plots & saving results")
    print("=" * 60)

    plot_loss_curves(train_losses, val_losses,
                     os.path.join(output_dir, "loss_curves.png"),
                     num_epochs, stopped_epoch)
    per_class_aucs, macro_auc = plot_roc_curves(
        y_test, test_probs, os.path.join(output_dir, "roc_curves.png"))
    plot_confusion_matrix(y_test, test_preds, os.path.join(output_dir, "confusion_matrix.png"))
    plot_per_class_f1(y_test, test_preds, os.path.join(output_dir, "f1_scores.png"))

    # Embedding cosine similarity heatmap by tumor grade
    clin_all, _ = encode_clinical_features(clinical_df, scaler=scaler, fit_scaler=False)
    final_model.eval()
    with torch.no_grad():
        img_t = torch.tensor(img_feats, dtype=torch.float32).to(device_mlp)
        clin_t = torch.tensor(clin_all, dtype=torch.float32).to(device_mlp)
        embeddings = final_model.get_fused_embedding(img_t, clin_t).cpu().numpy()

    sim_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            pairwise = cosine_similarity(embeddings[labels == i], embeddings[labels == j])
            sim_matrix[i, j] = pairwise.mean()

    plot_embedding_similarity_heatmap(
        sim_matrix, os.path.join(output_dir, "embedding_similarity.png"))

    save_summary_txt(
        os.path.join(output_dir, "experiment_summary.txt"),
        patient_ids, labels, train_idx, test_idx,
        y_train, y_test, clin_train, img_feats,
        best_params, search_results, test_preds, test_probs,
        macro_f1, bal_acc, acc, cm, per_class_aucs, macro_auc,
        train_losses, val_losses, elapsed, stopped_epoch,
    )

    save_data = {
        "config": {
            "model": "Swin-Tiny (frozen) + clinical MLP",
            "num_patients": len(patient_ids),
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "clinical_dim": int(clin_train.shape[1]),
            "swin_dim": int(img_feats.shape[1]),
            "num_epochs": num_epochs,
            "cv_folds": SWIN_NUM_CV_FOLDS,
        },
        "best_hyperparams": {
            k: (v if v is not None else "none") for k, v in (best_params or {}).items()
        },
        "cv_best_f1": float(search_results[0]["mean_f1"]),
        "early_stopping": {
            "patience": SWIN_PATIENCE,
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

    # Save artifacts for downstream pipelines (e.g. DMGI)
    np.save(os.path.join(output_dir, "fused_embeddings.npy"), embeddings)
    torch.save(final_model.state_dict(), os.path.join(output_dir, "swin_model.pt"))
    with open(os.path.join(output_dir, "best_hyperparams.json"), "w") as f:
        json.dump({k: (v if v is not None else "none") for k, v in (best_params or {}).items()}, f, indent=2)
    with open(os.path.join(output_dir, "clinical_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved downstream artifacts (fused_embeddings, model, hyperparams, scaler) to {output_dir}")
