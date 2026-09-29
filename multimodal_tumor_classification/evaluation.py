"""Metrics computation, plotting, and experiment summary generation."""

import numpy as np
from collections import Counter
from sklearn.metrics import (
    classification_report, confusion_matrix as sk_confusion_matrix,
    balanced_accuracy_score, f1_score, roc_curve, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import (
    LABEL_NAMES, NUM_CLASSES, SWIN_NUM_EPOCHS, SWIN_NUM_CV_FOLDS, SWIN_PATIENCE,
    BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
)


# =============================================================================
# METRICS
# =============================================================================

def evaluate_predictions(y_true, y_pred, label_names=LABEL_NAMES) -> dict:
    """Compute and print standard classification metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))

    print(f"Macro F1:          {macro_f1:.3f}")
    print(f"Balanced Accuracy: {bal_acc:.3f}")
    print(f"Accuracy:          {acc:.3f}")

    cm = sk_confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    print(f"{'':12s} {'Pred G1':>8s} {'Pred G2':>8s} {'Pred G3':>8s}")
    for i, name in enumerate(label_names):
        print(f"{name:12s} {cm[i, 0]:8d} {cm[i, 1]:8d} {cm[i, 2]:8d}")

    return {
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(bal_acc),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
    }


# =============================================================================
# PLOTS
# =============================================================================

def plot_loss_curves(train_losses, val_losses, out_path,
                     max_epochs=SWIN_NUM_EPOCHS, stopped_epoch=None):
    """Training and validation loss vs epoch."""
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train loss", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Validation loss", linewidth=1.5)

    best_epoch = int(np.argmin(val_losses)) + 1
    best_val = min(val_losses)
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.7,
               label=f"Best epoch ({best_epoch}, loss={best_val:.4f})")
    ax.plot(best_epoch, best_val, "g*", markersize=12)

    if stopped_epoch is not None and stopped_epoch < max_epochs:
        ax.axvline(stopped_epoch, color="red", linestyle=":", alpha=0.7,
                   label=f"Early stop (epoch {stopped_epoch})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training & Validation Loss (with Early Stopping)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_roc_curves(y_true, y_probs, out_path):
    """Per-class and macro-average ROC curves. Returns (per_class_aucs, macro_auc)."""
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    all_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(all_fpr)
    aucs = []

    for i, (name, color) in enumerate(zip(LABEL_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        a = roc_auc_score(y_bin[:, i], y_probs[:, i])
        aucs.append(a)
        mean_tpr += np.interp(all_fpr, fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=1.5,
                label=f"{name} (AUC = {a:.3f})")

    mean_tpr /= NUM_CLASSES
    macro_auc = float(np.mean(aucs))
    ax.plot(all_fpr, mean_tpr, color="navy", linewidth=2, linestyle="--",
            label=f"Macro avg (AUC = {macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return dict(zip(LABEL_NAMES, aucs)), macro_auc


def plot_confusion_matrix(y_true, y_pred, out_path):
    """Confusion matrix heatmap."""
    cm = sk_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(LABEL_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_per_class_f1(y_true, y_pred, out_path):
    """Bar chart of per-class F1 + macro F1."""
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(LABEL_NAMES) + 1)
    vals = list(f1s) + [macro]
    labels = LABEL_NAMES + ["Macro Avg"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#555555"]
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class and Macro F1 Scores")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_embedding_similarity_heatmap(sim_matrix, out_path):
    """3x3 heatmap of average pairwise cosine similarity between grade embeddings."""
    mask = np.tri(NUM_CLASSES, k=-1, dtype=bool)  # mask upper triangle (k=-1 on transposed)
    mask = ~mask & ~np.eye(NUM_CLASSES, dtype=bool)  # mask strictly upper triangle
    masked = np.ma.array(sim_matrix, mask=mask)

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.cm.Reds.copy()
    cmap.set_bad(color="white")
    im = ax.imshow(masked, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(LABEL_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Grade")
    ax.set_ylabel("Grade")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i in range(NUM_CLASSES):
        for j in range(i + 1):  # lower triangle + diagonal
            ax.text(j, i, f"{sim_matrix[i, j]:.3f}", ha="center", va="center",
                    color="white" if sim_matrix[i, j] > 0.5 else "black",
                    fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# =============================================================================
# TEXT SUMMARY (Swin baseline)
# =============================================================================

def save_summary_txt(out_path, patient_ids, labels, train_idx, test_idx,
                     y_train, y_test, clin_train, img_feats,
                     best_params, search_results, test_preds, test_probs,
                     macro_f1, bal_acc, acc, cm, per_class_aucs, macro_auc,
                     train_losses, val_losses, elapsed, stopped_epoch=None):
    """Write comprehensive experiment summary."""
    lines = []
    L = lines.append

    L("=" * 70)
    L("EXPERIMENT SUMMARY -- Multimodal Swin-Tiny Baseline")
    L("=" * 70)

    L("\n--- MODEL ARCHITECTURE ---")
    L("Image encoder:    Swin-Tiny (torchvision, ImageNet-1K pretrained, frozen)")
    L("Image embedding:  768-d (avg-pooled over 3 slices per patient)")
    L(f"Image projection: Linear(768, {best_params['proj_dim']}) + ReLU")
    L(f"Clinical input:   {clin_train.shape[1]}-d encoded features")
    L(f"  - Binary features (0/1):    {len(BINARY_FEATURES)}")
    cat_dims = sum(len(v) for v in CATEGORICAL_FEATURES.values())
    L(f"  - One-hot categorical:      {cat_dims}")
    L(f"  - Numerical (std-scaled):   {len(NUMERICAL_FEATURES)}")
    L(f"Fusion:           Concatenate (2 x {best_params['proj_dim']})")
    L(f"Classifier:       Linear -> ReLU -> Dropout({best_params['dropout']}) -> Linear(3)")

    L("\n--- DATA ---")
    L(f"Total patients:   {len(patient_ids)}")
    grade_dist = Counter(labels.tolist())
    L(f"Grade distribution: G1={grade_dist[0]}, G2={grade_dist[1]}, G3={grade_dist[2]}")
    L(f"Train / Test:     {len(train_idx)} / {len(test_idx)}")

    L("\n--- BEST HYPERPARAMETERS ---")
    for k, v in best_params.items():
        L(f"  {k}: {v if v is not None else 'none'}")
    L(f"  CV Macro F1: {search_results[0]['mean_f1']:.4f} +/- {search_results[0]['std_f1']:.4f}")

    L("\n--- TRAINING ---")
    L(f"Best val loss: {min(val_losses):.4f} at epoch {int(np.argmin(val_losses)) + 1}")
    if stopped_epoch is not None and stopped_epoch < SWIN_NUM_EPOCHS:
        L(f"Early stopped at epoch {stopped_epoch}")

    L("\n--- TEST RESULTS ---")
    L(f"Accuracy:          {acc:.4f}")
    L(f"Balanced Accuracy: {bal_acc:.4f}")
    L(f"Macro F1:          {macro_f1:.4f}")
    L(f"Macro AUC:         {macro_auc:.4f}")

    per_f1 = f1_score(y_test, test_preds, average=None, zero_division=0)
    for i, name in enumerate(LABEL_NAMES):
        L(f"  {name} F1: {per_f1[i]:.4f}  AUC: {per_class_aucs[name]:.4f}")

    L(f"\nConfusion Matrix:")
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
