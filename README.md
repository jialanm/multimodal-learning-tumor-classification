# Multimodal Breast Cancer Tumor Grading

Predict Nottingham histologic grade (Grade 1 / 2 / 3) from the [Duke Breast Cancer MRI](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903) dataset using multimodal learning: DCE-MRI imaging + clinical features.

Four baseline approaches are implemented:

1. **Ovis2 VLM few-shot** -- a vision-language model (AIDC-AI/Ovis2-4B) that reads DCE composite images and clinical metadata through natural language prompts.
2. **Swin-Tiny + Clinical MLP** -- a frozen Swin-Tiny image encoder fused with an MLP over encoded clinical features, trained with 5-fold cross-validated grid search.
3. **DMGI (Deep Multiplex Graph Infomax)** -- learns node embeddings on a multiplex graph of patient similarity via mutual information maximization and consensus regularization, with downstream logistic regression or fine-tuned MLP classification.
4. **Graph-Augmented Swin + MLP** -- combines the Swin-Tiny + Clinical MLP with graph smoothness regularization over clinical similarity graphs, providing transductive learning without label leakage.

## Results Summary

### Full-dataset results (899 patients)

| Method | Split | Macro F1 | Balanced Acc | Accuracy | Macro AUC |
|--------|-------|----------|--------------|----------|-----------|
| Swin-Tiny + MLP | 80/20 | 0.513 | 0.547 | 0.511 | 0.725 |
| **Graph-Augmented Swin + MLP** | **60/10/30** | **0.541** | **0.584** | **0.544** | **0.727** |
| Graph-Augmented Swin + MLP | 70/10/20 | 0.515 | 0.535 | 0.535 | 0.701 |
| Graph-Augmented Swin + MLP | 80/20 | 0.471 | 0.571 | 0.472 | 0.735 |
| DMGI (LogReg on frozen H) | 60/10/30 | 0.424 ± 0.020 | -- | 0.529 | -- |
| DMGI (fine-tuned MLP) | 60/10/30 | 0.451 ± 0.034 | -- | -- | -- |

### Ovis2 VLM few-shot results (subset)

| Method | Crop | Patients | Macro F1 | Balanced Acc | Accuracy |
|--------|------|----------|----------|--------------|----------|
| Ovis2-4B few-shot | Proportional (25% pad) | 60 | 0.383 | 0.388 | 0.433 |
| Ovis2-4B few-shot | None (full-size) | 60 | 0.290 | 0.301 | 0.300 |
| Ovis2-4B few-shot | Fixed 256x256 | 100 | 0.411 | 0.414 | 0.450 |

Detailed per-experiment results (JSON, plots, logs) are in `output/`.

## Project Structure

```
.
├── README.md
├── pyproject.toml
├── .gitignore
├── multimodal_tumor_classification/   # Python package
│   ├── __init__.py
│   ├── __main__.py                    # CLI entry point
│   ├── config.py                      # Paths, labels, crop modes, clinical features
│   ├── dicom_utils.py                 # DICOM loading, series identification
│   ├── clinical.py                    # Clinical xlsx parsing, text builder, feature encoding
│   ├── imaging.py                     # DCE composites, cropping, slice sampling
│   ├── dataset.py                     # Dataset building (orchestrates DICOM + clinical)
│   ├── prompts.py                     # Few-shot example selection, prompt formatting
│   ├── ovis2_pipeline.py              # Ovis2 VLM model loading + inference
│   ├── swin_pipeline.py               # Swin-Tiny + MLP training pipeline
│   ├── dmgi_pipeline.py               # DMGI multiplex graph infomax pipeline
│   ├── graph_augmented_pipeline.py    # Graph-augmented Swin + MLP pipeline
│   └── evaluation.py                  # Metrics, plotting, experiment summaries
├── data/
│   ├── Duke-Breast-Cancer-MRI/        # DICOM folders per patient (not included)
│   ├── Annotation_Boxes.xlsx          # Tumor bounding box annotations
│   └── Clinical_and_Other_Features_Full.xlsx
├── output/                            # Runtime artifacts (composites, caches, feature files)
│   ├── ovis2_proportional_crop/       # Composites + results (proportional crop)
│   ├── ovis2_nocrop/                  # Composites + results (no crop)
│   ├── ovis2_fixed256_crop/           # Composites + results (256x256 crop)
│   ├── swin_baseline/                 # Swin feature cache + plots
│   ├── swin_899/                      # Swin baseline on full dataset (899 patients)
│   ├── dmgi_baseline/                 # DMGI results + embeddings
│   ├── graph_augmented/               # Graph-augmented (60/10/30 split)
│   ├── graph_augmented_80_20/         # Graph-augmented (80/20 split)
│   └── graph_augmented_70_10_20/      # Graph-augmented (70/10/20 split)
```

## Data

This project uses the **Duke Breast Cancer MRI** dataset from The Cancer Imaging Archive (TCIA).

1. Download DICOM images from [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903) and place them under `data/Duke-Breast-Cancer-MRI/` (one subfolder per patient, e.g. `Breast_MRI_024/`).
2. The annotation and clinical spreadsheets (`Annotation_Boxes.xlsx`, `Clinical_and_Other_Features_Full.xlsx`) should also be placed in `data/`.

## Installation

### Prerequisites

- Python 3.10+
- MPS (Apple Silicon) or CUDA GPU recommended for the Ovis2 VLM pipeline. The Swin-Tiny baseline runs on CPU.

### Setup

```bash
# Create and activate a conda environment
conda create -n mml python=3.10 -y
conda activate mml

# Install PyTorch (adjust for your platform -- shown here for macOS MPS)
pip install torch==2.4.0 torchvision==0.19.0

# Install the package in editable mode
pip install -e .
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Swin-Tiny encoder, tensor operations |
| `transformers` | Ovis2-4B VLM loading |
| `pydicom` | DICOM image reading |
| `scikit-learn` | Metrics, cross-validation, preprocessing |
| `pandas`, `openpyxl` | Clinical spreadsheet parsing |
| `matplotlib` | Plot generation |
| `pillow` | Image I/O |
| `numpy` | Array operations |

All dependencies are declared in `pyproject.toml` and installed automatically with `pip install -e .`.

## Usage

### Ovis2 VLM few-shot classification

```bash
# Proportional bbox crop (25% padding) -- default
python -m multimodal_tumor_classification ovis2 --crop proportional

# No crop (full-size DCE composites)
python -m multimodal_tumor_classification ovis2 --crop none

# Fixed 256x256 crop centered on tumor
python -m multimodal_tumor_classification ovis2 --crop fixed256

# Limit to N patients (useful for testing)
python -m multimodal_tumor_classification ovis2 --crop proportional --num-patients 5
```

The Ovis2 pipeline runs three phases:
1. **DICOM processing** -- loads pre-contrast and 1st post-contrast series, builds RGB composites (R=pre, G=post1, B=subtraction), applies the selected crop, caches PNGs.
2. **Few-shot inference** -- loads Ovis2-4B, builds prompts with clinical metadata and 1 example per grade, classifies 3 slices per patient, majority-votes.
3. **Evaluation** -- prints classification report, macro F1, balanced accuracy, confusion matrix. Saves results JSON to `output/`.

### Swin-Tiny + Clinical MLP baseline

```bash
# Run with defaults (200 epochs, full grid search)
python -m multimodal_tumor_classification swin

# Quick test run
python -m multimodal_tumor_classification swin --epochs 5
```

This runs:
1. **Feature extraction** -- extracts 768-d Swin-Tiny embeddings per patient (average-pooled over 3 slices). Cached to disk.
2. **Clinical encoding** -- binary, one-hot, and standard-scaled numerical features (31-d total).
3. **Grid search** -- 36 hyperparameter combinations x 5-fold stratified CV.
4. **Final evaluation** -- trains best model on 80/20 stratified split. Generates confusion matrix, ROC curves, F1 bar chart, and loss curves.

### DMGI (Deep Multiplex Graph Infomax)

```bash
python -m multimodal_tumor_classification dmgi --output-dir output/dmgi_baseline
```

This runs:
1. **Feature extraction** -- reuses Swin-Tiny (768-d) + clinical (31-d) = 799-d node features with row-wise L1 normalization.
2. **Graph construction** -- builds 4 clinical similarity graphs (receptor/staging, demographics/progression, treatment, anatomical) via cosine similarity with thresholding.
3. **DMGI training** -- trains a multiplex GCN with mutual information maximization, consensus regularization, and supervised loss for up to 2000 epochs (patience=100).
4. **Evaluation** -- logistic regression and fine-tuned MLP on frozen consensus embeddings.

### Graph-Augmented Swin + MLP

```bash
# 60/10/30 train/val/test split
python -m multimodal_tumor_classification graph-aug --split 60-10-30

# 80/20 train/test split
python -m multimodal_tumor_classification graph-aug --split 80-20
```

This runs the Swin-Tiny + Clinical MLP with an additional graph smoothness regularization term that encourages similar patients (by clinical features) to have similar embeddings, without leaking labels.

### CLI installed entry point

After `pip install -e .`, you can also use:

```bash
tumor-classify ovis2 --crop proportional
tumor-classify swin
tumor-classify dmgi
tumor-classify graph-aug
```

## DICOM Image Processing

### From 3D volume to 2D composite

Each patient in the Duke dataset has multiple MRI series acquired during a Dynamic Contrast-Enhanced (DCE) protocol. The raw data is a set of 3D DICOM volumes (one per contrast phase), where each volume contains ~100-200 axial slices of shape ~512x512.

**Step 1: Series identification.** Two series are selected per patient based on `SeriesDescription` metadata:
- **Pre-contrast** (`ax dyn pre`) -- baseline scan before gadolinium injection
- **1st post-contrast** (`ax dyn 1st pass`) -- first scan after gadolinium injection, where tumors show contrast uptake

**Step 2: Slice selection.** The `Annotation_Boxes.xlsx` file provides the tumor's bounding box including `Start Slice` and `End Slice` (the axial range containing the tumor). Three slices are sampled around the tumor center (e.g., for a tumor spanning slices 89-112, center=100, we take slices 99, 100, 101).

**Step 3: DCE RGB composite.** Each selected 2D axial slice is converted from two grayscale images (pre and post-contrast) into a single RGB image:

| Channel | Source | What it shows |
|---------|--------|---------------|
| **Red** | Pre-contrast slice | Baseline tissue signal (anatomy) |
| **Green** | 1st post-contrast slice | Signal after gadolinium (enhanced vasculature) |
| **Blue** | Subtraction (post - pre) | Contrast uptake only (highlights enhancement regions) |

Each channel is independently min-max normalized to [0, 255]. The subtraction channel is floored at zero (negative values clipped). This encoding means:
- **Bright blue/cyan areas** = strong contrast uptake, often indicative of tumor vascularity
- **Gray/white areas** = similar signal in all phases (normal tissue)
- **Red-dominant areas** = signal present pre-contrast but reduced post-contrast

### Cropping strategies

The full axial slice is ~512x512 pixels but the tumor region is typically small. Three crop modes are supported:

| Mode | Description |
|------|-------------|
| `proportional` | Crop to the annotation bounding box + 25% padding (proportional to the larger bbox dimension, minimum 5px). Small tumors get tight crops, large tumors get wider margins. |
| `fixed256` | Crop a fixed 256x256 pixel window centered on the bounding box center. If the window extends beyond the image boundary, it shifts inward. |
| `none` | No crop -- use the full axial slice as-is. |

### Summary

```
Patient DICOM folder
  |
  |-- ax dyn pre (3D volume, ~160 slices)
  |-- ax dyn 1st pass (3D volume, ~160 slices)
  |
  v  Select 3 slices around tumor center (from annotation bbox)
  |
  v  For each slice:
  |    pre[slice]  -----> R channel (normalized 0-255)
  |    post[slice] -----> G channel (normalized 0-255)
  |    post - pre  -----> B channel (clipped >=0, normalized 0-255)
  |                           |
  |                           v
  |                    RGB composite image
  |                           |
  |                           v
  |                    Crop (proportional / fixed256 / none)
  |                           |
  |                           v
  |                    Saved as PNG
  v
3 PNG composites per patient --> fed to Ovis2 VLM or Swin-Tiny encoder
```

## Pipeline Overview

```
DICOM volumes (pre-contrast + 1st post-contrast)
    |
    v
DCE RGB Composite (R=pre, G=post1, B=subtraction)
    |
    |---> [Ovis2 VLM] Few-shot prompt + clinical text -> Grade prediction
    |
    |---> [Swin-Tiny + Clinical MLP]  (see Fusion MLP below)
    |
    |---> [DMGI] Swin+Clinical node features + multiplex clinical graphs -> Consensus embeddings -> LogReg/MLP
    |
    \---> [Graph-Augmented] Swin+Clinical MLP + graph smoothness regularization -> Grade prediction
```

### Fusion MLP architecture

The image and clinical modalities have very different dimensionalities (768-d vs 31-d). A projection layer maps each branch into a shared embedding space before fusion.

```
Image branch:     Swin-Tiny (frozen) -> 768-d -> Linear(768, proj_dim) -> ReLU -> proj_dim-d
                                                                                      |
Clinical branch:  31-d encoded features -------> Linear(31, proj_dim)  -> ReLU -> proj_dim-d
                                                                                      |
                                                                                Concatenate
                                                                                      |
                                                                               2 * proj_dim-d
                                                                                      |
                                                                                      v
                                                          Linear(2*proj_dim, hidden_dim) -> ReLU -> Dropout
                                                                                      |
                                                                                      v
                                                          Linear(hidden_dim, 3) -> Grade 1 / 2 / 3
```

**Key design choices:**

- **Projection layers** -- both the 768-d image embedding and the 31-d clinical vector are projected to the same dimensionality (`proj_dim`, default 32 or 64) via learned linear layers. This puts both modalities on equal footing before fusion, preventing the higher-dimensional image branch from dominating.
- **Concatenation fusion** -- the two projected embeddings are simply concatenated into a single `2 * proj_dim` vector. This is a straightforward early-fusion strategy that lets the downstream classifier learn cross-modal interactions.
- **Frozen image encoder** -- the Swin-Tiny backbone is pretrained on ImageNet and kept frozen. Only the projection layers and classifier head are trained, which avoids overfitting given the dataset size (~900 patients).
- **Clinical feature encoding** -- the 31-d clinical vector is composed of 20 binary features (0/1), 8 one-hot categorical features, and 3 standard-scaled numerical features.
- **Balanced batch sampling** -- a `WeightedRandomSampler` oversamples minority classes (Grade 1, Grade 3) so each training batch has roughly equal class representation, counteracting the Grade 2-heavy class imbalance.
- **Early stopping** -- training halts if the validation loss does not improve for 20 consecutive epochs (patience=20), and the best-performing weights are restored. This prevents overfitting on the small dataset.

## Key Findings from Baselines

### Image preprocessing

- **RGB composites outperform grayscale.** Encoding pre-contrast, post-contrast, and subtraction images into separate RGB channels produced better results than using single-channel grayscale inputs. The RGB composite integrates more information about tumor contrast uptake into a single image and emulates how radiologists analyze DCE-MRI by comparing pre- and post-contrast phases side by side.
- **Cropping to the tumor region improves performance.** Narrowing the field of view to the annotated bounding box (with either proportional or fixed-size padding) consistently improved classification accuracy over using full-size slices. The crop provides visual assistance that helps the model focus on the tumor rather than irrelevant background tissue.

### Unimodal vs. multimodal model performance
- **Multimodal model fails to identify Grade 1 tumors.** Grade 1 tumors closely resemble normal breast tissue with minimal contrast uptake, making them difficult to distinguish visually. The multimodal fusion model struggles with this class, misclassifying most Grade 1 cases as Grade 2. This highlights a key limitation that more advanced architectures (e.g., multiplex graph networks) may need to address.

## Compute Requirements

### Hardware

All experiments were run on an Apple Silicon Mac with MPS (Metal Performance Shaders). The Ovis2 VLM pipeline requires a GPU (MPS or CUDA); the Swin-Tiny baseline runs on CPU.

### Model sizes

| Model | Parameters | Disk / VRAM (approx.) |
|-------|------------|----------------------|
| Ovis2-4B (bfloat16) | 4B | ~8 GB |
| Swin-Tiny (ImageNet-1K, frozen) | 28M | ~110 MB |

### Runtime

| Pipeline | Patients | Total runtime | Notes |
|----------|----------|---------------|-------|
| Swin-Tiny + MLP | 899 | ~166 s | Feature extraction + 36-combo grid search (5-fold CV) + final training |
| Graph-Augmented Swin + MLP | 899 | ~166 s | Same architecture + graph smoothness regularization |
| DMGI | 899 | ~1559 s | 717 epochs (early stopped from 2000), 5-seed evaluation |
| Ovis2 proportional crop | 60 | ~15-20 min | 3 slices/patient, sequential few-shot inference on MPS |
| Ovis2 no crop | 60 | ~15-20 min | Same as above; full-size images are slower per call |
| Ovis2 fixed 256x256 | 100 | ~25-35 min | 3 slices/patient, sequential few-shot inference on MPS |

### Breakdown

- **DICOM processing** -- ~1-2 min for the full dataset (899 patients). Composites are cached as PNGs so this cost is paid only once.
- **Ovis2 model loading** -- ~30-60 s to download and load the 4B-parameter model onto MPS. Sub-second on subsequent runs if weights are cached locally.
- **Ovis2 per-patient inference** -- ~5-10 s per patient (3 forward passes with `max_new_tokens=16`). The main bottleneck is sequential autoregressive decoding on MPS.
- **Swin-Tiny feature extraction** -- ~90 s for 899 patients (2697 images). Features are cached to disk (`swin_features.npy`) for reuse.
- **Swin grid search** -- ~50 s for 36 hyperparameter combinations x 5 folds. Each fold trains a small MLP (< 1K trainable parameters) for up to 200 epochs with early stopping.
- **DMGI training** -- ~26 min for 717 epochs on the full graph (899 nodes, 4 relations). Includes 5-seed multi-run evaluation with LogReg and fine-tuned MLP.

### Disk space

| Item | Size |
|------|------|
| DICOM data (`data/Duke-Breast-Cancer-MRI/`) | ~250 GB (full dataset from TCIA) |
| Cached composites (PNGs, per crop mode) | ~5-15 MB |
| Swin feature cache (`swin_features.npy`) | ~2.7 MB |
| Ovis2 model weights (HuggingFace cache) | ~8 GB |
