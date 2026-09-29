"""Clinical data parsing, text formatting, and feature encoding."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (
    CLINICAL_FILE, LABEL_COLUMN, LABEL_MAP,
    CLINICAL_FEATURES_TEXT, BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
)


def load_clinical_dataframe(clinical_file: str = CLINICAL_FILE) -> pd.DataFrame:
    """
    Parse the clinical xlsx into a clean DataFrame.
    Standardized on Clinical_and_Other_Features_Full.xlsx format (iloc[0] = column names).
    """
    raw = pd.read_excel(clinical_file, header=None)
    col_names = raw.iloc[0].tolist()

    seen = {}
    unique_cols = []
    for c in col_names:
        c = str(c).strip() if pd.notna(c) else f"_unnamed_{len(unique_cols)}"
        if c in seen:
            seen[c] += 1
            unique_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            unique_cols.append(c)

    df = raw.iloc[2:].copy()
    df.columns = unique_cols
    df = df.reset_index(drop=True)

    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
    df = df.dropna(subset=[LABEL_COLUMN])
    df = df[df[LABEL_COLUMN].isin(LABEL_MAP.keys())]
    return df


def build_clinical_text(row: pd.Series) -> str:
    """Format clinical columns into a readable string for VLM prompts."""
    parts = []
    for col_name, label, decode_map in CLINICAL_FEATURES_TEXT:
        val = row.get(col_name)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        try:
            val_int = int(float(val))
        except (ValueError, TypeError):
            parts.append(f"{label}: {val}")
            continue
        if decode_map:
            parts.append(f"{label}: {decode_map.get(val_int, str(val))}")
        else:
            parts.append(f"{label}: {val_int}")
    return "; ".join(parts) if parts else "No clinical data"


def encode_clinical_features(df: pd.DataFrame, scaler: StandardScaler = None,
                             fit_scaler: bool = True) -> tuple:
    """
    Encode clinical features into a numeric array for the Swin MLP.

    Returns:
        (feature_array [N, D], fitted StandardScaler)
    """
    parts = []

    for col in BINARY_FEATURES:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float).values
        parts.append(vals.reshape(-1, 1))

    for col, categories in CATEGORICAL_FEATURES.items():
        raw_vals = df[col].values
        one_hot = np.zeros((len(df), len(categories)), dtype=np.float32)
        for i, val in enumerate(raw_vals):
            try:
                val_cast = int(float(val))
            except (ValueError, TypeError):
                val_cast = str(val).strip()
            for j, cat in enumerate(categories):
                if val_cast == cat or str(val_cast) == str(cat):
                    one_hot[i, j] = 1.0
                    break
        parts.append(one_hot)

    num_cols = []
    for col in NUMERICAL_FEATURES:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float).values
        num_cols.append(vals.reshape(-1, 1))
    num_array = np.hstack(num_cols)

    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        num_array = scaler.fit_transform(num_array)
    else:
        num_array = scaler.transform(num_array)
    parts.append(num_array)

    return np.hstack(parts).astype(np.float32), scaler
