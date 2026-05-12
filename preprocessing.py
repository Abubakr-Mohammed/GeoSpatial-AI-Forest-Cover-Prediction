# preprocessing.py
# Handles all data preparation steps:
#   1. Feature / target separation
#   2. Label remapping  (1-7  →  0-6)
#   3. Stratified train / val / test split
#   4. StandardScaler fitted ONLY on training data
#   5. Class-weight computation for imbalanced training

import numpy as np
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from config import (
    TARGET_COLUMN,
    CONTINUOUS_FEATURES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    NUM_CLASSES,
    MODEL_DIR,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_ratios() -> None:
    total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0 — got {total:.4f}. "
            "Check TRAIN_RATIO, VAL_RATIO, TEST_RATIO in config.py."
        )


def _remap_labels(series: pd.Series) -> np.ndarray:
    """Shift class labels from 1-based (1-7) to 0-based (0-6)."""
    return (series - 1).to_numpy(dtype=np.int32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           np.ndarray,  np.ndarray,  np.ndarray]:
    """Stratified split into train / val / test sets.

    Returns
    -------
    X_train, X_val, X_test : pd.DataFrame  (raw, unscaled features)
    y_train, y_val, y_test : np.ndarray    (0-based integer labels)
    """
    _validate_ratios()

    X = df.drop(columns=[TARGET_COLUMN])
    y = _remap_labels(df[TARGET_COLUMN])

    # First cut: train vs temp (val + test)
    val_test_ratio = VAL_RATIO + TEST_RATIO
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=val_test_ratio,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    # Second cut: val vs test (relative size within temp)
    relative_test = TEST_RATIO / val_test_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test,
        stratify=y_temp,
        random_state=RANDOM_SEED,
    )

    print(
        f"[preprocessing] Split complete:\n"
        f"  Train : {len(X_train):>7,} samples ({100*TRAIN_RATIO:.0f}%)\n"
        f"  Val   : {len(X_val):>7,} samples ({100*VAL_RATIO:.0f}%)\n"
        f"  Test  : {len(X_test):>7,} samples ({100*TEST_RATIO:.0f}%)"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the TRAINING set's continuous features only.

    The fitted scaler is also persisted to MODEL_DIR so it can be reloaded
    at inference time without re-fitting.
    """
    scaler = StandardScaler()
    scaler.fit(X_train[CONTINUOUS_FEATURES])

    os.makedirs(MODEL_DIR, exist_ok=True)
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[preprocessing] Scaler fitted and saved → {scaler_path}")
    return scaler


def apply_scaler(
    scaler: StandardScaler,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a pre-fitted scaler to all three splits.

    Only CONTINUOUS_FEATURES are scaled; binary columns pass through
    unchanged. Returns three numpy arrays ready for TensorFlow.
    """
    def _transform(X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        X[CONTINUOUS_FEATURES] = scaler.transform(X[CONTINUOUS_FEATURES])
        return X.to_numpy(dtype=np.float32)

    X_train_s = _transform(X_train)
    X_val_s   = _transform(X_val)
    X_test_s  = _transform(X_test)

    print(
        f"[preprocessing] Scaling applied.\n"
        f"  Feature matrix shape — Train: {X_train_s.shape}, "
        f"Val: {X_val_s.shape}, Test: {X_test_s.shape}"
    )
    return X_train_s, X_val_s, X_test_s


def compute_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights to counter label imbalance.

    Weights are passed to model.fit(class_weight=...) so that minority
    classes (e.g. Cottonwood/Willow — 0.5% of data) receive proportionally
    more gradient signal during training.

    Returns a dict  {class_index: weight}  indexed 0-based.
    """
    classes  = np.arange(NUM_CLASSES)
    weights  = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    weight_dict = dict(enumerate(weights))

    print("[preprocessing] Class weights (higher = rarer class):")
    from config import CLASS_NAMES
    for idx, w in weight_dict.items():
        print(f"  [{idx}] {CLASS_NAMES[idx]:<22s}: {w:.4f}")

    return weight_dict


def load_scaler(path: str | None = None) -> StandardScaler:
    """Reload a previously saved scaler from disk (for inference / evaluation)."""
    if path is None:
        path = os.path.join(MODEL_DIR, "scaler.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Scaler not found at '{path}'. Run preprocessing first."
        )
    scaler = joblib.load(path)
    print(f"[preprocessing] Scaler loaded from '{path}'.")
    return scaler


def preprocess(
    df: pd.DataFrame,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    dict[int, float],
    StandardScaler,
]:
    """Full preprocessing pipeline — single call convenience wrapper.

    Steps
    -----
    1. Stratified split
    2. Fit scaler on train, apply to all splits
    3. Compute class weights

    Returns
    -------
    X_train, X_val, X_test  : scaled float32 numpy arrays
    y_train, y_val, y_test  : int32 label arrays (0-based)
    class_weights           : dict for model.fit
    scaler                  : fitted StandardScaler instance
    """
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    scaler = fit_scaler(X_train)
    X_train_s, X_val_s, X_test_s = apply_scaler(scaler, X_train, X_val, X_test)
    class_weights = compute_class_weights(y_train)

    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, class_weights, scaler


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from load_data import load_raw_data
    df = load_raw_data()

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     class_weights, scaler) = preprocess(df)

    # Sanity checks
    print("\n── Sanity checks ───────────────────────────────────────────────────")

    # 1. No data leakage: scaler stats come only from train
    print(f"  Scaler mean (Elevation): {scaler.mean_[0]:.2f}")

    # 2. Label range
    print(f"  y_train range : {y_train.min()} – {y_train.max()}  (expected 0–6)")

    # 3. Stratification: class proportions similar across splits
    for name, y in [("Train", y_train), ("Val  ", y_val), ("Test ", y_test)]:
        dominant = np.bincount(y).argmax()
        pct = 100 * np.bincount(y)[dominant] / len(y)
        print(f"  {name} dominant class: {dominant}  ({pct:.1f}%)")

    # 4. Feature dtype
    print(f"  X_train dtype : {X_train.dtype}  (expected float32)")
    print(f"  X_train shape : {X_train.shape}")