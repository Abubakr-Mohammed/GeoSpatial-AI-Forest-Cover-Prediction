# model.py
# Defines, compiles, trains, and persists the deep neural network classifier.
#
# Design principles:
#   - Architecture is fully driven by config.py — no magic numbers here.
#   - build_model() is a pure function: same inputs → same network.
#   - Training concerns (callbacks, history) are separated from architecture.

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    NUM_CLASSES,
    HIDDEN_UNITS,
    DROPOUT_RATE,
    LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    MODEL_DIR,
    RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Fix Python, NumPy, and TensorFlow seeds for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

def build_model(
    input_dim: int,
    hidden_units: list[int] = HIDDEN_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE,
) -> keras.Model:
    """Build and compile a feed-forward DNN for multi-class classification.

    Architecture per hidden layer
    ─────────────────────────────
      Dense(n)  →  BatchNormalization  →  ReLU  →  Dropout(rate)

    BatchNormalization stabilises training on features with very different
    scales (even after StandardScaler, the 44 binary columns remain sparse).
    Dropout regularises to reduce overfitting on the large majority classes.

    Parameters
    ----------
    input_dim    : number of input features (54 for this dataset)
    hidden_units : list of neuron counts, one entry per hidden layer
    dropout_rate : fraction of units dropped during training
    learning_rate: initial Adam learning rate

    Returns
    -------
    Compiled keras.Model ready for model.fit()
    """
    set_seeds()

    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs

    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, use_bias=False, name=f"dense_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="forest_cover_dnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def get_callbacks(run_name: str = "baseline") -> list:
    """Return the standard callback stack used during training.

    Callbacks
    ---------
    EarlyStopping   : halt when val_loss stops improving; restore best weights.
    ModelCheckpoint : save the best model to disk during training.
    ReduceLROnPlateau: halve the learning rate after 3 stagnant val_loss epochs.
    TensorBoard     : logs for optional TensorBoard visualisation.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    log_dir      = os.path.join(MODEL_DIR, "logs", run_name)
    checkpoint   = os.path.join(MODEL_DIR, f"{run_name}_best.keras")

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float],
    run_name: str = "baseline",
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
) -> keras.callbacks.History:
    """Fit *model* on the training data and return the History object.

    Parameters
    ----------
    model        : compiled Keras model from build_model()
    X_train/val  : scaled float32 feature arrays
    y_train/val  : int32 label arrays (0-based)
    class_weights: from preprocessing.compute_class_weights()
    run_name     : label used for checkpoint filename and TensorBoard logs
    batch_size   : number of samples per gradient update
    epochs       : maximum training epochs (EarlyStopping may stop sooner)

    Returns
    -------
    keras.callbacks.History — contains loss/accuracy curves for plotting
    """
    print(f"\n[model] Starting training run: '{run_name}'")
    print(f"        Epochs (max): {epochs}  |  Batch size: {batch_size}")
    print(f"        Input shape : {X_train.shape}  |  Classes: {NUM_CLASSES}\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=get_callbacks(run_name),
        verbose=1,
    )
    return history


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_model(model: keras.Model, name: str = "final_model") -> str:
    """Save the full model (architecture + weights) to MODEL_DIR."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{name}.keras")
    model.save(path)
    print(f"[model] Model saved → {path}")
    return path


def load_model(name: str = "final_model") -> keras.Model:
    """Load a previously saved model from MODEL_DIR."""
    path = os.path.join(MODEL_DIR, f"{name}.keras")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model found at '{path}'. Train a model first."
        )
    model = keras.models.load_model(path)
    print(f"[model] Model loaded from '{path}'.")
    return model


# ---------------------------------------------------------------------------
# Quick architecture smoke-test (no training)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    INPUT_DIM = 54  # 10 continuous + 4 wilderness + 40 soil types

    model = build_model(input_dim=INPUT_DIM)
    model.summary()

    print("\n[model] Callback list:")
    for cb in get_callbacks("smoke_test"):
        print(f"  • {cb.__class__.__name__}")

    # Verify a forward pass with random data
    dummy_X = np.random.rand(8, INPUT_DIM).astype(np.float32)
    dummy_y = np.random.randint(0, NUM_CLASSES, size=(8,)).astype(np.int32)
    dummy_w = {i: 1.0 for i in range(NUM_CLASSES)}

    loss, acc = model.evaluate(dummy_X, dummy_y, verbose=0)
    print(f"\n[model] Dummy forward pass — loss: {loss:.4f}, acc: {acc:.4f}")
    print("[model] Architecture check passed ✓")