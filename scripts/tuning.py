# tuning.py
# Hyperparameter search for the forest cover DNN.
#
# Strategy: We use two complementary approaches:
#   1. RandomSearch  — broad exploration of the search space
#   2. Hyperband     — efficient follow-up that kills poor trials early
#
# What gets tuned:
#   - Number of hidden layers        (2 – 4)
#   - Neurons per layer              (64, 128, 256, 512)
#   - Dropout rate                   (0.1 – 0.5)
#   - Learning rate                  (1e-4 – 1e-2, log scale)
#   - Batch size                     (256, 512, 1024)

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

from config import (
    NUM_CLASSES,
    DROPOUT_RATE,
    LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    MODEL_DIR,
    RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# Search space — the model builder keras_tuner calls for each trial
# ---------------------------------------------------------------------------

def build_tunable_model(hp: kt.HyperParameters) -> keras.Model:
    """Define the search space and return a compiled model for one trial.

    keras_tuner calls this function repeatedly, each time with a different
    combination of hyperparameter values sampled from the ranges below.

    Parameters
    ----------
    hp : kt.HyperParameters
        Hyperparameter object injected by the tuner.

    Returns
    -------
    Compiled keras.Model for this trial.
    """
    # ── Hyperparameters to search ──────────────────────────────────────────
    n_layers = hp.Int(
        "n_layers", min_value=2, max_value=4, step=1,
        default=3,
    )
    units_choices = [64, 128, 256, 512]
    dropout = hp.Float(
        "dropout_rate", min_value=0.1, max_value=0.5, step=0.1,
        default=DROPOUT_RATE,
    )
    lr = hp.Float(
        "learning_rate", min_value=1e-4, max_value=1e-2,
        sampling="log", default=LEARNING_RATE,
    )

    # ── Build the graph ────────────────────────────────────────────────────
    inputs = keras.Input(shape=(54,), name="features")
    x = inputs

    for i in range(n_layers):
        units = hp.Choice(f"units_layer_{i}", values=units_choices, default=256)
        x = layers.Dense(units, use_bias=False, name=f"dense_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        x = layers.Dropout(dropout, name=f"dropout_{i}")(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Tuner factories
# ---------------------------------------------------------------------------

def get_random_search_tuner(
    tuner_dir: str,
    max_trials: int = 20,
    executions_per_trial: int = 1,
) -> kt.RandomSearch:
    """Broad random exploration of the search space.

    Good for a first pass — covers diverse corners of the hyperparameter
    space without any prior assumptions.

    Parameters
    ----------
    tuner_dir          : directory to store trial results and checkpoints
    max_trials         : total number of hyperparameter combinations to try
    executions_per_trial: how many times to train each config (>1 averages
                          out randomness; keep at 1 for speed)
    """
    return kt.RandomSearch(
        hypermodel=build_tunable_model,
        objective="val_accuracy",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=tuner_dir,
        project_name="random_search",
        seed=RANDOM_SEED,
        overwrite=False,          # Resume interrupted searches automatically
    )


def get_hyperband_tuner(
    tuner_dir: str,
    max_epochs: int = 30,
) -> kt.Hyperband:
    """Efficient search using the Hyperband early-stopping algorithm.

    Hyperband trains many configs for a few epochs, eliminates the worst
    performers, and allocates more epochs to promising ones. Much faster
    than full training for every trial.

    Parameters
    ----------
    tuner_dir  : directory to store trial results
    max_epochs : maximum epochs any single trial is allowed to run
    """
    return kt.Hyperband(
        hypermodel=build_tunable_model,
        objective="val_accuracy",
        max_epochs=max_epochs,
        factor=3,               # Reduction factor between Hyperband brackets
        directory=tuner_dir,
        project_name="hyperband",
        seed=RANDOM_SEED,
        overwrite=False,
    )


# ---------------------------------------------------------------------------
# Search runner
# ---------------------------------------------------------------------------

def run_search(
    tuner: kt.Tuner,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float],
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
) -> None:
    """Execute the hyperparameter search.

    Parameters
    ----------
    tuner        : a tuner instance from get_random_search_tuner() or
                   get_hyperband_tuner()
    X_train/val  : scaled float32 feature arrays
    y_train/val  : int32 label arrays (0-based)
    class_weights: from preprocessing.compute_class_weights()
    batch_size   : samples per gradient update
    epochs       : max epochs per trial (EarlyStopping may cut short)
    """
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    print(f"\n[tuning] Starting search: {tuner.__class__.__name__}")
    print(f"         Search space summary:")
    tuner.search_space_summary()

    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1,
    )
    print("\n[tuning] Search complete.")


# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------

def get_best_hyperparameters(
    tuner: kt.Tuner,
    top_n: int = 3,
) -> list[kt.HyperParameters]:
    """Return the top-n hyperparameter configurations found by the search."""
    best_hps = tuner.get_best_hyperparameters(num_trials=top_n)

    print(f"\n[tuning] Top {top_n} hyperparameter configurations:")
    for rank, hp in enumerate(best_hps, start=1):
        print(f"\n  ── Rank {rank} ──────────────────────────────────────────")
        n_layers = hp.get("n_layers")
        print(f"     Layers       : {n_layers}")
        for i in range(n_layers):
            print(f"     Layer {i} units : {hp.get(f'units_layer_{i}')}")
        print(f"     Dropout      : {hp.get('dropout_rate'):.2f}")
        print(f"     Learning rate: {hp.get('learning_rate'):.6f}")

    return best_hps


def build_best_model(
    tuner: kt.Tuner,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float],
    run_name: str = "tuned",
) -> tuple[keras.Model, keras.callbacks.History]:
    """Retrieve the best hyperparameters, build the model, and do a full
    training run with the complete callback stack from model.py.

    Returns
    -------
    (trained_model, history)
    """
    from model import get_callbacks, save_model

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n[tuning] Building final model with best hyperparameters...")

    model = build_tunable_model(best_hps)
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=best_hps.get("batch_size") if "batch_size" in best_hps.values else BATCH_SIZE,
        class_weight=class_weights,
        callbacks=get_callbacks(run_name),
        verbose=1,
    )

    save_model(model, name=run_name)
    return model, history


# ---------------------------------------------------------------------------
# Convenience: run both tuners sequentially
# ---------------------------------------------------------------------------

def run_full_tuning_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float],
    tuner_type: str = "hyperband",
    max_trials: int = 20,
) -> tuple[keras.Model, keras.callbacks.History]:
    """End-to-end tuning: search → best HPs → full training run.

    Parameters
    ----------
    tuner_type : "random" for RandomSearch, "hyperband" for Hyperband
    max_trials : only used when tuner_type="random"

    Returns
    -------
    (best_model, history)
    """
    tuner_dir = os.path.join(MODEL_DIR, "tuner")

    if tuner_type == "random":
        tuner = get_random_search_tuner(tuner_dir, max_trials=max_trials)
    elif tuner_type == "hyperband":
        tuner = get_hyperband_tuner(tuner_dir)
    else:
        raise ValueError(f"Unknown tuner_type '{tuner_type}'. Use 'random' or 'hyperband'.")

    run_search(tuner, X_train, y_train, X_val, y_val, class_weights)
    get_best_hyperparameters(tuner, top_n=3)
    model, history = build_best_model(
        tuner, X_train, y_train, X_val, y_val, class_weights,
        run_name=f"best_{tuner_type}",
    )
    return model, history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from load_data import load_raw_data
    from preprocessing import preprocess

    tf.random.set_seed(RANDOM_SEED)

    print("[tuning] Loading and preprocessing data...")
    df = load_raw_data()
    X_train, X_val, X_test, y_train, y_val, y_test, class_weights, _ = preprocess(df)

    # Default: Hyperband (fastest for exploration)
    # Switch to "random" for broader search at the cost of more time
    model, history = run_full_tuning_pipeline(
        X_train, y_train,
        X_val,   y_val,
        class_weights,
        tuner_type="random",
        max_trials=1,  # Only used for random search; ignored by Hyperband
    )

    print("\n[tuning] Pipeline complete.")
    print(f"  Final val accuracy : {max(history.history['val_accuracy']):.4f}")
    print(f"  Final val loss     : {min(history.history['val_loss']):.4f}")