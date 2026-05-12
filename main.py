# main.py
# Single entry point for the forest cover type classification pipeline.
#
# Usage
# -----
#   python3 main.py                        # full pipeline (train + evaluate)
#   python3 main.py --mode train           # train baseline model only
#   python3 main.py --mode tune            # hyperparameter search + train best
#   python3 main.py --mode evaluate        # evaluate a saved model on test set
#   python3 main.py --mode explore         # data exploration + plots only
#
# All behaviour is controlled via config.py — no need to edit this file.

import argparse
import os
import time

import numpy as np
import tensorflow as tf

from config import RANDOM_SEED, MODEL_DIR
from load_data import load_raw_data, summarise, plot_class_distribution
from load_data import plot_continuous_features, plot_correlation_heatmap
from preprocessing import preprocess
from model import build_model, train_model, save_model, load_model, set_seeds
from evaluate import run_full_evaluation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_header(title: str) -> None:
    """Print a clearly visible section header."""
    width = 60
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def _elapsed(start: float) -> str:
    secs = int(time.time() - start)
    return f"{secs // 60}m {secs % 60}s"


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_explore(df) -> None:
    """Stage 1 — Print summary statistics and save exploration plots."""
    _print_header("STAGE 1 · Data Exploration")
    summarise(df)
    plot_class_distribution(df)
    plot_continuous_features(df)
    plot_correlation_heatmap(df)
    print("\n[main] Exploration complete. Plots saved to plots/")


def stage_preprocess(df):
    """Stage 2 — Split, scale, and compute class weights."""
    _print_header("STAGE 2 · Preprocessing")
    return preprocess(df)


def stage_train(X_train, y_train, X_val, y_val, class_weights):
    """Stage 3 — Build and train the baseline model."""
    _print_header("STAGE 3 · Baseline Model Training")
    t0 = time.time()

    input_dim = X_train.shape[1]
    model     = build_model(input_dim=input_dim)
    model.summary()

    history = train_model(
        model, X_train, y_train, X_val, y_val,
        class_weights=class_weights,
        run_name="baseline",
    )

    save_model(model, name="baseline_final")
    print(f"\n[main] Training complete in {_elapsed(t0)}.")
    return model, history


def stage_tune(X_train, y_train, X_val, y_val, class_weights):
    """Stage 4 — Hyperparameter search and best-model training."""
    _print_header("STAGE 4 · Hyperparameter Tuning")
    t0 = time.time()

    from tuning import run_full_tuning_pipeline
    model, history = run_full_tuning_pipeline(
        X_train, y_train,
        X_val,   y_val,
        class_weights,
        tuner_type="random",
        max_trials=10,
    )

    print(f"\n[main] Tuning complete in {_elapsed(t0)}.")
    return model, history


def stage_evaluate(model, X_test, y_test, history=None):
    """Stage 5 — Full evaluation on the held-out test set."""
    _print_header("STAGE 5 · Evaluation")
    results = run_full_evaluation(model, X_test, y_test, history=history)

    _print_header("FINAL RESULTS")
    print(f"  Overall accuracy  : {results['report']['accuracy']:.4f}  "
          f"({results['report']['accuracy']*100:.2f}%)")
    print(f"  Macro F1-score    : {results['report']['macro avg']['f1-score']:.4f}")
    print(f"  Weighted F1-score : {results['report']['weighted avg']['f1-score']:.4f}")
    print(f"  Test loss         : {results['test_loss']:.4f}")

    print("\n  Per-class F1 scores:")
    for name in [r for r in results['report'] if r not in
                 ("accuracy", "macro avg", "weighted avg")]:
        f1 = results['report'][name]['f1-score']
        bar = "█" * int(f1 * 20)
        print(f"    {name:<22s}: {f1:.4f}  {bar}")

    return results


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------

def run_full_pipeline() -> None:
    """Train baseline → tune → evaluate. The complete end-to-end flow."""
    _print_header("FOREST COVER TYPE CLASSIFIER · Full Pipeline")
    t0 = time.time()

    df = load_raw_data()
    stage_explore(df)

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     class_weights, _) = stage_preprocess(df)

    # Baseline
    baseline_model, baseline_history = stage_train(
        X_train, y_train, X_val, y_val, class_weights
    )

    # Tuning
    tuned_model, tuned_history = stage_tune(
        X_train, y_train, X_val, y_val, class_weights
    )

    # Evaluate the tuned model (better than baseline)
    stage_evaluate(tuned_model, X_test, y_test, history=tuned_history)

    print(f"\n[main] Full pipeline complete in {_elapsed(t0)}. "
          f"All outputs saved to saved_models/ and plots/")


def run_train_only() -> None:
    """Train baseline model only — no tuning."""
    _print_header("FOREST COVER TYPE CLASSIFIER · Train Mode")
    df = load_raw_data()
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     class_weights, _) = stage_preprocess(df)
    model, history = stage_train(X_train, y_train, X_val, y_val, class_weights)
    stage_evaluate(model, X_test, y_test, history=history)


def run_tune_only() -> None:
    """Hyperparameter search only — assumes data has been preprocessed."""
    _print_header("FOREST COVER TYPE CLASSIFIER · Tune Mode")
    df = load_raw_data()
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     class_weights, _) = stage_preprocess(df)
    model, history = stage_tune(X_train, y_train, X_val, y_val, class_weights)
    stage_evaluate(model, X_test, y_test, history=history)


def run_evaluate_only() -> None:
    """Load the best saved model and evaluate on the test set."""
    _print_header("FOREST COVER TYPE CLASSIFIER · Evaluate Mode")
    df = load_raw_data()
    (_, _, X_test,
     _, _, y_test,
     _, _) = stage_preprocess(df)

    # Search for best available saved model
    for name in ["best_random_best", "best_hyperband_best",
                 "best_random", "best_hyperband",
                 "baseline_final", "final_model"]:
        path = os.path.join(MODEL_DIR, f"{name}.keras")
        if os.path.exists(path):
            model = load_model(name)
            break
    else:
        raise FileNotFoundError(
            "No saved model found in saved_models/. "
            "Run with --mode train or --mode tune first."
        )

    stage_evaluate(model, X_test, y_test, history=None)


def run_explore_only() -> None:
    """Data exploration and plots only — no training."""
    _print_header("FOREST COVER TYPE CLASSIFIER · Explore Mode")
    df = load_raw_data()
    stage_explore(df)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forest Cover Type Classifier",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "train", "tune", "evaluate", "explore"],
        default="full",
        help=(
            "full     — explore + train baseline + tune + evaluate  (default)\n"
            "train    — train baseline model and evaluate\n"
            "tune     — hyperparameter search and evaluate best model\n"
            "evaluate — evaluate the best saved model on the test set\n"
            "explore  — data exploration and plots only\n"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    set_seeds(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    args = parse_args()

    mode_map = {
        "full"    : run_full_pipeline,
        "train"   : run_train_only,
        "tune"    : run_tune_only,
        "evaluate": run_evaluate_only,
        "explore" : run_explore_only,
    }

    mode_map[args.mode]()