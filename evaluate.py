# evaluate.py
# Loads a trained model and measures its real-world performance on the
# held-out test set — data the model has never seen during training or tuning.
#
# Produces:
#   • Overall accuracy, loss
#   • Per-class precision, recall, F1-score
#   • Confusion matrix heatmap
#   • Training history curves (loss & accuracy)
#   • Per-class accuracy bar chart

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from config import CLASS_NAMES, PLOTS_DIR, MODEL_DIR, NUM_CLASSES


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the model on the test set and return predictions + loss.

    Parameters
    ----------
    model      : trained Keras model
    X_test     : scaled float32 feature array
    y_test     : int32 label array (0-based)
    batch_size : inference batch size (no effect on results, only speed)

    Returns
    -------
    y_pred     : predicted class indices (0-based)
    y_proba    : softmax probability array  shape (n_samples, 7)
    test_loss  : scalar cross-entropy loss on the test set
    """
    print("\n[evaluate] Running inference on test set...")
    results  = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    test_loss, test_acc = results[0], results[1]

    y_proba = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred  = np.argmax(y_proba, axis=1)

    print(f"[evaluate] Test loss     : {test_loss:.4f}")
    print(f"[evaluate] Test accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    return y_pred, y_proba, test_loss


# ---------------------------------------------------------------------------
# Metrics reporting
# ---------------------------------------------------------------------------

def print_classification_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Print and return a full per-class classification report.

    Metrics explained
    -----------------
    Precision : of all the times we predicted class X, how often were we right?
    Recall    : of all the actual class X samples, how many did we catch?
    F1-score  : harmonic mean of precision and recall — the balanced metric.
    Support   : how many test samples belong to each class.
    """
    print("\n[evaluate] Classification Report:")
    print("─" * 70)
    report_str = classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )
    print(report_str)

    report_dict = classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=True,
    )
    return report_dict


def print_per_class_accuracy(
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute and print per-class accuracy separately from the main report.

    Per-class accuracy = correct predictions for class X / total class X samples.
    Useful for spotting which specific cover types are hardest to classify.
    """
    print("\n[evaluate] Per-class accuracy:")
    print("─" * 50)
    per_class = {}
    for cls_idx in range(NUM_CLASSES):
        mask     = y_test == cls_idx
        if mask.sum() == 0:
            continue
        acc      = accuracy_score(y_test[mask], y_pred[mask])
        name     = CLASS_NAMES[cls_idx]
        bar      = "█" * int(acc * 30)
        print(f"  [{cls_idx}] {name:<22s}: {acc:.4f}  {bar}")
        per_class[name] = acc
    return per_class


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True,
    save: bool = True,
) -> None:
    """Heatmap of the confusion matrix.

    Parameters
    ----------
    normalize : if True, show row-normalized percentages (easier to read
                when classes are imbalanced); if False show raw counts.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)

    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt, title_suffix = ".2f", "(row-normalized)"
    else:
        cm_display = cm
        fmt, title_suffix = "d", "(raw counts)"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.4,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(f"Confusion matrix {title_suffix}", fontsize=12)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
        plt.savefig(path, dpi=150)
        print(f"[evaluate] Plot saved → {path}")
    plt.show()


def plot_training_history(
    history,
    save: bool = True,
) -> None:
    """Side-by-side plots of loss and accuracy over training epochs.

    The gap between train and val curves tells you about overfitting:
      - Curves close together → good generalisation
      - Large gap (train much better) → overfitting
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    hist = history.history

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    # ── Loss ──────────────────────────────────────────────────────────────
    ax1.plot(hist["loss"],     label="Train loss",      color="#4C72B0", linewidth=1.8)
    ax1.plot(hist["val_loss"], label="Val loss",  color="#DD8452",
             linewidth=1.8, linestyle="--")
    ax1.set_title("Loss over epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Sparse categorical cross-entropy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── Accuracy ───────────────────────────────────────────────────────────
    ax2.plot(hist["accuracy"],     label="Train accuracy",      color="#4C72B0", linewidth=1.8)
    ax2.plot(hist["val_accuracy"], label="Val accuracy",  color="#DD8452",
             linewidth=1.8, linestyle="--")
    ax2.set_title("Accuracy over epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("Training history", fontsize=13)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "training_history.png")
        plt.savefig(path, dpi=150)
        print(f"[evaluate] Plot saved → {path}")
    plt.show()


def plot_per_class_accuracy(
    per_class_acc: dict[str, float],
    save: bool = True,
) -> None:
    """Horizontal bar chart of per-class accuracy — easy to spot weak classes."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    names  = list(per_class_acc.keys())
    values = list(per_class_acc.values())
    colors = ["#2ecc71" if v >= 0.80 else "#e67e22" if v >= 0.60 else "#e74c3c"
              for v in values]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            min(val + 0.01, 0.98), bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", fontsize=9,
        )

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Per-class accuracy")
    ax.set_title("Per-class accuracy on test set\n"
                 "(green ≥ 80%,  orange ≥ 60%,  red < 60%)", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.axvline(x=0.80, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "per_class_accuracy.png")
        plt.savefig(path, dpi=150)
        print(f"[evaluate] Plot saved → {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Full evaluation pipeline — single call convenience wrapper
# ---------------------------------------------------------------------------

def run_full_evaluation(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    history=None,
) -> dict:
    """Run all evaluation steps in sequence.

    Parameters
    ----------
    model   : trained Keras model
    X_test  : scaled float32 test features
    y_test  : int32 test labels (0-based)
    history : keras History object (optional — skips curve plot if None)

    Returns
    -------
    results dict with keys: y_pred, y_proba, test_loss, report, per_class_acc
    """
    y_pred, y_proba, test_loss = evaluate_model(model, X_test, y_test)
    report       = print_classification_report(y_test, y_pred)
    per_class    = print_per_class_accuracy(y_test, y_pred)

    plot_confusion_matrix(y_test, y_pred, normalize=True)
    plot_per_class_accuracy(per_class)

    if history is not None:
        plot_training_history(history)

    return {
        "y_pred"       : y_pred,
        "y_proba"      : y_proba,
        "test_loss"    : test_loss,
        "report"       : report,
        "per_class_acc": per_class,
    }


# ---------------------------------------------------------------------------
# Entry point — loads saved model and evaluates on test set
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tensorflow as tf
    from load_data import load_raw_data
    from preprocessing import preprocess
    from model import load_model

    print("[evaluate] Loading data and preprocessing...")
    df = load_raw_data()
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     class_weights, _) = preprocess(df)

    # Load whichever saved model exists — tuned first, then baseline
    # Note: ModelCheckpoint appends '_best' to the run_name, so we check
    # both variants (e.g. 'best_random_best' and 'best_random')
    for model_name in ["best_random_best", "best_hyperband_best", "best_random", "best_hyperband", "baseline_best", "final_model"]:
        model_path = os.path.join(MODEL_DIR, f"{model_name}.keras")
        if os.path.exists(model_path):
            print(f"[evaluate] Found model: {model_name}")
            model = load_model(model_name)
            break
    else:
        raise FileNotFoundError(
            "No saved model found in saved_models/. "
            "Run model.py or tuning.py first."
        )

    results = run_full_evaluation(model, X_test, y_test, history=None)

    print("\n[evaluate] ── Summary ─────────────────────────────────────────")
    print(f"  Overall accuracy : {results['report']['accuracy']:.4f}")
    print(f"  Macro F1-score   : {results['report']['macro avg']['f1-score']:.4f}")
    print(f"  Weighted F1-score: {results['report']['weighted avg']['f1-score']:.4f}")