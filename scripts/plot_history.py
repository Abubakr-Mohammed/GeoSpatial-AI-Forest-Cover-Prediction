# plot_history.py
# Reads TensorBoard event files from saved_models/logs/ and saves
# training_history.png without needing TensorBoard to be installed.
#
# Usage:
#   python3 plot_history.py

import os
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record


def read_events(log_dir: str) -> dict[str, list]:
    """Read scalar values from all TFEvent files in a directory."""
    data = {}
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))

    for ef in sorted(event_files):
        try:
            for record in tf_record.tf_record_iterator(ef):
                event = event_pb2.Event.FromString(record)
                for value in event.summary.value:
                    tag = value.tag
                    if tag not in data:
                        data[tag] = []
                    # Handle both old and new TF summary formats
                    if value.HasField("simple_value"):
                        data[tag].append((event.step, value.simple_value))
                    elif value.HasField("tensor"):
                        import tensorflow as tf
                        scalar = tf.make_ndarray(value.tensor).item()
                        data[tag].append((event.step, scalar))
        except Exception:
            continue

    # Sort each tag's values by step
    return {tag: sorted(vals, key=lambda x: x[0]) for tag, vals in data.items()}


def plot_history_from_logs(
    log_base: str = "saved_models/logs",
    save_path: str = "plots/training_history.png",
) -> None:
    """Find train/val event logs, extract loss + accuracy, and plot them."""

    # Find the most recent run folder inside logs/
    run_dirs = sorted([
        d for d in os.listdir(log_base)
        if os.path.isdir(os.path.join(log_base, d))
    ])
    if not run_dirs:
        raise FileNotFoundError(f"No run folders found in '{log_base}'.")

    run_name = run_dirs[-1]
    print(f"[plot_history] Using run: {run_name}")

    train_dir = os.path.join(log_base, run_name, "train")
    val_dir   = os.path.join(log_base, run_name, "validation")

    train_data = read_events(train_dir)
    val_data   = read_events(val_dir)

    # Extract loss and accuracy series
    def get_series(data, key):
        matches = [v for k, v in data.items() if key.lower() in k.lower()]
        return [v for _, v in matches[0]] if matches else []

    train_loss = get_series(train_data, "loss")
    val_loss   = get_series(val_data,   "loss")
    train_acc  = get_series(train_data, "accuracy")
    val_acc    = get_series(val_data,   "accuracy")

    epochs = range(1, max(len(train_loss), len(train_acc)) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    # ── Loss ──────────────────────────────────────────────────────────────
    if train_loss:
        ax1.plot(range(1, len(train_loss)+1), train_loss,
                 label="Train loss", color="#4C72B0", linewidth=1.8)
    if val_loss:
        ax1.plot(range(1, len(val_loss)+1), val_loss,
                 label="Val loss", color="#DD8452", linewidth=1.8, linestyle="--")
    ax1.set_title("Loss over epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Sparse categorical cross-entropy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── Accuracy ──────────────────────────────────────────────────────────
    if train_acc:
        ax2.plot(range(1, len(train_acc)+1), train_acc,
                 label="Train accuracy", color="#4C72B0", linewidth=1.8)
    if val_acc:
        ax2.plot(range(1, len(val_acc)+1), val_acc,
                 label="Val accuracy", color="#DD8452", linewidth=1.8, linestyle="--")
    ax2.set_title("Accuracy over epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle(f"Training history — {run_name}", fontsize=13)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"[plot_history] Saved → {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_history_from_logs()