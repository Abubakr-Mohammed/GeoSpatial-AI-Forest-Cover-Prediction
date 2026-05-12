# load_data.py
# Responsible for loading the raw CSV and providing exploration utilities.
# Nothing here modifies the data — that is preprocessing.py's job.

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from config import DATA_PATH, TARGET_COLUMN, CLASS_NAMES, PLOTS_DIR


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV from *path* and return a DataFrame.

    Raises FileNotFoundError with a clear message if the file is missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Update DATA_PATH in config.py to point to your cover_data.csv."
        )
    df = pd.read_csv(path)
    print(f"[load_data] Loaded {len(df):,} rows × {df.shape[1]} columns from '{path}'.")
    return df


# ---------------------------------------------------------------------------
# Exploration helpers
# ---------------------------------------------------------------------------

def summarise(df: pd.DataFrame) -> None:
    """Print a concise statistical and structural summary of the DataFrame."""
    print("\n── Shape ──────────────────────────────────────────────────────────")
    print(f"  Rows: {df.shape[0]:,}   Columns: {df.shape[1]}")

    print("\n── Data types ─────────────────────────────────────────────────────")
    print(df.dtypes.value_counts().to_string())

    print("\n── Missing values ──────────────────────────────────────────────────")
    missing = df.isnull().sum()
    if missing.any():
        print(missing[missing > 0].to_string())
    else:
        print("  None — dataset is complete.")

    print("\n── Continuous feature statistics ───────────────────────────────────")
    from config import CONTINUOUS_FEATURES
    print(df[CONTINUOUS_FEATURES].describe().round(2).to_string())

    print("\n── Class distribution ──────────────────────────────────────────────")
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    for cls_idx, count in counts.items():
        name = CLASS_NAMES[int(cls_idx) - 1]
        pct  = 100 * count / len(df)
        bar  = "█" * int(pct / 2)
        print(f"  Class {cls_idx} ({name:<20s}): {count:>7,}  {pct:5.1f}%  {bar}")


def plot_class_distribution(df: pd.DataFrame, save: bool = True) -> None:
    """Bar chart of class frequencies."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    counts = df[TARGET_COLUMN].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = sns.color_palette("viridis", len(counts))
    bars = ax.bar(CLASS_NAMES, counts.values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2000,
            f"{val:,}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_title("Class distribution — forest cover types", fontsize=12)
    ax.set_ylabel("Sample count")
    ax.set_xlabel("Cover type")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "class_distribution.png")
        plt.savefig(path, dpi=150)
        print(f"[load_data] Plot saved → {path}")
    plt.show()


def plot_continuous_features(df: pd.DataFrame, save: bool = True) -> None:
    """Grid of histograms for the 10 continuous features."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    from config import CONTINUOUS_FEATURES

    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    axes = axes.flatten()

    for ax, feature in zip(axes, CONTINUOUS_FEATURES):
        ax.hist(df[feature], bins=50, color="#4C72B0", edgecolor="none", alpha=0.85)
        ax.set_title(feature, fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("Count", fontsize=7)
        ax.tick_params(labelsize=7)

    plt.suptitle("Distribution of continuous features", fontsize=12, y=1.01)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "continuous_features.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[load_data] Plot saved → {path}")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True) -> None:
    """Correlation heatmap for continuous features only."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    from config import CONTINUOUS_FEATURES

    corr = df[CONTINUOUS_FEATURES].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.4, ax=ax, annot_kws={"size": 7},
        vmin=-1, vmax=1,
    )
    ax.set_title("Feature correlation — continuous variables", fontsize=12)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
        plt.savefig(path, dpi=150)
        print(f"[load_data] Plot saved → {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point (run this file directly for a quick exploration report)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_raw_data()
    summarise(df)
    plot_class_distribution(df)
    plot_continuous_features(df)
    plot_correlation_heatmap(df)