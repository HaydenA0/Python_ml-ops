"""Exploratory Data Analysis — class distribution, demographics, and imaging position."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from python.pcxp_mlops.config import DEFAULT_CLASS_NAMES, get_paths

PLOTS_DIR = Path("plots")


def _ensure_plots_dir():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_data():
    paths = get_paths()
    df = pd.read_csv(paths.train_metadata_csv)
    df = df.drop_duplicates(subset=["patientId"]).reset_index(drop=True)
    return df


def _plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Target"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=["steelblue", "coral"], width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(DEFAULT_CLASS_NAMES)
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, str(v), ha="center", va="bottom")
    ratio = counts[1] / counts.sum()
    ax.text(
        0.5, max(counts.values) * 0.95,
        f"Positive ratio: {ratio:.2%}",
        ha="center", va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_distribution.png")
    plt.close(fig)


def _plot_age_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["age"].dropna(), bins=30, color="steelblue", edgecolor="white")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.set_title("Age Distribution")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "age_distribution.png")
    plt.close(fig)


def _plot_sex_distribution(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["sex"].value_counts()
    ax.bar(counts.index, counts.values, color=["cornflowerblue", "lightcoral"], width=0.4)
    ax.set_ylabel("Count")
    ax.set_title("Sex Distribution")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sex_distribution.png")
    plt.close(fig)


def _plot_position_distribution(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["position"].value_counts()
    ax.bar(counts.index, counts.values, color=["seagreen", "goldenrod"], width=0.4)
    ax.set_ylabel("Count")
    ax.set_title("Imaging Position Distribution")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "position_distribution.png")
    plt.close(fig)


def _print_summary(df):
    total = len(df)
    pos = df["Target"].sum()
    neg = total - pos
    print(f"Total patients (unique): {total}")
    print(f"  No Lung Opacity: {neg} ({neg/total:.2%})")
    print(f"  Lung Opacity:    {pos} ({pos/total:.2%})")
    print()
    print(f"Age — mean: {df['age'].mean():.1f}, median: {df['age'].median():.1f}, "
          f"std: {df['age'].std():.1f}")
    print(f"Sex — {df['sex'].value_counts().to_dict()}")
    print(f"Position — {df['position'].value_counts().to_dict()}")


def main():
    _ensure_plots_dir()
    df = _load_data()
    print("=" * 50)
    print("EDA — RSNA Pneumonia Detection Dataset")
    print("=" * 50)
    _print_summary(df)
    print()
    _plot_class_distribution(df)
    print("Saved plots/class_distribution.png")
    _plot_age_distribution(df)
    print("Saved plots/age_distribution.png")
    _plot_sex_distribution(df)
    print("Saved plots/sex_distribution.png")
    _plot_position_distribution(df)
    print("Saved plots/position_distribution.png")
    print("Done.")


if __name__ == "__main__":
    main()
