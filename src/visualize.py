"""
Visualization utilities for the ECG MLOps project.

Generates plots for:
1. Raw ECG waveform (all 12 leads) for a single recording.
2. Feature distribution across the five diagnostic superclasses.
3. Model performance comparison (AUC and F1 per class).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Add src/ to path so this script can be run directly
import sys
sys.path.insert(0, str(Path(__file__).parent))

from features import load_signal, LEADS
from data_loader import load_metadata, load_scp_statements
from labels import build_label_matrix, filter_labeled_records

OUTPUT_DIR = Path("notebooks")
OUTPUT_DIR.mkdir(exist_ok=True)

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SAMPLING_RATE = 100  # Hz


def plot_ecg_waveform(filename_lr: str, title: str = None, save: bool = True, save_path: str = "ecg_waveform.png"):    
    """
    Plot all 12 leads of a single ECG recording in clinical layout.

    The x-axis is time in seconds. Each lead is plotted in its own row,
    resembling the standard 12-lead ECG paper format.
    """
    signal = load_signal(filename_lr)
    n_samples = signal.shape[0]
    time_axis = np.arange(n_samples) / SAMPLING_RATE  # Convert samples to seconds

    fig, axes = plt.subplots(12, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(title or f"12-Lead ECG: {filename_lr}", fontsize=13, fontweight="bold")

    for i, (ax, lead) in enumerate(zip(axes, LEADS)):
        ax.plot(time_axis, signal[:, i], color="#1f77b4", linewidth=0.8)
        ax.set_ylabel(lead, fontsize=9, rotation=0, labelpad=25)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (seconds)", fontsize=10)
    plt.tight_layout()

    if save:
        path = OUTPUT_DIR / save_path
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.show()


def plot_label_distribution(labels: pd.DataFrame, save: bool = True):
    """
    Bar chart of record counts per diagnostic superclass.

    Shows both single-label and multi-label record counts to illustrate
    the multi-label nature of the dataset.
    """
    counts = labels[SUPERCLASSES].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color="#2196F3", edgecolor="white", linewidth=0.5)

    # Annotate each bar with its count
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{int(count):,}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_title("Record Count per Diagnostic Superclass\n(PTB-XL, 100 Hz, 21,388 labeled records)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Superclass", fontsize=11)
    ax.set_ylabel("Number of Records", fontsize=11)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save:
        path = OUTPUT_DIR / "label_distribution.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.show()


def plot_model_comparison(save: bool = True):
    """
    Side-by-side bar chart comparing LogReg and XGBoost
    on macro AUC and macro F1 for validation and test sets.
    """
    # Results from our training run
    results = {
        "Logistic Regression": {
            "Val AUC":  0.8175,
            "Test AUC": 0.7977,
            "Val F1":   0.5128,
            "Test F1":  0.4936,
        },
        "XGBoost": {
            "Val AUC":  0.8561,
            "Test AUC": 0.8397,
            "Val F1":   0.5958,
            "Test F1":  0.5817,
        },
    }

    metrics = ["Val AUC", "Test AUC", "Val F1", "Test F1"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#2196F3", "#FF5722"]
    for i, (model_name, scores) in enumerate(results.items()):
        values = [scores[m] for m in metrics]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name,
                      color=colors[i], edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    ax.set_title("Model Comparison: Logistic Regression vs XGBoost\n"
                 "Multi-label ECG Classification on PTB-XL",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save:
        path = OUTPUT_DIR / "model_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.show()


if __name__ == "__main__":
    print("Loading data...")
    df  = load_metadata()
    scp = load_scp_statements()

    label_matrix          = build_label_matrix(df, scp)
    df_labeled, labels    = filter_labeled_records(df, label_matrix)

    print("Generating plots...\n")

    # Plot 1: a NORM record and an MI record side by side
    norm_record = df_labeled[labels["NORM"] == 1].iloc[0]["filename_lr"]
    mi_record   = df_labeled[labels["MI"]   == 1].iloc[0]["filename_lr"]

    plot_ecg_waveform(norm_record, title="12-Lead ECG — NORM (Normal)",    save_path="ecg_waveform_norm.png")
    plot_ecg_waveform(mi_record,   title="12-Lead ECG — MI (Myocardial Infarction)", save_path="ecg_waveform_mi.png")

    # Plot 2: label distribution
    plot_label_distribution(labels)

    # Plot 3: model comparison
    plot_model_comparison()

    print("\nAll plots saved to notebooks/")