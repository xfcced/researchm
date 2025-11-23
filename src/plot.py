#!/usr/bin/env python3
"""
Visualization script for comparing Native vs Docker training performance.
Generates multiple plots from training logs.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Get project root and logs directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")


def load_data():
    """Load CSV and JSONL data from logs directory."""
    # Load epoch-level metrics
    epochs_df = pd.read_csv(os.path.join(LOGS_DIR, "metrics_epochs.csv"))

    # Load run-level summaries
    runs_data = []
    with open(os.path.join(LOGS_DIR, "runs_summary.jsonl"), "r") as f:
        for line in f:
            runs_data.append(json.loads(line))
    runs_df = pd.DataFrame(runs_data)

    return epochs_df, runs_df


def plot_training_loss_curves(epochs_df):
    """Plot training loss curves over epochs for native vs docker."""
    plt.figure(figsize=(12, 6))

    for env in epochs_df["env"].unique():
        env_data = epochs_df[epochs_df["env"] == env]

        # Group by epoch and calculate mean and std
        grouped = env_data.groupby("epoch")["train_loss"].agg(["mean", "std"])

        plt.plot(
            grouped.index,
            grouped["mean"],
            label=f"{env.capitalize()}",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        plt.fill_between(
            grouped.index,
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.2,
        )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Training Loss", fontsize=12)
    plt.title("Training Loss: Native vs Docker", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "training_loss_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_test_accuracy_curves(epochs_df):
    """Plot test accuracy curves over epochs for native vs docker."""
    plt.figure(figsize=(12, 6))

    for env in epochs_df["env"].unique():
        env_data = epochs_df[epochs_df["env"] == env]

        # Group by epoch and calculate mean and std
        grouped = env_data.groupby("epoch")["test_accuracy"].agg(["mean", "std"])

        plt.plot(
            grouped.index,
            grouped["mean"] * 100,
            label=f"{env.capitalize()}",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        plt.fill_between(
            grouped.index,
            (grouped["mean"] - grouped["std"]) * 100,
            (grouped["mean"] + grouped["std"]) * 100,
            alpha=0.2,
        )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title("Test Accuracy: Native vs Docker", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "test_accuracy_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_epoch_time_comparison(epochs_df):
    """Plot epoch time comparison between native and docker."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot
    sns.boxplot(
        data=epochs_df,
        x="env",
        y="epoch_time_sec",
        ax=ax1,
        hue="env",
        palette="Set2",
        legend=False,
    )
    ax1.set_xlabel("Environment", fontsize=12)
    ax1.set_ylabel("Epoch Time (seconds)", fontsize=12)
    ax1.set_title("Epoch Time Distribution", fontsize=13, fontweight="bold")
    ax1.set_xticks(range(len(epochs_df["env"].unique())))
    ax1.set_xticklabels([env.capitalize() for env in epochs_df["env"].unique()])

    # Violin plot
    sns.violinplot(
        data=epochs_df,
        x="env",
        y="epoch_time_sec",
        ax=ax2,
        hue="env",
        palette="Set2",
        legend=False,
    )
    ax2.set_xlabel("Environment", fontsize=12)
    ax2.set_ylabel("Epoch Time (seconds)", fontsize=12)
    ax2.set_title("Epoch Time Density", fontsize=13, fontweight="bold")
    ax2.set_xticks(range(len(epochs_df["env"].unique())))
    ax2.set_xticklabels([env.capitalize() for env in epochs_df["env"].unique()])

    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "epoch_time_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_throughput_comparison(epochs_df):
    """Plot training throughput (images/sec) comparison."""
    plt.figure(figsize=(12, 6))

    for env in epochs_df["env"].unique():
        env_data = epochs_df[epochs_df["env"] == env]
        grouped = env_data.groupby("epoch")["throughput_img_per_sec"].agg(
            ["mean", "std"]
        )

        plt.plot(
            grouped.index,
            grouped["mean"],
            label=f"{env.capitalize()}",
            linewidth=2,
            marker="s",
            markersize=4,
        )
        plt.fill_between(
            grouped.index,
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.2,
        )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Throughput (images/second)", fontsize=12)
    plt.title("Training Throughput: Native vs Docker", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "throughput_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_gpu_utilization(epochs_df):
    """Plot GPU utilization and memory usage comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # GPU Utilization
    for env in epochs_df["env"].unique():
        env_data = epochs_df[epochs_df["env"] == env]
        grouped = env_data.groupby("epoch")["gpu_util_avg"].agg(["mean", "std"])

        ax1.plot(
            grouped.index,
            grouped["mean"],
            label=f"{env.capitalize()}",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        ax1.fill_between(
            grouped.index,
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.2,
        )

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("GPU Utilization (%)", fontsize=12)
    ax1.set_title("GPU Utilization", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # GPU Memory
    for env in epochs_df["env"].unique():
        env_data = epochs_df[epochs_df["env"] == env]
        grouped = env_data.groupby("epoch")["gpu_mem_avg_MB"].agg(["mean", "std"])

        ax2.plot(
            grouped.index,
            grouped["mean"],
            label=f"{env.capitalize()}",
            linewidth=2,
            marker="s",
            markersize=4,
        )
        ax2.fill_between(
            grouped.index,
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.2,
        )

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("GPU Memory (MB)", fontsize=12)
    ax2.set_title("GPU Memory Usage", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "gpu_metrics.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_run_summary_comparison(runs_df):
    """Plot run-level summary statistics comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total training time
    sns.barplot(
        data=runs_df,
        x="env",
        y="train_total_time_sec",
        ax=axes[0, 0],
        hue="env",
        palette="Set2",
        errorbar="sd",
        legend=False,
    )
    axes[0, 0].set_xlabel("Environment", fontsize=11)
    axes[0, 0].set_ylabel("Total Training Time (s)", fontsize=11)
    axes[0, 0].set_title("Total Training Time per Run", fontsize=12, fontweight="bold")
    axes[0, 0].set_xticks(range(len(runs_df["env"].unique())))
    axes[0, 0].set_xticklabels([env.capitalize() for env in runs_df["env"].unique()])

    # Final test accuracy
    sns.barplot(
        data=runs_df,
        x="env",
        y="final_test_accuracy",
        ax=axes[0, 1],
        hue="env",
        palette="Set2",
        errorbar="sd",
        legend=False,
    )
    axes[0, 1].set_xlabel("Environment", fontsize=11)
    axes[0, 1].set_ylabel("Final Test Accuracy", fontsize=11)
    axes[0, 1].set_title("Final Test Accuracy per Run", fontsize=12, fontweight="bold")
    axes[0, 1].set_xticks(range(len(runs_df["env"].unique())))
    axes[0, 1].set_xticklabels([env.capitalize() for env in runs_df["env"].unique()])
    axes[0, 1].set_ylim([0.75, 0.85])

    # Average GPU utilization
    sns.barplot(
        data=runs_df,
        x="env",
        y="avg_gpu_util_epoch_mean",
        ax=axes[1, 0],
        hue="env",
        palette="Set2",
        errorbar="sd",
        legend=False,
    )
    axes[1, 0].set_xlabel("Environment", fontsize=11)
    axes[1, 0].set_ylabel("Avg GPU Utilization (%)", fontsize=11)
    axes[1, 0].set_title("Average GPU Utilization", fontsize=12, fontweight="bold")
    axes[1, 0].set_xticks(range(len(runs_df["env"].unique())))
    axes[1, 0].set_xticklabels([env.capitalize() for env in runs_df["env"].unique()])

    # Average epoch time
    sns.barplot(
        data=runs_df,
        x="env",
        y="avg_epoch_time_sec",
        ax=axes[1, 1],
        hue="env",
        palette="Set2",
        errorbar="sd",
        legend=False,
    )
    axes[1, 1].set_xlabel("Environment", fontsize=11)
    axes[1, 1].set_ylabel("Avg Epoch Time (s)", fontsize=11)
    axes[1, 1].set_title("Average Epoch Time", fontsize=12, fontweight="bold")
    axes[1, 1].set_xticks(range(len(runs_df["env"].unique())))
    axes[1, 1].set_xticklabels([env.capitalize() for env in runs_df["env"].unique()])

    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "run_summary_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_performance_overhead(runs_df):
    """Calculate and plot performance overhead of Docker vs Native."""
    native_stats = runs_df[runs_df["env"] == "native"]
    docker_stats = runs_df[runs_df["env"] == "docker"]

    metrics = {
        "Total Time": ("train_total_time_sec", "s"),
        "Avg Epoch Time": ("avg_epoch_time_sec", "s"),
        "GPU Utilization": ("avg_gpu_util_epoch_mean", "%"),
        "Final Accuracy": ("final_test_accuracy", ""),
    }

    overhead_data = []

    for metric_name, (col, unit) in metrics.items():
        native_mean = native_stats[col].mean()
        docker_mean = docker_stats[col].mean()

        if metric_name == "Final Accuracy":
            # For accuracy, we want difference, not overhead
            diff = (docker_mean - native_mean) * 100  # percentage points
            overhead_data.append(
                {"Metric": metric_name, "Difference": diff, "Label": f"{diff:+.2f}pp"}
            )
        else:
            overhead_pct = ((docker_mean - native_mean) / native_mean) * 100
            overhead_data.append(
                {
                    "Metric": metric_name,
                    "Difference": overhead_pct,
                    "Label": f"{overhead_pct:+.2f}%",
                }
            )

    overhead_df = pd.DataFrame(overhead_data)

    plt.figure(figsize=(10, 6))
    colors = ["red" if x > 0 else "green" for x in overhead_df["Difference"]]
    bars = plt.barh(
        overhead_df["Metric"], overhead_df["Difference"], color=colors, alpha=0.7
    )

    # Add value labels
    for i, (bar, label) in enumerate(zip(bars, overhead_df["Label"])):
        width = bar.get_width()
        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f" {label}",
            ha="left" if width > 0 else "right",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    plt.xlabel("Difference (%)", fontsize=12)
    plt.title("Docker vs Native Performance Difference", fontsize=14, fontweight="bold")
    plt.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "performance_overhead.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main function to generate all plots."""
    # Create plots directory if it doesn't exist
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading data...")
    epochs_df, runs_df = load_data()

    print(f"\nLoaded {len(epochs_df)} epoch records and {len(runs_df)} run summaries")
    print(f"Environments: {epochs_df['env'].unique()}")
    print(
        f"Runs per environment: {epochs_df.groupby('env')['run_id'].nunique().to_dict()}"
    )

    print("\nGenerating plots...")

    # Generate all plots
    plot_training_loss_curves(epochs_df)
    plot_test_accuracy_curves(epochs_df)
    plot_epoch_time_comparison(epochs_df)
    plot_throughput_comparison(epochs_df)
    plot_gpu_utilization(epochs_df)
    plot_run_summary_comparison(runs_df)
    plot_performance_overhead(runs_df)

    print(f"\n✅ All plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
