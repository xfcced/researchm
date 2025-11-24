#!/usr/bin/env python3
import os
import sys
import subprocess


# ========================================
# Configuration
# ========================================
DOCKER_IMAGE = "test:latest"


def main():
    # Get parameters from command line or use defaults
    NUM_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    NUM_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    print("\n" + "=" * 60)
    print("  Benchmark: Native vs Docker")
    print("=" * 60)
    print(f"  Epochs per training: {NUM_EPOCHS}")
    print(f"  Runs per environment: {NUM_RUNS}")
    print("=" * 60 + "\n")

    # Clean old logs
    os.makedirs("logs", exist_ok=True)
    for log_file in ["../logs/metrics_epochs.csv", "../logs/runs_summary.jsonl"]:
        if os.path.exists(log_file):
            os.remove(log_file)

    # Run Native environment
    print("\n" + "=" * 60)
    print("  Starting NATIVE environment training")
    print("=" * 60 + "\n")
    subprocess.run(
        [
            "mamba",
            "run",
            "-n",
            "researchm",
            "python",
            "src/train.py",
            str(NUM_RUNS),
            str(NUM_EPOCHS),
        ]
    )

    # Run Docker environment
    print("\n" + "=" * 60)
    print("  Starting DOCKER environment training")
    print("=" * 60 + "\n")
    subprocess.run(
        [
            "docker",
            "run",
            "--gpus",
            "all",
            "-v",
            f"{os.getcwd()}:/workspace",
            DOCKER_IMAGE,
            "micromamba",
            "run",
            "-n",
            "researchm",
            "python",
            "src/train.py",
            str(NUM_RUNS),
            str(NUM_EPOCHS),
        ]
    )

    print("\n" + "=" * 60)
    print("  ✅ Benchmark completed!")
    print("=" * 60)
    print("  📊 Log files:")
    print("     - logs/metrics_epochs.csv")
    print("     - logs/runs_summary.jsonl")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
