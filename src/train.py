import os
import sys
import time
import csv
import json
import subprocess
import psutil
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ========================================
# Global Configuration
# ========================================
NUM_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
NUM_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 30

# Get project root directory (parent of parent of train.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")


# -----------------------------
# 0. Detect Docker Environment
# -----------------------------
def is_docker():
    """Check if running inside a Docker container"""
    # Method 1: Check for .dockerenv file
    if os.path.exists("/.dockerenv"):
        return True

    # Method 2: Check cgroup for docker or containerd
    try:
        with open("/proc/1/cgroup", "r") as f:
            content = f.read()
            return "docker" in content or "containerd" in content
    except Exception:
        pass

    return False


ENV_NAME = "docker" if is_docker() else "native"
print(f"Running environment: {ENV_NAME}")


# -----------------------------
# System Metrics Collection
# -----------------------------
def collect_system_metrics():
    """Collect CPU, RAM, GPU utilization and memory"""
    # CPU and RAM
    cpu_util = psutil.cpu_percent(interval=0.1)
    ram_used_mb = psutil.virtual_memory().used / (1024**2)

    # GPU utilization and memory using nvidia-smi
    try:
        result = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ]
            )
            .decode()
            .strip()
        )
        gpu_util_str, gpu_mem_str = result.split(",")
        gpu_util = float(gpu_util_str)
        gpu_mem_mb = float(gpu_mem_str)
    except Exception:
        # Fallback if nvidia-smi is not available
        gpu_util = -1.0
        gpu_mem_mb = -1.0

    return cpu_util, ram_used_mb, gpu_util, gpu_mem_mb


# -----------------------------
# 1. Data Loading
# -----------------------------
# Data augmentation for training set
train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)

# No augmentation for test set
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)

train_ds = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform,
)

test_ds = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform,
)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)

print(f"Is GPU: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()


# -----------------------------
# 4. Training Function
# -----------------------------
def train_one_epoch(model, optimizer, epoch):
    model.train()
    total_loss = 0
    epoch_start = time.time()

    # System metrics accumulators
    gpu_util_sum = gpu_mem_sum = cpu_util_sum = ram_used_sum = 0.0
    metric_samples = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Sample system metrics every 10 batches
        if batch_idx % 10 == 0:
            cpu_util, ram_used_mb, gpu_util, gpu_mem_mb = collect_system_metrics()
            cpu_util_sum += cpu_util
            ram_used_sum += ram_used_mb
            gpu_util_sum += gpu_util
            gpu_mem_sum += gpu_mem_mb
            metric_samples += 1

    avg_train_loss = total_loss / len(train_loader)
    epoch_time_sec = time.time() - epoch_start

    # Calculate averaged system metrics
    if metric_samples > 0:
        gpu_util_avg = gpu_util_sum / metric_samples
        gpu_mem_avg_MB = gpu_mem_sum / metric_samples
        cpu_util_avg = cpu_util_sum / metric_samples
        ram_used_avg_MB = ram_used_sum / metric_samples
    else:
        gpu_util_avg = gpu_mem_avg_MB = cpu_util_avg = ram_used_avg_MB = None

    print(f"Epoch {epoch}: train loss = {avg_train_loss:.4f}")

    return (
        avg_train_loss,
        epoch_time_sec,
        gpu_util_avg,
        gpu_mem_avg_MB,
        cpu_util_avg,
        ram_used_avg_MB,
    )


# -----------------------------
# 5. Testing Function
# -----------------------------
def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy


# -----------------------------
# 6. Main Training Loop with Logging
# -----------------------------
# Create logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)

# Epoch-level CSV logging setup
epoch_csv_path = os.path.join(LOGS_DIR, "metrics_epochs.csv")
epoch_fieldnames = [
    "env",
    "run_id",
    "epoch",
    "train_loss",
    "test_accuracy",
    "epoch_time_sec",
    "throughput_img_per_sec",
    "gpu_util_avg",
    "gpu_mem_avg_MB",
    "cpu_util_avg",
    "ram_used_avg_MB",
]

# Write CSV header if file doesn't exist
if not os.path.exists(epoch_csv_path):
    with open(epoch_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_fieldnames)
        writer.writeheader()

# Execute multiple training runs
for run_id in range(1, NUM_RUNS + 1):
    print(f"\n{'='*60}")
    print(f"Starting RUN {run_id}/{NUM_RUNS}")
    print(f"{'='*60}\n")

    # Re-initialize model, optimizer, and scheduler for each run
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Run-level tracking
    run_start = time.time()
    num_training_samples = len(train_ds)
    batch_size = train_loader.batch_size
    learning_rate = optimizer.param_groups[0]["lr"]

    # Accumulators for run-level summary
    epoch_times = []
    gpu_utils = []
    gpu_mems = []
    final_test_accuracy = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        (
            avg_train_loss,
            epoch_time_sec,
            gpu_util_avg,
            gpu_mem_avg_MB,
            cpu_util_avg,
            ram_used_avg_MB,
        ) = train_one_epoch(model, optimizer, epoch)
        test_accuracy = evaluate(model)
        scheduler.step()

        # Calculate throughput
        throughput_img_per_sec = num_training_samples / epoch_time_sec

        # Log epoch metrics to CSV
        epoch_row = {
            "env": ENV_NAME,
            "run_id": run_id,
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "test_accuracy": test_accuracy,
            "epoch_time_sec": epoch_time_sec,
            "throughput_img_per_sec": throughput_img_per_sec,
            "gpu_util_avg": gpu_util_avg,
            "gpu_mem_avg_MB": gpu_mem_avg_MB,
            "cpu_util_avg": cpu_util_avg,
            "ram_used_avg_MB": ram_used_avg_MB,
        }

        with open(epoch_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=epoch_fieldnames)
            writer.writerow(epoch_row)

        # Accumulate for run-level summary
        epoch_times.append(epoch_time_sec)
        if gpu_util_avg is not None:
            gpu_utils.append(gpu_util_avg)
        if gpu_mem_avg_MB is not None:
            gpu_mems.append(gpu_mem_avg_MB)
        final_test_accuracy = test_accuracy

    # Calculate run-level summary
    run_total_time_sec = time.time() - run_start
    avg_epoch_time_sec = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    avg_gpu_util_epoch_mean = sum(gpu_utils) / len(gpu_utils) if gpu_utils else -1.0
    avg_gpu_mem_epoch_mean = sum(gpu_mems) / len(gpu_mems) if gpu_mems else -1.0

    run_summary = {
        "env": ENV_NAME,
        "run_id": run_id,
        "num_epochs": NUM_EPOCHS,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer_type": optimizer.__class__.__name__,
        "train_total_time_sec": run_total_time_sec,
        "avg_epoch_time_sec": avg_epoch_time_sec,
        "avg_gpu_util_epoch_mean": avg_gpu_util_epoch_mean,
        "avg_gpu_mem_epoch_mean": avg_gpu_mem_epoch_mean,
        "final_test_accuracy": final_test_accuracy,
    }

    # Print run summary
    print("\n" + "=" * 50)
    print(f"RUN {run_id} SUMMARY")
    print("=" * 50)
    for key, value in run_summary.items():
        print(f"{key}: {value}")
    print("=" * 50)

    # Append to JSON Lines file
    with open(os.path.join(LOGS_DIR, "runs_summary.jsonl"), "a") as f:
        f.write(json.dumps(run_summary) + "\n")

print(f"\n{'='*60}")
print(f"All {NUM_RUNS} runs completed!")
print(f"{'='*60}\n")
