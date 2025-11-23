# PyTorch CIFAR-10 Training

## Environment Setup

### Using Mamba (Native)

1. Create the environment:
```bash
mamba env create -f environment.yml
```

2. Activate the environment:
```bash
mamba activate pytorch-training
```

3. Run training:
```bash
python src/train.py
```

### Using Docker

1. Build the Docker image:
```bash
docker build -t pytorch-training .
```

2. Run with GPU support:
```bash
docker run --gpus all -v $(pwd):/workspace pytorch-training
```

3. Run interactively:
```bash
docker run --gpus all -it -v $(pwd):/workspace pytorch-training bash
```

## Benchmark: Native vs Docker

Run automated benchmarks to compare native and Docker performance:

```bash
# Use default parameters (5 runs, 30 epochs)
python src/benchmark.py

# Custom parameters (NUM_RUNS NUM_EPOCHS)
python src/benchmark.py 3 10
```

This will:
1. Run training in native environment with configured parameters
2. Run training in Docker environment with the same parameters
3. Save all metrics to `logs/` directory




## Configuration

Both `train.py` and `benchmark.py` accept command-line parameters:

```bash
# Run training with custom parameters (NUM_RUNS NUM_EPOCHS)
python src/train.py 3 50

# Run benchmark with custom parameters (NUM_RUNS NUM_EPOCHS)
python src/benchmark.py 5 30
```

**Default values:**
- `train.py`: 3 runs, 50 epochs
- `benchmark.py`: 5 runs, 30 epochs

## Output

Training metrics are saved to the `logs/` directory:

### `logs/metrics_epochs.csv`
Per-epoch detailed metrics for all runs.

| Field | Description |
|-------|-------------|
| `env` | Environment (native/docker) |
| `run_id` | Training run identifier |
| `epoch` | Epoch number |
| `train_loss` | Training loss (averaged over batches) |
| `test_accuracy` | Test set accuracy (0-1) |
| `epoch_time_sec` | Epoch duration in seconds |
| `throughput_img_per_sec` | Training throughput (images/second) |
| `gpu_util_avg` | Average GPU utilization (%) |
| `gpu_mem_avg_MB` | Average GPU memory usage (MB) |
| `cpu_util_avg` | Average CPU utilization (%) |
| `ram_used_avg_MB` | Average RAM usage (MB) |

### `logs/runs_summary.jsonl`
Run-level summary statistics (JSON Lines format, one run per line).

| Field | Description |
|-------|-------------|
| `env` | Environment (native/docker) |
| `run_id` | Training run identifier |
| `num_epochs` | Total number of epochs |
| `batch_size` | Training batch size |
| `learning_rate` | Initial learning rate |
| `optimizer_type` | Optimizer class name |
| `train_total_time_sec` | Total training time in seconds |
| `avg_epoch_time_sec` | Average time per epoch |
| `avg_gpu_util_epoch_mean` | Average GPU utilization across epochs |
| `avg_gpu_mem_epoch_mean` | Average GPU memory usage across epochs |
| `final_test_accuracy` | Final test accuracy (0-1) |
