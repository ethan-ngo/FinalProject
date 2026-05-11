# GPU Portfolio Optimization

GPU-accelerated mean-variance portfolio optimization using custom CUDA kernels and cuBLAS. Computes a covariance matrix and runs gradient descent on the GPU, then compares performance against a CPU baseline.

---

## How to Run

**1. Fetch stock return data**
```bash
make fetch_data
```
This downloads a returns CSV via `fetch_data.py` (uses `yfinance`).

**2. Build**
```bash
make all       # builds both cpu_baseline and portfolio_gpu
make cpu       # CPU baseline only
make gpu       # GPU binary only
```

**3. Run**
```bash
# GPU (with CPU comparison built in)
./portfolio_gpu data/returns.csv <lambda> <max_iter> <learning_rate>
# Example:
./portfolio_gpu data/returns.csv 1.0 2000 0.001

# CPU baseline only
./cpu_baseline data/returns.csv 1.0 2000 0.001
```

**4. Full benchmark sweep**
```bash
make benchmark
```
Runs both binaries across a range of portfolio sizes and saves plots to `plots/`.

---

## Dependencies

| Dependency | Notes |
|---|---|
| CUDA Toolkit | Includes `nvcc`, `cuBLAS`, `cudart`, and Thrust |
| cuBLAS | Used for DGEMV in the gradient descent loop |
| Thrust | Bundled with CUDA ≥ 10 |
| C++17 | Required by both `nvcc` and `g++` |
| Python 3 | For `fetch_data.py` and `benchmark.py` |
| `yfinance` | Python package — auto-installed by `make fetch_data` |
| `pandas` | Python package — auto-installed by `make fetch_data` |

Install Python dependencies manually if needed:
```bash
pip install yfinance pandas
```

---

## Hardware / Software Environment

- **GPU**: NVIDIA GPU with compute capability ≥ sm_75 (e.g. Turing / Ampere / Ada). The Makefile targets `sm_75` by default.
- **CUDA**: Built against CUDA 12.8 via NVIDIA HPC SDK 25.3. Paths in the Makefile assume an NVHPC install at `/opt/nvidia/hpc_sdk/Linux_x86_64/25.3`; adjust `CUDA_HOME` and `CUBLAS_DIR` if your setup differs.
- **OS**: Linux x86_64
- **Cluster note**: The comment in `main.cu` shows the `srun` command used on the Delta cluster (NCSA) with an A40 GPU:
  ```
  srun --account=bchn-delta-gpu --partition=gpuA40x4-interactive \
       --nodes=1 --gpus-per-node=1 --tasks=1 --tasks-per-node=16 \
       --cpus-per-task=1 --mem=20g --pty bash
  ```

---

## Key Files

| File | Purpose |
|---|---|
| `gpu_cov.cu` / `gpu_cov.cuh` | **Main CUDA kernel** — computes the N×N covariance matrix on the GPU |
| `gpu_gradient.cu` / `gpu_gradient.cuh` | GPU gradient descent loop using cuBLAS DGEMV + custom update kernels |
| `main.cu` | Driver — loads data, runs both GPU and CPU pipelines, prints speedup summary |
| `cpu_baseline.cpp` / `cpu_baseline_inline.h` | CPU reference implementation for benchmarking |
| `benchmark.py` | Benchmark sweep script — calls both binaries across portfolio sizes and generates plots |
| `fetch_data.py` | Downloads historical return data from Yahoo Finance |
| `get_tickers.py` | Scrapes a variety of sources for ticker symbols. [WIP] |
| `Makefile` | Build system |
