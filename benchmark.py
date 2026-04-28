#!/usr/bin/env python3
"""
benchmark.py
------------
Runs cpu_baseline and the GPU main binary for several values of N (asset count),
parses their timing output, and produces two plots:
    1. Runtime vs N  (CPU and GPU separately, log-log)
    2. Speedup factor vs N

Usage (after compiling):
    python benchmark.py [--csv data/returns.csv] [--ns 50,100,200,500]
                        [--cpu_bin ./cpu_baseline]
                        [--gpu_bin ./portfolio_gpu]
                        [--out_dir plots/]
"""

import argparse
import subprocess
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
def parse_timing(output: str, prefix: str):
    """Extract cov, opt, total from a *_TIMING line."""
    pat = rf"{prefix}_TIMING N=(\d+) cov=([\d.]+) opt=([\d.]+) total=([\d.]+)"
    m = re.search(pat, output)
    if not m:
        return None
    return dict(N=int(m.group(1)),
                cov=float(m.group(2)),
                opt=float(m.group(3)),
                total=float(m.group(4)))

# ------------------------------------------------------------------ #
def run_binary(binary, csv, n_assets, subsample=True, **kwargs):
    """Run a binary after optionally subsampling the CSV to n_assets columns."""
    import pandas as pd, tempfile, shutil

    if subsample:
        df = pd.read_csv(csv, index_col=0)
        # cap at available assets
        n_assets = min(n_assets, df.shape[1])
        df_sub = df.iloc[:, :n_assets]
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        df_sub.to_csv(tmp.name)
        tmp.close()
        csv_arg = tmp.name
    else:
        csv_arg = csv

    cmd = [binary, csv_arg,
           str(kwargs.get("lambda_", 1.0)),
           str(kwargs.get("max_iter", 1000)),
           str(kwargs.get("lr", 1e-3))]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr

    if subsample:
        os.unlink(csv_arg)

    return output

# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",     default="data/returns.csv")
    ap.add_argument("--ns",      default="50,100,200,300,500")
    ap.add_argument("--cpu_bin", default="./cpu_baseline")
    ap.add_argument("--gpu_bin", default="./portfolio_gpu")
    ap.add_argument("--out_dir", default="plots")
    ap.add_argument("--max_iter",type=int, default=1000)
    ap.add_argument("--lr",      type=float, default=1e-3)
    ap.add_argument("--lambda_", type=float, default=1.0,  dest="lambda_")
    args = ap.parse_args()

    ns = [int(x) for x in args.ns.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    cpu_cov, cpu_opt, cpu_tot = [], [], []
    gpu_cov, gpu_opt, gpu_tot = [], [], []
    valid_ns = []

    kw = dict(lambda_=args.lambda_, max_iter=args.max_iter, lr=args.lr)

    for n in ns:
        print(f"\n=== N = {n} ===")

        # CPU
        try:
            out_cpu = run_binary(args.cpu_bin, args.csv, n, **kw)
            tc = parse_timing(out_cpu, "CPU")
            if tc is None:
                print(f"  [warn] CPU timing not found for N={n}")
                continue
        except Exception as e:
            print(f"  [warn] CPU run failed for N={n}: {e}")
            continue

        # GPU
        try:
            out_gpu = run_binary(args.gpu_bin, args.csv, n, **kw)
            tg = parse_timing(out_gpu, "GPU")
            if tg is None:
                print(f"  [warn] GPU timing not found for N={n}")
                continue
        except Exception as e:
            print(f"  [warn] GPU run failed for N={n}: {e}")
            continue

        print(f"  CPU total={tc['total']:.1f} ms   GPU total={tg['total']:.1f} ms"
              f"   speedup={tc['total']/tg['total']:.1f}×")

        cpu_cov.append(tc["cov"]);  cpu_opt.append(tc["opt"]);  cpu_tot.append(tc["total"])
        gpu_cov.append(tg["cov"]);  gpu_opt.append(tg["opt"]);  gpu_tot.append(tg["total"])
        valid_ns.append(n)

    if not valid_ns:
        print("No successful runs — check binary paths and CSV.")
        return

    ns_arr = np.array(valid_ns)

    # ---- Plot 1: Runtime vs N
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.loglog(ns_arr, cpu_cov, "b-o",  label="CPU cov")
    ax.loglog(ns_arr, cpu_opt, "b--s", label="CPU GD")
    ax.loglog(ns_arr, cpu_tot, "b-D",  label="CPU total", linewidth=2)
    ax.loglog(ns_arr, gpu_cov, "r-o",  label="GPU cov")
    ax.loglog(ns_arr, gpu_opt, "r--s", label="GPU GD")
    ax.loglog(ns_arr, gpu_tot, "r-D",  label="GPU total", linewidth=2)
    ax.set_xlabel("Number of assets (N)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Runtime vs N (log-log)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":")

    # ---- Plot 2: Speedup
    ax = axes[1]
    speedup_cov = np.array(cpu_cov) / np.array(gpu_cov)
    speedup_opt = np.array(cpu_opt) / np.array(gpu_opt)
    speedup_tot = np.array(cpu_tot) / np.array(gpu_tot)

    ax.plot(ns_arr, speedup_cov, "g-o",  label="Covariance speedup")
    ax.plot(ns_arr, speedup_opt, "m-s",  label="GD speedup")
    ax.plot(ns_arr, speedup_tot, "k-D",  label="Total speedup", linewidth=2)
    ax.axhline(1, color="gray", ls="--", label="1× baseline")
    ax.set_xlabel("Number of assets (N)")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("GPU Speedup vs N")
    ax.legend()
    ax.grid(True, ls=":")

    fig.suptitle("Portfolio Optimisation: CPU vs GPU", fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "speedup.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot → {out_path}")

    # ---- Print table
    print("\n{:>8}  {:>12}  {:>12}  {:>10}  {:>10}  {:>10}".format(
          "N", "CPU total", "GPU total", "Cov ×", "GD ×", "Total ×"))
    print("-" * 70)
    for i, n in enumerate(valid_ns):
        print("{:>8}  {:>12.1f}  {:>12.1f}  {:>10.1f}  {:>10.1f}  {:>10.1f}".format(
              n, cpu_tot[i], gpu_tot[i],
              speedup_cov[i], speedup_opt[i], speedup_tot[i]))

if __name__ == "__main__":
    main()