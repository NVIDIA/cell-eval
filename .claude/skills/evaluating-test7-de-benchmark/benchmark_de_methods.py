#!/usr/bin/env python3
"""Test 7 — DE method benchmark: wall time, CPU RAM, and GPU VRAM per call.

For each configured DE method (pdex, pydeseq2), runs DE on N randomly sampled
perturbations from the real dataset, measuring:
  - wall-clock time per call
  - peak RSS delta per call (via /proc/self/status on Linux, resource on macOS)
  - GPU VRAM delta per call (via nvidia-smi; 0 if no GPU or method doesn't use GPU)

Outputs:
  <outdir>/test7_benchmark_summary__<dataset>.csv   — one row per (method, perturbation, repeat)
  <outdir>/test7_benchmark_report__<dataset>.md     — human-readable summary table
  <outdir>/plots/test7_time__<dataset>.png          — box plot: wall time per method
  <outdir>/plots/test7_ram__<dataset>.png           — box plot: peak RAM delta per method
  <outdir>/plots/test7_gpu__<dataset>.png           — box plot: GPU VRAM delta per method

Usage:
  uv run python benchmark_de_methods.py \
      --adata /path/to.h5ad --pert-col gene --control non-targeting \
      --replicate-col batch --methods pdex,pydeseq2 \
      --n-perts 10 --outdir .
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import tracemalloc

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cell_eval._de_backends import build_de_frame


# ── GPU helpers ────────────────────────────────────────────────────────────────

def _gpu_vram_mb() -> float:
    """Current GPU VRAM used in MB, summed across all devices. Returns 0.0 if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        return sum(float(x) for x in out.decode().strip().splitlines() if x.strip())
    except Exception:
        return 0.0


# ── RAM helpers ────────────────────────────────────────────────────────────────

def _rss_mb() -> float:
    """Current process RSS in MB."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ── DE call ────────────────────────────────────────────────────────────────────

def _run_one(adata: ad.AnnData, method: str, pert: str, control: str,
             pert_col: str, replicate_col: str | None,
             num_threads: int) -> dict:
    """Run one DE call and return timing + resource measurements."""
    mask = adata.obs[pert_col].isin([pert, control])
    sub = adata[mask]

    rss_before = _rss_mb()
    gpu_before = _gpu_vram_mb()

    tracemalloc.start()
    t0 = time.perf_counter()

    if method == "pdex":
        result = build_de_frame(
            mode="real", adata=sub, control_pert=control,
            pert_col=pert_col, num_threads=num_threads,
            allow_discrete=False, de_method="pdex",
            de_kwargs=None, counts_layer=None, replicate_col=None,
        )
    elif method == "pydeseq2":
        result = build_de_frame(
            mode="real", adata=sub, control_pert=control,
            pert_col=pert_col, num_threads=num_threads,
            allow_discrete=True, de_method="pydeseq2",
            de_kwargs=None, counts_layer="_raw_counts", replicate_col=replicate_col,
        )
    else:
        raise ValueError(f"unknown method: {method}")

    wall_s = time.perf_counter() - t0
    _, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rss_after = _rss_mb()
    gpu_after = _gpu_vram_mb()

    n_cells_pert = int((sub.obs[pert_col] == pert).sum())
    n_cells_ctrl = int((sub.obs[pert_col] == control).sum())

    return {
        "method":        method,
        "perturbation":  pert,
        "n_cells_pert":  n_cells_pert,
        "n_cells_ctrl":  n_cells_ctrl,
        "n_genes_out":   len(result),
        "wall_s":        wall_s,
        "ram_delta_mb":  max(rss_after - rss_before, 0.0),
        "peak_traced_mb": peak_traced / 1e6,
        "gpu_delta_mb":  max(gpu_after - gpu_before, 0.0),
    }


# ── plotting ────────────────────────────────────────────────────────────────────

METHOD_COLOR = {"pdex": "#0173B2", "pydeseq2": "#DE8F05"}


def _boxplot(df: pd.DataFrame, col: str, ylabel: str, title: str, out_png: str,
             methods: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    data = [df[df["method"] == m][col].dropna().values for m in methods]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", lw=1.8),
                    whiskerprops=dict(color="#555"), capprops=dict(color="#555"),
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(METHOD_COLOR.get(m, "0.5"))
        patch.set_alpha(0.6)

    rng = np.random.default_rng(0)
    for i, (m, d) in enumerate(zip(methods, data), 1):
        jitter = rng.uniform(-0.12, 0.12, size=len(d))
        ax.scatter(i + jitter, d, s=25, color=METHOD_COLOR.get(m, "0.5"),
                   edgecolors="white", linewidths=0.4, alpha=0.85, zorder=3)

    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot: {os.path.abspath(out_png)}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adata",         required=True)
    ap.add_argument("--pert-col",      default="gene")
    ap.add_argument("--control",       default="non-targeting")
    ap.add_argument("--replicate-col", default="batch",
                    help="pseudobulk replicate column for pydeseq2 (required by the backend)")
    ap.add_argument("--counts-layer",  default=None,
                    help="raw-counts layer name (auto-detects 'counts' if present)")
    ap.add_argument("--methods",       default="pdex,pydeseq2",
                    help="comma-sep DE backends to benchmark (default: pdex,pydeseq2)")
    ap.add_argument("--n-perts",       type=int, default=10,
                    help="number of perturbations to sample for benchmarking (default: 10)")
    ap.add_argument("--seed",          type=int, default=0)
    ap.add_argument("--num-threads",   type=int, default=8)
    ap.add_argument("--outdir",        default=".")
    a = ap.parse_args()

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    dataset = os.path.splitext(os.path.basename(a.adata))[0]

    print(f"loading {a.adata} …")
    adata = ad.read_h5ad(a.adata)
    counts_layer = a.counts_layer
    if counts_layer is None and "counts" in adata.layers:
        counts_layer = "counts"
    print(f"loaded {adata.n_obs} cells × {adata.n_vars} genes")

    # Build a single AnnData: X = log-norm, layers['_raw_counts'] = raw integer counts.
    # Both methods use the same object so the cell/gene selection is identical.
    adata_main = adata.copy()
    raw_src = adata_main.layers[counts_layer] if counts_layer else adata_main.X.copy()
    adata_main.layers["_raw_counts"] = raw_src
    sc.pp.normalize_total(adata_main, inplace=True)
    sc.pp.log1p(adata_main)

    # Sample perturbations
    all_perts = [p for p in adata.obs[a.pert_col].astype(str).unique()
                 if p != a.control]
    rng = np.random.default_rng(a.seed)
    n = min(a.n_perts, len(all_perts))
    perts = list(rng.choice(all_perts, size=n, replace=False))
    print(f"benchmarking {n} perturbations × {len(methods)} methods")

    rows = []
    for method in methods:
        print(f"\n=== {method} ===")
        for pert in perts:
            print(f"  {pert} … ", end="", flush=True)
            try:
                row = _run_one(adata_main, method, pert, a.control,
                               a.pert_col, a.replicate_col, a.num_threads)
                print(f"{row['wall_s']:.1f}s  RAM+{row['ram_delta_mb']:.0f}MB"
                      f"  GPU+{row['gpu_delta_mb']:.0f}MB")
                rows.append(row)
            except Exception as e:
                print(f"FAILED: {e}")

    df = pd.DataFrame(rows)
    os.makedirs(a.outdir, exist_ok=True)
    base = os.path.join(a.outdir, f"test7_benchmark_summary__{dataset}")
    df.to_csv(base + ".csv", index=False)

    # Report
    summary = (df.groupby("method")[["wall_s", "ram_delta_mb", "peak_traced_mb", "gpu_delta_mb"]]
               .agg(["mean", "std", "min", "max"]).round(1))
    md_lines = [f"# Test 7 — DE method benchmark: {dataset}\n",
                f"n_perts={n}  methods={methods}\n",
                f"Dataset: {adata.n_obs} cells × {adata.n_vars} genes\n\n",
                "## Summary\n\n",
                summary.to_markdown(), "\n\n",
                "## Per-perturbation\n\n",
                df[["method", "perturbation", "n_cells_pert", "n_cells_ctrl",
                    "wall_s", "ram_delta_mb", "peak_traced_mb", "gpu_delta_mb"]]
                .to_markdown(index=False)]
    report_path = base + ".md"
    with open(report_path, "w") as f:
        f.write("\n".join(str(x) for x in md_lines))
    print(f"\nreport: {report_path}")

    # Plots
    plots_dir = os.path.join(a.outdir, "plots")
    _boxplot(df, "wall_s", "wall time (s)", "Test 7 — wall time per DE call",
             os.path.join(plots_dir, f"test7_time__{dataset}.png"), methods)
    _boxplot(df, "ram_delta_mb", "RSS delta (MB)",
             "Test 7 — peak RSS increase per DE call",
             os.path.join(plots_dir, f"test7_ram__{dataset}.png"), methods)
    _boxplot(df, "gpu_delta_mb", "GPU VRAM delta (MB)",
             "Test 7 — GPU VRAM increase per DE call",
             os.path.join(plots_dir, f"test7_gpu__{dataset}.png"), methods)

    print(f"\nwrote: {base}.csv  |  {base}.md")


if __name__ == "__main__":
    main()
