"""Full Arc dataset (8.59M × 2000 per side) through cell-eval's cpu (illico) pipeline.

Mirrors ``run_arc_celleval_gpu.py`` but sets ``CELL_EVAL_BACKEND=cpu`` so both
pdex and the metric backends use the cpu (numba) implementations.

Two wall times are reported:
  * ``bench_wall_clock`` — MetricsEvaluator init + compute_all_metrics. The
    cell-eval pipeline cost.
  * ``wall_clock_total`` — includes the h5ad dense reads + numba sparsify
    that happen before MetricsEvaluator. Excluded from the bench because disk
    I/O is a function of how the Arc files are stored, not cell-eval.

Expect this to be substantially slower than the gpu run — last full pass was
~30 m bench wall on Arc.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

os.environ["CELL_EVAL_BACKEND"] = "cpu"

import anndata as ad
import numba as nb
import numpy as np
import polars as pl
import scipy.sparse as sp

from cell_eval import MetricsEvaluator

REAL = "/home/sdicks/git/rapids-singlecell-notebooks/arc/adata_real.h5ad"
PRED = "/home/sdicks/git/rapids-singlecell-notebooks/arc/adata_pred.h5ad"
OUTDIR = "/home/sdicks/git/cell-eval/cell-eval-arc-celleval-cpu"
PERT_COL = "drugname_drugconc"
CONTROL = "[('DMSO_TF', 0.0, 'uM')]"

os.makedirs(OUTDIR, exist_ok=True)


@nb.njit(parallel=True, cache=True)
def _nnz_per_row(X):
    N, M = X.shape
    counts = np.empty(N, dtype=np.int32)
    for i in nb.prange(N):  # ty: ignore[not-iterable]
        c = 0
        for j in range(M):
            if X[i, j] != 0:
                c += 1
        counts[i] = c
    return counts


@nb.njit(parallel=True, cache=True)
def _fill_csr(X, indptr, indices, data):
    N, M = X.shape
    for i in nb.prange(N):  # ty: ignore[not-iterable]
        pos = indptr[i]
        for j in range(M):
            v = X[i, j]
            if v != 0:
                indices[pos] = j
                data[pos] = v
                pos += 1


def dense_to_csr(X: np.ndarray) -> sp.csr_matrix:
    if X.dtype != np.float32:
        X = np.ascontiguousarray(X, dtype=np.float32)
    elif not X.flags.c_contiguous:
        X = np.ascontiguousarray(X)
    N, M = X.shape
    counts = _nnz_per_row(X)
    indptr = np.empty(N + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(counts, dtype=np.int32, out=indptr[1:])
    nnz = int(indptr[-1])
    if nnz > 2_147_483_647:
        raise ValueError(f"nnz={nnz} exceeds int32 max")
    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float32)
    _fill_csr(X, indptr, indices, data)
    return sp.csr_matrix((data, indices, indptr), shape=(N, M))


def fmt(seconds: float) -> str:
    if seconds >= 3600:
        h, rem = divmod(seconds, 3600); m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {s:.1f}s"
    if seconds >= 60:
        m, s = divmod(seconds, 60); return f"{int(m)}m {s:.1f}s"
    return f"{seconds:.2f}s"


def load_and_sparsify(path: str, label: str, timings: dict[str, float]) -> ad.AnnData:
    t = time.perf_counter()
    adata = ad.read_h5ad(path)
    timings[f"{label}:read_h5ad"] = time.perf_counter() - t
    print(f"{label}: dense X dtype={adata.X.dtype} shape={adata.X.shape}, "
          f"converting to CSR", flush=True)
    t = time.perf_counter()
    adata.X = dense_to_csr(adata.X)
    timings[f"{label}:dense_to_csr"] = time.perf_counter() - t
    nnz = adata.X.nnz
    print(f"{label}: CSR nnz={nnz:,} "
          f"({nnz / (adata.n_obs * adata.n_vars):.2%} density)", flush=True)
    return adata


def main() -> None:
    print(f"CELL_EVAL_BACKEND={os.environ['CELL_EVAL_BACKEND']}", flush=True)
    setup_t0 = time.perf_counter()
    timings: dict[str, float] = {}

    adata_real = load_and_sparsify(REAL, "real", timings)
    adata_pred = load_and_sparsify(PRED, "pred", timings)
    gc.collect()
    timings["setup_total_io_and_sparsify"] = time.perf_counter() - setup_t0

    bench_t0 = time.perf_counter()
    t = time.perf_counter()
    ev = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL,
        pert_col=PERT_COL,
        outdir=OUTDIR,
        num_threads=-1,
    )
    timings["metrics_evaluator_init_inc_de"] = time.perf_counter() - t
    print(f"MetricsEvaluator (incl. DE): {fmt(timings['metrics_evaluator_init_inc_de'])}", flush=True)

    metric_configs = {
        "discrimination_score_l1": {"exclude_target_gene": False},
        "discrimination_score_l2": {"exclude_target_gene": False},
        "discrimination_score_cosine": {"exclude_target_gene": False},
    }

    t = time.perf_counter()
    full, agg = ev.compute(
        profile="full",
        metric_configs=metric_configs,
        write_csv=True,
        break_on_error=False,
    )
    timings["compute_all_metrics"] = time.perf_counter() - t
    print(f"Pipeline compute(): {fmt(timings['compute_all_metrics'])}", flush=True)

    bench_wall = time.perf_counter() - bench_t0
    total_wall = time.perf_counter() - setup_t0
    timings["bench_wall_clock"] = bench_wall
    timings["wall_clock_total"] = total_wall

    with open(os.path.join(OUTDIR, "celleval_cpu_timings.json"), "w") as f:
        json.dump(timings, f, indent=2)

    print("\n=== aggregated metrics (mean row) ===", flush=True)
    mean_row = agg.filter(pl.col("statistic") == "mean").drop("statistic").row(0, named=True)
    for k, v in mean_row.items():
        try:
            print(f"  {k:35s} {float(v):.6f}", flush=True)
        except (TypeError, ValueError):
            print(f"  {k:35s} {v}", flush=True)

    lines = [
        "# cell-eval (cpu / illico backend) — Arc benchmark\n",
        f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%S %Z')}`",
        f"- Inputs: `{REAL}` + `{PRED}`",
        f"- `pert_col`: `{PERT_COL}` · `control`: `{CONTROL}`",
        f"- Backend: `{os.environ['CELL_EVAL_BACKEND']}` "
        f"(pdex via `cell_eval.pdex._illico`, metrics via `cell_eval.metrics._anndata._cpu`)",
        f"- **Bench wall (cell-eval only)**: **{fmt(bench_wall)}**",
        f"- Total wall incl. h5ad load + sparsify: {fmt(total_wall)}\n",
        "## Stages\n",
        "| Stage | Wall time |",
        "|---|---:|",
    ]
    for k in [
        "real:read_h5ad", "real:dense_to_csr",
        "pred:read_h5ad", "pred:dense_to_csr",
        "setup_total_io_and_sparsify",
        "metrics_evaluator_init_inc_de",
        "compute_all_metrics",
        "bench_wall_clock",
        "wall_clock_total",
    ]:
        if k in timings:
            lines.append(f"| `{k}` | {fmt(timings[k])} |")
    lines.append("\n## Aggregated metrics (mean over perturbations)\n")
    lines.append("| Metric | Mean |")
    lines.append("|---|---:|")
    for k, v in mean_row.items():
        try:
            lines.append(f"| `{k}` | {float(v):.6f} |")
        except (TypeError, ValueError):
            lines.append(f"| `{k}` | {v} |")
    with open(os.path.join(OUTDIR, "celleval_cpu_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nReport: {OUTDIR}/celleval_cpu_report.md", flush=True)
    print(f"Bench wall: {fmt(bench_wall)}  ·  Total: {fmt(total_wall)}", flush=True)


if __name__ == "__main__":
    main()
