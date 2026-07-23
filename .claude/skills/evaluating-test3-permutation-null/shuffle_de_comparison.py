#!/usr/bin/env python3
"""Per-perturbation DE gene count scatter — pdex vs pydeseq2 — on a perturbed-vs-perturbed null.

NULL DESIGN
-----------
1. Take all non-control (perturbed) cells and shuffle their labels (globally or within-batch).
2. This produces N "fake perturbation groups" — each is a random mix of cells from
   all real perturbations (in that batch, for within-batch mode).
3. For each fake group X: randomly select a different fake group Y as reference,
   then run DE(X vs Y).  Both arms are similarly-sized perturbed cells, so the
   perturbed-vs-control confound cancels and pseudobulk sample sizes are balanced.
   Neither arm has perturbation-specific signal (both are random mixes).
4. A calibrated method calls ~0 DE genes per comparison.
   This is a genuine null analogous to test_2's control A/B split.

Two variants are always produced:
  global  — labels shuffled across all batches ignoring batch membership
  within  — labels shuffled independently within each batch stratum

Output
------
  <outdir>/plots/test_3_shuffle_de_comparison__global.png
  <outdir>/plots/test_3_shuffle_de_comparison__within.png

Usage
-----
  uv run --project /path/to/cell-eval \
    python /path/to/shuffle_de_comparison.py \
    --config config.yaml --outdir .
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os

import anndata as ad
import numpy as np
import polars as pl
import scanpy as sc
import scipy.sparse as sp
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from cell_eval._de_backends import build_de_frame

log = logging.getLogger("shuffle_de_cmp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LFC_ARCHIVE_VERSION = 5
_T3_WORKER_STATE = {}


# ── helpers ────────────────────────────────────────────────────────────────────

def _looks_raw(X) -> bool:
    """Robust raw-integer detection: X looks like raw counts if (a sample of) its
    finite values are all (close to) integers. Matches the shared de_helpers heuristic."""
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        return False
    if data.size > 5_000_000:
        rng = np.random.default_rng(0)
        data = data[rng.integers(0, data.size, 5_000_000)]
    return bool(np.allclose(data, np.rint(data)))


def _shuffle_within_batch(labels: np.ndarray, batch: np.ndarray | None,
                          rng: np.random.Generator) -> np.ndarray:
    """Shuffle labels within each batch stratum (or globally if batch is None)."""
    out = labels.copy()
    if batch is None:
        idx = np.arange(len(labels))
        rng.shuffle(idx)
        return out[idx]
    for b in np.unique(batch):
        idx = np.where(batch == b)[0]
        if idx.size < 2:
            continue
        perm = idx.copy()
        rng.shuffle(perm)
        out[idx] = out[perm]
    return out


def _sig_mask(lfc, fdr, fdr_thr, lfc_thr) -> np.ndarray:
    return np.isfinite(fdr) & (fdr <= fdr_thr) & (np.abs(lfc) >= lfc_thr)


def _cpm_eligible_genes(adata: ad.AnnData, groupby: str, pert_label: str,
                         ctrl_label: str, counts_layer, *, min_cpm: float = 5.0) -> set:
    """Return the set of genes with mean CPM ≥ min_cpm in either perturbed or control cells.

    Shared eligibility filter applied identically to both pdex and pyDESeq2.  Uses raw
    counts from counts_layer when available; otherwise uses expm1(adata.X) for log-normalised
    X or adata.X directly for integer-count X.
    """
    obs_g = adata.obs[groupby].astype(str).to_numpy()
    if counts_layer and counts_layer in adata.layers:
        X = adata.layers[counts_layer]
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        X = X.tocsr()
    else:
        X = adata.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        X = X.tocsr()
        sample = X.data[:min(5_000, X.data.size)] if X.data.size > 0 else np.array([0.0])
        if not np.allclose(sample, np.rint(sample)):
            X = X.copy()
            X.data = np.expm1(X.data)

    pos_p = np.where(obs_g == pert_label)[0]
    pos_c = np.where(obs_g == ctrl_label)[0]
    if pos_p.size == 0 or pos_c.size == 0:
        return set(adata.var_names)

    def mean_cpm(pos):
        group = X[pos]
        library_sizes = np.asarray(group.sum(axis=1)).ravel().astype(float)
        scales = np.divide(
            1e6, library_sizes, out=np.zeros_like(library_sizes),
            where=library_sizes > 0,
        )
        return np.asarray(group.multiply(scales[:, None]).mean(axis=0)).ravel()

    m_pert = mean_cpm(pos_p)
    m_ctrl = mean_cpm(pos_c)
    eligible_idx = np.where((m_pert >= min_cpm) | (m_ctrl >= min_cpm))[0]
    return set(adata.var_names[eligible_idx])


def _run_de(adata: ad.AnnData, method: str, groupby: str, reference: str,
            n_threads: int, counts_layer, replicate_col,
            non_parametric_engine: str = "pdex") -> pl.DataFrame:
    """Run DE and preserve every returned LFC for complete-case analyses."""
    return build_de_frame(
        mode="real",
        adata=adata,
        control_pert=reference,
        pert_col=groupby,
        num_threads=n_threads,
        allow_discrete=(method == "pydeseq2"),
        de_method=method,
        de_kwargs=({"engine": non_parametric_engine} if method == "pdex" else None),
        counts_layer=counts_layer,
        replicate_col=replicate_col,
    )


def _count_sig_for_target(de, target_label: str, fdr_thr: float, lfc_thr: float,
                          cpm_eligible: set[str] | None = None):
    """Count CPM-eligible DE genes and return their set for a target label."""
    sub = de.filter(de["target"] == target_label)
    if sub.height == 0:
        return 0, set()
    lfc = sub["log2_fold_change"].to_numpy().astype(float)
    fdr = sub["fdr"].to_numpy().astype(float)
    feats = sub["feature"].to_numpy().astype(str)
    sig = _sig_mask(lfc, fdr, fdr_thr, lfc_thr)
    if cpm_eligible is not None:
        sig &= np.fromiter((g in cpm_eligible for g in feats), dtype=bool, count=len(feats))
    return int(sig.sum()), set(feats[sig])


def _jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 1.0


# ── main null computation ──────────────────────────────────────────────────────

def _run_one_fake_comparison(task, *, adata_raw, adata_norm, shuffled,
                             fdr_thr, lfc_thr, n_threads, counts_layer,
                             replicate_col, min_cells, gene_names, methods,
                             non_parametric_engine):
    """Run one shuffled target/reference comparison."""
    p, ref_label = task
    gene_idx = {g: i for i, g in enumerate(gene_names or [])}
    n_genes = len(gene_names or [])
    mask_p = shuffled == p
    mask_ref = shuffled == ref_label
    n_p, n_ref = int(mask_p.sum()), int(mask_ref.sum())
    if n_p < min_cells or n_ref < min_cells:
        return p, None

    combined = mask_p | mask_ref
    a_norm = adata_norm[combined].copy()
    a_norm.obs["_group"] = np.where(mask_p[combined], "target", "ref")
    a_raw = adata_raw[combined].copy()
    a_raw.obs["_group"] = a_norm.obs["_group"].values
    cpm_eligible = _cpm_eligible_genes(
        a_raw, "_group", "target", "ref", counts_layer,
    )
    log.info("  %s vs %s  (n_target=%d, n_ref=%d)", p, ref_label, n_p, n_ref)

    n_pdex, genes_pdex = 0, set()
    n_pydx, genes_pydx = 0, set()
    lfc_pdex_arr = np.full(n_genes, np.nan)
    lfc_pydx_arr = np.full(n_genes, np.nan)

    if "pdex" in methods:
        try:
            de_pdex = _run_de(a_norm, "pdex", "_group", "ref",
                              n_threads, counts_layer, replicate_col, non_parametric_engine)
            n_pdex, genes_pdex = _count_sig_for_target(
                de_pdex, "target", fdr_thr, lfc_thr, cpm_eligible,
            )
            if gene_idx:
                sub = de_pdex.filter(de_pdex["target"] == "target")
                for feat, lfc_val in zip(
                    sub["feature"].to_list(), sub["log2_fold_change"].to_list()
                ):
                    idx = gene_idx.get(str(feat))
                    if idx is not None:
                        lfc_pdex_arr[idx] = np.nan if lfc_val is None else float(lfc_val)
        except Exception as exc:  # noqa: BLE001
            log.warning("pdex shuffled null for %s: %s", p, exc)

    if "pydeseq2" in methods:
        try:
            de_pydx = _run_de(a_raw, "pydeseq2", "_group", "ref",
                              n_threads, counts_layer, replicate_col, non_parametric_engine)
            n_pydx, genes_pydx = _count_sig_for_target(
                de_pydx, "target", fdr_thr, lfc_thr, cpm_eligible,
            )
            if gene_idx:
                sub = de_pydx.filter(de_pydx["target"] == "target")
                for feat, lfc_val in zip(
                    sub["feature"].to_list(), sub["log2_fold_change"].to_list()
                ):
                    idx = gene_idx.get(str(feat))
                    if idx is not None:
                        lfc_pydx_arr[idx] = np.nan if lfc_val is None else float(lfc_val)
        except Exception as exc:  # noqa: BLE001
            log.warning("pydeseq2 shuffled null for %s: %s", p, exc)

    result = {
        "n_pdex": n_pdex, "genes_pdex": genes_pdex,
        "n_pydx": n_pydx, "genes_pydx": genes_pydx,
        "n_cells": n_p, "reference": ref_label,
        "n_reference_cells": n_ref,
        "lfc_pdex": lfc_pdex_arr, "lfc_pydx": lfc_pydx_arr,
    }
    log.info("  fake-%s  n=%d  pdex_n_de=%d  pydx_n_de=%d",
             p, n_p, n_pdex, n_pydx)
    return p, result


def _run_one_fake_comparison_worker(task):
    return _run_one_fake_comparison(task, **_T3_WORKER_STATE)


def _run_shuffled_null(adata_raw: ad.AnnData, adata_norm: ad.AnnData,
                       perts: list[str], pc: str,
                       block_cols: list[str], seed: int,
                       fdr_thr: float, lfc_thr: float,
                       n_threads: int, counts_layer, replicate_col,
                       shuffle_mode: str = "within",
                       min_cells: int = 10,
                       gene_names: list[str] | None = None,
                       methods: list[str] | None = None,
                       non_parametric_engine: str = "pdex",
                       comparison_workers: int = 1,
                       ) -> dict[str, dict]:
    """
    Shuffle all non-control labels (globally or within-batch) → N fake groups.
    For each fake group X: randomly pick a different fake group Y and run DE(X vs Y).
    Returns dict[pert_label] = {n_pdex, n_pydx, genes_pdex, genes_pydx, n_cells,
                                lfc_pdex, lfc_pydx}  where lfc_* are full-gene LFC arrays
    aligned to gene_names (NaN for genes with no result). Used for the correlation matrix.
    """
    labels = adata_norm.obs[pc].astype(str).to_numpy()
    bc = [c for c in block_cols if c in adata_norm.obs.columns]
    if shuffle_mode == "within" and bc:
        batch = adata_norm.obs[bc[0]].astype(str).to_numpy()
    else:
        batch = None  # global: ignore batch

    rng = np.random.default_rng(seed)
    shuffled = _shuffle_within_batch(labels.copy(), batch, rng)
    adata_norm = adata_norm.copy()
    adata_raw  = adata_raw.copy()
    adata_norm.obs["_shuf"] = shuffled
    adata_raw.obs["_shuf"]  = shuffled

    if methods is None:
        methods = ["pdex", "pydeseq2"]
    results: dict[str, dict] = {}
    fake_labels = sorted(set(shuffled))

    # For each fake group X, randomly pick a different fake group Y as reference
    rng_pair = np.random.default_rng(seed + 1)
    pairs: dict[str, str] = {}
    for p in fake_labels:
        others = [q for q in fake_labels if q != p]
        pairs[p] = others[int(rng_pair.integers(0, len(others)))]

    tasks = [(p, pairs[p]) for p in fake_labels]
    worker_state = {
        "adata_raw": adata_raw, "adata_norm": adata_norm,
        "shuffled": shuffled, "fdr_thr": fdr_thr, "lfc_thr": lfc_thr,
        "n_threads": n_threads, "counts_layer": counts_layer,
        "replicate_col": replicate_col, "min_cells": min_cells,
        "gene_names": gene_names, "methods": methods,
        "non_parametric_engine": non_parametric_engine,
    }
    use_pool = comparison_workers > 1 and not (
        "pdex" in methods and non_parametric_engine == "rsc"
    )
    if use_pool:
        _T3_WORKER_STATE.clear()
        _T3_WORKER_STATE.update(worker_state)
        pool = mp.get_context("fork").Pool(processes=comparison_workers)
        fitted = pool.imap(_run_one_fake_comparison_worker, tasks, chunksize=1)
    else:
        pool = None
        fitted = (
            _run_one_fake_comparison(task, **worker_state) for task in tasks
        )
    try:
        for p, result in fitted:
            if result is not None:
                results[p] = result
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return results


def _write_lfc_vectors(results, gene_names, mode, method, engine, method_key, out_path):
    """Stream one long-form Parquet row per shuffled-comparison-feature LFC pair."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("method", pa.string()),
        ("engine", pa.string()),
        ("shuffle_mode", pa.string()),
        ("comparison", pa.string()),
        ("reference", pa.string()),
        ("n_cells", pa.int32()),
        ("n_reference_cells", pa.int32()),
        ("feature_index", pa.int32()),
        ("feature", pa.string()),
        ("lfc", pa.float64()),
    ])
    writer = pq.ParquetWriter(out_path, schema, compression="zstd")
    n_rows = 0
    try:
        for comparison, d in results.items():
            values = np.asarray(d[method_key], dtype=float)
            n = len(values)
            table = pa.Table.from_pydict({
                "method": [method] * n,
                "engine": [engine] * n,
                "shuffle_mode": [mode] * n,
                "comparison": [comparison] * n,
                "reference": [str(d["reference"])] * n,
                "n_cells": np.full(n, int(d["n_cells"]), dtype=np.int32),
                "n_reference_cells": np.full(
                    n, int(d["n_reference_cells"]), dtype=np.int32
                ),
                "feature_index": np.arange(n, dtype=np.int32),
                "feature": gene_names,
                "lfc": values,
            }, schema=schema)
            writer.write_table(table)
            n_rows += n
    finally:
        writer.close()
    return n_rows


# ── plotting ───────────────────────────────────────────────────────────────────

def _plot(results: dict[str, dict], out_png: str, title: str,
          fdr_thr: float, lfc_thr: float) -> None:
    perts = [p for p in results]
    if not perts:
        log.warning("No perturbations to plot; skipping %s", out_png)
        return

    x = np.array([results[p]["n_pdex"] for p in perts], dtype=float)
    y = np.array([results[p]["n_pydx"]  for p in perts], dtype=float)
    n_cells = np.array([results[p]["n_cells"] for p in perts], dtype=float)
    jaccard = np.array([_jaccard(results[p]["genes_pdex"], results[p]["genes_pydx"])
                        for p in perts])

    nc_norm = (n_cells - n_cells.min()) / max(n_cells.max() - n_cells.min(), 1)
    dot_sizes = 40 + nc_norm * 360

    cmap = plt.cm.viridis
    cnorm = Normalize(vmin=n_cells.min(), vmax=n_cells.max())

    fig, ax = plt.subplots(figsize=(8, 7))

    lim_max = max(int(x.max()), int(y.max()), 5) * 1.3 + 1
    ax.fill_between([0, lim_max], [0, 0],       [0, lim_max],      color="#dce8f5", alpha=0.35, zorder=0)
    ax.fill_between([0, lim_max], [0, lim_max], [lim_max, lim_max], color="#f5ece0", alpha=0.35, zorder=0)
    ax.plot([0, lim_max], [0, lim_max], "--", color="#777777", lw=1.2, zorder=2)
    ax.axhline(0, color="#bbbbbb", lw=0.5, zorder=1)
    ax.axvline(0, color="#bbbbbb", lw=0.5, zorder=1)

    ax.scatter(x, y, s=dot_sizes, c=n_cells, cmap=cmap, norm=cnorm,
               alpha=0.85, linewidths=0.5, edgecolors="white", zorder=3)

    ax.set_xlim(-lim_max * 0.04, lim_max)
    ax.set_ylim(-lim_max * 0.04, lim_max)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    cx, cy = float(np.mean(x)), float(np.mean(y))
    for i, p in enumerate(perts):
        xi, yi = float(x[i]), float(y[i])
        vx, vy = xi - cx, yi - cy
        mag = max(np.hypot(vx, vy), 1e-6)
        nx_, ny_ = vx / mag, vy / mag
        lx = float(np.clip(xi + nx_ * lim_max * 0.16, lim_max * 0.01, lim_max * 0.93))
        ly = float(np.clip(yi + ny_ * lim_max * 0.16, lim_max * 0.02, lim_max * 0.97))
        ax.annotate(
            f"{p}\nJ={jaccard[i]:.2f}",
            xy=(xi, yi), xytext=(lx, ly),
            fontsize=6.5,
            arrowprops=dict(arrowstyle="-", color="#444444", lw=0.7, shrinkA=0, shrinkB=3),
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
            zorder=5,
        )

    cb = fig.colorbar(ScalarMappable(norm=cnorm, cmap=cmap), ax=ax, pad=0.02)
    cb.set_label("cell count per fake-perturbation group", fontsize=9)

    ax.legend(handles=[
        plt.Line2D([0], [0], ls="--", color="#777777", lw=1.2, label="equal calling"),
        plt.Rectangle((0, 0), 1, 1, fc="#f5ece0", alpha=0.6, label="pyDESeq2 calls more"),
        plt.Rectangle((0, 0), 1, 1, fc="#dce8f5", alpha=0.6, label="pdex calls more"),
    ], fontsize=7.5, loc="upper left")

    ax.set_xlabel("# DE genes — pdex (cell-level Wilcoxon)", fontsize=10)
    ax.set_ylabel("# DE genes — pyDESeq2 (pseudobulk)", fontsize=10)
    ax.set_title(
        f"{title}\n"
        f"FDR < {fdr_thr}, |LFC| ≥ {lfc_thr}  ·  dot size ∝ cell count  ·  "
        f"J = Jaccard(pdex DE ∩ pyDESeq2 DE)\n"
        f"null: shuffled group X vs random shuffled group Y — calibrated → cloud at origin",
        fontsize=8.5,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_png)


# ── correlation matrix plot ────────────────────────────────────────────────────

def _cc_mask_and_ranked(results: dict, gene_names: list[str],
                        methods: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Compute the shared complete-case gene mask and a variance-ranked index array.

    complete-case: gene has finite LFC in BOTH pdex AND pydeseq2 for EVERY fake perturbation.
    Ranking: cross-method LFC variance pooled from all fake-perts × both methods — symmetric,
    neither method's calling dominates. Returns (cc_mask, ranked_cc_idx) where cc_mask is a
    boolean over gene_names and ranked_cc_idx is the descending-variance order within cc_mask.
    """
    n = len(gene_names)
    cc = np.ones(n, dtype=bool)
    method_keys = [k for k, m in (("lfc_pdex", "pdex"), ("lfc_pydx", "pydeseq2")) if m in methods]
    for p in results:
        for key in method_keys:
            v = results[p].get(key)
            if v is not None:
                cc &= np.isfinite(v)
            else:
                cc[:] = False  # method absent entirely — no complete-case genes
    cc_idx = np.where(cc)[0]
    if cc_idx.size == 0:
        return cc, np.array([], dtype=int)
    # Pool LFC from all methods × fake-perts for variance ranking
    pool = np.stack([results[p][key][cc_idx]
                     for p in results for key in method_keys
                     if results[p].get(key) is not None])
    var_cc = np.nanvar(pool, axis=0)
    ranked_cc_idx = cc_idx[np.argsort(-var_cc)]
    return cc, ranked_cc_idx


def _plot_corr_matrix(results: dict[str, dict], method_key: str,
                      out_png: str, title: str,
                      genes_mask: np.ndarray | None = None,
                      gene_set_label: str = "") -> None:
    """Pairwise Spearman correlation matrix between fake-pert LFC vectors.

    Each cell [i, j] = Spearman r between the LFC vector of fake pert i and fake pert j
    (both compared against their randomly-chosen reference within the same shuffled null).
    A calibrated null — clean diagonal, zero elsewhere.

    genes_mask: boolean array over gene_names. When provided, both methods use IDENTICAL
    genes so the two panels are feature-comparable. Should be the shared complete-case mask
    from _cc_mask_and_ranked. When None, falls back to pairwise-finite filtering (old behaviour).
    """
    from scipy import stats

    perts = [p for p in results if results[p].get(method_key) is not None]
    if len(perts) < 2:
        log.warning("Not enough perturbations for correlation matrix (%s); skipping %s",
                    method_key, out_png)
        return
    perts = sorted(perts)
    n = len(perts)

    vecs = [results[p][method_key] for p in perts]
    if genes_mask is not None:
        vecs = [v[genes_mask] for v in vecs]
        n_features = int(genes_mask.sum())
    else:
        n_features = len(vecs[0]) if vecs else 0

    if n_features < 20:
        log.warning("Skipping %s: only %d genes available", out_png, n_features)
        return

    mat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            a, b = vecs[i], vecs[j]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 20:
                mat[i, j] = mat[j, i] = 0.0
                continue
            r, _ = stats.spearmanr(a[ok], b[ok])
            mat[i, j] = mat[j, i] = float(r)
    np.fill_diagonal(mat, 1.0)

    diagonal_mask = np.eye(n, dtype=bool)
    mean_diagonal = float(np.nanmean(mat[diagonal_mask]))
    mean_off = float(np.nanmean(mat[~diagonal_mask]))

    fig, ax = plt.subplots(figsize=(max(7, n * 0.22 + 1.5), max(6, n * 0.22 + 1.5)))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(perts, rotation=90, fontsize=max(4, min(7, 120 // n)))
    ax.set_yticks(range(n))
    ax.set_yticklabels(perts, fontsize=max(4, min(7, 120 // n)))
    ax.set_xlabel("fake perturbation (shuffled labels)", fontsize=9)
    ax.set_ylabel("fake perturbation (shuffled labels)", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    glabel = gene_set_label or f"{n_features} genes"
    cb.set_label(f"Spearman r (LFC)  [{glabel}]", fontsize=8)
    method_label = "pdex (cell-level Wilcoxon)" if method_key == "lfc_pdex" else "pydeseq2 (pseudobulk DESeq2)"
    ax.set_title(f"{title}\n{method_label}  (mean diagonal ρ = {mean_diagonal:.3f}; "
                 f"off-diagonal ρ = {mean_off:.3f})\n"
                 f"calibrated null — clean diagonal, zero elsewhere",
                 fontsize=8.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_png)


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config",        default="config.yaml")
    ap.add_argument("--adata",         default=None)
    ap.add_argument("--pert-col",      default=None)
    ap.add_argument("--control",       default=None)
    ap.add_argument("--replicate-col", default=None)
    ap.add_argument("--block-cols",    default=None, help="comma-separated batch column(s)")
    ap.add_argument("--fdr",           type=float, default=None)
    ap.add_argument("--lfc",           type=float, default=None)
    ap.add_argument("--n-threads",     type=int,   default=None)
    ap.add_argument("--comparison-workers", type=int, default=1,
                    help="parallel shuffled comparisons for CPU PyDESeq2; use --n-threads 1")
    ap.add_argument("--seed",          type=int,   default=None)
    ap.add_argument("--outdir",        default=".")
    ap.add_argument("--methods",        default="pdex,pydeseq2",
                    help="comma-sep DE backends to run (default: pdex,pydeseq2)")
    ap.add_argument("--non-parametric-engine", choices=("pdex", "rsc"), default=None,
                    help="non-parametric engine; overrides config (default: pdex)")
    ap.add_argument("--replot",        action="store_true",
                    help="skip DE; regenerate corr-matrix plots from saved test_3_lfc_matrix__<mode>.npz files")
    ap.add_argument("--shuffle-mode",  default="both",
                    choices=["global", "within", "both"],
                    help="global: ignore batch; within: shuffle within each batch; both: produce both plots")
    args = ap.parse_args()
    if args.comparison_workers < 1:
        ap.error("--comparison-workers must be at least 1")
    os.makedirs(args.outdir, exist_ok=True)

    cfg: dict = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        log.info("Loaded config from %s", args.config)

    if args.adata:          cfg["adata_path"]    = args.adata
    if args.pert_col:       cfg["pert_col"]      = args.pert_col
    if args.control:        cfg["control_pert"]  = args.control
    if args.replicate_col:  cfg["replicate_col"] = args.replicate_col
    if args.block_cols:     cfg["block_cols"]    = [c.strip() for c in args.block_cols.split(",")]
    if args.fdr is not None:      cfg["fdr_threshold"] = args.fdr
    if args.lfc is not None:      cfg["lfc_threshold"] = args.lfc
    if args.n_threads is not None: cfg["num_threads"]  = args.n_threads
    if args.seed is not None:      cfg["seed"]         = args.seed
    if args.non_parametric_engine is not None: cfg["non_parametric_engine"] = args.non_parametric_engine

    cfg.setdefault("fdr_threshold", 0.05)
    cfg.setdefault("lfc_threshold", 0.1)
    cfg.setdefault("num_threads", 8)
    cfg.setdefault("seed", 0)
    cfg.setdefault("block_cols", [])
    cfg.setdefault("counts_layer", None)
    cfg.setdefault("replicate_col", None)
    cfg.setdefault("non_parametric_engine", "pdex")
    if cfg["non_parametric_engine"] not in {"pdex", "rsc"}:
        ap.error(
            "config non_parametric_engine must be 'pdex' or 'rsc'"
        )

    available_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    )
    worker_cap = max(1, available_cpus // max(1, int(cfg["num_threads"])))
    if args.comparison_workers > worker_cap:
        log.warning(
            "Capping --comparison-workers from %d to %d (%d CPUs, %d threads/fit)",
            args.comparison_workers, worker_cap, available_cpus, cfg["num_threads"],
        )
        args.comparison_workers = worker_cap

    fdr_thr    = cfg["fdr_threshold"]
    lfc_thr    = cfg["lfc_threshold"]
    pc         = cfg["pert_col"]
    ctrl       = cfg["control_pert"]
    block_cols = cfg.get("block_cols", [])
    min_cells  = 2 * cfg.get("min_cells_per_group", 10)

    log.info("Loading %s", cfg["adata_path"])
    adata = ad.read_h5ad(cfg["adata_path"])
    log.info("Loaded %d cells × %d genes", *adata.shape)

    perts = [str(p) for p in adata.obs[pc].unique() if str(p) != ctrl]
    # keep only perturbed cells — control is never used in this null
    pert_only = adata[adata.obs[pc].astype(str).isin(perts)].copy()
    log.info("%d perturbations, %d perturbed cells (control dropped)", len(perts), pert_only.n_obs)

    adata_raw = pert_only.copy()
    adata_norm = pert_only.copy()
    if _looks_raw(adata_norm.X):
        sc.pp.normalize_total(adata_norm, inplace=True)
        sc.pp.log1p(adata_norm)
        log.info("Normalised for pdex")

    dataset_name = os.path.splitext(os.path.basename(cfg["adata_path"]))[0]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    modes = []
    if args.shuffle_mode in ("global", "both"):
        modes.append("global")
    if args.shuffle_mode in ("within", "both"):
        modes.append("within")

    # --replot: skip DE and regenerate corr-matrix plots from saved .npz files
    if args.replot:
        for mode in modes:
            npz_path = os.path.join(args.outdir, f"test_3_lfc_matrix__{mode}.npz")
            if not os.path.exists(npz_path):
                log.warning("--replot: %s not found, skipping", npz_path)
                continue
            d = np.load(npz_path, allow_pickle=False)
            archive_version = (int(np.asarray(d["format_version"]).item())
                               if "format_version" in d else None)
            if archive_version != LFC_ARCHIVE_VERSION:
                d.close()
                raise ValueError(
                    f"Stale or unsupported Test-3 LFC archive {npz_path}: "
                    f"format_version={archive_version!r}. Rerun without --replot "
                    "to regenerate unfiltered LFC vectors."
                )
            archive_engine = (str(np.asarray(d["non_parametric_engine"]).item())
                              if "non_parametric_engine" in d else None)
            if "pdex" in methods and archive_engine != cfg["non_parametric_engine"]:
                d.close()
                raise ValueError(
                    f"Test-3 archive non_parametric_engine={archive_engine!r}, expected "
                    f"{cfg['non_parametric_engine']!r}; rerun without --replot"
                )
            pert_names = d["pert_names"].tolist()
            gene_names_r = d["gene_names"].tolist() if "gene_names" in d else []
            results_replot = {p: {"lfc_pdex": d["lfc_pdex"][i] if "lfc_pdex" in d else None,
                                  "lfc_pydx": d["lfc_pydx"][i] if "lfc_pydx" in d else None}
                              for i, p in enumerate(pert_names)}
            d.close()
            mode_label = "global shuffle" if mode == "global" else "within-batch shuffle"
            corr_title = f"calibrated null — clean diagonal, zero elsewhere\n{mode_label} — {dataset_name}"
            os.makedirs(os.path.join(args.outdir, "plots"), exist_ok=True)
            cc_mask_r, _ = _cc_mask_and_ranked(results_replot, gene_names_r, methods)
            n_cc_r = int(cc_mask_r.sum())
            cc_label_r = f"all complete-case genes (n={n_cc_r})"
            for method_key, method_tag in (("lfc_pdex", "pdex"), ("lfc_pydx", "pydeseq2")):
                _plot_corr_matrix(
                    results_replot, method_key,
                    out_png=os.path.join(args.outdir, "plots",
                                         f"test_3_corr_matrix__{mode}__{method_tag}.png"),
                    title=corr_title,
                    genes_mask=cc_mask_r, gene_set_label=cc_label_r,
                )
            log.info("Replotted corr matrices for mode=%s", mode)
        log.info("Done (--replot).")
        return

    gene_names = adata.var_names.tolist()

    for mode in modes:
        log.info("=== shuffle_mode=%s ===", mode)
        results = _run_shuffled_null(
            adata_raw, adata_norm, perts, pc,
            block_cols=block_cols, seed=cfg["seed"],
            fdr_thr=fdr_thr, lfc_thr=lfc_thr,
            n_threads=cfg["num_threads"],
            counts_layer=cfg["counts_layer"],
            replicate_col=cfg.get("replicate_col"),
            shuffle_mode=mode,
            min_cells=min_cells,
            gene_names=gene_names,
            methods=methods,
            non_parametric_engine=cfg["non_parametric_engine"],
            comparison_workers=args.comparison_workers,
        )
        # save LFC matrices so corr-matrix plots can be regenerated without re-running DE
        perts_with_lfc = [p for p in results if results[p].get("lfc_pdex") is not None]
        lfc_save: dict = {
            "format_version": np.asarray(LFC_ARCHIVE_VERSION, dtype=np.int64),
            "gene_names": np.array(gene_names),
            "pert_names": np.array(perts_with_lfc),
            "non_parametric_engine": np.asarray(cfg["non_parametric_engine"]),
        }
        if "pdex" in methods:
            lfc_save["lfc_pdex"] = np.stack([results[p]["lfc_pdex"] for p in perts_with_lfc])
        if "pydeseq2" in methods:
            lfc_save["lfc_pydx"] = np.stack([results[p]["lfc_pydx"] for p in perts_with_lfc])
        npz_path = os.path.join(args.outdir, f"test_3_lfc_matrix__{mode}.npz")
        np.savez_compressed(npz_path, **lfc_save)
        log.info("Saved LFC matrix → %s", npz_path)
        for method_key, method_tag in (
            ("lfc_pdex", "pdex"), ("lfc_pydx", "pydeseq2")
        ):
            if method_tag not in methods:
                continue
            lfc_path = os.path.join(
                args.outdir, f"test3_lfc_vectors_{mode}_{method_tag}.parquet"
            )
            n_lfc_rows = _write_lfc_vectors(
                results, gene_names, mode, method_tag,
                cfg["non_parametric_engine"] if method_tag == "pdex" else "cpu",
                method_key, lfc_path
            )
            log.info("Saved LFC vectors → %s (%d rows)", lfc_path, n_lfc_rows)

        mode_label = "global shuffle (across all batches)" if mode == "global" else f"within-batch shuffle (block_cols={block_cols})"
        if "pdex" in methods and "pydeseq2" in methods:
            _plot(
                results,
                out_png=os.path.join(args.outdir, "plots", f"test_3_shuffle_de_comparison__{mode}.png"),
                title=(f"DE gene count: pyDESeq2 vs pdex — {dataset_name}\n"
                       f"Null: shuffled fake-pert X vs random shuffled fake-pert Y  ({mode_label})"),
                fdr_thr=fdr_thr, lfc_thr=lfc_thr,
            )
        cc_mask, ranked_cc_idx = _cc_mask_and_ranked(results, gene_names, methods)
        n_cc = int(cc_mask.sum())
        log.info("Test-3 complete-case genes (both methods, all fake-perts): %d / %d",
                 n_cc, len(gene_names))

        corr_title = (f"Test-3 permutation-null LFC correlation — {dataset_name}\n"
                      f"{mode_label}")
        method_map = [("lfc_pdex", "pdex"), ("lfc_pydx", "pydeseq2")]

        # Primary: all complete-case genes shared between both methods
        cc_label = f"all complete-case genes (n={n_cc}; from {len(gene_names)} total)"
        for method_key, method_tag in [(k, t) for k, t in method_map if t in methods]:
            _plot_corr_matrix(
                results, method_key,
                out_png=os.path.join(args.outdir, "plots",
                                     f"test_3_corr_matrix__{mode}__{method_tag}.png"),
                title=corr_title,
                genes_mask=cc_mask, gene_set_label=cc_label,
            )

        # Sensitivity caps: top 400 / 2000 / 4000 by cross-method LFC variance
        for cap in (400, 2000, 4000):
            if cap >= n_cc:
                log.info("Test-3 sensitivity cap=%d skipped (cap ≥ n_cc=%d)", cap, n_cc)
                continue
            cap_mask = np.zeros(len(gene_names), dtype=bool)
            cap_mask[ranked_cc_idx[:cap]] = True
            cap_label = f"top {cap} genes by cross-method LFC variance (n={cap}; from {n_cc} cc)"
            for method_key, method_tag in [(k, t) for k, t in method_map if t in methods]:
                _plot_corr_matrix(
                    results, method_key,
                    out_png=os.path.join(args.outdir, "plots",
                                         f"test_3_corr_matrix__{mode}_top{cap}__{method_tag}.png"),
                    title=corr_title,
                    genes_mask=cap_mask, gene_set_label=cap_label,
                )
    log.info("Done.")


if __name__ == "__main__":
    main()
