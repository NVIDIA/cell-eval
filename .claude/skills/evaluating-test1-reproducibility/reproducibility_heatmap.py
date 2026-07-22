#!/usr/bin/env python
"""Test-1 reproducibility visualisations, wired to adata_Validation.h5ad.

Reuses the SHARED Test-1 helpers from the metaskill runner (`stratified_split`, `_de_two`,
`maybe_normalize`) so the split-half DE exactly matches the real Test 1 — then persists the
per-perturbation, per-gene split-A / split-B log2FC (which the runner doesn't save) and renders:

  Layer 1 — global heatmap: rows = all perturbations sorted by split-half ρ (lowest ρ at top),
            columns = union DE genes sorted by split-A mean LFC, two panels side by side (split A |
            split B). High-ρ perturbations (bottom) show matching A/B patterns; low-ρ ones (top) don't.

  Layer 2 — per-perturbation zoom for a few cases (high-ρ / low-ρ / borderline): the LFC_A-vs-LFC_B
            scatter next to that perturbation's two heatmap rows.

Colour: diverging blue–white–red centred at 0, capped at ±2 log2FC (genes near 0 in both splits are
white; colour in B where A is white — or vice versa — is irreproducibility made visible).

Run: `python reproducibility_heatmap.py --adata <h5ad>` (de_method=pdex by default).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import multiprocessing
import os
import pickle
import sys

# Backend thread counts are passed explicitly below. Prevent OpenBLAS, MKL,
# OpenMP, NumExpr, and Accelerate from silently creating a full-machine pool
# inside every repeat process (a common source of 100x oversubscription).
# Re-exec once so the limits exist before Python startup/site hooks; setting
# them only before the NumPy import is insufficient on environments whose
# sitecustomize imports a numerical library.
def _cli_positive_int(option: str, default: int) -> int:
    """Read a simple integer CLI option before argparse/numerical imports."""
    for index, token in enumerate(sys.argv[1:], start=1):
        if token == option and index + 1 < len(sys.argv):
            try:
                return max(1, int(sys.argv[index + 1]))
            except ValueError:
                return default
        if token.startswith(option + "="):
            try:
                return max(1, int(token.split("=", 1)[1]))
            except ValueError:
                return default
    return default


_REQUESTED_BACKEND_THREADS = _cli_positive_int("--threads", 8)
_THREAD_ENV_LIMITS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    # pdex calls numba.set_num_threads(cfg['num_threads']); Numba rejects a
    # value above this startup maximum. Match the explicit backend setting
    # while leaving unrelated numerical pools single-threaded.
    "NUMBA_NUM_THREADS": str(_REQUESTED_BACKEND_THREADS),
    "POLARS_MAX_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "TOKIO_WORKER_THREADS": "1",
    "ASYNC_STD_THREAD_COUNT": "1",
    "MALLOC_CONF": "background_thread:false,narenas:2",
}
if (
    os.environ.get("CELL_EVAL_THREAD_GUARD") != "1"
    and any(
        os.environ.get(name) != value
        for name, value in _THREAD_ENV_LIMITS.items()
    )
):
    guarded_env = os.environ.copy()
    guarded_env.update(_THREAD_ENV_LIMITS)
    guarded_env["CELL_EVAL_THREAD_GUARD"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], guarded_env)
for _thread_env, _thread_limit in _THREAD_ENV_LIMITS.items():
    os.environ[_thread_env] = _thread_limit

import anndata as ad
import numpy as np
import polars as pl
from scipy import stats

_RT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "de_helpers.py")

CAP = 2.0  # ±log2FC colour cap
SIGNATURE_CACHE_VERSION = 4
CPM_NORMALIZATION = "per_cell_then_mean"
MIN_CPM = 5.0

# Fork workers inherit these read-mostly objects without serialising a large
# AnnData for every repeat. Each worker owns its copy-on-write mutations (pdex
# normalisation) and returns only the repeat's compact signature dictionaries.
_REPEAT_ADATA = None
_REPEAT_CFG = None
_REPEAT_RT = None


def _load_runner():
    """Import the shared metaskill runner as a module (reuse its exact Test-1 helpers)."""
    spec = importlib.util.spec_from_file_location("rt_shared", _RT)
    rt = importlib.util.module_from_spec(spec)
    sys.modules["rt_shared"] = rt
    spec.loader.exec_module(rt)
    return rt


def _compute_cpm_elig(adata, pos_P, pos_C, cfg, *, min_cpm=MIN_CPM):
    """Return a set of gene names with mean CPM ≥ min_cpm in either pert or ctrl cells.

    Computed from raw counts (cfg['counts_layer'] if present) so the result is independent
    of whether adata.X has been log-normalised by maybe_normalize. Applied once per
    perturbation and shared across DE backends so the CPM gate is method-neutral.
    """
    import scipy.sparse as sp_local
    counts_layer = cfg.get("counts_layer")
    if counts_layer and counts_layer in adata.layers:
        X = adata.layers[counts_layer]
        if not sp_local.issparse(X):
            X = sp_local.csr_matrix(X)
        X = X.tocsr()
    else:
        X = adata.X
        if not sp_local.issparse(X):
            X = sp_local.csr_matrix(X)
        X = X.tocsr()
        sample = X.data[:min(5_000, X.data.size)] if X.data.size > 0 else np.array([0.0])
        if not np.allclose(sample, np.rint(sample)):
            X = X.copy(); X.data = np.expm1(X.data)
    def mean_cpm(pos):
        group = X[pos]
        library_sizes = np.asarray(group.sum(axis=1)).ravel().astype(float)
        scales = np.divide(
            1e6, library_sizes, out=np.zeros_like(library_sizes),
            where=library_sizes > 0,
        )
        return np.asarray(group.multiply(scales[:, None]).mean(axis=0)).ravel()

    m_pert = mean_cpm(pos_P)
    m_ctrl = mean_cpm(pos_C)
    return set(adata.var_names[np.where((m_pert >= min_cpm) | (m_ctrl >= min_cpm))[0]])


def split_half_signatures(adata, cfg, rt, *, seed=0):
    """For each perturbation: batch-stratified A/B split (controls also split), run DE_A = A vs
    ctrl_half_A and DE_B = B vs ctrl_half_B via the shared `_de_two`, and collect per-gene LFC_A/LFC_B
    (+ FDRs) and the split-half Spearman ρ. Returns a dict pert -> {rho, lfc_a, lfc_b, fdr_a, fdr_b}
    (each an aligned pandas Series indexed by gene)."""
    rt.maybe_normalize(adata, cfg)  # pdex expects log-norm; no-op for pydeseq2
    pc, mcg, ctrl = cfg["pert_col"], cfg["min_cells_per_group"], cfg["control_pert"]
    labels = adata.obs[pc].astype(str).to_numpy()
    pos_C = np.where(labels == ctrl)[0]
    ctrl_obs = adata.obs.iloc[pos_C]
    perts = [g for g in sorted(set(labels)) if g != ctrl]
    out = {}
    for i, pert in enumerate(perts):
        pos_P = np.where(labels == pert)[0]
        if pos_P.size < 2 * mcg:
            continue
        arm = rt.stratified_split(adata.obs.iloc[pos_P], cfg["block_cols"], seed + i)
        ai, bi = np.where(arm == "A")[0], np.where(arm == "B")[0]
        carm = rt.stratified_split(ctrl_obs, cfg["block_cols"], seed + i + 7)
        cai, cbi = np.where(carm == "A")[0], np.where(carm == "B")[0]
        if min(ai.size, bi.size, cai.size, cbi.size) < mcg:
            continue
        try:
            de_A = rt._de_two(adata, pos_P[ai], pos_C[cai], cfg, "pert", "ctrl")
            de_B = rt._de_two(adata, pos_P[bi], pos_C[cbi], cfg, "pert", "ctrl")
        except Exception as e:  # noqa: BLE001
            print(f"  {pert}: DE failed ({e})"); continue
        j = de_A.join(de_B, on="feature", how="inner", suffix="_b")
        la = j["log2_fold_change"].to_numpy().astype(float)
        lb = j["log2_fold_change_b"].to_numpy().astype(float)
        ok = np.isfinite(la) & np.isfinite(lb)
        rho_pairwise = (float(stats.spearmanr(la[ok], lb[ok]).statistic)
                        if ok.sum() >= 5 else float("nan"))
        genes = j["feature"].to_numpy().astype(str)
        elig = _compute_cpm_elig(adata, pos_P, pos_C, cfg)
        out[pert] = {"rho": float("nan"), "rho_pairwise": rho_pairwise,
                     "genes": genes, "lfc_a": la, "lfc_b": lb,
                     "fdr_a": j["fdr"].to_numpy().astype(float),
                     "fdr_b": j["fdr_b"].to_numpy().astype(float),
                     "cpm_elig": np.array([g in elig for g in genes]),
                     "n_cells": int(pos_P.size)}
        print(f"  {pert:20s} provisional-overlap ρ={rho_pairwise:.3f}  ({pos_P.size} cells)")
    if not out:
        raise ValueError(
            f"No perturbation produced a valid split-half signature. Check control "
            f"label {ctrl!r}, --min-cells={mcg}, block columns, and backend input."
        )
    return out


def _write_lfc_vectors(signatures_by_repeat, method, base_seed, out_path):
    """Stream one long-form Parquet row per repeat-perturbation-feature LFC pair."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("method", pa.string()),
        ("repeat", pa.int32()),
        ("seed", pa.int64()),
        ("perturbation", pa.string()),
        ("n_cells", pa.int32()),
        ("feature_index", pa.int32()),
        ("feature", pa.string()),
        ("lfc_a", pa.float64()),
        ("lfc_b", pa.float64()),
    ])
    writer = pq.ParquetWriter(out_path, schema, compression="zstd")
    n_rows = 0
    try:
        for repeat, sigs in enumerate(signatures_by_repeat):
            for perturbation, d in sigs.items():
                n = len(d["genes"])
                table = pa.Table.from_pydict({
                    "method": [method] * n,
                    "repeat": np.full(n, repeat, dtype=np.int32),
                    "seed": np.full(n, base_seed + repeat, dtype=np.int64),
                    "perturbation": [perturbation] * n,
                    "n_cells": np.full(n, int(d["n_cells"]), dtype=np.int32),
                    "feature_index": np.arange(n, dtype=np.int32),
                    "feature": d["genes"],
                    "lfc_a": d["lfc_a"],
                    "lfc_b": d["lfc_b"],
                }, schema=schema)
                writer.write_table(table)
                n_rows += n
    finally:
        writer.close()
    return n_rows


def _run_repeat_worker(task):
    """Process-pool entry point for one complete split repeat."""
    repeat, seed = task
    if _REPEAT_ADATA is None or _REPEAT_CFG is None or _REPEAT_RT is None:
        raise RuntimeError("repeat worker globals were not initialized")
    print(f"\n=== {_REPEAT_CFG['de_method']} split-half DE "
          f"(repeat={repeat}, seed={seed}) ===", flush=True)
    return repeat, split_half_signatures(
        _REPEAT_ADATA, _REPEAT_CFG, _REPEAT_RT, seed=seed
    )


def _signature_cache_path(cache_dir, method, dataset):
    return os.path.join(cache_dir, f"test1_signatures_{method}__{dataset}.pkl")


def _write_signature_cache(path, signatures_by_repeat, *, method, dataset, args,
                           counts_layer):
    """Atomically persist completed repeats, including partial runs."""
    payload = {
        "format_version": SIGNATURE_CACHE_VERSION,
        "method": method,
        "dataset": dataset,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "pert_col": args.pert_col,
        "control": args.control,
        "block_cols": args.block_cols,
        "cpm_normalization": CPM_NORMALIZATION,
        "min_cpm": MIN_CPM,
        "counts_layer": counts_layer,
        "signatures_by_repeat": dict(sorted(signatures_by_repeat.items())),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _read_signature_cache(path, *, method, dataset, args, counts_layer):
    """Load and validate a complete or partial local signature checkpoint."""
    with open(path, "rb") as handle:
        payload = pickle.load(handle)  # trusted local checkpoint from this skill
    version = payload.get("format_version")
    if version != SIGNATURE_CACHE_VERSION:
        raise ValueError(
            f"Unsupported signature cache format {version!r} in {path}; "
            f"version {SIGNATURE_CACHE_VERSION} is required after the per-cell CPM change"
        )
    expected = {
        "method": method,
        "dataset": dataset,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "pert_col": args.pert_col,
        "control": args.control,
        "block_cols": args.block_cols,
        "cpm_normalization": CPM_NORMALIZATION,
        "min_cpm": MIN_CPM,
        "counts_layer": counts_layer,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Signature cache metadata mismatch in {path}: {mismatches}")
    signatures = payload.get("signatures_by_repeat")
    if not isinstance(signatures, dict):
        raise ValueError(f"Invalid version-{version} repeat signature payload in {path}")
    cleaned = {}
    for repeat, value in signatures.items():
        repeat = int(repeat)
        if repeat < 0 or repeat >= args.n_repeats or not isinstance(value, dict):
            raise ValueError(f"Invalid repeat {repeat!r} in {path}")
        for group, signature in value.items():
            if not isinstance(signature, dict):
                raise ValueError(f"Invalid signature {group!r} in repeat {repeat} of {path}")
            cpm_elig = signature.get("cpm_elig")
            genes = signature.get("genes")
            if cpm_elig is None or genes is None or len(cpm_elig) != len(genes):
                raise ValueError(
                    f"Signature cache {path} predates the CPM-eligibility schema "
                    f"(repeat={repeat}, group={group!r}); delete it and recompute"
                )
        cleaned[repeat] = value
    return cleaned


def _available_memory_bytes():
    """Best-effort available-memory estimate, respecting a cgroup limit."""
    available = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            fields = {
                line.split(":", 1)[0]: int(line.split()[1]) * 1024
                for line in handle if ":" in line
            }
        available = fields.get("MemAvailable")
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open("/sys/fs/cgroup/memory.max", encoding="utf-8") as handle:
            raw_limit = handle.read().strip()
        with open("/sys/fs/cgroup/memory.current", encoding="utf-8") as handle:
            current = int(handle.read().strip())
        if raw_limit != "max":
            cgroup_available = max(0, int(raw_limit) - current)
            available = (cgroup_available if available is None
                         else min(available, cgroup_available))
    except (OSError, ValueError):
        pass
    return available


def _matrix_storage_bytes(matrix):
    """Approximate bytes owned by a dense or SciPy-style sparse matrix."""
    total = int(getattr(matrix, "nbytes", 0) or 0)
    if total:
        return total
    return sum(
        int(getattr(getattr(matrix, name, None), "nbytes", 0) or 0)
        for name in ("data", "indices", "indptr", "row", "col")
    )


def _adata_storage_bytes(adata):
    """Estimate matrix storage that a fork worker may dirty."""
    matrices = [adata.X, *adata.layers.values()]
    if adata.raw is not None:
        matrices.append(adata.raw.X)
    return sum(_matrix_storage_bytes(matrix) for matrix in matrices)


def _genes_finite_in_every_signature(sigs_by_group: dict) -> set[str]:
    """Genes with finite split-A and split-B effects in every supplied signature.

    Correlation matrices must use one fixed feature panel for all perturbation
    pairs.  Taking a pairwise finite intersection would silently change the
    correlated genes from cell to cell, so feature selection is restricted to
    this global complete-case set before ranking or capping.
    """
    common = None
    for sigs in sigs_by_group.values():
        for d in sigs.values():
            genes = np.asarray(d["genes"], dtype=str)
            finite = np.isfinite(d["lfc_a"]) & np.isfinite(d["lfc_b"])
            available = set(genes[finite])
            common = available if common is None else common.intersection(available)
            if not common:
                return set()
    return set() if common is None else common


def _union_de_genes(sigs, cfg, max_genes):
    """Union of genes DE (FDR<thr & |LFC|>thr) in split A or B of ANY perturbation, capped to the
    `max_genes` most dynamic (highest cross-perturbation variance of LFC_A), for a readable heatmap."""
    fdr, lfc = cfg["fdr_threshold"], cfg["lfc_threshold"]
    union = set()
    for pert, d in sigs.items():
        sig = ((d["fdr_a"] <= fdr) & (np.abs(d["lfc_a"]) >= lfc)) | ((d["fdr_b"] <= fdr) & (np.abs(d["lfc_b"]) >= lfc))
        sig &= _cpm_elig_mask(d)
        for g in d["genes"][sig]:
            union.add(g)
    union.intersection_update(_genes_finite_in_every_signature({"signatures": sigs}))
    union = sorted(union)
    # rank union genes by cross-perturbation variance of LFC_A (most informative first)
    idx = {pert: {g: i for i, g in enumerate(d["genes"])} for pert, d in sigs.items()}
    var = {}
    for g in union:
        vals = [sigs[p]["lfc_a"][idx[p][g]] for p in sigs if g in idx[p]]
        var[g] = float(np.nanvar(vals)) if vals else 0.0
    union = sorted(union, key=lambda g: -var[g])[:max_genes]
    return union


def _matrix(sigs, perts, genes, which):
    """[len(perts) × len(genes)] LFC matrix for split `which` ∈ {'lfc_a','lfc_b'} (NaN where absent)."""
    M = np.full((len(perts), len(genes)), np.nan)
    for r, p in enumerate(perts):
        d = sigs[p]
        gi = {g: i for i, g in enumerate(d["genes"])}
        for c, g in enumerate(genes):
            if g in gi:
                M[r, c] = d[which][gi[g]]
    return M


def layer1_heatmap(sigs, out_png, cfg, *, max_genes=2000, title_prefix="Test-1", unit="perturbation"):
    """Global heatmap: rows = units sorted by ρ (lowest at top), cols = union DE genes sorted by
    split-A mean LFC, two panels (split A | split B), diverging blue–white–red capped ±CAP.
    `unit`/`title_prefix` let the guide-level Test 4 reuse this (unit='guide')."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    perts = sorted(sigs, key=lambda p: (np.inf if not np.isfinite(sigs[p]["rho"]) else sigs[p]["rho"]))
    genes = _union_de_genes(sigs, cfg, max_genes)
    A = _matrix(sigs, perts, genes, "lfc_a")
    B = _matrix(sigs, perts, genes, "lfc_b")
    order = np.argsort(np.nanmean(np.where(np.isfinite(A), A, np.nan), axis=0))  # cols by mean split-A LFC
    A, B, genes = A[:, order], B[:, order], [genes[i] for i in order]
    ylab = [f"{p}  (ρ={sigs[p]['rho']:.2f})" for p in perts]

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.22 * len(perts) + 1.5)), sharey=True)
    for ax, M, title in ((axes[0], A, "split A"), (axes[1], B, "split B")):
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-CAP, vmax=CAP, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"union DE genes (n={len(genes)}, sorted by split-A mean LFC)")
        ax.set_xticks([])
    axes[0].set_yticks(range(len(perts)))
    axes[0].set_yticklabels(ylab, fontsize=6)
    fig.suptitle(f"{title_prefix} within-condition reproducibility ({cfg['de_method']}) — split A vs split B\n"
                 f"rows = {unit}s sorted by split-half ρ (low ρ at top = irreproducible)", fontsize=11)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(f"log2FC (capped ±{CAP:g}; white = 0)")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_png, perts, genes


def _cpm_elig_mask(signature: dict) -> np.ndarray:
    """Return the required per-gene CPM mask, rejecting stale signature data."""
    cpm_elig = signature.get("cpm_elig")
    if cpm_elig is None or len(cpm_elig) != len(signature["genes"]):
        raise ValueError(
            "Signature is missing a valid cpm_elig mask; recompute stale cached signatures"
        )
    return np.asarray(cpm_elig, dtype=bool)


def _union_de_genes_ranked(sigs_by_method, cfg):
    """Return ALL union-DE complete-case genes ranked by cross-method LFC variance (desc).

    G = union of genes called DE (split A or B, FDR+LFC thresholds) by any method × group,
    intersected with the complete-case set (finite lfc_a AND lfc_b for every method × group).
    Note: union membership depends on both methods' FDR calling — a gene enters G if either
    method calls it DE in any split. Only the variance ranking is symmetric across methods:
    it pools split-A and split-B LFCs from ALL methods so no single method's FDR drives
    the rank order. The pooled variance includes between-method disagreement.
    Returns the full ranked list; callers slice [:n] for any desired cap size.
    """
    fdr, lfc = cfg["fdr_threshold"], cfg["lfc_threshold"]
    union = set()
    for sigs in sigs_by_method.values():
        for d in sigs.values():
            sig = ((d["fdr_a"] <= fdr) & (np.abs(d["lfc_a"]) >= lfc)) | \
                  ((d["fdr_b"] <= fdr) & (np.abs(d["lfc_b"]) >= lfc))
            sig &= _cpm_elig_mask(d)
            union.update(d["genes"][sig])
    union.intersection_update(_genes_finite_in_every_signature(sigs_by_method))
    union = sorted(union)
    if not union:
        return []
    idx = {m: {p: {g: i for i, g in enumerate(d["genes"])} for p, d in sigs.items()}
           for m, sigs in sigs_by_method.items()}
    var = {}
    for g in union:
        vals = ([sigs_by_method[m][p]["lfc_a"][idx[m][p][g]]
                 for m in sigs_by_method for p in sigs_by_method[m] if g in idx[m][p]] +
                [sigs_by_method[m][p]["lfc_b"][idx[m][p][g]]
                 for m in sigs_by_method for p in sigs_by_method[m] if g in idx[m][p]])
        var[g] = float(np.nanvar(vals)) if vals else 0.0
    return sorted(union, key=lambda g: -var[g])


def _union_de_genes_multi(sigs_by_method, cfg, max_genes):
    """Capped union-DE gene list for heatmap column ordering (layer1/layer2 display).

    Delegates to _union_de_genes_ranked for gene selection, then sorts the top-max_genes
    by mean split-A LFC (blue→red column order) for heatmap readability.
    For correlation matrices use _union_de_genes_ranked directly and slice [:n].
    """
    ranked = _union_de_genes_ranked(sigs_by_method, cfg)
    top = ranked[:max_genes]
    if not top:
        return []
    idx = {m: {p: {g: i for i, g in enumerate(d["genes"])} for p, d in sigs.items()}
           for m, sigs in sigs_by_method.items()}
    meanA = {}
    for g in top:
        vals = [sigs_by_method[m][p]["lfc_a"][idx[m][p][g]]
                for m in sigs_by_method for p in sigs_by_method[m] if g in idx[m][p]]
        meanA[g] = float(np.nanmean(vals)) if vals else 0.0
    return sorted(top, key=lambda g: meanA[g])  # column order: blue (down) → red (up)


def _rho_on_gene_panel(d: dict, genes: list[str]) -> float:
    """Split-half Spearman rho on one fixed, explicitly supplied gene panel."""
    gi = {g: i for i, g in enumerate(d["genes"])}
    la = np.asarray([d["lfc_a"][gi[g]] if g in gi else np.nan for g in genes])
    lb = np.asarray([d["lfc_b"][gi[g]] if g in gi else np.nan for g in genes])
    ok = np.isfinite(la) & np.isfinite(lb)
    return float(stats.spearmanr(la[ok], lb[ok]).statistic) if ok.sum() >= 5 else float("nan")


def _set_shared_panel_rhos(sigs_by_method: dict, genes: list[str]) -> None:
    """Replace headline rho values with correlations on a global shared panel."""
    for sigs in sigs_by_method.values():
        for d in sigs.values():
            d["rho"] = _rho_on_gene_panel(d, genes)
            d["rho_n_genes"] = len(genes)


def _zoom_row(sc, hm, d, genes, cfg, row_label, *, rho_genes=None):
    """Draw one (method, perturbation) row: LFC_A-vs-LFC_B scatter (left) + split-A/B heatmap strips
    (right, over the shared global gene order). Returns the heatmap image handle."""
    fdr, lfc = cfg["fdr_threshold"], cfg["lfc_threshold"]
    gi = {g: i for i, g in enumerate(d["genes"])}
    scatter_genes = genes if rho_genes is None else rho_genes
    la = np.asarray([d["lfc_a"][gi[g]] if g in gi else np.nan for g in scatter_genes])
    lb = np.asarray([d["lfc_b"][gi[g]] if g in gi else np.nan for g in scatter_genes])
    fa = np.asarray([d["fdr_a"][gi[g]] if g in gi else np.nan for g in scatter_genes])
    fb = np.asarray([d["fdr_b"][gi[g]] if g in gi else np.nan for g in scatter_genes])
    cpm = _cpm_elig_mask(d)
    ce = np.asarray([cpm[gi[g]] if g in gi else False for g in scatter_genes])
    finite = np.isfinite(la) & np.isfinite(lb)
    sig = finite & ce & (((fa <= fdr) & (np.abs(la) >= lfc)) |
                         ((fb <= fdr) & (np.abs(lb) >= lfc)))
    sc.axhline(0, color="0.8", lw=0.7); sc.axvline(0, color="0.8", lw=0.7)
    sc.plot([-CAP, CAP], [-CAP, CAP], "--", color="0.6", lw=0.8)
    sc.scatter(la[finite & ~sig], lb[finite & ~sig], s=5, color="0.7", alpha=0.4, label="non-DE")
    sc.scatter(la[sig], lb[sig], s=12, color="#1a3c6e", alpha=0.7, label="DE (A∪B)")
    sc.set_xlim(-CAP - 0.2, CAP + 0.2); sc.set_ylim(-CAP - 0.2, CAP + 0.2)
    sc.set_xlabel("LFC split A"); sc.set_ylabel("LFC split B")
    sc.set_title(f"{row_label}\nρ={d['rho']:.2f} (shared n={d.get('rho_n_genes', len(scatter_genes))})",
                 fontsize=8.5)
    rowA = np.array([[d["lfc_a"][gi[g]] if g in gi else np.nan for g in genes]])
    rowB = np.array([[d["lfc_b"][gi[g]] if g in gi else np.nan for g in genes]])
    im = hm.imshow(np.vstack([rowA, rowB]), aspect="auto", cmap="RdBu_r", vmin=-CAP, vmax=CAP,
                   interpolation="nearest")
    hm.set_yticks([0, 1]); hm.set_yticklabels(["A", "B"], fontsize=8); hm.set_xticks([])
    hm.set_title(f"{row_label} — split A (top) vs split B (bottom), union DE genes", fontsize=8.5)
    return im


def layer2_zoom_compare(sigs_by_method, out_png, cfg, genes, *, methods, per_page=1,
                        title_prefix="Test-1", rho_genes=None):
    """Per-perturbation zoom COMPARING backends: each PNG holds `per_page` perturbation(s); every
    perturbation contributes ONE ROW PER METHOD (e.g. a pdex row and a pydeseq2 row) — scatter +
    split-A/B heatmap strips over a shared gene order — so the two backends' reproducibility for the
    same perturbation (same A/B split) sit directly above/below each other. Perturbations are sorted by
    the FIRST method's ρ (low ρ first). One PNG per perturbation by default."""
    if len(genes) == 0:
        return []

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sort_m = methods[0]
    common = [p for p in sigs_by_method[sort_m] if all(p in sigs_by_method[m] for m in methods)]
    common.sort(key=lambda p: (np.inf if not np.isfinite(sigs_by_method[sort_m][p]["rho"])
                               else sigs_by_method[sort_m][p]["rho"]))
    base = out_png[:-4] if out_png.endswith(".png") else out_png
    pages = [common[i:i + per_page] for i in range(0, len(common), per_page)]
    outs = []
    for pi, page in enumerate(pages, 1):
        rows = [(p, m) for p in page for m in methods]   # group by perturbation, one row per method
        fig, axes = plt.subplots(len(rows), 2, figsize=(11, 3.2 * len(rows)),
                                 gridspec_kw={"width_ratios": [1, 2.4], "hspace": 0.6, "wspace": 0.15},
                                 squeeze=False)
        im = None
        for r, (p, m) in enumerate(rows):
            im = _zoom_row(axes[r][0], axes[r][1], sigs_by_method[m][p], genes, cfg,
                           f"{p} · {m}", rho_genes=rho_genes)
        fig.colorbar(im, ax=axes[:, 1].tolist(), fraction=0.02, pad=0.02, label=f"log2FC (±{CAP:g})")
        if len(page) == 1:
            p = page[0]
            rr = " vs ".join(f"{m} ρ={sigs_by_method[m][p]['rho']:.2f}" for m in methods)
            sup = f"{title_prefix} reproducibility zoom — {p}  ({rr})   [{pi}/{len(pages)}]"
        else:
            sup = f"{title_prefix} reproducibility zoom — {', '.join(methods)}   [{pi}/{len(pages)}]"
        fig.suptitle(sup, fontsize=11)
        png = f"{base}_{pi:02d}.png"
        fig.savefig(png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        outs.append(png)
    return outs


def _all_shared_genes(sigs_by_method: dict) -> list[str]:
    """Fixed complete-case genes shared by every supplied signature."""
    return sorted(_genes_finite_in_every_signature(sigs_by_method))


def _gene_set_diagnostics(sigs_by_method: dict, cfg: dict) -> dict:
    """Report and return gene-set filtering counts for inclusion in plot titles.

    Computes, prints to terminal, and returns:
      union_before  — union-DE genes before CPM and complete-case filtering
      dropped_cpm   — genes removed by every DE call's CPM eligibility mask
      dropped_nonfinite — CPM-retained genes lacking a finite LFC somewhere
      dropped_de    — total genes removed by CPM or complete-case filtering
      retained_de   — final complete-case union-DE count
      all_before    — all genes appearing in any DE output
      dropped_all   — removed for the all-eligible panel
      retained_all  — final complete-case all-gene panel count
    Also breaks down dropped_de by which method's absence caused the removal.
    """
    fdr, lfc = cfg["fdr_threshold"], cfg["lfc_threshold"]

    # Identify method name for each top-level key (handles "pdex__repeat0" style too)
    def _method_of(key: str) -> str:
        if "pydeseq2" in key:
            return "pydeseq2"
        if "pdex" in key:
            return "pdex"
        return key

    # Raw union-DE and the union after applying each signature's CPM gate.
    union_before: set[str] = set()
    union_after_cpm: set[str] = set()
    for sigs in sigs_by_method.values():
        for d in sigs.values():
            sig = ((d["fdr_a"] <= fdr) & (np.abs(d["lfc_a"]) >= lfc)) | \
                  ((d["fdr_b"] <= fdr) & (np.abs(d["lfc_b"]) >= lfc))
            union_before.update(d["genes"][sig])
            union_after_cpm.update(d["genes"][sig & _cpm_elig_mask(d)])

    # Complete-case per method (intersection within method, then across methods)
    cc_by_method: dict[str, set[str]] = {}
    for ent_key, sigs in sigs_by_method.items():
        m = _method_of(ent_key)
        cc: set[str] | None = None
        for d in sigs.values():
            finite = set(d["genes"][np.isfinite(d["lfc_a"]) & np.isfinite(d["lfc_b"])])
            cc = finite if cc is None else cc.intersection(finite)
        cc = cc or set()
        if m in cc_by_method:
            cc_by_method[m] = cc_by_method[m].intersection(cc)
        else:
            cc_by_method[m] = cc
    cc_global = set.intersection(*cc_by_method.values()) if cc_by_method else set()

    retained_de = union_after_cpm & cc_global
    dropped_cpm = union_before - union_after_cpm
    dropped_nonfinite = union_after_cpm - cc_global
    dropped_de = union_before - retained_de

    # All genes appearing in any signature (for the all-eligible panel)
    all_before: set[str] = set()
    for sigs in sigs_by_method.values():
        for d in sigs.values():
            all_before.update(d["genes"])
    retained_all = all_before & cc_global
    dropped_all = all_before - cc_global

    pct_de = len(dropped_de) / len(union_before) * 100 if union_before else 0.0
    pct_all = len(dropped_all) / len(all_before) * 100 if all_before else 0.0

    # Per-method attribution of the nonfinite-LFC drops (only for two methods).
    method_breakdown = ""
    if len(cc_by_method) == 2:
        ms = list(cc_by_method)
        cc0, cc1 = cc_by_method[ms[0]], cc_by_method[ms[1]]
        d0_only = len(dropped_nonfinite & cc1)        # absent in ms[0], present in ms[1]
        d1_only = len(dropped_nonfinite & cc0)        # absent in ms[1], present in ms[0]
        d_both  = len(dropped_nonfinite - cc0 - cc1)  # absent in both
        method_breakdown = (
            f"  dropped breakdown: {ms[0]}-only={d0_only}, "
            f"{ms[1]}-only={d1_only}, both={d_both}"
        )
        mi_tag = f" [{ms[0]}-only={d0_only}, {ms[1]}-only={d1_only}, both={d_both}]"
    else:
        mi_tag = ""

    print(f"\n{'─' * 64}")
    print("Gene-set diagnostics:")
    print(f"  union-DE before CPM/complete-case: {len(union_before)}")
    print(f"  dropped by CPM eligibility:        {len(dropped_cpm)}")
    print(f"  dropped (≥1 nonfinite LFC):        {len(dropped_nonfinite)}{mi_tag}")
    print(f"  dropped total:                     {len(dropped_de)} ({pct_de:.1f}%)")
    print(f"  final complete-case union-DE:   {len(retained_de)}")
    if method_breakdown:
        print(method_breakdown)
    print(f"  all genes (any DE output):      {len(all_before)}")
    print(f"  dropped (≥1 nonfinite LFC):     {len(dropped_all)} ({pct_all:.1f}%)")
    print(f"  final complete-case all-genes:  {len(retained_all)}")
    print(f"{'─' * 64}")

    de_sfx = (f"from {len(union_before)} union-DE; dropped {len(dropped_de)} "
              f"({pct_de:.0f}%; CPM={len(dropped_cpm)}, nonfinite={len(dropped_nonfinite)})")
    all_sfx = (f"from {len(all_before)} total; dropped {len(dropped_all)} "
               f"({pct_all:.0f}%)")
    return {
        "union_before": len(union_before),
        "dropped_cpm": len(dropped_cpm),
        "dropped_nonfinite": len(dropped_nonfinite),
        "dropped_de":   len(dropped_de),
        "retained_de":  len(retained_de),
        "all_before":   len(all_before),
        "dropped_all":  len(dropped_all),
        "retained_all": len(retained_all),
        "de_sfx":       de_sfx,
        "all_sfx":      all_sfx,
    }


def _shared_unit_order(sigs_by_method: dict, methods: list[str]):
    """Return normalized repeat lists and one unit order shared by every panel."""
    repeats_by_method = {}
    common_units: set[str] | None = None
    for method in methods:
        method_sigs = sigs_by_method[method]
        repeats = method_sigs if isinstance(method_sigs, (list, tuple)) else [method_sigs]
        if not repeats:
            raise ValueError(f"No split-half signatures supplied for method {method!r}")
        repeats_by_method[method] = repeats
        for sigs in repeats:
            units = set(sigs)
            common_units = units if common_units is None else common_units.intersection(units)
    if not common_units:
        raise ValueError("No units are shared by every method and repeat")

    def rho_sort_key(unit):
        values = [
            sigs[unit]["rho"]
            for method in methods
            for sigs in repeats_by_method[method]
        ]
        finite = [value for value in values if np.isfinite(value)]
        return (float(np.mean(finite)) if finite else np.inf, unit)

    return repeats_by_method, sorted(common_units, key=rho_sort_key)


def _filter_signatures_to_shared_units(sigs_by_method: dict, methods: list[str]):
    """Preserve input shape while removing units absent from any method/repeat."""
    repeats_by_method, units = _shared_unit_order(sigs_by_method, methods)
    filtered = {}
    for method in methods:
        original = sigs_by_method[method]
        repeats = [
            {unit: sigs[unit] for unit in units}
            for sigs in repeats_by_method[method]
        ]
        filtered[method] = repeats if isinstance(original, (list, tuple)) else repeats[0]
    return filtered


def layer3_corr_matrix(sigs_by_method, out_png, cfg, genes, *, methods,
                       title_prefix="Test-1", unit="perturbation",
                       gene_set_label: str = "union DE genes"):
    """Mean split-A × split-B signature correlation matrix per backend.

    ``sigs_by_method[method]`` may be one signature dict (backwards
    compatibility for guide-level Test 4) or a list of independently split
    signature dicts. Spearman and Pearson matrices are computed separately for
    every repeat, then averaged cell-by-cell. Colors encode the mean Spearman
    matrix. Panel titles report finite diagonal and off-diagonal means from the
    averaged Spearman and Pearson matrices. The diagonal is within-perturbation
    reproducibility; strong off-diagonals mean signatures are not
    perturbation-specific.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    repeats_by_method, perts = _shared_unit_order(sigs_by_method, methods)
    n = len(perts)
    fig, axes = plt.subplots(1, len(methods), figsize=(6.8 * len(methods), 6.4), squeeze=False)
    im = None
    repeat_counts = []
    matrix_archive = {
        "methods": np.asarray(methods, dtype=str),
        "gene_set_label": np.asarray(gene_set_label, dtype=str),
        "targets": np.asarray(perts, dtype=str),
    }
    if not isinstance(genes, dict):
        matrix_archive["features"] = np.asarray(genes, dtype=str)
    for ax, m in zip(axes[0], methods):
        method_genes = genes[m] if isinstance(genes, dict) else genes
        matrix_archive[f"features__{m}"] = np.asarray(method_genes, dtype=str)
        repeat_sigs = repeats_by_method[m]
        repeat_counts.append(len(repeat_sigs))
        repeat_spearman = []
        repeat_pearson = []
        for sigs in repeat_sigs:
            A = _matrix(sigs, perts, method_genes, "lfc_a")
            B = _matrix(sigs, perts, method_genes, "lfc_b")
            if not np.isfinite(A).all() or not np.isfinite(B).all():
                raise ValueError(
                    f"Method {m!r} does not have a fixed finite {len(method_genes)}-gene "
                    "panel across all perturbations and repeats; select features from the "
                    "global complete-case gene set before calculating correlations"
                )
            if len(method_genes) < 5:
                C_r = np.full((n, n), np.nan)
                P_r = np.full((n, n), np.nan)
            else:
                # Only coefficients are needed. Row-wise ranking followed by
                # normalized matrix multiplication is equivalent to pairwise
                # scipy.stats correlations, but avoids calculating thousands
                # of unused p-values for large fixed panels.
                ranked_A = stats.rankdata(A, axis=1)
                ranked_B = stats.rankdata(B, axis=1)

                def cross_corr(left, right):
                    left = left - left.mean(axis=1, keepdims=True)
                    right = right - right.mean(axis=1, keepdims=True)
                    denom = np.sqrt(
                        np.sum(left * left, axis=1)[:, None]
                        * np.sum(right * right, axis=1)[None, :]
                    )
                    with np.errstate(divide="ignore", invalid="ignore"):
                        corr = (left @ right.T) / denom
                    corr[denom == 0] = np.nan
                    return np.clip(corr, -1.0, 1.0)

                C_r = cross_corr(ranked_A, ranked_B)
                P_r = cross_corr(A, B)
            repeat_spearman.append(C_r)
            repeat_pearson.append(P_r)
        with np.errstate(invalid="ignore"):
            C = np.nanmean(np.stack(repeat_spearman), axis=0)
            P = np.nanmean(np.stack(repeat_pearson), axis=0)
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        # annotate each diagonal cell with that unit's cell count (n cells before the A/B split)
        for i, p in enumerate(perts):
            counts = [sigs[p].get("n_cells") for sigs in repeat_sigs]
            counts = [x for x in counts if x is not None]
            if counts:
                nc = int(round(float(np.mean(counts))))
                ax.text(i, i, str(nc), ha="center", va="center", fontsize=4,
                        color="white", fontweight="bold")
        ax.set_xticks(range(n)); ax.set_xticklabels(perts, rotation=90, fontsize=5)
        ax.set_yticks(range(n)); ax.set_yticklabels(perts, fontsize=5)
        ax.set_xlabel(f"split B {unit}"); ax.set_ylabel(f"split A {unit}")
        diagonal_mask = np.eye(n, dtype=bool)
        spearman_diagonal = float(np.nanmean(C[diagonal_mask]))
        spearman_offdiagonal = float(np.nanmean(C[~diagonal_mask]))
        pearson_diagonal = float(np.nanmean(P[diagonal_mask]))
        pearson_offdiagonal = float(np.nanmean(P[~diagonal_mask]))
        matrix_archive[f"targets__{m}"] = np.asarray(perts, dtype=str)
        matrix_archive[f"spearman__{m}"] = C
        matrix_archive[f"pearson__{m}"] = P
        print(
            f"  {m} {gene_set_label}: mean across {len(repeat_sigs)} repeat(s); "
            f"Spearman diag={spearman_diagonal:.4f}, off={spearman_offdiagonal:.4f}; "
            f"Pearson diag={pearson_diagonal:.4f}, off={pearson_offdiagonal:.4f}"
        )
        ax.set_title(
            f"{m}\n"
            f"Spearman: diag={spearman_diagonal:.2f}, off={spearman_offdiagonal:.2f}; "
            f"Pearson: diag={pearson_diagonal:.2f}, off={pearson_offdiagonal:.2f}",
            fontsize=9,
        )
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02,
                 label=f"Spearman(split A, split B) over {gene_set_label}")
    repeats_label = (str(repeat_counts[0]) if len(set(repeat_counts)) == 1
                     else "/".join(map(str, repeat_counts)))
    fig.suptitle(f"{title_prefix} mean across {repeats_label} repeats — "
                 f"split-A × split-B signature correlation — {gene_set_label}\n"
                 f"diagonal = within-{unit} reproducibility (bright = reproducible), "
                 "off-diagonal = cross-" + unit + "; heatmap colors = Spearman", fontsize=10)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    matrix_path = os.path.splitext(out_png)[0] + ".npz"
    np.savez_compressed(matrix_path, **matrix_archive)
    print(f"  saved correlation values: {matrix_path}")
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True, help="input perturbation .h5ad path")
    ap.add_argument("--methods", default="pdex,pydeseq2",
                    help="comma-sep DE backends to compute + compare in the zoom (one row each)")
    ap.add_argument("--pert-col", default="gene")
    ap.add_argument("--control", default="non-targeting")
    ap.add_argument("--replicate-col", default="batch")
    ap.add_argument("--block-cols", default="batch")
    ap.add_argument("--counts-layer", default=None, help="raw-counts layer (auto-uses 'counts' if present)")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--lfc", type=float, default=0.1)
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--max-genes", type=int, default=2000)
    ap.add_argument("--zoom-per-page", type=int, default=1, help="perturbations per zoom PNG (each adds 1 row per method)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-repeats", type=int, default=5,
                    help="independent A/B splits used for the averaged Layer-3 correlation maps")
    ap.add_argument("--threads", type=int, default=8,
                    help="worker threads passed to each DE backend fit")
    ap.add_argument("--parallel-repeats", type=int, default=1,
                    help="repeat-level fork workers; use up to n-repeats on large multicore hosts")
    ap.add_argument("--signature-cache-dir", default=None,
                    help="optional directory for method-level repeat signature checkpoints")
    ap.add_argument("--resume-signatures", action="store_true",
                    help="load matching method checkpoints when present; compute missing ones")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    run_lock_handle = None
    try:
        import fcntl
        run_lock_path = os.path.join(a.outdir, ".test1_reproducibility.lock")
        run_lock_handle = open(run_lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(run_lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another Test-1 process is already using {a.outdir!r}; "
                "wait for it or choose a different --outdir"
            ) from error
        run_lock_handle.seek(0)
        run_lock_handle.truncate()
        run_lock_handle.write(f"pid={os.getpid()}\n")
        run_lock_handle.flush()
    except ImportError:
        print("[safety] fcntl unavailable; same-outdir process locking is disabled")
    if a.n_repeats < 1:
        ap.error("--n-repeats must be at least 1")
    if a.threads < 1:
        ap.error("--threads must be at least 1")
    if a.parallel_repeats < 1:
        ap.error("--parallel-repeats must be at least 1")
    available_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    )
    # Repeat workers each invoke a backend configured with ``--threads``.
    # Keep a quarter of the host free for the OS, plotting, and interactive
    # work; otherwise workers × threads can make the machine unreachable.
    cpu_safe_workers = max(1, int(available_cpus * 0.75) // a.threads)
    available_memory = _available_memory_bytes()
    input_bytes = os.path.getsize(a.adata)
    # A worker can dirty normalized/count arrays inherited by fork. A 1.5x
    # on-disk estimate and a 50% available-memory budget are deliberately
    # conservative; underestimation reduces concurrency rather than risking OOM.
    memory_per_worker = max(int(input_bytes * 1.5), 512 * 1024**2)
    memory_safe_workers = (
        max(1, int(available_memory * 0.50) // memory_per_worker)
        if available_memory is not None else 1
    )
    safe_repeat_workers = min(cpu_safe_workers, memory_safe_workers)
    memory_label = (
        f"{available_memory / 1024**3:.1f} GiB available"
        if available_memory is not None else "unknown available memory"
    )
    print(
        f"[resources] {available_cpus} CPUs in affinity; {memory_label}; "
        f"safe repeat-worker cap={safe_repeat_workers} "
        f"(CPU={cpu_safe_workers}, memory={memory_safe_workers})"
    )
    if a.parallel_repeats > safe_repeat_workers:
        print(
            f"[safety] capping --parallel-repeats from {a.parallel_repeats} "
            f"to {safe_repeat_workers}: CPU cap={cpu_safe_workers} "
            f"({available_cpus} CPUs, {a.threads} threads/backend); "
            f"memory cap={memory_safe_workers}"
        )
        a.parallel_repeats = safe_repeat_workers
    if available_memory is not None and input_bytes > available_memory * 0.60:
        raise MemoryError(
            f"Input file is {input_bytes / 1024**3:.1f} GiB but only "
            f"{available_memory / 1024**3:.1f} GiB is currently available. "
            "Refusing an in-memory load; free memory or use a larger machine."
        )
    rt = _load_runner()
    methods = [m for m in a.methods.split(",") if m]
    unknown_methods = sorted(set(methods) - {"pdex", "pydeseq2"})
    if not methods or unknown_methods:
        ap.error(
            "--methods must contain pdex and/or pydeseq2; "
            f"unknown={unknown_methods}"
        )
    # run pydeseq2 (needs raw .X) BEFORE pdex (maybe_normalize log-transforms .X in place)
    methods_order = [m for m in methods if m != "pdex"] + (["pdex"] if "pdex" in methods else [])
    counts_layer = a.counts_layer

    adata = ad.read_h5ad(a.adata)
    if a.pert_col not in adata.obs:
        raise ValueError(
            f"--pert-col {a.pert_col!r} is absent; available obs columns: "
            f"{adata.obs.columns.tolist()}"
        )
    missing_blocks = [
        c for c in a.block_cols.split(",") if c and c not in adata.obs
    ]
    if missing_blocks:
        raise ValueError(f"Missing --block-cols in adata.obs: {missing_blocks}")
    labels_present = set(adata.obs[a.pert_col].astype(str))
    if a.control not in labels_present:
        raise ValueError(
            f"Control label {a.control!r} is absent from {a.pert_col!r}; "
            f"observed {len(labels_present)} labels"
        )
    loaded_bytes = _adata_storage_bytes(adata)
    postload_available = _available_memory_bytes()
    if postload_available is not None:
        loaded_worker_bytes = max(
            memory_per_worker, int(max(loaded_bytes, 1) * 1.5)
        )
        postload_memory_workers = max(
            1, int(postload_available * 0.50) // loaded_worker_bytes
        )
        postload_safe_workers = min(cpu_safe_workers, postload_memory_workers)
        if a.parallel_repeats > postload_safe_workers:
            print(
                f"[safety] post-load matrix estimate caps repeat workers from "
                f"{a.parallel_repeats} to {postload_safe_workers}: "
                f"{loaded_bytes / 1024**3:.1f} GiB matrix storage, "
                f"{postload_available / 1024**3:.1f} GiB available"
            )
            a.parallel_repeats = postload_safe_workers
    if counts_layer is None and "counts" in adata.layers:
        counts_layer = "counts"   # robust raw source (survives pdex's in-place normalize)
    print(f"loaded {a.adata}: {adata.n_obs} cells × {adata.n_vars} genes  (methods={methods})")
    ds = os.path.splitext(os.path.basename(a.adata))[0]  # dataset tag appended to every output file

    def cfg_for(m):
        return {"pert_col": a.pert_col, "control_pert": a.control, "replicate_col": a.replicate_col,
                "block_cols": [c for c in a.block_cols.split(",") if c], "de_method": m,
                "allow_discrete": m == "pydeseq2", "normalize_if_raw": m == "pdex",
                "counts_layer": counts_layer, "min_cells_per_group": a.min_cells,
                "fdr_threshold": a.fdr, "lfc_threshold": a.lfc, "seed": a.seed,
                "num_threads": a.threads}

    plots_dir = os.path.join(a.outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    repeat_sigs_by_method = {}
    display_sigs_by_method = {}
    inference_has_run = False
    for m in methods_order:
        repeat_results = []
        all_tasks = [(repeat, a.seed + repeat) for repeat in range(a.n_repeats)]
        cache_path = (
            _signature_cache_path(a.signature_cache_dir, m, ds)
            if a.signature_cache_dir else None
        )
        if a.resume_signatures and cache_path and os.path.exists(cache_path):
            cached_repeats = _read_signature_cache(
                cache_path, method=m, dataset=ds, args=a, counts_layer=counts_layer
            )
            repeat_results = sorted(cached_repeats.items())
            print(
                f"\n=== {m}: loaded {len(repeat_results)}/{a.n_repeats} "
                f"repeats from {cache_path} ==="
            )
        completed = {repeat for repeat, _ in repeat_results}
        tasks = [task for task in all_tasks if task[0] not in completed]
        use_parallel_repeats = (
            a.parallel_repeats > 1
            and a.n_repeats > 1
            and not inference_has_run
        )
        if (a.parallel_repeats > 1 and a.n_repeats > 1
                and inference_has_run and tasks):
            print(
                f"[safety] {m}: using sequential repeats because a threaded backend "
                "already ran in this process; a second fork pool can deadlock. "
                "For concurrent repeats in every backend, run methods separately "
                "with --signature-cache-dir and --resume-signatures."
            )
        if not tasks:
            pass
        elif not use_parallel_repeats:
            for repeat, repeat_seed in tasks:
                print(f"\n=== {m} split-half DE (repeat={repeat}, seed={repeat_seed}) ===")
                repeat_results.append((repeat, split_half_signatures(
                    adata, cfg_for(m), rt, seed=repeat_seed
                )))
                if cache_path:
                    _write_signature_cache(
                        cache_path, dict(repeat_results), method=m, dataset=ds, args=a,
                        counts_layer=counts_layer
                    )
                    print(f"checkpointed {m} repeat {repeat}: {cache_path}")
            inference_has_run = True
        else:
            # The explicit fork context is intentional: it shares the loaded
            # AnnData copy-on-write instead of pickling it into every worker.
            # This option is therefore supported on POSIX hosts only.
            pool_failed = False
            if "fork" not in multiprocessing.get_all_start_methods():
                print("[safety] POSIX fork unavailable; falling back to sequential repeats")
                pool_failed = True
            else:
                global _REPEAT_ADATA, _REPEAT_CFG, _REPEAT_RT
                _REPEAT_ADATA, _REPEAT_CFG, _REPEAT_RT = adata, cfg_for(m), rt
                workers = min(a.parallel_repeats, len(tasks))
                ctx = multiprocessing.get_context("fork")
                try:
                    with concurrent.futures.ProcessPoolExecutor(
                            max_workers=workers, mp_context=ctx) as pool:
                        futures = {
                            pool.submit(_run_repeat_worker, task): task for task in tasks
                        }
                        for future in concurrent.futures.as_completed(futures):
                            item = future.result()
                            repeat_results.append(item)
                            if cache_path:
                                _write_signature_cache(
                                    cache_path, dict(repeat_results), method=m,
                                    dataset=ds, args=a, counts_layer=counts_layer
                                )
                                print(
                                    f"checkpointed {m} repeat {item[0]}: {cache_path}"
                                )
                except Exception as error:
                    print(
                        f"[safety] repeat pool failed ({error!r}); completed repeats "
                        "remain checkpointed and missing repeats will run sequentially"
                    )
                    pool_failed = True
                finally:
                    _REPEAT_ADATA = _REPEAT_CFG = _REPEAT_RT = None
            if pool_failed:
                completed = {repeat for repeat, _ in repeat_results}
                for repeat, repeat_seed in tasks:
                    if repeat in completed:
                        continue
                    print(
                        f"\n=== {m} split-half DE fallback "
                        f"(repeat={repeat}, seed={repeat_seed}) ==="
                    )
                    repeat_results.append((repeat, split_half_signatures(
                        adata, cfg_for(m), rt, seed=repeat_seed
                    )))
                    if cache_path:
                        _write_signature_cache(
                            cache_path, dict(repeat_results), method=m,
                            dataset=ds, args=a, counts_layer=counts_layer
                        )
            inference_has_run = True
        repeat_results.sort(key=lambda x: x[0])
        if len(repeat_results) != a.n_repeats:
            raise RuntimeError(
                f"{m}: expected {a.n_repeats} repeats, got {len(repeat_results)}"
            )
        empty_repeats = [repeat for repeat, sigs in repeat_results if not sigs]
        if empty_repeats:
            raise ValueError(
                f"{m}: empty signatures in repeats {empty_repeats}; remove the "
                "invalid cache and check control/min-cell settings"
            )
        repeats = [sigs for _, sigs in repeat_results]
        if cache_path:
            _write_signature_cache(
                cache_path, dict(repeat_results), method=m, dataset=ds, args=a,
                counts_layer=counts_layer
            )
            print(f"saved complete signature checkpoint: {cache_path}")
        repeat_sigs_by_method[m] = repeats
        display_sigs_by_method[m] = repeats[0]
        lfc_path = os.path.join(a.outdir, f"test1_lfc_vectors_{m}__{ds}.parquet")
        n_lfc_rows = _write_lfc_vectors(repeats, m, a.seed, lfc_path)
        print(f"LFC vectors: {os.path.abspath(lfc_path)} ({n_lfc_rows} rows)")

    # Remove method-only/repeat-only units before they can nominate DE genes,
    # constrain the complete-case panel, or contribute headline rho values.
    repeat_sigs_by_method = _filter_signatures_to_shared_units(
        repeat_sigs_by_method, methods,
    )
    display_sigs_by_method = {
        method: repeats[0] for method, repeats in repeat_sigs_by_method.items()
    }

    # Select the shared correlation gene set from every method and repeat.
    flattened_sigs = {
        f"{m}__repeat{repeat}": sigs
        for m, repeats in repeat_sigs_by_method.items()
        for repeat, sigs in enumerate(repeats)
    }
    ranked_de = _union_de_genes_ranked(flattened_sigs, cfg_for(methods[0]))
    genes_all = _all_shared_genes(flattened_sigs)
    _set_shared_panel_rhos(flattened_sigs, genes_all)
    diag = _gene_set_diagnostics(flattened_sigs, cfg_for(methods[0]))

    # Headline rho values, their CSVs, and zoom titles all use the same global
    # complete-case panel across every method, perturbation, split, and repeat.
    # Layer 1/2 remain explicitly labelled repeat-0 diagnostics; FDRs are not
    # averaged because their arithmetic mean is not a defined DEG call.
    for m in methods_order:
        repeats = repeat_sigs_by_method[m]
        rho_rows = [
            {"repeat": repeat, "seed": a.seed + repeat,
             "perturbation": p, "rho": d["rho"],
             "rho_n_genes": d.get("rho_n_genes"), "n_cells": d.get("n_cells")}
            for repeat, sigs in enumerate(repeats)
            for p, d in sigs.items()
        ]
        p1 = os.path.join(plots_dir, f"test1_heatmap_{m}__{ds}.png")
        layer1_heatmap(repeats[0], p1, cfg_for(m), max_genes=a.max_genes,
                       title_prefix=f"Test-1 repeat 0 of {a.n_repeats}")
        print("Layer 1 heatmap:", os.path.abspath(p1))
        pl.DataFrame(rho_rows).sort(["repeat", "rho"]).write_csv(
            os.path.join(a.outdir, f"test1_rho_{m}__{ds}.csv"))

    # Layer-2 zoom uses the capped union-DE set (heatmap column readability).
    genes_zoom = _union_de_genes_multi(flattened_sigs, cfg_for(methods[0]), a.max_genes)

    # Layer-2 zoom: repeat 0 only, labelled as such; one row per method.
    if genes_zoom:
        outs = layer2_zoom_compare(display_sigs_by_method,
                                   os.path.join(plots_dir, f"test1_zoom__{ds}.png"),
                                   cfg_for(methods[0]), genes_zoom, methods=methods,
                                   per_page=a.zoom_per_page,
                                   title_prefix=f"Test-1 repeat 0 of {a.n_repeats}",
                                   rho_genes=genes_all)
        print(f"\nLayer 2 zoom: {len(outs)} PNG(s) — one row per method "
              f"({', '.join(methods)}) per perturbation")
    else:
        print("\nLayer 2 zoom: skipped (shared CPM-eligible union-DE panel is empty)")

    # A cache-building one-method invocation must not overwrite the canonical
    # combined matrix: its DE-gene union is method-specific and therefore not
    # directly comparable with the final shared union.
    matrix_method_suffix = "" if len(methods) > 1 else f"_{methods[0]}"

    # Both gene sets require complete finite lfc_a/lfc_b in ALL methods × ALL
    # perturbations × ALL repeats, so pdex and pydeseq2 are evaluated on the
    # exact same feature panel in each plot.
    #
    # Compute the full ranked union-DE set once: G = union of genes called DE by any
    # method × perturbation × split, intersected with the complete-case set, ranked by
    # cross-method cross-split LFC variance (method-neutral — does not use FDR from either
    # method alone). Callers slice this list for sensitivity caps.
    def _emit_layer3(genes, suffix, label):
        p = os.path.join(plots_dir, f"test1_corr_matrix{suffix}{matrix_method_suffix}__{ds}.png")
        layer3_corr_matrix(repeat_sigs_by_method, p, cfg_for(methods[0]), genes,
                           methods=methods, gene_set_label=label)
        print(f"Layer 3 {suffix or 'primary'}: {os.path.abspath(p)}")

    # Layer-3a — Union-DE genes, PRIMARY: all union-DE complete-case genes, no cap.
    if len(ranked_de) < 5:
        print(f"Layer 3 primary union-DE: skipped (only {len(ranked_de)} genes in complete-case union)")
    else:
        de_label_full = (
            f"union-DE genes — called by ≥1 method in ≥1 perturbation "
            f"(n={len(ranked_de)}; {diag['de_sfx']})"
            if len(methods) > 1 else
            f"union DE genes (n={len(ranked_de)}; {diag['de_sfx']})"
        )
        _emit_layer3(ranked_de, "", de_label_full)

    # Layer-3a sensitivity: caps at 400 / 2000 / 4000 (ranked by cross-method LFC variance).
    # Skip a cap when it equals or exceeds the full set (would be identical to the primary).
    for cap in (400, 2000, 4000):
        if cap >= len(ranked_de):
            print(f"Layer 3 sensitivity cap={cap}: skipped (cap ≥ full union-DE n={len(ranked_de)})")
            continue
        capped = ranked_de[:cap]
        if len(capped) < 5:
            print(f"Layer 3 sensitivity cap={cap}: skipped (only {len(capped)} genes available)")
            continue
        _emit_layer3(capped, f"_top{cap}",
                     f"union-DE genes — top {cap} by cross-method LFC variance "
                     f"(n={len(capped)}; {diag['de_sfx']})")

    # Layer-3b — All eligible genes: unbiased overall LFC agreement, complete-case
    # across every method, perturbation, split, and repeat (no FDR/LFC filter).
    p3b = os.path.join(
        plots_dir, f"test1_corr_matrix_all_genes{matrix_method_suffix}__{ds}.png"
    )
    layer3_corr_matrix(repeat_sigs_by_method, p3b, cfg_for(methods[0]), genes_all, methods=methods,
                       gene_set_label=f"all eligible genes — complete-case, no FDR filter "
                                      f"(n={len(genes_all)}; {diag['all_sfx']})")
    print("Layer 3b all-eligible comparison:", os.path.abspath(p3b))


if __name__ == "__main__":
    main()
