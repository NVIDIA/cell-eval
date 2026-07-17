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
        rho = float(stats.spearmanr(la[ok], lb[ok]).statistic) if ok.sum() >= 5 else float("nan")
        genes = j["feature"].to_numpy().astype(str)
        out[pert] = {"rho": rho, "genes": genes, "lfc_a": la, "lfc_b": lb,
                     "fdr_a": j["fdr"].to_numpy().astype(float),
                     "fdr_b": j["fdr_b"].to_numpy().astype(float),
                     "n_cells": int(pos_P.size)}
        print(f"  {pert:20s} ρ={rho:.3f}  ({pos_P.size} cells)")
    if not out:
        raise ValueError(
            f"No perturbation produced a valid split-half signature. Check control "
            f"label {ctrl!r}, --min-cells={mcg}, block columns, and backend input."
        )
    return out


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


def _write_signature_cache(path, signatures_by_repeat, *, method, dataset, args):
    """Atomically persist completed repeats, including partial runs."""
    payload = {
        "format_version": 2,
        "method": method,
        "dataset": dataset,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "pert_col": args.pert_col,
        "control": args.control,
        "block_cols": args.block_cols,
        "signatures_by_repeat": dict(sorted(signatures_by_repeat.items())),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _read_signature_cache(path, *, method, dataset, args):
    """Load and validate a complete or partial local signature checkpoint."""
    with open(path, "rb") as handle:
        payload = pickle.load(handle)  # trusted local checkpoint from this skill
    expected = {
        "method": method,
        "dataset": dataset,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "pert_col": args.pert_col,
        "control": args.control,
        "block_cols": args.block_cols,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Signature cache metadata mismatch in {path}: {mismatches}")
    version = payload.get("format_version")
    if version == 1:
        signatures = payload.get("signatures")
        if not isinstance(signatures, list) or len(signatures) != args.n_repeats:
            raise ValueError(f"Invalid version-1 repeat signature payload in {path}")
        return dict(enumerate(signatures))
    if version != 2:
        raise ValueError(f"Unsupported signature cache format {version!r} in {path}")
    signatures = payload.get("signatures_by_repeat")
    if not isinstance(signatures, dict):
        raise ValueError(f"Invalid version-2 repeat signature payload in {path}")
    cleaned = {}
    for repeat, value in signatures.items():
        repeat = int(repeat)
        if repeat < 0 or repeat >= args.n_repeats or not isinstance(value, dict):
            raise ValueError(f"Invalid repeat {repeat!r} in {path}")
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
    per_gene_lfcA = {}
    for pert, d in sigs.items():
        sig = ((d["fdr_a"] <= fdr) & (np.abs(d["lfc_a"]) >= lfc)) | ((d["fdr_b"] <= fdr) & (np.abs(d["lfc_b"]) >= lfc))
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


def _union_de_genes_multi(sigs_by_method, cfg, max_genes):
    """Union DE genes across ALL methods (a gene DE in split A or B of any perturbation in any method),
    ordered by the across-perturbation, across-method mean split-A LFC, capped to `max_genes` by
    cross-perturbation variance so both methods' heatmap strips share one column order."""
    fdr, lfc = cfg["fdr_threshold"], cfg["lfc_threshold"]
    union = set()
    for sigs in sigs_by_method.values():
        for d in sigs.values():
            sig = ((d["fdr_a"] <= fdr) & (np.abs(d["lfc_a"]) >= lfc)) | ((d["fdr_b"] <= fdr) & (np.abs(d["lfc_b"]) >= lfc))
            union.update(d["genes"][sig])
    union.intersection_update(_genes_finite_in_every_signature(sigs_by_method))
    union = sorted(union)
    idx = {m: {p: {g: i for i, g in enumerate(d["genes"])} for p, d in sigs.items()}
           for m, sigs in sigs_by_method.items()}
    var, meanA = {}, {}
    for g in union:
        vals = [sigs_by_method[m][p]["lfc_a"][idx[m][p][g]]
                for m in sigs_by_method for p in sigs_by_method[m] if g in idx[m][p]]
        var[g] = float(np.nanvar(vals)) if vals else 0.0
        meanA[g] = float(np.nanmean(vals)) if vals else 0.0
    top = sorted(union, key=lambda g: -var[g])[:max_genes]
    return sorted(top, key=lambda g: meanA[g])  # column order: blue (down) → red (up)


def _zoom_row(sc, hm, d, genes, cfg, row_label):
    """Draw one (method, perturbation) row: LFC_A-vs-LFC_B scatter (left) + split-A/B heatmap strips
    (right, over the shared global gene order). Returns the heatmap image handle."""
    fdr, lfc = cfg["fdr_threshold"], cfg["lfc_threshold"]
    sig = ((d["fdr_a"] <= fdr) & (np.abs(d["lfc_a"]) >= lfc)) | ((d["fdr_b"] <= fdr) & (np.abs(d["lfc_b"]) >= lfc))
    la, lb = d["lfc_a"], d["lfc_b"]
    sc.axhline(0, color="0.8", lw=0.7); sc.axvline(0, color="0.8", lw=0.7)
    sc.plot([-CAP, CAP], [-CAP, CAP], "--", color="0.6", lw=0.8)
    sc.scatter(la[~sig], lb[~sig], s=5, color="0.7", alpha=0.4, label="non-DE")
    sc.scatter(la[sig], lb[sig], s=12, color="#1a3c6e", alpha=0.7, label="DE (A∪B)")
    sc.set_xlim(-CAP - 0.2, CAP + 0.2); sc.set_ylim(-CAP - 0.2, CAP + 0.2)
    sc.set_xlabel("LFC split A"); sc.set_ylabel("LFC split B")
    sc.set_title(f"{row_label}\nρ={d['rho']:.2f}", fontsize=8.5)
    gi = {g: i for i, g in enumerate(d["genes"])}
    rowA = np.array([[la[gi[g]] if g in gi else np.nan for g in genes]])
    rowB = np.array([[lb[gi[g]] if g in gi else np.nan for g in genes]])
    im = hm.imshow(np.vstack([rowA, rowB]), aspect="auto", cmap="RdBu_r", vmin=-CAP, vmax=CAP,
                   interpolation="nearest")
    hm.set_yticks([0, 1]); hm.set_yticklabels(["A", "B"], fontsize=8); hm.set_xticks([])
    hm.set_title(f"{row_label} — split A (top) vs split B (bottom), union DE genes", fontsize=8.5)
    return im


def layer2_zoom_compare(sigs_by_method, out_png, cfg, genes, *, methods, per_page=1, title_prefix="Test-1"):
    """Per-perturbation zoom COMPARING backends: each PNG holds `per_page` perturbation(s); every
    perturbation contributes ONE ROW PER METHOD (e.g. a pdex row and a pydeseq2 row) — scatter +
    split-A/B heatmap strips over a shared gene order — so the two backends' reproducibility for the
    same perturbation (same A/B split) sit directly above/below each other. Perturbations are sorted by
    the FIRST method's ρ (low ρ first). One PNG per perturbation by default."""
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
            im = _zoom_row(axes[r][0], axes[r][1], sigs_by_method[m][p], genes, cfg, f"{p} · {m}")
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

    fig, axes = plt.subplots(1, len(methods), figsize=(6.8 * len(methods), 6.4), squeeze=False)
    im = None
    repeat_counts = []
    matrix_archive = {
        "methods": np.asarray(methods, dtype=str),
        "gene_set_label": np.asarray(gene_set_label, dtype=str),
    }
    if not isinstance(genes, dict):
        matrix_archive["features"] = np.asarray(genes, dtype=str)
    for ax, m in zip(axes[0], methods):
        method_genes = genes[m] if isinstance(genes, dict) else genes
        matrix_archive[f"features__{m}"] = np.asarray(method_genes, dtype=str)
        method_sigs = sigs_by_method[m]
        repeat_sigs = method_sigs if isinstance(method_sigs, (list, tuple)) else [method_sigs]
        if not repeat_sigs:
            raise ValueError(f"No split-half signatures supplied for method {m!r}")
        repeat_counts.append(len(repeat_sigs))
        common_perts = set(repeat_sigs[0])
        for sigs in repeat_sigs[1:]:
            common_perts.intersection_update(sigs)
        if not common_perts:
            raise ValueError(f"No perturbations shared by all repeats for method {m!r}")
        mean_rho = {
            p: float(np.nanmean([sigs[p]["rho"] for sigs in repeat_sigs]))
            for p in common_perts
        }
        perts = sorted(common_perts, key=lambda p: (
            np.inf if not np.isfinite(mean_rho[p]) else mean_rho[p]
        ))
        n = len(perts)
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
        rho_rows = []
        all_tasks = [(repeat, a.seed + repeat) for repeat in range(a.n_repeats)]
        cache_path = (
            _signature_cache_path(a.signature_cache_dir, m, ds)
            if a.signature_cache_dir else None
        )
        if a.resume_signatures and cache_path and os.path.exists(cache_path):
            cached_repeats = _read_signature_cache(
                cache_path, method=m, dataset=ds, args=a
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
                        cache_path, dict(repeat_results), method=m, dataset=ds, args=a
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
                                    dataset=ds, args=a
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
                            dataset=ds, args=a
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
                cache_path, dict(repeat_results), method=m, dataset=ds, args=a
            )
            print(f"saved complete signature checkpoint: {cache_path}")
        for repeat, sigs in repeat_results:
            repeat_seed = a.seed + repeat
            rho_rows.extend({"repeat": repeat, "seed": repeat_seed,
                             "perturbation": p, "rho": d["rho"],
                             "n_cells": d.get("n_cells")}
                            for p, d in sigs.items())
        repeat_sigs_by_method[m] = repeats
        display_sigs_by_method[m] = repeats[0]
        # Layer 1/2 remain an explicitly labelled single-split diagnostic. FDRs
        # are not averaged because their arithmetic mean is not a defined DEG call.
        p1 = os.path.join(plots_dir, f"test1_heatmap_{m}__{ds}.png")
        layer1_heatmap(repeats[0], p1, cfg_for(m), max_genes=a.max_genes,
                       title_prefix=f"Test-1 repeat 0 of {a.n_repeats}")
        print("Layer 1 heatmap:", os.path.abspath(p1))
        pl.DataFrame(rho_rows).sort(["repeat", "rho"]).write_csv(
            os.path.join(a.outdir, f"test1_rho_{m}__{ds}.csv"))

    # Select the shared correlation gene set from every method and repeat.
    flattened_sigs = {
        f"{m}__repeat{repeat}": sigs
        for m, repeats in repeat_sigs_by_method.items()
        for repeat, sigs in enumerate(repeats)
    }
    genes = _union_de_genes_multi(flattened_sigs, cfg_for(methods[0]), a.max_genes)

    # Layer-2 zoom: repeat 0 only, labelled as such; one row per method.
    outs = layer2_zoom_compare(display_sigs_by_method,
                               os.path.join(plots_dir, f"test1_zoom__{ds}.png"),
                               cfg_for(methods[0]), genes, methods=methods,
                               per_page=a.zoom_per_page,
                               title_prefix=f"Test-1 repeat 0 of {a.n_repeats}")
    print(f"\nLayer 2 zoom: {len(outs)} PNG(s) — one row per method ({', '.join(methods)}) per perturbation")

    # A cache-building one-method invocation must not overwrite the canonical
    # combined matrix: its DE-gene union is method-specific and therefore not
    # directly comparable with the final shared union.
    matrix_method_suffix = "" if len(methods) > 1 else f"_{methods[0]}"

    # Canonical and only Layer-3a DE comparison: nominate genes from either
    # backend, require complete finite effects in both, and evaluate both
    # methods on the exact same fixed panel. Do not emit method-specific
    # "best within method" DE matrices because they are not head-to-head
    # comparisons and can be dominated by different feature selection.
    p3 = os.path.join(
        plots_dir, f"test1_corr_matrix{matrix_method_suffix}__{ds}.png"
    )
    de_label = (
        f"shared cross-method DE union (n={len(genes)})"
        if len(methods) > 1 else f"union DE genes (n={len(genes)})"
    )
    layer3_corr_matrix(
        repeat_sigs_by_method,
        p3,
        cfg_for(methods[0]),
        genes,
        methods=methods,
        gene_set_label=de_label,
    )
    print("Layer 3a canonical DE comparison:", os.path.abspath(p3))

    # Layer-3b: the only all-gene comparison uses one complete-case panel
    # shared by every method, perturbation, split, and repeat.
    genes_all = _all_shared_genes(flattened_sigs)
    p3b = os.path.join(
        plots_dir, f"test1_corr_matrix_all_genes{matrix_method_suffix}__{ds}.png"
    )
    layer3_corr_matrix(repeat_sigs_by_method, p3b, cfg_for(methods[0]), genes_all, methods=methods,
                       gene_set_label=f"all genes shared across methods (n={len(genes_all)})")
    print("Layer 3b canonical all-gene comparison:", os.path.abspath(p3b))


if __name__ == "__main__":
    main()
