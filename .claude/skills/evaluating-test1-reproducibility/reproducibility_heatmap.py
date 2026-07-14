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
import importlib.util
import os
import sys

import anndata as ad
import numpy as np
import polars as pl
from scipy import stats

_RT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "de_helpers.py")

CAP = 2.0  # ±log2FC colour cap


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
    return out


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


def layer1_heatmap(sigs, out_png, cfg, *, max_genes=400, title_prefix="Test-1", unit="perturbation"):
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


def _all_shared_genes(sigs_by_method: dict, min_perts_frac: float = 0.5) -> list[str]:
    """Genes with finite LFC in both splits in ≥min_perts_frac of perturbations for every method.

    Excludes genes where pydeseq2 (or any method) failed to converge in most perturbations,
    which would otherwise make the correlation matrix NaN-dominated."""
    candidate_sets = []
    for sigs in sigs_by_method.values():
        gene_finite: dict[str, int] = {}
        for d in sigs.values():
            gi = {g: i for i, g in enumerate(d["genes"])}
            for g, idx in gi.items():
                if np.isfinite(d["lfc_a"][idx]) and np.isfinite(d["lfc_b"][idx]):
                    gene_finite[g] = gene_finite.get(g, 0) + 1
        min_count = max(1, int(len(sigs) * min_perts_frac))
        candidate_sets.append({g for g, c in gene_finite.items() if c >= min_count})
    if not candidate_sets:
        return []
    return sorted(set.intersection(*candidate_sets))


def _lowest_expressed_genes(adata: ad.AnnData, candidate_genes: list[str],
                             n: int = 400, counts_layer: str | None = None,
                             ctrl_label: str | None = None, pert_col: str | None = None) -> list[str]:
    """Return the bottom-n genes from candidate_genes by mean expression.

    Uses the raw counts layer (or adata.X) on control cells only when ctrl_label/pert_col are
    given, otherwise on all cells.  Call this BEFORE pdex normalises adata.X in place."""
    import scipy.sparse as sp
    if counts_layer and counts_layer in adata.layers:
        X = adata.layers[counts_layer]
    else:
        X = adata.X
    if ctrl_label and pert_col and pert_col in adata.obs.columns:
        mask = (adata.obs[pert_col].astype(str) == ctrl_label).to_numpy()
        X = X[mask]
    mean_expr = np.asarray(X.mean(axis=0)).ravel() if sp.issparse(X) else np.asarray(X).mean(axis=0).ravel()
    g2i = {g: i for i, g in enumerate(adata.var_names)}
    ranked = sorted([(g, mean_expr[g2i[g]]) for g in candidate_genes if g in g2i], key=lambda x: x[1])
    return [g for g, _ in ranked[:n]]


def layer3_corr_matrix(sigs_by_method, out_png, cfg, genes, *, methods,
                       title_prefix="Test-1", unit="perturbation",
                       gene_set_label: str = "union DE genes"):
    """Split-A × split-B signature correlation matrix per backend (one panel each). Entry (i,j) =
    Spearman correlation of perturbation i's split-A LFC vector vs perturbation j's split-B LFC vector,
    over `genes`. **The diagonal is within-perturbation A/B reproducibility** (should be bright);
    strong off-diagonal means signatures are not perturbation-specific. Perturbations are ordered by
    split-half ρ (low ρ first). `gene_set_label` is shown in the colourbar label and the suptitle."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(methods), figsize=(6.8 * len(methods), 6.4), squeeze=False)
    im = None
    for ax, m in zip(axes[0], methods):
        sigs = sigs_by_method[m]
        perts = sorted(sigs, key=lambda p: (np.inf if not np.isfinite(sigs[p]["rho"]) else sigs[p]["rho"]))
        A = _matrix(sigs, perts, genes, "lfc_a")
        B = _matrix(sigs, perts, genes, "lfc_b")
        n = len(perts)
        C = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                ok = np.isfinite(A[i]) & np.isfinite(B[j])
                if ok.sum() >= 5:
                    C[i, j] = stats.spearmanr(A[i][ok], B[j][ok]).statistic
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        # annotate each diagonal cell with that unit's cell count (n cells before the A/B split)
        for i, p in enumerate(perts):
            nc = sigs[p].get("n_cells")
            if nc is not None:
                ax.text(i, i, str(nc), ha="center", va="center", fontsize=4, color="white", fontweight="bold")
        ax.set_xticks(range(n)); ax.set_xticklabels(perts, rotation=90, fontsize=5)
        ax.set_yticks(range(n)); ax.set_yticklabels(perts, fontsize=5)
        ax.set_xlabel(f"split B {unit}"); ax.set_ylabel(f"split A {unit}")
        diag = np.nanmean(np.diag(C))
        ax.set_title(f"{m}  (mean diagonal ρ = {diag:.2f})", fontsize=10)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02,
                 label=f"Spearman(split A, split B) over {gene_set_label}")
    fig.suptitle(f"{title_prefix} split-A × split-B signature correlation — {gene_set_label}\n"
                 f"diagonal = within-{unit} reproducibility (bright = reproducible), "
                 "off-diagonal = cross-" + unit, fontsize=10)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
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
    ap.add_argument("--max-genes", type=int, default=400)
    ap.add_argument("--zoom-per-page", type=int, default=1, help="perturbations per zoom PNG (each adds 1 row per method)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    rt = _load_runner()
    methods = [m for m in a.methods.split(",") if m]
    # run pydeseq2 (needs raw .X) BEFORE pdex (maybe_normalize log-transforms .X in place)
    methods_order = [m for m in methods if m != "pdex"] + (["pdex"] if "pdex" in methods else [])
    counts_layer = a.counts_layer

    adata = ad.read_h5ad(a.adata)
    if counts_layer is None and "counts" in adata.layers:
        counts_layer = "counts"   # robust raw source (survives pdex's in-place normalize)
    print(f"loaded {a.adata}: {adata.n_obs} cells × {adata.n_vars} genes  (methods={methods})")
    ds = os.path.splitext(os.path.basename(a.adata))[0]  # dataset tag appended to every output file

    # Pre-compute per-gene mean expression on control cells BEFORE pdex normalises adata.X in-place.
    # Used later to select lowest-expressed genes for the third correlation matrix.
    _ctrl_mean_order: list[str] | None = None
    try:
        _ctrl_mean_order = _lowest_expressed_genes(
            adata, adata.var_names.tolist(), n=adata.n_vars,
            counts_layer=counts_layer, ctrl_label=a.control, pert_col=a.pert_col)
    except Exception as _e:
        print(f"  [warn] could not pre-compute gene expression order: {_e}")

    def cfg_for(m):
        return {"pert_col": a.pert_col, "control_pert": a.control, "replicate_col": a.replicate_col,
                "block_cols": [c for c in a.block_cols.split(",") if c], "de_method": m,
                "allow_discrete": m == "pydeseq2", "normalize_if_raw": m == "pdex",
                "counts_layer": counts_layer, "min_cells_per_group": a.min_cells,
                "fdr_threshold": a.fdr, "lfc_threshold": a.lfc, "seed": a.seed, "num_threads": 8}

    plots_dir = os.path.join(a.outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    sigs_by_method = {}
    for m in methods_order:
        print(f"\n=== {m} split-half DE (seed={a.seed}) ===")
        sigs_by_method[m] = split_half_signatures(adata, cfg_for(m), rt, seed=a.seed)
        # per-backend Layer-1 heatmap + ρ table
        p1 = os.path.join(plots_dir, f"test1_heatmap_{m}__{ds}.png")
        layer1_heatmap(sigs_by_method[m], p1, cfg_for(m), max_genes=a.max_genes)
        print("Layer 1 heatmap:", os.path.abspath(p1))
        pl.DataFrame({"perturbation": list(sigs_by_method[m]),
                      "rho": [sigs_by_method[m][p]["rho"] for p in sigs_by_method[m]]}).sort("rho").write_csv(
            os.path.join(a.outdir, f"test1_rho_{m}__{ds}.csv"))

    # Layer-2 zoom: one PNG per perturbation, one row per method (shared gene order across methods)
    genes = _union_de_genes_multi(sigs_by_method, cfg_for(methods[0]), a.max_genes)
    outs = layer2_zoom_compare(sigs_by_method, os.path.join(plots_dir, f"test1_zoom__{ds}.png"),
                               cfg_for(methods[0]), genes, methods=methods, per_page=a.zoom_per_page)
    print(f"\nLayer 2 zoom: {len(outs)} PNG(s) — one row per method ({', '.join(methods)}) per perturbation")

    # Layer-3a: union DE genes (existing behaviour)
    p3 = os.path.join(plots_dir, f"test1_corr_matrix__{ds}.png")
    layer3_corr_matrix(sigs_by_method, p3, cfg_for(methods[0]), genes, methods=methods,
                       gene_set_label=f"union DE genes (n={len(genes)})")
    print("Layer 3a correlation (union DE genes):", os.path.abspath(p3))

    # Layer-3b: all genes shared across both methods' DE output
    genes_all = _all_shared_genes(sigs_by_method)
    p3b = os.path.join(plots_dir, f"test1_corr_matrix_all_genes__{ds}.png")
    layer3_corr_matrix(sigs_by_method, p3b, cfg_for(methods[0]), genes_all, methods=methods,
                       gene_set_label=f"all genes (n={len(genes_all)})")
    print("Layer 3b correlation (all genes):", os.path.abspath(p3b))

    # Layer-3c: lowest-expressed genes (bottom max_genes by mean raw count in control cells)
    if _ctrl_mean_order is not None:
        genes_low = [g for g in _ctrl_mean_order if g in set(genes_all)][:a.max_genes]
    else:
        genes_low = genes_all[:a.max_genes]  # fallback: first alphabetically
    p3c = os.path.join(plots_dir, f"test1_corr_matrix_low_expr__{ds}.png")
    layer3_corr_matrix(sigs_by_method, p3c, cfg_for(methods[0]), genes_low, methods=methods,
                       gene_set_label=f"lowest-expressed genes (n={len(genes_low)})")
    print("Layer 3c correlation (lowest-expressed genes):", os.path.abspath(p3c))


if __name__ == "__main__":
    main()
