#!/usr/bin/env python
"""Test-4 within-GUIDE reproducibility visualisations — exactly what Test 1 plots, but the unit is an
individual sgRNA (guide) instead of a perturbation.

Test 4 IS Test 1 at the guide level: split EACH guide's cells into halves A/B (control cells also
split), run DE_A = guide_A vs ctrl_A and DE_B = guide_B vs ctrl_B, and ask whether the two independent
half-signatures agree (split-half Spearman LFC ρ). This renders the same three layers as
`evaluating-test1-reproducibility/reproducibility_heatmap.py` — and in fact **reuses that module's
plotting functions directly** (single source of truth), passing `unit='guide'` / `title_prefix='Test-4'`:

  Layer 1 — global heatmap: rows = guides (labeled `GENE.gN`) sorted by split-half ρ (low ρ at top),
            columns = union DE genes, two panels split A | split B.
  Layer 2 — per-guide zoom: one PNG per guide, one row per backend (pdex, pydeseq2) = LFC_A-vs-LFC_B
            scatter + split-A/B heatmap strips.
  Layer 3 — split-A × split-B guide correlation matrix per backend: diagonal = within-guide A/B
            reproducibility (bright = reproducible), off-diagonal = cross-guide.

Both backends are computed on the SAME A/B splits for an apples-to-apples comparison.

Run: `python guide_split_reproducibility.py --adata <h5ad> --sgrna-col <col> --pert-col gene ...`
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

_T1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "evaluating-test1-reproducibility", "reproducibility_heatmap.py")
_DH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "de_helpers.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_runner():
    return _load(_DH, "de_helpers_t4")


# reuse Test-1's plotting layers verbatim (unit='guide')
t1 = _load(_T1, "t1_repro")


def guide_split_half_signatures(adata, cfg, rt, *, max_guides=None, max_control=None, seed=0):
    """Test-1's split-half signature, per GUIDE: for each guide with ≥ 2×min_cells_per_group cells,
    batch-stratified A/B split (control also split), DE_A = guide_A vs ctrl_A and DE_B = guide_B vs
    ctrl_B via the shared `_de_two`, collecting per-gene LFC_A/LFC_B (+ FDRs) and the split-half ρ.
    Returns the SAME dict schema Test-1 uses: `GENE.gN` -> {rho, genes, lfc_a, lfc_b, fdr_a, fdr_b}."""
    rt.maybe_normalize(adata, cfg)
    pc, mcg, ctrl = cfg["pert_col"], cfg["min_cells_per_group"], cfg["control_pert"]
    sg = cfg["sgrna_col"]
    tgc = cfg.get("target_gene_col") or pc
    labels = adata.obs[pc].astype(str).to_numpy()
    guide_of = adata.obs[sg].astype(str).to_numpy()
    gene_of = adata.obs[tgc].astype(str).to_numpy()
    noncontrol = labels != ctrl

    pos_C = np.where(labels == ctrl)[0]
    if max_control and pos_C.size > max_control:
        pos_C = np.sort(np.random.default_rng(seed).choice(pos_C, size=max_control, replace=False))
    ctrl_obs = adata.obs.iloc[pos_C]

    # enumerate guides per gene → compact GENE.gN labels; keep only guides with enough cells to split
    candidates = []  # (n_cells, label, gene, guide, pos_G)
    for gene in sorted(set(gene_of[noncontrol])):
        if gene == ctrl:
            continue
        gus = [gu for gu in sorted(set(guide_of[(gene_of == gene) & noncontrol])) if gu != ctrl]
        for k, gu in enumerate(gus, 1):
            pos_G = np.where((guide_of == gu) & (gene_of == gene) & noncontrol)[0]
            if pos_G.size >= 2 * mcg:
                candidates.append((int(pos_G.size), f"{gene}.g{k}", gene, gu, pos_G))
    # optionally cap to the most-powered guides (by cell count) for a readable heatmap/matrix
    candidates.sort(key=lambda t: -t[0])
    if max_guides:
        candidates = candidates[:max_guides]
    candidates.sort(key=lambda t: (t[2], t[1]))  # stable display order by gene then guide

    out = {}
    for i, (ncell, lbl, gene, gu, pos_G) in enumerate(candidates):
        arm = rt.stratified_split(adata.obs.iloc[pos_G], cfg["block_cols"], seed + i)
        ai, bi = np.where(arm == "A")[0], np.where(arm == "B")[0]
        carm = rt.stratified_split(ctrl_obs, cfg["block_cols"], seed + i + 7)
        cai, cbi = np.where(carm == "A")[0], np.where(carm == "B")[0]
        if min(ai.size, bi.size, cai.size, cbi.size) < mcg:
            continue
        try:
            de_A = rt._de_two(adata, pos_G[ai], pos_C[cai], cfg, "pert", "ctrl")
            de_B = rt._de_two(adata, pos_G[bi], pos_C[cbi], cfg, "pert", "ctrl")
        except Exception as e:  # noqa: BLE001
            print(f"  {lbl}: DE failed ({e})"); continue
        j = de_A.join(de_B, on="feature", how="inner", suffix="_b")
        la = j["log2_fold_change"].to_numpy().astype(float)
        lb = j["log2_fold_change_b"].to_numpy().astype(float)
        ok = np.isfinite(la) & np.isfinite(lb)
        rho = float(stats.spearmanr(la[ok], lb[ok]).statistic) if ok.sum() >= 5 else float("nan")
        out[lbl] = {"rho": rho, "genes": j["feature"].to_numpy().astype(str),
                    "lfc_a": la, "lfc_b": lb,
                    "fdr_a": j["fdr"].to_numpy().astype(float),
                    "fdr_b": j["fdr_b"].to_numpy().astype(float),
                    "n_cells": int(ncell)}
        print(f"  {lbl:16s} ρ={rho:.3f}  ({pos_G.size} cells)  {gu[:36]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True)
    ap.add_argument("--methods", default="pdex,pydeseq2")
    ap.add_argument("--pert-col", default="gene")
    ap.add_argument("--control", default="non-targeting")
    ap.add_argument("--sgrna-col", required=True, help="guide / sgRNA column (the UNIT for Test 4)")
    ap.add_argument("--target-gene-col", default=None, help="gene each guide targets (defaults to --pert-col)")
    ap.add_argument("--replicate-col", default="batch")
    ap.add_argument("--block-cols", default="batch")
    ap.add_argument("--counts-layer", default=None)
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--lfc", type=float, default=0.1)
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--max-guides", type=int, default=None, help="cap #guides (most-powered first) for readability")
    ap.add_argument("--max-control", type=int, default=None, help="subsample control cells (speed)")
    ap.add_argument("--max-genes", type=int, default=400, help="cap heatmap columns")
    ap.add_argument("--zoom-per-page", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    rt = _load_runner()
    methods = [m for m in a.methods.split(",") if m]
    order = [m for m in methods if m != "pdex"] + (["pdex"] if "pdex" in methods else [])
    counts_layer = a.counts_layer

    adata = ad.read_h5ad(a.adata)
    if counts_layer is None and "counts" in adata.layers:
        counts_layer = "counts"
    ds = os.path.splitext(os.path.basename(a.adata))[0]  # dataset tag on every output file
    print(f"loaded {a.adata}: {adata.n_obs} cells × {adata.n_vars} genes  (methods={methods})")

    def cfg_for(m):
        return {"pert_col": a.pert_col, "control_pert": a.control, "replicate_col": a.replicate_col,
                "sgrna_col": a.sgrna_col, "target_gene_col": a.target_gene_col or a.pert_col,
                "block_cols": [c for c in a.block_cols.split(",") if c], "de_method": m,
                "allow_discrete": m == "pydeseq2", "normalize_if_raw": m == "pdex",
                "counts_layer": counts_layer, "min_cells_per_group": a.min_cells,
                "fdr_threshold": a.fdr, "lfc_threshold": a.lfc, "seed": a.seed, "num_threads": 8}

    plots_dir = os.path.join(a.outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    sigs_by_method = {}
    for m in order:
        print(f"\n=== {m} within-guide split-half DE (seed={a.seed}) ===")
        sigs_by_method[m] = guide_split_half_signatures(adata, cfg_for(m), rt, max_guides=a.max_guides,
                                                        max_control=a.max_control, seed=a.seed)
        p1 = os.path.join(plots_dir, f"test4_guide_heatmap_{m}__{ds}.png")
        t1.layer1_heatmap(sigs_by_method[m], p1, cfg_for(m), max_genes=a.max_genes,
                          title_prefix="Test-4", unit="guide")
        print("Layer 1 heatmap:", os.path.abspath(p1))
        pl.DataFrame({"guide": list(sigs_by_method[m]),
                      "rho": [sigs_by_method[m][g]["rho"] for g in sigs_by_method[m]]}).sort("rho").write_csv(
            os.path.join(a.outdir, f"test4_rho_{m}__{ds}.csv"))

    genes = t1._union_de_genes_multi(sigs_by_method, cfg_for(methods[0]), a.max_genes)
    outs = t1.layer2_zoom_compare(sigs_by_method, os.path.join(plots_dir, f"test4_zoom__{ds}.png"),
                                  cfg_for(methods[0]), genes, methods=methods, per_page=a.zoom_per_page,
                                  title_prefix="Test-4")
    print(f"\nLayer 2 zoom: {len(outs)} PNG(s) — one row per method ({', '.join(methods)}) per guide")

    p3 = os.path.join(plots_dir, f"test4_corr_matrix__{ds}.png")
    t1.layer3_corr_matrix(sigs_by_method, p3, cfg_for(methods[0]), genes, methods=methods,
                          title_prefix="Test-4", unit="guide")
    print("Layer 3 correlation matrices:", os.path.abspath(p3))


if __name__ == "__main__":
    main()
