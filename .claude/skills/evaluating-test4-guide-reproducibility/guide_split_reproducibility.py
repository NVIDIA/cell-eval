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
        rho_pairwise = (float(stats.spearmanr(la[ok], lb[ok]).statistic)
                        if ok.sum() >= 5 else float("nan"))
        genes = j["feature"].to_numpy().astype(str)
        elig = t1._compute_cpm_elig(adata, pos_G, pos_C, cfg)
        out[lbl] = {"rho": float("nan"), "rho_pairwise": rho_pairwise,
                    "gene": gene, "guide": gu,
                    "genes": genes, "lfc_a": la, "lfc_b": lb,
                    "fdr_a": j["fdr"].to_numpy().astype(float),
                    "fdr_b": j["fdr_b"].to_numpy().astype(float),
                    "cpm_elig": np.array([g in elig for g in genes]),
                    "n_cells": int(ncell)}
        print(f"  {lbl:16s} provisional-overlap ρ={rho_pairwise:.3f}  ({pos_G.size} cells)  {gu[:36]}")
    return out


def _write_lfc_vectors(sigs, method, out_path):
    """Stream one long-form Parquet row per guide-feature LFC pair."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("method", pa.string()),
        ("guide_label", pa.string()),
        ("target_gene", pa.string()),
        ("guide", pa.string()),
        ("n_cells", pa.int32()),
        ("feature_index", pa.int32()),
        ("feature", pa.string()),
        ("lfc_a", pa.float64()),
        ("lfc_b", pa.float64()),
    ])
    writer = pq.ParquetWriter(out_path, schema, compression="zstd")
    n_rows = 0
    try:
        for label, d in sigs.items():
            n = len(d["genes"])
            table = pa.Table.from_pydict({
                "method": [method] * n,
                "guide_label": [label] * n,
                "target_gene": [str(d["gene"])] * n,
                "guide": [str(d["guide"])] * n,
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
        sigs = guide_split_half_signatures(
            adata, cfg_for(m), rt, max_guides=a.max_guides,
            max_control=a.max_control, seed=a.seed,
        )
        sigs_by_method[m] = sigs
        lfc_path = os.path.join(a.outdir, f"test4_lfc_vectors_{m}__{ds}.parquet")
        n_lfc_rows = _write_lfc_vectors(sigs, m, lfc_path)
        print(f"LFC vectors: {os.path.abspath(lfc_path)} ({n_lfc_rows} rows)")

    # A guide must exist in every backend before it can nominate genes, constrain
    # the complete-case feature panel, or receive a headline cross-method rho.
    sigs_by_method = t1._filter_signatures_to_shared_units(sigs_by_method, methods)

    # Compute gene sets: both use the same complete-case intersection so pdex and
    # pydeseq2 are evaluated on the exact same feature panel in each plot.
    #
    # ranked_de: full union-DE set (genes called DE by ≥1 method × guide × split),
    # ranked by cross-method cross-split LFC variance — method-neutral, no cap.
    # Callers slice [:n] for sensitivity caps.
    ranked_de = t1._union_de_genes_ranked(sigs_by_method, cfg_for(methods[0]))
    genes_all = t1._all_shared_genes(sigs_by_method)
    t1._set_shared_panel_rhos(sigs_by_method, genes_all)
    # capped set for layer2 heatmap readability (sorted by mean LFC for display)
    genes_zoom = t1._union_de_genes_multi(sigs_by_method, cfg_for(methods[0]), a.max_genes)
    diag = t1._gene_set_diagnostics(sigs_by_method, cfg_for(methods[0]))

    for m in order:
        p1 = os.path.join(plots_dir, f"test4_guide_heatmap_{m}__{ds}.png")
        t1.layer1_heatmap(sigs_by_method[m], p1, cfg_for(m), max_genes=a.max_genes,
                          title_prefix="Test-4", unit="guide")
        print("Layer 1 heatmap:", os.path.abspath(p1))
        pl.DataFrame({
            "guide": list(sigs_by_method[m]),
            "rho": [sigs_by_method[m][g]["rho"] for g in sigs_by_method[m]],
            "rho_n_genes": [sigs_by_method[m][g].get("rho_n_genes") for g in sigs_by_method[m]],
        }).sort("rho").write_csv(os.path.join(a.outdir, f"test4_rho_{m}__{ds}.csv"))

    if genes_zoom:
        outs = t1.layer2_zoom_compare(
            sigs_by_method, os.path.join(plots_dir, f"test4_zoom__{ds}.png"),
            cfg_for(methods[0]), genes_zoom, methods=methods,
            per_page=a.zoom_per_page, title_prefix="Test-4", rho_genes=genes_all,
        )
        print(f"\nLayer 2 zoom: {len(outs)} PNG(s) — one row per method "
              f"({', '.join(methods)}) per guide")
    else:
        print("\nLayer 2 zoom: skipped (shared CPM-eligible union-DE panel is empty)")

    def _emit_layer3(genes, suffix, label):
        p = os.path.join(plots_dir, f"test4_corr_matrix{suffix}__{ds}.png")
        t1.layer3_corr_matrix(sigs_by_method, p, cfg_for(methods[0]), genes,
                               methods=methods, title_prefix="Test-4", unit="guide",
                               gene_set_label=label)
        print(f"Layer 3 {suffix or 'primary (union-DE)'}: {os.path.abspath(p)}")

    # Layer 3a — Union-DE PRIMARY: all union-DE complete-case genes, no cap.
    if len(ranked_de) < 5:
        print(f"Layer 3 primary union-DE: skipped (only {len(ranked_de)} genes in complete-case union)")
    else:
        de_label_full = (
            f"union-DE genes — called by ≥1 method in ≥1 guide "
            f"(n={len(ranked_de)}; {diag['de_sfx']})"
            if len(methods) > 1 else
            f"union DE genes (n={len(ranked_de)}; {diag['de_sfx']})"
        )
        _emit_layer3(ranked_de, "", de_label_full)

    # Layer 3a sensitivity: caps at 400 / 2000 / 4000 (ranked by cross-method LFC variance).
    # Skip when cap ≥ full set (would duplicate the primary).
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

    # Layer 3b — All eligible genes: unbiased overall LFC agreement, complete-case across all
    # methods and guides (no FDR/LFC filter; same panel for pdex and pydeseq2).
    _emit_layer3(genes_all, "_all_genes",
                 f"all eligible genes — complete-case, no FDR filter "
                 f"(n={len(genes_all)}; {diag['all_sfx']})")


if __name__ == "__main__":
    main()
