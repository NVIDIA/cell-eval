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
            reproducibility (dark red = higher positive correlation), off-diagonal = cross-guide.

Both backends are computed on the SAME A/B splits for an apples-to-apples comparison.

Run: `python guide_split_reproducibility.py --adata <h5ad> --sgrna-col <col> --pert-col gene ...`
"""
from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from de_backends import write_resolved_config

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

_T4_WORKER_STATE = {}
DE_MASK_CACHE_VERSION = 2


def _fit_guide_splits(task, adata, cfg, rt, pos_C, ctrl_obs, seed):
    """Fit both halves for one guide without applying the shared CPM mask."""
    i, (ncell, lbl, gene, gu, pos_G) = task
    arm = rt.stratified_split(adata.obs.iloc[pos_G], cfg["block_cols"], seed + i)
    ai, bi = np.where(arm == "A")[0], np.where(arm == "B")[0]
    carm = rt.stratified_split(ctrl_obs, cfg["block_cols"], seed + i + 7)
    cai, cbi = np.where(carm == "A")[0], np.where(carm == "B")[0]
    mcg = cfg["min_cells_per_group"]
    if min(ai.size, bi.size, cai.size, cbi.size) < mcg:
        return task, None, "insufficient cells after split"
    try:
        de_A = rt._de_two(adata, pos_G[ai], pos_C[cai], cfg, "pert", "ctrl")
        de_B = rt._de_two(adata, pos_G[bi], pos_C[cbi], cfg, "pert", "ctrl")
    except Exception as exc:  # noqa: BLE001
        return task, None, str(exc)
    j = de_A.join(de_B, on="feature", how="inner", suffix="_b")
    la = j["log2_fold_change"].to_numpy().astype(float)
    lb = j["log2_fold_change_b"].to_numpy().astype(float)
    ok = np.isfinite(la) & np.isfinite(lb)
    rho_pairwise = (float(stats.spearmanr(la[ok], lb[ok]).statistic)
                    if ok.sum() >= 5 else float("nan"))
    return task, {
        "rho_pairwise": rho_pairwise,
        "genes": j["feature"].to_numpy().astype(str),
        "lfc_a": la,
        "lfc_b": lb,
        "fdr_a": j["fdr"].to_numpy().astype(float),
        "fdr_b": j["fdr_b"].to_numpy().astype(float),
    }, None


def _fit_guide_splits_worker(task):
    return _fit_guide_splits(task, **_T4_WORKER_STATE)


def guide_split_half_signatures(
    adata, cfg, rt, *, max_guides=None, max_control=None, seed=0,
    guide_workers=1,
):
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

    # Enumerate (gene, guide) groups in one pass. Repeated full-column masks per
    # guide dominate runtime on large screens even when DE itself is GPU-fast.
    non_pos = np.where(noncontrol)[0]
    grouped = pd.DataFrame({
        "gene": gene_of[noncontrol],
        "guide": guide_of[noncontrol],
    }).groupby(["gene", "guide"], sort=True, observed=True).indices
    by_gene = {}
    for (gene, guide), local_pos in grouped.items():
        if gene == ctrl or guide == ctrl:
            continue
        by_gene.setdefault(str(gene), []).append(
            (str(guide), non_pos[np.asarray(local_pos, dtype=int)])
        )

    # compact GENE.gN labels; keep only guides with enough cells to split
    candidates = []  # (n_cells, label, gene, guide, pos_G)
    for gene in sorted(by_gene):
        for k, (gu, pos_G) in enumerate(sorted(by_gene[gene]), 1):
            if pos_G.size >= 2 * mcg:
                candidates.append((int(pos_G.size), f"{gene}.g{k}", gene, gu, pos_G))
    # optionally cap to the most-powered guides (by cell count) for a readable heatmap/matrix
    candidates.sort(key=lambda t: -t[0])
    if max_guides:
        candidates = candidates[:max_guides]
    candidates.sort(key=lambda t: (t[2], t[1]))  # stable display order by gene then guide

    tasks = list(enumerate(candidates))
    if cfg["de_method"] == "pydeseq2" and guide_workers > 1:
        _T4_WORKER_STATE.clear()
        _T4_WORKER_STATE.update(
            adata=adata, cfg=cfg, rt=rt, pos_C=pos_C,
            ctrl_obs=ctrl_obs, seed=seed,
        )
        pool = mp.get_context("fork").Pool(processes=guide_workers)
        results = pool.imap(_fit_guide_splits_worker, tasks, chunksize=1)
    else:
        pool = None
        results = (
            _fit_guide_splits(task, adata, cfg, rt, pos_C, ctrl_obs, seed)
            for task in tasks
        )

    out = {}
    try:
        for task, fitted, error in results:
            _, (ncell, lbl, gene, gu, pos_G) = task
            if fitted is None:
                print(f"  {lbl}: DE failed ({error})")
                continue
            genes = fitted["genes"]
            elig = rt._compute_cpm_elig(adata, pos_G, pos_C, cfg)
            out[lbl] = {"rho": float("nan"),
                        "rho_pairwise": fitted["rho_pairwise"],
                        "gene": gene, "guide": gu,
                        "genes": genes, "lfc_a": fitted["lfc_a"],
                        "lfc_b": fitted["lfc_b"],
                        "fdr_a": fitted["fdr_a"], "fdr_b": fitted["fdr_b"],
                        "cpm_elig": np.array([g in elig for g in genes]),
                        "n_cells": int(ncell)}
            print(f"  {lbl:16s} provisional-overlap ρ={fitted['rho_pairwise']:.3f}  "
                  f"({pos_G.size} cells)  {gu[:36]}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return out


def _write_lfc_vectors(sigs, method, engine, out_path):
    """Stream one long-form Parquet row per guide-feature LFC pair."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("method", pa.string()),
        ("engine", pa.string()),
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
                "engine": [engine] * n,
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


def _write_de_mask_cache(sigs, method, engine, cfg, dataset, out_path):
    """Write compact, versioned per-guide split DE masks.

    Masks are aligned to one shared feature universe and bit-packed along the
    feature axis.  ``present`` distinguishes an absent backend estimate from a
    present non-DE feature.  The cached DE masks are tied to the recorded
    FDR/LFC/CPM semantics; they are not reusable with different thresholds.
    """
    labels = list(sigs)
    feature_set = set()
    for d in sigs.values():
        feature_set.update(np.asarray(d["genes"], dtype=str))
    features = np.asarray(sorted(feature_set), dtype=str)
    feature_index = {feature: i for i, feature in enumerate(features)}
    shape = (len(labels), len(features))
    present = np.zeros(shape, dtype=bool)
    cpm_elig = np.zeros(shape, dtype=bool)
    de_a = np.zeros(shape, dtype=bool)
    de_b = np.zeros(shape, dtype=bool)
    target_genes = []
    guides = []
    n_cells = np.empty(len(labels), dtype=np.int32)
    fdr_threshold = float(cfg["fdr_threshold"])
    lfc_threshold = float(cfg["lfc_threshold"])

    for row, label in enumerate(labels):
        d = sigs[label]
        genes = np.asarray(d["genes"], dtype=str)
        columns = np.fromiter(
            (feature_index[feature] for feature in genes),
            dtype=np.int64,
            count=len(genes),
        )
        la = np.asarray(d["lfc_a"], dtype=float)
        lb = np.asarray(d["lfc_b"], dtype=float)
        fa = np.asarray(d["fdr_a"], dtype=float)
        fb = np.asarray(d["fdr_b"], dtype=float)
        ce = np.asarray(d["cpm_elig"], dtype=bool)
        if not all(len(values) == len(genes) for values in (la, lb, fa, fb, ce)):
            raise ValueError(f"{label}: signature arrays have inconsistent lengths")
        present[row, columns] = True
        cpm_elig[row, columns] = ce
        de_a[row, columns] = (
            ce
            & np.isfinite(la)
            & np.isfinite(fa)
            & (fa <= fdr_threshold)
            & (np.abs(la) >= lfc_threshold)
        )
        de_b[row, columns] = (
            ce
            & np.isfinite(lb)
            & np.isfinite(fb)
            & (fb <= fdr_threshold)
            & (np.abs(lb) >= lfc_threshold)
        )
        target_genes.append(str(d["gene"]))
        guides.append(str(d["guide"]))
        n_cells[row] = int(d["n_cells"])

    pack = lambda mask: np.packbits(mask, axis=1, bitorder="little")
    np.savez_compressed(
        out_path,
        format_version=np.asarray(DE_MASK_CACHE_VERSION, dtype=np.int32),
        method=np.asarray(method, dtype=str),
        engine=np.asarray(engine, dtype=str),
        dataset=np.asarray(dataset, dtype=str),
        guide_labels=np.asarray(labels, dtype=str),
        target_genes=np.asarray(target_genes, dtype=str),
        guides=np.asarray(guides, dtype=str),
        n_cells=n_cells,
        features=features,
        n_features=np.asarray(len(features), dtype=np.int32),
        bitorder=np.asarray("little", dtype=str),
        present_packed=pack(present),
        cpm_elig_packed=pack(cpm_elig),
        de_a_packed=pack(de_a),
        de_b_packed=pack(de_b),
        fdr_threshold=np.asarray(fdr_threshold, dtype=float),
        lfc_threshold=np.asarray(lfc_threshold, dtype=float),
        cpm_normalization=np.asarray(t1.CPM_NORMALIZATION, dtype=str),
        min_cpm=np.asarray(t1.MIN_CPM, dtype=float),
        counts_layer=np.asarray(str(cfg.get("counts_layer") or ""), dtype=str),
    )
    return {
        "n_guides": len(labels),
        "n_features": len(features),
        "n_de_a": int(de_a.sum()),
        "n_de_b": int(de_b.sum()),
    }


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
    ap.add_argument("--threads", type=int, default=8,
                    help="worker threads supplied to each DE backend fit")
    ap.add_argument("--guide-workers", type=int, default=1,
                    help="parallel guide fits for PyDESeq2; use --threads 1 with this")
    ap.add_argument("--pairwise-workers", type=int, default=8,
                    help="fork workers for pair-specific DE-union correlation rows")
    ap.add_argument("--non-parametric-engine", choices=("pdex", "rsc"), default="pdex",
                    help="non-parametric engine; pdex uses Arc pdex and rsc uses RAPIDS GPU Wilcoxon")
    ap.add_argument("--max-genes", type=int, default=400, help="cap heatmap columns")
    ap.add_argument("--zoom-per-page", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=".")
    ap.add_argument(
        "--expression-state", choices=("raw_counts", "log1p_normalized"), default="",
        help="user-confirmed state of adata.X; recorded in the resolved YAML",
    )
    ap.add_argument(
        "--run-root", default="",
        help="confirmed run root for configs/ and logs/ (defaults to --outdir)",
    )
    a = ap.parse_args()
    if a.pairwise_workers < 1:
        ap.error("--pairwise-workers must be at least 1")
    available_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    )
    if a.pairwise_workers > available_cpus:
        print(
            f"[safety] capping --pairwise-workers from {a.pairwise_workers} "
            f"to {available_cpus}"
        )
        a.pairwise_workers = available_cpus
    rt = _load_runner()
    methods = [m for m in a.methods.split(",") if m]
    order = [m for m in methods if m != "pdex"] + (["pdex"] if "pdex" in methods else [])
    counts_layer = a.counts_layer

    adata = ad.read_h5ad(a.adata)
    if counts_layer is None and "counts" in adata.layers:
        counts_layer = "counts"
    elif (counts_layer is None and "pdex" in methods and a.non_parametric_engine != "rsc"
          and rt._looks_raw_integer(adata)):
        counts_layer = "_raw_counts_for_de"
        adata.layers[counts_layer] = adata.X.copy()
        print(f"preserved raw counts in temporary layer {counts_layer!r} for CPM")
    ds = os.path.splitext(os.path.basename(a.adata))[0]  # dataset tag on every output file
    print(f"loaded {a.adata}: {adata.n_obs} cells × {adata.n_vars} genes  (methods={methods})")
    a.run_root = os.path.abspath(os.path.expanduser(a.run_root or a.outdir))
    write_resolved_config(
        run_root=a.run_root,
        workflow="de_test4_guide_reproducibility",
        dataset=ds,
        resolved={
            "arguments": vars(a),
            "methods": methods,
            "effective_counts_layer": counts_layer or "X",
            "target_gene_col": a.target_gene_col or a.pert_col,
            "results_outdir": os.path.abspath(a.outdir),
        },
    )

    def cfg_for(m):
        return {"pert_col": a.pert_col, "control_pert": a.control, "replicate_col": a.replicate_col,
                "sgrna_col": a.sgrna_col, "target_gene_col": a.target_gene_col or a.pert_col,
                "block_cols": [c for c in a.block_cols.split(",") if c], "de_method": m,
                "allow_discrete": m == "pydeseq2", "normalize_if_raw": m == "pdex",
                "non_parametric_engine": a.non_parametric_engine,
                "counts_layer": counts_layer, "min_cells_per_group": a.min_cells,
                "fdr_threshold": a.fdr, "lfc_threshold": a.lfc, "seed": a.seed,
                "num_threads": a.threads}

    plots_dir = os.path.join(a.outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    sigs_by_method = {}
    for m in order:
        print(f"\n=== {m} within-guide split-half DE (seed={a.seed}) ===")
        sigs = guide_split_half_signatures(
            adata, cfg_for(m), rt, max_guides=a.max_guides,
            max_control=a.max_control, seed=a.seed,
            guide_workers=a.guide_workers,
        )
        sigs_by_method[m] = sigs
        lfc_path = os.path.join(a.outdir, f"test4_lfc_vectors_{m}__{ds}.parquet")
        engine = a.non_parametric_engine if m == "pdex" else "cpu"
        n_lfc_rows = _write_lfc_vectors(sigs, m, engine, lfc_path)
        print(f"LFC vectors: {os.path.abspath(lfc_path)} ({n_lfc_rows} rows)")
        mask_path = os.path.join(a.outdir, f"test4_de_masks_{m}__{ds}.npz")
        mask_stats = _write_de_mask_cache(
            sigs, m, engine, cfg_for(m), ds, mask_path
        )
        print(
            f"DE mask cache: {os.path.abspath(mask_path)} "
            f"({mask_stats['n_guides']} guides × {mask_stats['n_features']} features; "
            f"split-A DE={mask_stats['n_de_a']}, split-B DE={mask_stats['n_de_b']})"
        )

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
                               gene_set_label=label,
                               emit_boxplots=(suffix == ""))
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

    # Layer 3c — exact per-cell feature selection:
    # DE_A(guide i) union DE_B(guide j), with the configured CPM/FDR/LFC gates.
    method_suffix = "" if len(methods) > 1 else f"_{methods[0]}"
    pair_path = os.path.join(
        plots_dir,
        f"test4_corr_matrix_pair_specific_de_union"
        f"{method_suffix}__{ds}.png",
    )
    t1.pair_specific_de_union_corr_matrix(
        sigs_by_method, pair_path, cfg_for(methods[0]),
        methods=methods, title_prefix="Test-4", unit="guide",
        workers=a.pairwise_workers,
    )
    print("Layer 3c pair-specific DE-union:", os.path.abspath(pair_path))


if __name__ == "__main__":
    main()
