#!/usr/bin/env python
"""Test-5 same-gene independent-sgRNA visualisations — the Test-1 heatmap/correlation views, but the
unit is an individual sgRNA (guide) instead of a perturbation, and the question is Test 5's:
**do INDEPENDENT guides targeting the SAME gene produce concordant signatures?** (a metric that rewards
biology should say yes; same-gene guides should agree far more than unrelated guide pairs.)

It reuses the SHARED runner helpers (`maybe_normalize`, `_de_two`, `run_de`) so each guide's DE
signature (guide's cells vs control cells) matches the real pipeline, then renders — for BOTH backends
on the same cells, apples-to-apples:

  Layer 1 — global heatmap: rows = guides (labeled `GENE-guide`, grouped so same-gene guides are
            adjacent), columns = union DE genes sorted by mean LFC. Same-gene guides should show
            matching rows; a guide whose row doesn't match its gene-mates is off-target/inefficacious.

  Layer 2 — per-gene zoom: for each gene with ≥2 guides, one PNG with one row per backend =
            LFC(guide_i) vs LFC(guide_j) scatter + a heatmap strip with one row per guide of that gene.
            This is "one guide's signature vs another guide targeting the same perturbation."

  Layer 3 — guide × guide correlation matrix (one panel per backend): entry (i,j) = Spearman(guide_i
            LFC, guide_j LFC) over the union DE genes, guides ordered by gene with gene-block
            separators. **Bright within-gene blocks = guides of a gene agree (on-target, reproducible);
            bright cross-gene off-diagonal = a shared program / low specificity.**

Guides are compared against the SAME (full) control population, so a uniform positive baseline across
all pairs is expected — the informative signal is the CONTRAST between within-gene blocks and the
cross-gene background.

Run: `python samegene_guide_heatmap.py --adata <h5ad> --sgrna-col <col> --pert-col gene ...`
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections import OrderedDict

import anndata as ad
import numpy as np
import polars as pl
from scipy import stats

_RT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "de_helpers.py")

CAP = 2.0  # ±log2FC colour cap


def _load_runner():
    spec = importlib.util.spec_from_file_location("rt_shared", _RT)
    rt = importlib.util.module_from_spec(spec)
    sys.modules["rt_shared"] = rt
    spec.loader.exec_module(rt)
    return rt


def guide_signatures(adata, cfg, rt, *, min_guides=2, max_genes=None, max_control=None, seed=0):
    """For each guide of a gene with ≥`min_guides` qualifying guides: DE = guide's cells vs control
    cells (full control), via the shared `_de_two`. Returns (sigs, gene_groups):
      sigs        — OrderedDict `GENE-guide` -> {gene, guide, genes, lfc, fdr, n_cells} (grouped by gene)
      gene_groups — list of (gene, [guide_label, ...]) in the same order.
    Only genes with ≥min_guides guides (each ≥min_cells_per_group cells) are kept — the same-gene block
    structure is the whole point."""
    rt.maybe_normalize(adata, cfg)
    pc, mcg, ctrl = cfg["pert_col"], cfg["min_cells_per_group"], cfg["control_pert"]
    sg = cfg["sgrna_col"]
    tgc = cfg.get("target_gene_col") or pc
    labels = adata.obs[pc].astype(str).to_numpy()
    guide_of = adata.obs[sg].astype(str).to_numpy()
    gene_of = adata.obs[tgc].astype(str).to_numpy()

    pos_C = np.where(labels == ctrl)[0]
    if max_control and pos_C.size > max_control:
        pos_C = np.random.default_rng(seed).choice(pos_C, size=max_control, replace=False)
    pos_C.sort()

    # guides per (non-control) gene that clear the cell threshold
    gene_to_guides = {}
    for g in sorted(set(gene_of)):
        if g == ctrl:
            continue
        gd = [gu for gu in sorted(set(guide_of[gene_of == g]))
              if gu != ctrl and int(np.sum((guide_of == gu) & (gene_of == g))) >= mcg]
        if len(gd) >= min_guides:
            gene_to_guides[g] = gd
    # order genes by (n_guides desc, gene) and optionally cap for readability/runtime
    genes_sorted = sorted(gene_to_guides, key=lambda g: (-len(gene_to_guides[g]), g))
    if max_genes:
        genes_sorted = genes_sorted[:max_genes]

    sigs, gene_groups = OrderedDict(), []
    for gene in genes_sorted:
        kept = []
        for k, gu in enumerate(gene_to_guides[gene], 1):
            pos_G = np.where((guide_of == gu) & (gene_of == gene))[0]
            try:
                de = rt._de_two(adata, pos_G, pos_C, cfg, "pert", "ctrl")
            except Exception as e:  # noqa: BLE001
                print(f"  {gene}.g{k}: DE failed ({e})"); continue
            lbl = f"{gene}.g{k}"  # compact, readable label; raw sgRNA id kept in 'guide'
            sigs[lbl] = {"gene": gene, "guide": gu,
                         "genes": de["feature"].to_numpy().astype(str),
                         "lfc": de["log2_fold_change"].to_numpy().astype(float),
                         "fdr": de["fdr"].to_numpy().astype(float),
                         "n_cells": int(pos_G.size)}
            kept.append(lbl)
            print(f"  {lbl:16s} {gu[:40]:40s} ({pos_G.size} cells)")
        if len(kept) >= min_guides:
            gene_groups.append((gene, kept))
        else:
            for lbl in kept:
                sigs.pop(lbl, None)
    return sigs, gene_groups


def _union_de_genes(sigs, cfg, max_genes):
    """Union of genes DE (FDR<thr & |LFC|>thr) in ANY guide, capped to the `max_genes` most dynamic
    (highest cross-guide variance of LFC), ordered by mean LFC (blue→red)."""
    fdr, lfc = cfg["fdr_threshold"], cfg["lfc_threshold"]
    union = set()
    for d in sigs.values():
        sig = (d["fdr"] <= fdr) & (np.abs(d["lfc"]) >= lfc)
        union.update(d["genes"][sig])
    union = sorted(union)
    idx = {lbl: {g: i for i, g in enumerate(d["genes"])} for lbl, d in sigs.items()}
    var, mean = {}, {}
    for g in union:
        vals = [sigs[lbl]["lfc"][idx[lbl][g]] for lbl in sigs if g in idx[lbl]]
        var[g] = float(np.nanvar(vals)) if vals else 0.0
        mean[g] = float(np.nanmean(vals)) if vals else 0.0
    top = sorted(union, key=lambda g: -var[g])[:max_genes]
    return sorted(top, key=lambda g: mean[g])


def _matrix(sigs, labels, genes):
    """[len(labels) × len(genes)] LFC matrix (NaN where a gene is absent for that guide)."""
    M = np.full((len(labels), len(genes)), np.nan)
    for r, lbl in enumerate(labels):
        gi = {g: i for i, g in enumerate(sigs[lbl]["genes"])}
        for c, g in enumerate(genes):
            if g in gi:
                M[r, c] = sigs[lbl]["lfc"][gi[g]]
    return M


def _ordered_labels(gene_groups):
    return [lbl for _, guides in gene_groups for lbl in guides]


def _gene_boundaries(gene_groups):
    """row index of the first guide of each gene (for drawing separators / gene tick positions)."""
    bounds, centers, names, r = [], [], [], 0
    for gene, guides in gene_groups:
        bounds.append(r)
        centers.append(r + (len(guides) - 1) / 2)
        names.append(gene)
        r += len(guides)
    return bounds, centers, names, r


def layer1_heatmap(sigs, gene_groups, out_png, cfg, method, *, max_genes=400):
    """Rows = guides (labeled GENE-guide, grouped by gene), cols = union DE genes sorted by mean LFC.
    Horizontal lines separate genes so same-gene guide blocks are visible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = _ordered_labels(gene_groups)
    genes = _union_de_genes(sigs, cfg, max_genes)
    M = _matrix(sigs, labels, genes)
    bounds, _, _, n = _gene_boundaries(gene_groups)

    fig, ax = plt.subplots(figsize=(11, max(4, 0.24 * len(labels) + 1.5)))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-CAP, vmax=CAP, interpolation="nearest")
    for b in bounds[1:]:
        ax.axhline(b - 0.5, color="k", lw=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xticks([])
    ax.set_xlabel(f"union DE genes (n={len(genes)}, sorted by mean LFC)")
    ax.set_title(f"Test-5 guide-level signatures ({method}) — rows grouped by gene\n"
                 "same-gene guides adjacent; matching rows within a block = reproducible on-target effect",
                 fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(f"log2FC (capped ±{CAP:g}; white = 0)")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _zoom_row(sc, hm, sigs, guides, genes, cfg, backend):
    """One backend row for a gene: LFC(guide_i) vs LFC(guide_j) scatter (first two guides) + a heatmap
    strip with one row per guide of the gene (shared global gene order). Returns the heatmap handle."""
    fdr, lfc = cfg["fdr_threshold"], cfg["lfc_threshold"]
    g1, g2 = guides[0], guides[1]
    d1, d2 = sigs[g1], sigs[g2]
    # align g1,g2 on shared features for the scatter
    common = np.intersect1d(d1["genes"], d2["genes"])
    i1 = {g: i for i, g in enumerate(d1["genes"])}
    i2 = {g: i for i, g in enumerate(d2["genes"])}
    la = np.array([d1["lfc"][i1[g]] for g in common])
    lb = np.array([d2["lfc"][i2[g]] for g in common])
    fa = np.array([d1["fdr"][i1[g]] for g in common])
    fb = np.array([d2["fdr"][i2[g]] for g in common])
    sig = ((fa <= fdr) & (np.abs(la) >= lfc)) | ((fb <= fdr) & (np.abs(lb) >= lfc))
    ok = np.isfinite(la) & np.isfinite(lb)
    rho = float(stats.spearmanr(la[ok], lb[ok]).statistic) if ok.sum() >= 5 else float("nan")
    # mean pairwise ρ across ALL guides of the gene
    pair_rhos = []
    for a in range(len(guides)):
        for b in range(a + 1, len(guides)):
            da, db = sigs[guides[a]], sigs[guides[b]]
            c = np.intersect1d(da["genes"], db["genes"])
            ia = {g: k for k, g in enumerate(da["genes"])}; ib = {g: k for k, g in enumerate(db["genes"])}
            xa = np.array([da["lfc"][ia[g]] for g in c]); xb = np.array([db["lfc"][ib[g]] for g in c])
            m = np.isfinite(xa) & np.isfinite(xb)
            if m.sum() >= 5:
                pair_rhos.append(stats.spearmanr(xa[m], xb[m]).statistic)
    mean_rho = float(np.nanmean(pair_rhos)) if pair_rhos else float("nan")

    sc.axhline(0, color="0.8", lw=0.7); sc.axvline(0, color="0.8", lw=0.7)
    sc.plot([-CAP, CAP], [-CAP, CAP], "--", color="0.6", lw=0.8)
    sc.scatter(la[~sig & ok], lb[~sig & ok], s=5, color="0.7", alpha=0.4, label="non-DE")
    sc.scatter(la[sig & ok], lb[sig & ok], s=12, color="#1a3c6e", alpha=0.7, label="DE (either)")
    sc.set_xlim(-CAP - 0.2, CAP + 0.2); sc.set_ylim(-CAP - 0.2, CAP + 0.2)
    sc.set_xlabel(f"LFC {g1}"); sc.set_ylabel(f"LFC {g2}")
    sc.set_title(f"{backend}: {g1} vs {g2}\nρ={rho:.2f}"
                 + (f" · mean pairwise ρ={mean_rho:.2f} ({len(guides)} guides)" if len(guides) > 2 else ""),
                 fontsize=8.5)

    rows = []
    for gu in guides:
        d = sigs[gu]; gi = {g: i for i, g in enumerate(d["genes"])}
        rows.append([d["lfc"][gi[g]] if g in gi else np.nan for g in genes])
    im = hm.imshow(np.array(rows), aspect="auto", cmap="RdBu_r", vmin=-CAP, vmax=CAP, interpolation="nearest")
    hm.set_yticks(range(len(guides))); hm.set_yticklabels([g.split(".", 1)[-1] for g in guides], fontsize=7)
    hm.set_xticks([])
    hm.set_title(f"{backend} — one row per guide, union DE genes", fontsize=8.5)
    return im


def layer2_zoom(sigs_by_method, gene_groups, out_png, cfg, genes, *, methods, per_page=1):
    """Per-gene zoom PNG(s): each gene (≥2 guides) contributes one row per backend (scatter of its first
    two guides + a heatmap strip with one row per guide). One gene per PNG by default."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # genes present (with ≥2 guides) in every method
    common = []
    for gene, guides in gene_groups:
        gs = {m: [g for g in guides if g in sigs_by_method[m]] for m in methods}
        if all(len(gs[m]) >= 2 for m in methods):
            common.append((gene, gs))
    base = out_png[:-4] if out_png.endswith(".png") else out_png
    pages = [common[i:i + per_page] for i in range(0, len(common), per_page)]
    outs = []
    for pi, page in enumerate(pages, 1):
        rows = [(gene, gs, m) for gene, gs in page for m in methods]
        fig, axes = plt.subplots(len(rows), 2, figsize=(11, 3.2 * len(rows)),
                                 gridspec_kw={"width_ratios": [1, 2.4], "hspace": 0.6, "wspace": 0.15},
                                 squeeze=False)
        im = None
        for r, (gene, gs, m) in enumerate(rows):
            im = _zoom_row(axes[r][0], axes[r][1], sigs_by_method[m], gs[m], genes, cfg, m)
        fig.colorbar(im, ax=axes[:, 1].tolist(), fraction=0.02, pad=0.02, label=f"log2FC (±{CAP:g})")
        sup = (f"Test-5 same-gene guide reproducibility — {page[0][0]}   [{pi}/{len(pages)}]"
               if len(page) == 1 else
               f"Test-5 same-gene guide reproducibility — {', '.join(methods)}   [{pi}/{len(pages)}]")
        fig.suptitle(sup, fontsize=11)
        png = f"{base}_{pi:02d}.png"
        fig.savefig(png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        outs.append(png)
    return outs


def layer3_corr_matrix(sigs_by_method, gene_groups, out_png, cfg, genes, *, methods):
    """Guide × guide Spearman-LFC correlation per backend (one panel each), guides ordered by gene with
    gene-block separators. Within-gene blocks bright ⇒ guides of a gene agree; cross-gene off-diagonal
    bright ⇒ shared program / low specificity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(methods), figsize=(7.0 * len(methods), 6.8), squeeze=False)
    im = None
    for ax, m in zip(axes[0], methods):
        sigs = sigs_by_method[m]
        groups = [(g, [lbl for lbl in guides if lbl in sigs]) for g, guides in gene_groups]
        groups = [(g, gl) for g, gl in groups if len(gl) >= 1]
        labels = _ordered_labels(groups)
        M = _matrix(sigs, labels, genes)
        n = len(labels)
        C = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                ok = np.isfinite(M[i]) & np.isfinite(M[j])
                if ok.sum() >= 5:
                    C[i, j] = stats.spearmanr(M[i][ok], M[j][ok]).statistic
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        bounds, centers, names, _ = _gene_boundaries(groups)
        for b in bounds[1:]:
            ax.axhline(b - 0.5, color="k", lw=0.5); ax.axvline(b - 0.5, color="k", lw=0.5)
        ax.set_xticks(centers); ax.set_xticklabels(names, rotation=90, fontsize=5)
        ax.set_yticks(centers); ax.set_yticklabels(names, fontsize=5)
        ax.set_xlabel("guide (grouped by gene)"); ax.set_ylabel("guide (grouped by gene)")
        # mean within-gene off-diagonal vs mean cross-gene (use the stored target gene, not the label)
        gene_of_lbl = {lbl: sigs[lbl]["gene"] for lbl in labels}
        within, cross = [], []
        for i in range(n):
            for j in range(n):
                if i == j or not np.isfinite(C[i, j]):
                    continue
                (within if gene_of_lbl[labels[i]] == gene_of_lbl[labels[j]] else cross).append(C[i, j])
        wm = float(np.nanmean(within)) if within else float("nan")
        cm = float(np.nanmean(cross)) if cross else float("nan")
        ax.set_title(f"{m}  within-gene ρ̄={wm:.2f} vs cross-gene ρ̄={cm:.2f}", fontsize=10)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02,
                 label="Spearman(guide i, guide j) over union DE genes")
    fig.suptitle("Test-5 guide × guide signature correlation — bright WITHIN-gene blocks = same-gene "
                 "guides agree (on-target); bright cross-gene = shared program / low specificity", fontsize=11)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_png


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
    ap.add_argument("--min-guides", type=int, default=2, help="only keep genes with ≥ this many qualifying guides")
    ap.add_argument("--max-genes", type=int, default=None, help="cap #genes-with-guides (by #guides desc) for readability")
    ap.add_argument("--max-control", type=int, default=None, help="subsample control cells to this many (speed)")
    ap.add_argument("--max-de-genes", type=int, default=400, help="cap columns in the heatmaps")
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
    sigs_by_method, groups_by_method = {}, {}
    for m in order:
        print(f"\n=== {m} guide-level DE (min_guides={a.min_guides}) ===")
        sigs, gene_groups = guide_signatures(adata, cfg_for(m), rt, min_guides=a.min_guides,
                                             max_genes=a.max_genes, max_control=a.max_control, seed=a.seed)
        sigs_by_method[m] = sigs
        groups_by_method[m] = gene_groups
        p1 = os.path.join(plots_dir, f"test5_guide_heatmap_{m}__{ds}.png")
        layer1_heatmap(sigs, gene_groups, p1, cfg_for(m), m, max_genes=a.max_de_genes)
        print("Layer 1 heatmap:", os.path.abspath(p1))

    # shared gene ordering (union of guides across methods, grouped by gene) and gene column order
    genes = _union_de_genes({**sigs_by_method[methods[0]]}, cfg_for(methods[0]), a.max_de_genes)
    gene_groups = groups_by_method[methods[0]]

    outs = layer2_zoom(sigs_by_method, gene_groups, os.path.join(plots_dir, f"test5_zoom__{ds}.png"),
                       cfg_for(methods[0]), genes, methods=methods, per_page=a.zoom_per_page)
    print(f"\nLayer 2 zoom: {len(outs)} PNG(s) — one row per method per gene")

    p3 = os.path.join(plots_dir, f"test5_corr_matrix__{ds}.png")
    layer3_corr_matrix(sigs_by_method, gene_groups, p3, cfg_for(methods[0]), genes, methods=methods)
    print("Layer 3 correlation matrices:", os.path.abspath(p3))


if __name__ == "__main__":
    main()
