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
            separators. **Dark-red within-gene off-diagonal cells = guides of a gene agree
            (on-target, reproducible); dark-red cross-gene off-diagonal cells = a shared program /
            low specificity.**

Guides are compared against the SAME (full) control population, so a uniform positive baseline across
all pairs is expected — the informative signal is the CONTRAST between within-gene blocks and the
cross-gene background.

Run: `python samegene_guide_heatmap.py --adata <h5ad> --sgrna-col <col> --pert-col gene ...`
"""
from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import os
import sys
from collections import OrderedDict

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from de_backends import de_method_label, write_resolved_config

_RT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "de_helpers.py")

CAP = 2.0  # ±log2FC colour cap

_T5_WORKER_STATE = {}


def _display_method(method, cfg, *, verbose=False):
    return de_method_label(
        method,
        non_parametric_engine=cfg.get("non_parametric_engine", "pdex"),
        verbose=verbose,
    )


def _load_runner():
    spec = importlib.util.spec_from_file_location("rt_shared", _RT)
    rt = importlib.util.module_from_spec(spec)
    sys.modules["rt_shared"] = rt
    spec.loader.exec_module(rt)
    return rt


def _compute_cpm_elig_t5(adata, pos_P, pos_C, cfg, *, min_cpm=5.0):
    """Return a set of gene names CPM-eligible (shared across DE backends, method-neutral)."""
    return cfg["_runtime"]._compute_cpm_elig(
        adata, pos_P, pos_C, cfg, min_cpm=min_cpm,
    )


def _fit_guide_signature(task, adata, cfg, rt, pos_C):
    """Fit one guide against control; CPM eligibility is added in the parent."""
    gene, k, gu, pos_G = task
    try:
        de = rt._de_two(adata, pos_G, pos_C, cfg, "pert", "ctrl")
    except Exception as exc:  # noqa: BLE001
        return task, None, str(exc)
    return task, {
        "genes": de["feature"].to_numpy().astype(str),
        "lfc": de["log2_fold_change"].to_numpy().astype(float),
        "fdr": de["fdr"].to_numpy().astype(float),
    }, None


def _fit_guide_signature_worker(task):
    return _fit_guide_signature(task, **_T5_WORKER_STATE)


def guide_signatures(
    adata, cfg, rt, *, min_guides=2, max_genes=None, max_control=None,
    seed=0, guide_workers=1,
):
    """For each guide of a gene with ≥`min_guides` qualifying guides: DE = guide's cells vs control
    cells (full control), via the shared `_de_two`. Returns (sigs, gene_groups):
      sigs        — OrderedDict `GENE-guide` -> {gene, guide, genes, lfc, fdr, n_cells} (grouped by gene)
      gene_groups — list of (gene, [guide_label, ...]) in the same order.
    Only genes with ≥min_guides guides (each ≥min_cells_per_group cells) are kept — the same-gene block
    structure is the whole point."""
    rt.maybe_normalize(adata, cfg)
    cfg["_runtime"] = rt
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

    # Enumerate (gene, guide) groups in one pass. Repeated full-column masks per
    # guide dominate runtime on large screens even when DE itself is GPU-fast.
    grouped = pd.DataFrame({
        "gene": gene_of,
        "guide": guide_of,
    }).groupby(["gene", "guide"], sort=True, observed=True).indices
    positions = {
        (str(gene), str(guide)): np.asarray(pos, dtype=int)
        for (gene, guide), pos in grouped.items()
        if gene != ctrl and guide != ctrl
    }

    # guides per (non-control) gene that clear the cell threshold
    gene_to_guides = {}
    for g in sorted({gene for gene, _ in positions}):
        gd = sorted(
            guide for gene, guide in positions
            if gene == g and positions[(gene, guide)].size >= mcg
        )
        if len(gd) >= min_guides:
            gene_to_guides[g] = gd
    # order genes by (n_guides desc, gene) and optionally cap for readability/runtime
    genes_sorted = sorted(gene_to_guides, key=lambda g: (-len(gene_to_guides[g]), g))
    if max_genes:
        genes_sorted = genes_sorted[:max_genes]

    tasks = [
        (gene, k, gu, positions[(gene, gu)])
        for gene in genes_sorted
        for k, gu in enumerate(gene_to_guides[gene], 1)
    ]
    if cfg["de_method"] == "pydeseq2" and guide_workers > 1:
        _T5_WORKER_STATE.clear()
        _T5_WORKER_STATE.update(
            adata=adata, cfg=cfg, rt=rt, pos_C=pos_C,
        )
        pool = mp.get_context("fork").Pool(processes=guide_workers)
        results = pool.imap(_fit_guide_signature_worker, tasks, chunksize=1)
    else:
        pool = None
        results = (
            _fit_guide_signature(task, adata, cfg, rt, pos_C)
            for task in tasks
        )

    sigs = OrderedDict()
    kept_by_gene = {gene: [] for gene in genes_sorted}
    try:
        for task, fitted, error in results:
            gene, k, gu, pos_G = task
            lbl = f"{gene}.g{k}"
            if fitted is None:
                print(f"  {lbl}: DE failed ({error})", flush=True)
                continue
            genes_arr = fitted["genes"]
            elig = _compute_cpm_elig_t5(adata, pos_G, pos_C, cfg)
            sigs[lbl] = {"gene": gene, "guide": gu,
                         "genes": genes_arr,
                         "lfc": fitted["lfc"],
                         "fdr": fitted["fdr"],
                         "cpm_elig": np.array([g in elig for g in genes_arr]),
                         "n_cells": int(pos_G.size)}
            kept_by_gene[gene].append(lbl)
            print(f"  {lbl:16s} {gu[:40]:40s} ({pos_G.size} cells)", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    gene_groups = []
    for gene in genes_sorted:
        kept = kept_by_gene[gene]
        if len(kept) >= min_guides:
            gene_groups.append((gene, kept))
        else:
            for lbl in kept:
                sigs.pop(lbl, None)
    return sigs, gene_groups


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
        ("lfc", pa.float64()),
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
                "lfc": d["lfc"],
            }, schema=schema)
            writer.write_table(table)
            n_rows += n
    finally:
        writer.close()
    return n_rows


def _cpm_elig_mask_t5(signature: dict) -> np.ndarray:
    """Return the required CPM mask for a guide signature."""
    cpm_elig = signature.get("cpm_elig")
    if cpm_elig is None or len(cpm_elig) != len(signature["genes"]):
        raise ValueError("Guide signature is missing a valid cpm_elig mask; recompute it")
    return np.asarray(cpm_elig, dtype=bool)


def _union_de_genes(sigs, cfg, max_genes):
    """Union of genes DE (FDR<thr & |LFC|>thr) in ANY guide, capped to the `max_genes` most dynamic
    (highest cross-guide variance of LFC), ordered by mean LFC (blue→red). Single-method version
    used for layer1 per-method heatmap column ordering."""
    fdr, lfc = cfg["fdr_threshold"], cfg["lfc_threshold"]
    union = set()
    for d in sigs.values():
        sig = (d["fdr"] <= fdr) & (np.abs(d["lfc"]) >= lfc)
        sig &= _cpm_elig_mask_t5(d)
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


def _genes_finite_all_t5(sigs_by_method: dict) -> set[str]:
    """Genes with finite LFC in EVERY method × guide combination (complete-case for layer3)."""
    common: set[str] | None = None
    for sigs in sigs_by_method.values():
        for d in sigs.values():
            finite = set(d["genes"][np.isfinite(d["lfc"])])
            common = finite if common is None else common.intersection(finite)
    return common or set()


def _union_de_genes_ranked_t5(sigs_by_method: dict, cfg: dict) -> list[str]:
    """All union-DE complete-case genes ranked by cross-method LFC variance (desc).

    G = union of genes called DE (FDR+LFC) in any method × guide, intersected with the
    complete-case set (finite LFC in EVERY method × guide). Ranking is symmetric: LFC
    pooled across all methods × guides. Union membership depends on both methods' FDR.
    Returns full ranked list; callers slice [:n] for any cap size.
    """
    fdr, lfc = cfg["fdr_threshold"], cfg["lfc_threshold"]
    union = set()
    for sigs in sigs_by_method.values():
        for d in sigs.values():
            sig = (d["fdr"] <= fdr) & (np.abs(d["lfc"]) >= lfc)
            sig &= _cpm_elig_mask_t5(d)
            union.update(d["genes"][sig])
    union.intersection_update(_genes_finite_all_t5(sigs_by_method))
    union = sorted(union)
    if not union:
        return []
    idx_by = {m: {lbl: {g: i for i, g in enumerate(d["genes"])} for lbl, d in sigs.items()}
              for m, sigs in sigs_by_method.items()}
    var = {}
    for g in union:
        vals = [sigs_by_method[m][lbl]["lfc"][idx_by[m][lbl][g]]
                for m in sigs_by_method for lbl in sigs_by_method[m]
                if g in idx_by[m][lbl]]
        var[g] = float(np.nanvar(vals)) if vals else 0.0
    return sorted(union, key=lambda g: -var[g])


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


def _common_gene_groups_t5(sigs_by_method: dict, methods: list[str], *, min_guides: int = 2):
    """One deterministic guide set/order shared by every backend panel."""
    min_guides = max(2, int(min_guides))
    common_labels = set.intersection(*(set(sigs_by_method[m]) for m in methods))
    if not common_labels:
        raise ValueError("No guides are shared by every requested method")
    by_gene: dict[str, list[str]] = {}
    for label in common_labels:
        genes = {str(sigs_by_method[m][label]["gene"]) for m in methods}
        if len(genes) != 1:
            raise ValueError(f"Guide {label!r} has inconsistent target genes across methods: {genes}")
        by_gene.setdefault(genes.pop(), []).append(label)
    groups = [
        (gene, sorted(by_gene[gene]))
        for gene in sorted(by_gene)
        if len(by_gene[gene]) >= min_guides
    ]
    if not groups:
        raise ValueError(
            f"No target genes have at least {min_guides} guides shared by every method"
        )
    return groups


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
    ax.set_title(
                 f"Test-5 guide-level signatures ({_display_method(method, cfg)}) "
                 "— rows grouped by gene\n"
                 "same-gene guides adjacent; matching rows within a block = reproducible on-target effect",
                 fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(f"log2FC (capped ±{CAP:g}; white = 0)")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _zoom_row(sc, hm, sigs, guides, genes, cfg, backend):
    """One backend row for a gene: LFC(guide_i) vs LFC(guide_j) scatter (first two guides) + a heatmap
    strip with one row per guide of the gene (shared global gene order). Returns the heatmap handle.
    Rho is computed over the shared `genes` panel (finite in both guides) so pdex and pydeseq2 values
    are feature-comparable rather than each using their own pairwise-finite intersection."""
    fdr, lfc = cfg["fdr_threshold"], cfg["lfc_threshold"]
    g1, g2 = guides[0], guides[1]
    d1, d2 = sigs[g1], sigs[g2]
    i1 = {g: i for i, g in enumerate(d1["genes"])}
    i2 = {g: i for i, g in enumerate(d2["genes"])}
    # Restrict to the shared gene panel (LFC = NaN for genes absent from a guide's DE output)
    la = np.array([d1["lfc"][i1[g]] if g in i1 else np.nan for g in genes])
    lb = np.array([d2["lfc"][i2[g]] if g in i2 else np.nan for g in genes])
    fa = np.array([d1["fdr"][i1[g]] if g in i1 else np.nan for g in genes])
    fb = np.array([d2["fdr"][i2[g]] if g in i2 else np.nan for g in genes])
    cpm1 = _cpm_elig_mask_t5(d1)
    cpm2 = _cpm_elig_mask_t5(d2)
    ce1 = np.array([cpm1[i1[g]] if g in i1 else False for g in genes])
    ce2 = np.array([cpm2[i2[g]] if g in i2 else False for g in genes])
    sig = (ce1 & np.isfinite(fa) & (fa <= fdr) & (np.abs(la) >= lfc)) | \
          (ce2 & np.isfinite(fb) & (fb <= fdr) & (np.abs(lb) >= lfc))
    ok = np.isfinite(la) & np.isfinite(lb)
    rho = float(stats.spearmanr(la[ok], lb[ok]).statistic) if ok.sum() >= 5 else float("nan")
    # mean pairwise ρ across ALL guides of the gene, over the shared panel
    pair_rhos = []
    for a in range(len(guides)):
        for b in range(a + 1, len(guides)):
            da, db = sigs[guides[a]], sigs[guides[b]]
            ia = {g: k for k, g in enumerate(da["genes"])}; ib = {g: k for k, g in enumerate(db["genes"])}
            xa = np.array([da["lfc"][ia[g]] if g in ia else np.nan for g in genes])
            xb = np.array([db["lfc"][ib[g]] if g in ib else np.nan for g in genes])
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
    if len(genes) == 0:
        return []

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
            im = _zoom_row(
                axes[r][0], axes[r][1], sigs_by_method[m], gs[m], genes,
                cfg, _display_method(m, cfg),
            )
        fig.colorbar(im, ax=axes[:, 1].tolist(), fraction=0.02, pad=0.02, label=f"log2FC (±{CAP:g})")
        sup = (f"Test-5 same-gene guide reproducibility — {page[0][0]}   [{pi}/{len(pages)}]"
               if len(page) == 1 else
               f"Test-5 same-gene guide reproducibility — "
               f"{', '.join(_display_method(m, cfg) for m in methods)}   "
               f"[{pi}/{len(pages)}]")
        fig.suptitle(sup, fontsize=11)
        png = f"{base}_{pi:02d}.png"
        fig.savefig(png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        outs.append(png)
    return outs


def layer3_corr_matrix(sigs_by_method, gene_groups, out_png, cfg, genes, *, methods,
                       gene_set_label: str = "", min_guides: int = 2):
    """Guide × guide Spearman-LFC correlation per backend (one panel each), guides ordered by gene with
    gene-block separators. Dark-red within-gene off-diagonal cells imply guide agreement; dark-red
    cross-gene off-diagonal cells imply a shared program / low specificity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Ignore backend-specific group omissions and derive one paired axis directly
    # from signatures present in every method.
    gene_groups = _common_gene_groups_t5(
        sigs_by_method, methods, min_guides=min_guides,
    )
    labels = _ordered_labels(gene_groups)
    fig, axes = plt.subplots(1, len(methods), figsize=(7.0 * len(methods), 6.8), squeeze=False)
    im = None
    for ax, m in zip(axes[0], methods):
        sigs = sigs_by_method[m]
        M = _matrix(sigs, labels, genes)
        n = len(labels)
        if len(genes) < 5:
            C = np.full((n, n), np.nan)
        elif not np.isfinite(M).all():
            raise ValueError(
                f"Method {m!r} does not have a fixed finite {len(genes)}-gene "
                "panel across all guides"
            )
        else:
            ranked = stats.rankdata(M, axis=1)
            ranked -= ranked.mean(axis=1, keepdims=True)
            norms = np.sqrt(np.sum(ranked * ranked, axis=1))
            denom = norms[:, None] * norms[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                C = (ranked @ ranked.T) / denom
            C[denom == 0] = np.nan
            C = np.clip(C, -1.0, 1.0)
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        bounds, centers, names, _ = _gene_boundaries(gene_groups)
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
        diagonal_mask = np.eye(n, dtype=bool)
        dm = float(np.nanmean(C[diagonal_mask]))
        om = float(np.nanmean(C[~diagonal_mask]))
        ax.set_title(
            f"{_display_method(m, cfg)}: diagonal ρ̄={dm:.2f}; "
            f"off-diagonal ρ̄={om:.2f}\n"
            f"within-gene off-diagonal ρ̄={wm:.2f}; cross-gene off-diagonal ρ̄={cm:.2f}",
            fontsize=9,
        )
    cbar_label = (f"Spearman(guide i, guide j) — {gene_set_label}"
                  if gene_set_label else "Spearman(guide i, guide j) over shared gene panel")
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02, label=cbar_label)
    sup = (
        "Test-5 guide × guide signature correlation — dark-red WITHIN-gene "
        "off-diagonals = higher same-gene agreement"
    )
    if gene_set_label:
        sup += f"\n{gene_set_label}"
    fig.suptitle(sup, fontsize=11)
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
    ap.add_argument("--threads", type=int, default=8,
                    help="worker threads supplied to each DE backend fit")
    ap.add_argument("--guide-workers", type=int, default=1,
                    help="parallel guide fits for PyDESeq2; divide available CPUs across workers")
    ap.add_argument("--non-parametric-engine", choices=("pdex", "rsc"), default="pdex",
                    help="non-parametric engine; pdex uses Arc pdex and rsc uses RAPIDS GPU Wilcoxon")
    ap.add_argument("--max-de-genes", type=int, default=400, help="cap columns in the heatmaps")
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
        workflow="de_test5_samegene_sgrna",
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
        print(f"\n=== {m} guide-level DE (min_guides={a.min_guides}) ===")
        sigs, gene_groups = guide_signatures(adata, cfg_for(m), rt, min_guides=a.min_guides,
                                             max_genes=a.max_genes, max_control=a.max_control,
                                             seed=a.seed, guide_workers=a.guide_workers)
        sigs_by_method[m] = sigs
        lfc_path = os.path.join(a.outdir, f"test5_lfc_vectors_{m}__{ds}.parquet")
        engine = a.non_parametric_engine if m == "pdex" else "cpu"
        n_lfc_rows = _write_lfc_vectors(sigs, m, engine, lfc_path)
        print(f"LFC vectors: {os.path.abspath(lfc_path)} ({n_lfc_rows} rows)")
        p1 = os.path.join(plots_dir, f"test5_guide_heatmap_{m}__{ds}.png")
        layer1_heatmap(sigs, gene_groups, p1, cfg_for(m), m, max_genes=a.max_de_genes)
        print("Layer 1 heatmap:", os.path.abspath(p1))

    # Gene sets for cross-method comparison.
    # ranked_de: union of genes called DE by any method × guide, intersected with the
    # complete-case set (finite LFC in EVERY method × guide), ranked by cross-method LFC
    # variance. Both methods use IDENTICAL genes in every layer3 plot.
    # genes_zoom: capped shared complete-case union (for layer2 readability).
    gene_groups = _common_gene_groups_t5(
        sigs_by_method, methods, min_guides=a.min_guides,
    )
    common_labels = _ordered_labels(gene_groups)
    shared_sigs_by_method = {
        m: OrderedDict((label, sigs_by_method[m][label]) for label in common_labels)
        for m in methods
    }
    ranked_de = _union_de_genes_ranked_t5(shared_sigs_by_method, cfg_for(methods[0]))
    genes_all = sorted(_genes_finite_all_t5(shared_sigs_by_method))
    genes_zoom = ranked_de[:a.max_de_genes]
    print(f"\nGene sets: union-DE (uncapped) n={len(ranked_de)}, all-eligible n={len(genes_all)}")

    if genes_zoom:
        outs = layer2_zoom(shared_sigs_by_method, gene_groups,
                           os.path.join(plots_dir, f"test5_zoom__{ds}.png"),
                           cfg_for(methods[0]), genes_zoom, methods=methods,
                           per_page=a.zoom_per_page)
        print(f"\nLayer 2 zoom: {len(outs)} PNG(s) — one row per method per gene")
    else:
        print("\nLayer 2 zoom: skipped (shared CPM-eligible union-DE panel is empty)")

    def _emit_l3(genes, suffix, label):
        p = os.path.join(plots_dir, f"test5_corr_matrix{suffix}__{ds}.png")
        layer3_corr_matrix(shared_sigs_by_method, gene_groups, p, cfg_for(methods[0]), genes,
                           methods=methods, gene_set_label=label, min_guides=a.min_guides)
        print(f"Layer 3 {suffix or 'primary (union-DE)'}: {os.path.abspath(p)}")

    # Primary: uncapped complete union-DE
    if len(ranked_de) < 5:
        print(f"Layer 3 primary union-DE: skipped (only {len(ranked_de)} genes)")
    else:
        _emit_l3(ranked_de, "", f"union-DE genes, no cap (n={len(ranked_de)})")

    # Sensitivity caps at 400 / 2000 / 4000 (cross-method LFC variance ranking).
    # Skip when cap ≥ full set (would duplicate the primary).
    for cap in (400, 2000, 4000):
        if cap >= len(ranked_de):
            print(f"Layer 3 sensitivity cap={cap}: skipped (cap ≥ full union-DE n={len(ranked_de)})")
            continue
        capped = ranked_de[:cap]
        if len(capped) < 5:
            print(f"Layer 3 sensitivity cap={cap}: skipped")
            continue
        _emit_l3(capped, f"_top{cap}", f"union-DE top-{cap} by cross-method LFC variance (n={len(capped)})")

    # All-eligible: complete-case genes, no DE filter — unbiased reference panel
    if len(genes_all) >= 5:
        _emit_l3(genes_all, "_all_genes",
                 f"all eligible genes — complete-case, no FDR filter (n={len(genes_all)})")


if __name__ == "__main__":
    main()
