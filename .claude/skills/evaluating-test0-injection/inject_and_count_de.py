#!/usr/bin/env python
"""Test-0 injection + anchor-recovery (TPR/FPR) check, wired to cell_eval2.h5ad.

Two pieces, as requested:

1. ``inject_effect_on_controls`` — the Test-0 *injection*: take the control cells, split them into
   two equal arms (A = reference, B = pretend-perturbation), and multiply the **raw counts** of a
   chosen anchor-gene set in arm B by ``2**delta`` (log2 fold-change ``delta``). Returns an AnnData of
   the two arms (``obs['_arm'] ∈ {'A','B'}``) ready to hand to differential expression, plus the
   anchor gene names. ``delta=0`` reproduces the untouched null.

2. ``recover_injected`` — the recovery check, **pooled over N repeats** (default 100): each repeat uses
   a different seed, so both the control/control A/B split AND the anchor set change; at each δ, DE is
   run on the arms (B vs A) with **wilcoxon** (cell-eval ``pdex``, cell-level rank-sum) and **pydeseq2**
   (pseudobulk DESeq2 Wald), and we record how many injected anchors are recovered (TPR) vs how many
   untouched genes are falsely called (FPR). Reported as mean±SD over the repeats.

DE uses the *real* ``cell_eval`` code paths (``build_de_frame``), not a reimplementation. Run
standalone: ``python inject_and_count_de.py --adata <h5ad>`` (100 repeats by default).
"""
from __future__ import annotations

import argparse
import os

import anndata as ad
import numpy as np
import polars as pl
import scipy.sparse as sp

from cell_eval._de_backends import build_de_frame


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _counts_matrix(adata, counts_layer):
    """Dense-ish raw-count matrix (cells × genes) from a layer or .X."""
    X = adata.layers[counts_layer] if counts_layer and counts_layer in adata.layers else adata.X
    return X


def _dense(X):
    return np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)


def _pdex_expr_filter(de_frame, adata, groupby, pert_label, ctrl_label, *, min_cpm=5.0):
    """Remove pdex DE genes where mean CPM < min_cpm in both groups.
    adata.X must be log1p-normalized; expm1 recovers the pre-log values."""
    obs_g = adata.obs[groupby].astype(str).to_numpy()
    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsr()

    def _mean_norm(idx):
        sub = X[idx, :].copy()
        sub.data = np.expm1(sub.data)
        return sub

    Xp = _mean_norm(np.where(obs_g == pert_label)[0])
    Xc = _mean_norm(np.where(obs_g == ctrl_label)[0])
    median_lib = float(np.asarray(Xc.sum(axis=1)).mean())
    min_norm = min_cpm / 1e6 * median_lib
    m_pert = np.asarray(Xp.mean(axis=0)).ravel()
    m_ctrl = np.asarray(Xc.mean(axis=0)).ravel()
    g2i = {g: i for i, g in enumerate(adata.var_names)}
    keep = {str(g) for g in de_frame["feature"].to_list()
            if (i := g2i.get(str(g))) is not None
            and (m_pert[i] >= min_norm or m_ctrl[i] >= min_norm)}
    return de_frame.filter(pl.col("feature").is_in(keep))


# --------------------------------------------------------------------------- #
# (1) injection on control cells
# --------------------------------------------------------------------------- #
def _stratified_split(obs, block_cols, rng) -> np.ndarray:
    """Split rows of obs 50/50 into 'A'/'B', stratified within each block_cols group.

    For each plate/batch, exactly half the cells go to arm A and half to arm B.
    This ensures arm A and arm B have the same number of cells from every replicate.
    """
    out = np.empty(len(obs), dtype=object)
    cols = [c for c in block_cols if c in obs.columns]
    if cols:
        groups = obs.groupby(cols, observed=True, sort=False).indices.values()
    else:
        groups = [np.arange(len(obs))]
    for idx in groups:
        idx = np.asarray(idx, dtype=int)
        perm = rng.permutation(idx.size)
        half = idx.size // 2
        out[idx[perm[:half]]] = "A"
        out[idx[perm[half:]]] = "B"
    return out.astype(str)


def inject_effect_on_controls(adata, delta, *, n_genes=12, seed=0,
                              pert_col="gene", control_label="non-targeting",
                              counts_layer="counts", max_cells_per_arm=3000,
                              block_cols=None):
    """Inject a known log2 fold-change ``delta`` into ``n_genes`` anchor genes in arm B of a
    control-vs-control split (the Test-0 ground truth).

    Returns ``(inj, anchors)`` where ``inj`` is an AnnData of the two arms (raw counts in ``.X``,
    ``obs['_arm'] ∈ {'A','B'}``; arm B's anchor counts multiplied by ``2**delta`` and rounded to int)
    and ``anchors`` is the list of injected gene names. Feed ``inj`` to DE with groupby=``_arm``,
    reference=``A`` — the injected anchors should show up as DE (TPR), untouched genes should not (FPR).

    When ``block_cols`` is given (e.g. ['plate']), the A/B split is stratified within each block so
    each arm has equal cell counts from every replicate — pseudobulk by replicate_col then gives each
    arm the same number of pseudobulk samples.
    """
    rng = np.random.default_rng(seed)
    ctrl = adata[adata.obs[pert_col].astype(str) == control_label].copy()
    n = ctrl.n_obs
    if max_cells_per_arm and n > 2 * max_cells_per_arm:
        keep = rng.choice(n, size=2 * max_cells_per_arm, replace=False)
        ctrl = ctrl[np.sort(keep)].copy()
        n = ctrl.n_obs
    # stratified split within each block (plate/batch) so each arm has equal replicate representation
    arm = _stratified_split(ctrl.obs, block_cols or [], rng)
    ctrl.obs["_arm"] = arm
    # work on raw counts
    C = _dense(_counts_matrix(ctrl, counts_layer)).astype(float)
    anchors_idx = np.sort(rng.choice(ctrl.n_vars, size=min(n_genes, ctrl.n_vars), replace=False))
    b = arm == "B"
    C[np.ix_(b, anchors_idx)] = np.rint(C[np.ix_(b, anchors_idx)] * (2.0 ** delta))
    inj = ad.AnnData(X=sp.csr_matrix(C), obs=ctrl.obs.copy(), var=ctrl.var.copy())
    anchors = list(ctrl.var_names[anchors_idx])
    return inj, anchors


# --------------------------------------------------------------------------- #
# (2) injection recovery: inject known anchors, run DE on the arms, measure TPR/FPR
# --------------------------------------------------------------------------- #
def _recover_one(adata, delta, seed, *, n_genes, pert_col, control_label, replicate_col,
                 counts_layer, fdr, lfc, num_threads, block_cols=None):
    """One injection→DE recovery at (delta, seed): returns per-method dicts (TPR/FPR + raw counts)."""
    import scanpy as sc

    inj, anchors = inject_effect_on_controls(adata, delta, n_genes=n_genes, seed=seed,
                                             pert_col=pert_col, control_label=control_label,
                                             counts_layer=counts_layer, block_cols=block_cols)
    anchor_set = set(anchors)
    n_untouched = inj.n_vars - len(anchor_set)
    ln = inj.copy()
    sc.pp.normalize_total(ln)
    sc.pp.log1p(ln)
    de_w = build_de_frame(mode="real", adata=ln, control_pert="A", pert_col="_arm",
                          num_threads=num_threads, allow_discrete=False, de_method="pdex",
                          de_kwargs=None, counts_layer=None, replicate_col=replicate_col)
    de_w = _pdex_expr_filter(de_w, ln, "_arm", "B", "A")
    de_p = build_de_frame(mode="real", adata=inj, control_pert="A", pert_col="_arm",
                          num_threads=num_threads, allow_discrete=True, de_method="pydeseq2",
                          de_kwargs=None, counts_layer=None, replicate_col=replicate_col)
    out = []
    for method, de in (("wilcoxon", de_w), ("pydeseq2", de_p)):
        sig = de.filter((pl.col("fdr") <= fdr) & (pl.col("log2_fold_change").abs() >= lfc))
        sig_genes = {str(f) for f in sig["feature"].to_list()}
        rec = len(sig_genes & anchor_set)
        fp = len(sig_genes - anchor_set)
        out.append({"delta": float(delta), "seed": int(seed), "method": method,
                    "n_anchors": len(anchor_set), "n_anchors_recovered": rec,
                    "TPR": rec / len(anchor_set), "n_false_positives": fp, "FPR": fp / n_untouched})
    return out


def recover_injected(adata, *, deltas=(0.0, 0.5, 1.0, 2.0), n_genes=12, n_repeats=100, seed=0,
                     pert_col="gene", control_label="non-targeting", replicate_col="batch",
                     counts_layer="counts", fdr=0.05, lfc=0.1, num_threads=8, per_repeat_out=None,
                     block_cols=None):
    """Sanity-check the injection, **pooled over ``n_repeats`` repeats** (default 100). Each repeat uses a
    different seed (``seed + r``), so both the control/control A/B split AND the ``n_genes`` anchor set
    change; at each δ, DE (arm B vs arm A) is run with **wilcoxon (pdex)** and **pydeseq2** and we record
    how many injected anchors are recovered (TPR) vs how many untouched genes are falsely called (FPR).

    Expectation: δ=0 → TPR≈FPR≈α (nothing real); δ≥1 → TPR→~1 on detectable anchors, FPR≈0 (12 injected
    genes barely shift library size ⇒ no compositional coupling). Returns ``(agg, per_repeat)``:
    ``agg`` = one row per (delta, method) with mean±SD TPR/FPR over the repeats + the pooled TPR
    (total anchors recovered / total injected); ``per_repeat`` = the raw per-(repeat, delta, method) rows.
    """
    rows = []
    for r in range(n_repeats):
        for delta in deltas:
            rows += _recover_one(adata, delta, seed + r, n_genes=n_genes, pert_col=pert_col,
                                 control_label=control_label, replicate_col=replicate_col,
                                 counts_layer=counts_layer, fdr=fdr, lfc=lfc, num_threads=num_threads,
                                 block_cols=block_cols)
    per = pl.DataFrame(rows)
    if per_repeat_out:
        per.write_csv(per_repeat_out)
    agg = (per.group_by(["delta", "method"]).agg(
        pl.len().alias("n_repeats"),
        pl.col("TPR").mean().alias("mean_TPR"), pl.col("TPR").std().alias("sd_TPR"),
        (pl.col("n_anchors_recovered").sum() / pl.col("n_anchors").sum()).alias("pooled_TPR"),
        pl.col("FPR").mean().alias("mean_FPR"), pl.col("FPR").std().alias("sd_FPR"),
        pl.col("n_false_positives").mean().alias("mean_false_positives"),
    ).sort(["delta", "method"]))
    return agg, per


def plot_recovery_box(per_repeat_csv, out_png, *, metric="n_anchors_recovered"):
    """Box plot of injection recovery across repeats: x = injected log2FC (δ), y = ``metric``, one box
    per method (pdex/Wilcoxon vs pydeseq2) at each δ, with the individual per-repeat points overlaid as a
    jittered scatter. Reads the per-repeat CSV written by ``recover_injected``.

    ``metric`` ∈ {``n_anchors_recovered`` (absolute # of the N injected anchors recovered — default),
    ``n_false_positives`` (DE genes detected that are NOT anchors — wrongly-called untouched genes),
    ``n_false_negatives`` (anchors NOT recovered = N − recovered), ``TPR``, ``FPR``}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    per = pl.read_csv(per_repeat_csv)
    n_anc = int(per["n_anchors"].max())
    if metric == "n_false_negatives":  # derive from what recover_injected already stored
        per = per.with_columns((pl.col("n_anchors") - pl.col("n_anchors_recovered")).alias("n_false_negatives"))
    deltas = sorted(per["delta"].unique().to_list())
    methods = ["wilcoxon", "pydeseq2"]
    color = {"wilcoxon": "#4C78A8", "pydeseq2": "#F58518"}   # colourblind-safe blue / orange
    label = {"wilcoxon": "pdex (cell-level Wilcoxon)", "pydeseq2": "pydeseq2 (pseudobulk DESeq2)"}
    ylab = {"n_anchors_recovered": f"anchors recovered (of {n_anc} injected)",
            "n_false_positives": "DE genes detected NOT among anchors (false positives)",
            "n_false_negatives": f"anchors NOT recovered — false negatives (of {n_anc})",
            "TPR": "anchor recovery TPR (fraction)",
            "FPR": "false-positive rate (untouched genes)"}.get(metric, metric)
    jit = np.random.default_rng(0)
    w = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for mi, m in enumerate(methods):
        off = (mi - 0.5) * w
        data = [per.filter((pl.col("delta") == d) & (pl.col("method") == m))[metric].to_numpy() for d in deltas]
        pos = [i + off for i in range(len(deltas))]
        bp = ax.boxplot(data, positions=pos, widths=w * 0.88, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.2))
        for box in bp["boxes"]:
            box.set(facecolor=color[m], alpha=0.45, edgecolor=color[m])
        for whisk in bp["whiskers"] + bp["caps"]:
            whisk.set(color=color[m])
        for i, arr in enumerate(data):
            xs = jit.normal(i + off, 0.035, size=len(arr))
            ax.scatter(xs, arr, s=16, color=color[m], edgecolor="white", linewidth=0.4, zorder=3)
    n_rep = int(per.group_by(["delta", "method"]).len()["len"].max())
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels([f"{d:g}" for d in deltas])
    ax.set_xlabel("injected log2 fold-change (δ)  ·  δ=0 = null")
    ax.set_ylabel(ylab)
    if metric in ("n_anchors_recovered", "n_false_negatives"):
        ax.set_ylim(-0.5, n_anc + 0.5)
        ax.set_yticks(range(0, n_anc + 1))
    elif metric == "n_false_positives":
        top = max(1, int(per["n_false_positives"].max()))
        ax.set_ylim(-0.5, top + 0.5)
        ax.set_yticks(range(0, top + 1))
    elif metric in ("TPR", "FPR"):
        ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="y", ls=":", color="0.8", zorder=0)
    ax.set_title(f"Test-0 injection recovery — pdex vs pydeseq2\n{n_rep} repeats × {n_anc} anchors "
                 "(control-vs-control), one point per repeat", fontsize=10)
    ax.legend(handles=[Patch(facecolor=color[m], alpha=0.6, label=label[m]) for m in methods],
              fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# (3) expression-bin-stratified injection: same delta, anchors sampled from each bin
# --------------------------------------------------------------------------- #
def _recover_one_binned(adata, delta, seed, *, n_genes_per_bin, n_bins, pert_col, control_label,
                        replicate_col, counts_layer, fdr, lfc, num_threads, block_cols=None):
    """Per-bin injection→DE recovery: for each bin independently inject n_genes_per_bin
    anchors (only those 12 genes) and run a separate DE experiment. Returns per-bin-per-method rows."""
    import pandas as pd
    import scanpy as sc
    rng = np.random.default_rng(seed)

    # --- shared A/B arm split of control cells (one split per seed, reused across all bins) ---
    ctrl = adata[adata.obs[pert_col].astype(str) == control_label].copy()
    n = ctrl.n_obs
    max_cells_per_arm = 3000
    if n > 2 * max_cells_per_arm:
        keep = rng.choice(n, size=2 * max_cells_per_arm, replace=False)
        ctrl = ctrl[np.sort(keep)].copy()
        n = ctrl.n_obs
    # stratified split within each block (plate/batch) so each arm has equal replicate representation
    arm = _stratified_split(ctrl.obs, block_cols or [], rng)
    ctrl.obs["_arm"] = arm

    C = _dense(_counts_matrix(ctrl, counts_layer)).astype(float)
    a_mask = arm == "A"
    b_mask = arm == "B"
    gene_means = C[a_mask].mean(axis=0)

    # log-scale equal-width bins (same logic as before)
    nonzero_idx = np.where(gene_means > 0)[0]
    bin_labels = np.full(ctrl.n_vars, -1, dtype=int)
    if len(nonzero_idx) >= n_bins:
        log_means = np.log10(gene_means[nonzero_idx])
        edges = np.linspace(log_means.min(), log_means.max(), n_bins + 1)
        q = pd.cut(pd.Series(log_means), bins=edges, labels=False, include_lowest=True)
        bin_labels[nonzero_idx] = np.asarray(q)
    actual_bins = sorted({b for b in bin_labels if b >= 0})

    rows = []
    for b in actual_bins:
        bin_gene_idx = np.where(bin_labels == b)[0]
        k = min(n_genes_per_bin, len(bin_gene_idx))
        if k == 0:
            continue

        # sample anchors from THIS bin only
        anchors_idx = rng.choice(bin_gene_idx, size=k, replace=False)
        anchor_set = set(ctrl.var_names[anchors_idx].tolist())
        bin_mean = float(gene_means[bin_gene_idx].mean())

        # inject ONLY this bin's anchors into arm B — separate experiment per bin
        C_inj = C.copy()
        C_inj[np.ix_(b_mask, anchors_idx)] = np.rint(
            C_inj[np.ix_(b_mask, anchors_idx)] * (2.0 ** delta)
        )
        inj = ad.AnnData(X=sp.csr_matrix(C_inj), obs=ctrl.obs.copy(), var=ctrl.var.copy())

        ln = inj.copy()
        sc.pp.normalize_total(ln)
        sc.pp.log1p(ln)
        de_w = build_de_frame(mode="real", adata=ln, control_pert="A", pert_col="_arm",
                              num_threads=num_threads, allow_discrete=False, de_method="pdex",
                              de_kwargs=None, counts_layer=None, replicate_col=replicate_col)
        de_w = _pdex_expr_filter(de_w, ln, "_arm", "B", "A")
        de_p = build_de_frame(mode="real", adata=inj, control_pert="A", pert_col="_arm",
                              num_threads=num_threads, allow_discrete=True, de_method="pydeseq2",
                              de_kwargs=None, counts_layer=None, replicate_col=replicate_col)

        for method, de in (("wilcoxon", de_w), ("pydeseq2", de_p)):
            sig_genes = {str(f) for f in de.filter(
                (pl.col("fdr") <= fdr) & (pl.col("log2_fold_change").abs() >= lfc)
            )["feature"].to_list()}
            rec = len(sig_genes & anchor_set)
            fp = len(sig_genes - anchor_set)
            n_non = ctrl.n_vars - k
            rows.append({
                "delta": float(delta), "seed": int(seed), "method": method,
                "bin": b, "bin_mean_count": bin_mean,
                "n_anchors": k, "n_recovered": rec,
                "TPR": rec / k,
                "n_non_anchors": n_non, "n_false_positives": fp,
                "FPR": fp / n_non if n_non > 0 else float("nan"),
            })
    return rows


def recover_injected_binned(adata, *, deltas=(0.5, 1.0, 2.0), n_genes_per_bin=5, n_bins=10,
                            n_repeats=30, seed=0, pert_col, control_label, replicate_col,
                            counts_layer, fdr, lfc, num_threads, per_repeat_out=None,
                            block_cols=None):
    """Run binned injection over n_repeats × deltas and return (agg, per_repeat) DataFrames.
    Each repeat samples fresh anchors from each expression-quantile bin of control cells,
    injects delta into arm B, runs DE, and records per-bin TPR."""
    rows = []
    for r in range(n_repeats):
        for delta in deltas:
            rows += _recover_one_binned(adata, delta, seed + r, n_genes_per_bin=n_genes_per_bin,
                                        n_bins=n_bins, pert_col=pert_col, control_label=control_label,
                                        replicate_col=replicate_col, counts_layer=counts_layer,
                                        fdr=fdr, lfc=lfc, num_threads=num_threads,
                                        block_cols=block_cols)
            print(f"  [binned] delta={delta:g}  repeat={r + 1}/{n_repeats}", flush=True)
    per = pl.DataFrame(rows)
    if per_repeat_out:
        per.write_csv(per_repeat_out)
    agg = (per.group_by(["delta", "method", "bin"]).agg(
        pl.col("bin_mean_count").mean().alias("bin_mean_count"),
        pl.col("TPR").mean().alias("mean_TPR"),
        pl.col("TPR").std().alias("sd_TPR"),
        pl.len().alias("n_repeats"),
    ).sort(["delta", "method", "bin"]))
    return agg, per


def plot_tpr_by_bin(per_csv, out_png, *, deltas=None, fdr=0.05, lfc=0.1):
    """Line plot of anchors recovered vs expression bin, one panel per method, lines coloured by delta.

    X-axis: expression bin (0=lowest mean count … n_bins-1=highest).
    Y-axis: number of injected anchor genes recovered per bin (out of n_genes_per_bin).
    Each point = mean ± 1 SD over repeats.  δ=0 (null) is shown as a dashed baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per = pl.read_csv(per_csv)
    if deltas is None:
        deltas = sorted(per["delta"].unique().to_list())
    methods = [m for m in ["wilcoxon", "pydeseq2"] if m in per["method"].unique().to_list()]
    method_color = {"wilcoxon": "#4C78A8", "pydeseq2": "#F58518"}
    method_label = {"wilcoxon": "pdex (cell-level Wilcoxon)", "pydeseq2": "pydeseq2 (pseudobulk DESeq2)"}
    # fixed colour palette for nonzero deltas — no yellow
    _DELTA_PALETTE = ["#1f77b4", "#d95f02", "#d62728", "#9467bd", "#2ca02c"]
    nonzero = [d for d in deltas if d != 0.0]
    delta_color = {d: _DELTA_PALETTE[i % len(_DELTA_PALETTE)] for i, d in enumerate(nonzero)}

    bins = sorted(per["bin"].unique().to_list())
    # x-axis: log10(mean raw count) per bin — equal spacing in log space
    bin_means_per_bin = (per.group_by("bin").agg(pl.col("bin_mean_count").mean())
                         .sort("bin").to_pandas().set_index("bin")["bin_mean_count"].to_dict())
    x_pos = {b: np.log10(bin_means_per_bin[b]) for b in bins}
    xlabels = [f"{bin_means_per_bin[b]:.2f}" for b in bins]

    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4.5), sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        # dashed null line at delta=0 if present
        if 0.0 in deltas:
            sub0 = per.filter((pl.col("delta") == 0.0) & (pl.col("method") == method))
            agg0 = (sub0.group_by("bin")
                    .agg(pl.col("n_recovered").mean().alias("mean_rec"))
                    .sort("bin"))
            xv = np.array([x_pos[b] for b in agg0["bin"].to_list()])
            ax.plot(xv, agg0["mean_rec"].to_numpy(),
                    color="gray", lw=1, ls="--", label="δ=0 (null)")
        for delta in nonzero:
            sub = per.filter((pl.col("delta") == delta) & (pl.col("method") == method))
            agg = (sub.group_by("bin")
                   .agg(pl.col("n_recovered").mean().alias("mean_rec"),
                        pl.col("n_recovered").std().alias("sd_rec"),
                        pl.col("n_anchors").first().alias("n_anchors"))
                   .sort("bin"))
            xv = np.array([x_pos[b] for b in agg["bin"].to_list()])
            y = agg["mean_rec"].to_numpy()
            e = agg["sd_rec"].to_numpy()
            n_total = int(agg["n_anchors"].max())
            col = delta_color[delta]
            ax.plot(xv, y, marker="o", color=col, lw=1.8, label=f"δ={delta:g}")
            ax.fill_between(xv, np.clip(y - e, 0, n_total), np.clip(y + e, 0, n_total),
                            color=col, alpha=0.15)
        n_anchors_per_bin = int(per.filter(pl.col("delta") != 0.0)["n_anchors"].max())
        ax.set_xticks(list(x_pos.values()))
        ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
        ax.set_xlabel("mean raw UMI count of genes in bin (arm A control cells, log scale)")
        ax.set_ylabel(f"anchors recovered (out of {n_anchors_per_bin} per bin)")
        ax.set_ylim(-0.2, n_anchors_per_bin + 0.5)
        ax.set_title(method_label.get(method, method), fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", ls=":", color="0.85")

    n_rep = int(per["seed"].n_unique())
    fig.suptitle(f"Test-0: anchors recovered by expression bin — same δ injected across all bins\n"
                 f"{n_rep} repeats · FDR<{fdr} · |LFC|>{lfc}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_fpr_by_bin(per_csv, out_png, *, deltas=None, fdr=0.05, lfc=0.1):
    """Line plot of false positives vs expression bin, one panel per method.

    X-axis: expression bin (0=lowest … n_bins-1=highest, by mean raw count in arm A).
    Y-axis: number of *untouched* (non-anchor) genes in each bin called DE at FDR threshold.
    δ=0 is the null; higher δ shows how injection of nearby genes
    in arm B bleeds into false positives in the same expression range."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per = pl.read_csv(per_csv)
    if deltas is None:
        deltas = sorted(per["delta"].unique().to_list())
    methods = [m for m in ["wilcoxon", "pydeseq2"] if m in per["method"].unique().to_list()]
    method_color = {"wilcoxon": "#4C78A8", "pydeseq2": "#F58518"}
    method_label = {"wilcoxon": "pdex (cell-level Wilcoxon)", "pydeseq2": "pydeseq2 (pseudobulk DESeq2)"}
    _DELTA_PALETTE = ["#808080", "#1f77b4", "#d95f02", "#d62728", "#9467bd", "#2ca02c"]
    all_ds = sorted(per["delta"].unique().to_list())
    delta_color = {d: _DELTA_PALETTE[i % len(_DELTA_PALETTE)] for i, d in enumerate(all_ds)}

    bins = sorted(per["bin"].unique().to_list())
    bin_means_per_bin = (per.group_by("bin").agg(pl.col("bin_mean_count").mean())
                         .sort("bin").to_pandas().set_index("bin")["bin_mean_count"].to_dict())
    x_pos = {b: np.log10(bin_means_per_bin[b]) for b in bins}
    xlabels = [f"{bin_means_per_bin[b]:.2f}" for b in bins]

    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4.5), sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        for delta in all_ds:
            sub = per.filter((pl.col("delta") == delta) & (pl.col("method") == method))
            agg = (sub.group_by("bin")
                   .agg(pl.col("n_false_positives").mean().alias("mean_FP"),
                        pl.col("n_false_positives").std().alias("sd_FP"))
                   .sort("bin"))
            xv = np.array([x_pos[b] for b in agg["bin"].to_list()])
            y = agg["mean_FP"].to_numpy()
            e = agg["sd_FP"].to_numpy()
            col = delta_color[delta]
            ls = "--" if delta == 0.0 else "-"
            ax.plot(xv, y, marker="o", color=col, lw=1.8, ls=ls, label=f"δ={delta:g}")
            ax.fill_between(xv, np.clip(y - e, 0, None), y + e,
                            color=col, alpha=0.15)
        ax.set_xticks(list(x_pos.values()))
        ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
        ax.set_xlabel("mean raw UMI count of genes in bin (arm A control cells, log scale)")
        ax.set_ylabel("DE genes detected (not anchors)")
        ax.set_ylim(-0.5, None)
        ax.set_title(method_label.get(method, method), fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", ls=":", color="0.85")

    n_rep = int(per["seed"].n_unique())
    fig.suptitle(f"Test-0: DE genes detected per bin that are not anchors\n"
                 f"{n_rep} repeats · FDR<{fdr} · |LFC|>{lfc}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True, help="input perturbation .h5ad path")
    ap.add_argument("--pert-col", default="gene")
    ap.add_argument("--control", default="non-targeting")
    ap.add_argument("--replicate-col", default="batch")
    ap.add_argument("--block-cols", default=None,
                    help="comma-sep column(s) to stratify the A/B arm split within (e.g. 'plate'). "
                         "Ensures each arm has equal cell counts from every replicate before "
                         "pydeseq2 pseudobulks by --replicate-col.")
    ap.add_argument("--counts-layer", default="counts")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--lfc", type=float, default=0.1)
    ap.add_argument("--n-genes", type=int, default=12, help="anchor genes injected per repeat")
    ap.add_argument("--n-repeats", type=int, default=50, help="repeats: each seed = a fresh control/control split + fresh anchors")
    ap.add_argument("--n-genes-per-bin", type=int, default=12, help="anchors per expression bin for binned TPR analysis")
    ap.add_argument("--n-bins", type=int, default=10, help="number of expression quantile bins")
    ap.add_argument("--n-repeats-binned", type=int, default=10, help="repeats for binned TPR analysis")
    ap.add_argument("--deltas", default="0,0.5,1,2", help="comma-sep log2FC tiers to inject")
    ap.add_argument("--out", default="injection_recovery.csv")
    a = ap.parse_args()
    deltas = tuple(float(x) for x in a.deltas.split(","))

    adata = ad.read_h5ad(a.adata)
    print(f"loaded {a.adata}: {adata.n_obs} cells × {adata.n_vars} genes")
    # fall back to .X if the requested counts layer doesn't exist
    counts_layer = a.counts_layer if (a.counts_layer and a.counts_layer in adata.layers) else None
    if counts_layer != a.counts_layer:
        print(f"[warn] counts_layer '{a.counts_layer}' not in adata.layers; using .X directly")

    # every output file carries the dataset name so runs on different datasets never collide
    dataset = os.path.splitext(os.path.basename(a.adata))[0]
    # treat --out as outdir when it has no file extension, otherwise use its parent
    out_abs = os.path.abspath(a.out)
    outdir = out_abs if not os.path.splitext(os.path.basename(a.out))[1] else (os.path.dirname(out_abs) or ".")
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    base_csv = os.path.join(outdir, f"injection_recovery__{dataset}")
    base_png = os.path.join(plots_dir, f"injection_recovery__{dataset}")
    agg_csv, per_csv = base_csv + ".csv", base_csv + "__per_repeat.csv"

    # (1) injection recovery, POOLED OVER n_repeats (each seed = a new control/control split + new anchors)
    block_cols = [c.strip() for c in a.block_cols.split(",") if c.strip()] if a.block_cols else []
    agg, per = recover_injected(adata, deltas=deltas, n_genes=a.n_genes, n_repeats=a.n_repeats,
                                seed=0, pert_col=a.pert_col, control_label=a.control,
                                replicate_col=a.replicate_col, counts_layer=counts_layer,
                                fdr=a.fdr, lfc=a.lfc, num_threads=8,
                                per_repeat_out=per_csv, block_cols=block_cols)
    print(f"\n[injection recovery] {a.n_repeats} repeats × {a.n_genes} anchors, control-vs-control, "
          f"FDR<{a.fdr} & |log2FC|>{a.lfc}. TPR = injected anchors recovered; FPR = untouched genes falsely called.")
    with pl.Config(tbl_rows=40, float_precision=4):
        print(agg)
    agg.write_csv(agg_csv)
    print(f"written: {os.path.abspath(agg_csv)}  (+ per-repeat: {per_csv})")

    # (2) box plots: x = injected log2FC, boxes = pdex vs pydeseq2 (per-repeat points overlaid).
    #     recovered = absolute # anchors recovered (of N); false_neg = anchors NOT recovered (N − recovered).
    for metric, tag in (("n_anchors_recovered", "recovered_box"), ("n_false_positives", "false_pos_box")):
        png = base_png + f"__{tag}.png"
        plot_recovery_box(per_csv, png, metric=metric)
        print(f"plot:    {os.path.abspath(png)}")

    # (3) expression-bin-stratified TPR: same delta injected into anchors from each expression bin
    print(f"\n[binned TPR] {a.n_repeats_binned} repeats × {a.n_genes_per_bin} anchors/bin × {a.n_bins} bins …")
    binned_deltas = tuple(d for d in deltas if d != 0.0)  # no point in null for TPR-by-bin (it's ~0 everywhere)
    binned_deltas_with_null = deltas  # include null for reference baseline
    bin_per_csv = base_csv + "__binned_per_repeat.csv"
    _, bin_per = recover_injected_binned(
        adata, deltas=binned_deltas_with_null,
        n_genes_per_bin=a.n_genes_per_bin, n_bins=a.n_bins, n_repeats=a.n_repeats_binned,
        seed=200, pert_col=a.pert_col, control_label=a.control,
        replicate_col=a.replicate_col, counts_layer=counts_layer,
        fdr=a.fdr, lfc=a.lfc, num_threads=8, per_repeat_out=bin_per_csv,
        block_cols=block_cols)
    bin_png = base_png + "__tpr_by_bin.png"
    plot_tpr_by_bin(bin_per_csv, bin_png, fdr=a.fdr, lfc=a.lfc)
    print(f"plot:    {os.path.abspath(bin_png)}")
    fpr_png = base_png + "__fpr_by_bin.png"
    plot_fpr_by_bin(bin_per_csv, fpr_png, fdr=a.fdr, lfc=a.lfc)
    print(f"plot:    {os.path.abspath(fpr_png)}")


if __name__ == "__main__":
    main()
