#!/usr/bin/env python3
"""
overview.py — DE gene count comparison: pyDESeq2 vs pdex per perturbation.

Scatter plot saved to ``plots/test01_overview__<dataset>.png`` (every output filename is suffixed with
the input .h5ad basename so runs on different datasets never overwrite each other):
  x = # DE genes per perturbation (pdex, cell-level Wilcoxon)
  y = # DE genes per perturbation (pyDESeq2, pseudobulk)
  size = cell count for that perturbation
  diagonal = equal calling; above = pyDESeq2 calls more; below = pdex calls more

Usage:
    uv run python overview.py --config config.yaml
    uv run python overview.py --config config.yaml --plot-only  # re-render from cached tables
"""
from __future__ import annotations

import argparse
import logging
import os

import anndata as ad
import matplotlib
import numpy as np
import polars as pl
import scanpy as sc
import scipy.sparse as sp
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cell_eval._de_backends import build_de_frame  # noqa: E402

log = logging.getLogger("overview")
OVERVIEW_CACHE_VERSION = 2

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _looks_raw_integer(adata: ad.AnnData) -> bool:
    X = adata.X
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        return False
    if data.size > 5_000_000:
        rng = np.random.default_rng(0)
        data = data[rng.integers(0, data.size, 5_000_000)]
    return bool(np.allclose(data, np.rint(data)))


def _make_norm_copy(adata: ad.AnnData) -> ad.AnnData:
    a = adata.copy()
    if _looks_raw_integer(a):
        sc.pp.normalize_total(a, inplace=True)
        sc.pp.log1p(a)
    return a


def _cpm_filter_multi(de_frame: pl.DataFrame, adata: ad.AnnData,
                      groupby: str, ctrl_label: str, cfg: dict,
                      *, min_cpm: float = 5.0) -> pl.DataFrame:
    """Per-perturbation CPM eligibility filter applied identically to both pdex and pyDESeq2.

    Removes genes with mean CPM < min_cpm in BOTH the perturbation's cells AND control cells.
    Uses raw counts from cfg['counts_layer'] when available; auto-detects log-normalised X
    (non-integer values) and applies expm1; otherwise uses X directly as raw counts.
    """
    obs_g = adata.obs[groupby].astype(str).to_numpy()
    counts_layer = cfg.get("counts_layer")
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
    var_names = list(adata.var_names)

    def mean_cpm(idx: np.ndarray) -> np.ndarray:
        group = X[idx, :]
        library_sizes = np.asarray(group.sum(axis=1)).ravel().astype(float)
        scales = np.divide(
            1e6, library_sizes, out=np.zeros_like(library_sizes),
            where=library_sizes > 0,
        )
        return np.asarray(group.multiply(scales[:, None]).mean(axis=0)).ravel()

    ctrl_idx = np.where(obs_g == ctrl_label)[0]
    m_ctrl = mean_cpm(ctrl_idx)

    targets = de_frame["target"].unique().to_list() if "target" in de_frame.columns else []
    if not targets:
        ctrl_pass = {g for g, mc in zip(var_names, m_ctrl) if mc >= min_cpm}
        return de_frame.filter(pl.col("feature").is_in(ctrl_pass))

    pass_pairs: set[tuple[str, str]] = set()
    for tgt in targets:
        tidx = np.where(obs_g == str(tgt))[0]
        if len(tidx) == 0:
            continue
        m_pert = mean_cpm(tidx)
        for g, mc, mp in zip(var_names, m_ctrl, m_pert):
            if mc >= min_cpm or mp >= min_cpm:
                pass_pairs.add((str(tgt), g))

    keep_mask = pl.Series([
        (str(t), str(f)) in pass_pairs
        for t, f in zip(de_frame["target"].to_list(), de_frame["feature"].to_list())
    ])
    return de_frame.filter(keep_mask)


def _run_de(adata: ad.AnnData, cfg: dict, method: str) -> pl.DataFrame:
    """Run DE without any post-processing filter."""
    return build_de_frame(
        mode="real",
        adata=adata,
        control_pert=cfg["control_pert"],
        pert_col=cfg["pert_col"],
        num_threads=cfg.get("num_threads", 8),
        allow_discrete=(method == "pydeseq2"),
        de_method=method,
        de_kwargs=None,
        counts_layer=cfg.get("counts_layer"),
        replicate_col=cfg.get("replicate_col") if method == "pydeseq2" else None,
    )


def _sig_sets(de: pl.DataFrame, fdr_t: float, lfc_t: float) -> dict[str, set]:
    """Return {perturbation: set(significant gene names)} from a DE frame."""
    out: dict[str, set] = {}
    for pert in de["target"].unique().to_list():
        sub = de.filter(pl.col("target") == pert)
        lfc = sub["log2_fold_change"].to_numpy().astype(float)
        fdr = sub["fdr"].to_numpy().astype(float)
        feat = sub["feature"].to_numpy().astype(str)
        sig = np.isfinite(fdr) & (fdr <= fdr_t) & np.isfinite(lfc) & (np.abs(lfc) >= lfc_t)
        out[pert] = set(feat[sig])
    return out


def _count_sig(de: pl.DataFrame, fdr_t: float, lfc_t: float) -> dict[str, int]:
    """Return {perturbation: n_significant_genes} from a DE frame."""
    return {p: len(s) for p, s in _sig_sets(de, fdr_t, lfc_t).items()}


def _jaccard(sets_a: dict[str, set], sets_b: dict[str, set]) -> dict[str, float]:
    """Per-perturbation Jaccard |A∩B|/|A∪B| between the two backends' significant gene sets."""
    out: dict[str, float] = {}
    for p in set(sets_a) | set(sets_b):
        a, b = sets_a.get(p, set()), sets_b.get(p, set())
        u = a | b
        out[p] = (len(a & b) / len(u)) if u else float("nan")
    return out


def _mean_expr_per_gene(adata: ad.AnnData, pert: str, pert_col: str) -> np.ndarray:
    """Mean expression per gene across cells belonging to `pert`."""
    mask = (adata.obs[pert_col] == pert).values
    X = adata.X[mask]
    if sp.issparse(X):
        return np.asarray(X.mean(axis=0)).ravel()
    return np.asarray(X, dtype=float).mean(axis=0)


def _pick_representative_perts(
    de_pyd: pl.DataFrame,
    de_pdx: pl.DataFrame,
    cell_counts: dict[str, int],
    ctrl: str,
) -> dict[str, str]:
    """Return ordered dict {rank_label: pert_name} for most / median / least cells,
    restricted to perturbations present in both DE frames and excluding control."""
    both = set(de_pyd["target"].unique().to_list()) & set(de_pdx["target"].unique().to_list())
    both.discard(ctrl)
    cands = sorted(both, key=lambda p: cell_counts.get(p, 0))
    n = len(cands)
    if n < 3:
        return {}
    picks = {
        f"least cells  (n={cell_counts.get(cands[0], 0)})":    cands[0],
        f"median cells  (n={cell_counts.get(cands[n//2], 0)})": cands[n // 2],
        f"most cells  (n={cell_counts.get(cands[-1], 0)})":    cands[-1],
    }
    return picks


def _build_ma_tables(
    adata_raw: ad.AnnData,
    adata_norm: ad.AnnData,
    de_pyd: pl.DataFrame,
    de_pdx: pl.DataFrame,
    perts: list[str],
    cfg: dict,
    tables_dir: str,
    ds: str,
    de_pdx_raw: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """For each method, build a gene-level table (feature, perturbation, mean_expr, lfc, fdr)
    covering the 3 representative perturbations and cache to disk.

    For pdex: lfc_map comes from de_pdx_raw (unfiltered, so ALL genes appear as grey dots in
    the MA plot, preserving the low-expression fan shape); fdr_map comes from de_pdx (filtered,
    so only expression-filtered genes can be marked as DE/red).
    """
    pc = cfg["pert_col"]
    genes = list(adata_raw.var_names)

    def _build(adata: ad.AnnData, de_for_lfc: pl.DataFrame, de_for_fdr: pl.DataFrame, method: str) -> pl.DataFrame:
        rows: list[dict] = []
        for pert in perts:
            means = _mean_expr_per_gene(adata, pert, pc)
            sub_lfc = de_for_lfc.filter(pl.col("target") == pert)
            sub_fdr = de_for_fdr.filter(pl.col("target") == pert)
            lfc_map = dict(zip(sub_lfc["feature"].to_list(), sub_lfc["log2_fold_change"].to_numpy().astype(float)))
            fdr_map = dict(zip(sub_fdr["feature"].to_list(), sub_fdr["fdr"].to_numpy().astype(float)))
            for g, m in zip(genes, means):
                rows.append({
                    "feature": g,
                    "perturbation": pert,
                    "mean_expr": float(m),
                    "log2_fold_change": lfc_map.get(g, float("nan")),
                    "fdr": fdr_map.get(g, float("nan")),
                })
        df = pl.DataFrame(rows)
        df.write_csv(os.path.join(tables_dir, f"overview_ma_{method}__{ds}.csv"))
        return df

    log.info("Building MA tables (mean expr × LFC) for representative perturbations …")
    # pdex: use raw (unfiltered) frame for LFC so all genes appear as grey dots;
    #       use filtered frame for FDR so only high-confidence DE genes are shown red.
    de_lfc_pdx = de_pdx_raw if de_pdx_raw is not None else de_pdx
    return _build(adata_raw, de_pyd, de_pyd, "pydeseq2"), _build(adata_raw, de_lfc_pdx, de_pdx, "pdex")


def _plot_ma_panels(
    ma_df: pl.DataFrame,
    picks: dict[str, str],
    fdr_t: float,
    lfc_t: float,
    method: str,
    x_label: str,
    out_path: str,
) -> str:
    """1 × 3 figure: mean expression vs LFC for each representative perturbation."""
    method_label = {
        "pdex": "pdex (cell-level Wilcoxon)",
        "pydeseq2": "pydeseq2 (pseudobulk DESeq2)",
    }[method]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (rank_label, pert) in zip(axes, picks.items()):
        sub = ma_df.filter(pl.col("perturbation") == pert)
        x    = sub["mean_expr"].to_numpy().astype(float)
        y    = sub["log2_fold_change"].to_numpy().astype(float)
        fdrs = sub["fdr"].to_numpy().astype(float)

        x1 = x + 1  # pseudocount for log scale
        ok    = np.isfinite(x1) & np.isfinite(y)
        is_de = ok & np.isfinite(fdrs) & (fdrs <= fdr_t) & (np.abs(y) >= lfc_t)
        n_de  = int(is_de.sum())

        ax.scatter(x1[ok & ~is_de], y[ok & ~is_de], s=3,  alpha=0.30, color="#AAAAAA",
                   rasterized=True, label="non-DE", zorder=2)
        ax.scatter(x1[ok & is_de],  y[ok & is_de],  s=14, alpha=0.85, color="#E63946",
                   rasterized=True, label=f"DE  (n={n_de})", zorder=3)

        ax.axhline(0,      color="#555555", lw=0.9, ls="--", zorder=1)
        ax.axhline( lfc_t, color="#E63946", lw=0.5, ls=":",  alpha=0.55, zorder=1)
        ax.axhline(-lfc_t, color="#E63946", lw=0.5, ls=":",  alpha=0.55, zorder=1)

        ax.set_xscale("log")
        ax.set_xlabel(x_label + " + 1  (log scale)", fontsize=9)
        ax.set_ylabel("log2 fold change", fontsize=9)
        ax.set_title(f"{pert}\n{rank_label}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.88)
        ax.grid(alpha=0.15, lw=0.5)

    fig.suptitle(
        f"{method_label}  —  mean expression vs LFC for 3 representative perturbations\n"
        f"grey = all genes  ·  red = DE genes  (FDR < {fdr_t},  |LFC| ≥ {lfc_t})",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

_PYD_COLOR = "#4472C4"
_PDX_COLOR = "#ED7D31"


def _plot_scatter(
    n_sig_pyd: dict[str, int],
    n_sig_pdx: dict[str, int],
    cell_counts: dict[str, int],
    cfg: dict,
    ax,
    jaccard: dict[str, float] | None = None,
) -> None:
    ctrl = cfg["control_pert"]
    perts = [p for p in n_sig_pyd if p != ctrl and p in n_sig_pdx]
    jaccard = jaccard or {}

    x = np.array([n_sig_pdx[p] for p in perts], dtype=float)
    y = np.array([n_sig_pyd[p] for p in perts], dtype=float)
    nc = np.array([cell_counts.get(p, 50) for p in perts], dtype=float)

    size_norm = (nc - nc.min()) / (nc.max() - nc.min() + 1e-9)
    dot_sizes = 40 + size_norm * 320

    sc = ax.scatter(x, y, s=dot_sizes, alpha=0.78, c=nc,
                    cmap="viridis", edgecolors="white", linewidths=0.5, zorder=3)

    lim_max = max(x.max(), y.max()) * 1.16   # headroom so edge labels (name + J=) aren't clipped
    lim_min = 0.0
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color="#888888", lw=1.2, ls="--", zorder=2, label="equal calling")
    ax.fill_between([lim_min, lim_max], [lim_min, lim_max], lim_max,
                    color=_PYD_COLOR, alpha=0.04, label="pyDESeq2 calls more")
    ax.fill_between([lim_min, lim_max], lim_min, [lim_min, lim_max],
                    color=_PDX_COLOR, alpha=0.04, label="pdex calls more")

    # label = "NAME  J=<Jaccard(DE_pdex, DE_pydeseq2)>"; leader ARROWS to dots (non-overlapping)
    def _lbl(p):
        j = jaccard.get(p)
        return f"{p}  J={j:.2f}" if (j is not None and np.isfinite(j)) else p
    try:
        from adjustText import adjust_text
        texts = [ax.text(xi, yi, _lbl(lbl), fontsize=7.5) for xi, yi, lbl in zip(x, y, perts)]
        adjust_text(texts, x=list(x), y=list(y), ax=ax,
                    arrowprops=dict(arrowstyle="->", color="0.5", lw=0.6),
                    expand=(1.25, 1.6), force_text=(0.4, 0.6), min_arrow_len=4)
    except Exception:  # noqa: BLE001 — fallback: fixed-offset annotation + arrow
        for xi, yi, lbl in zip(x, y, perts):
            ax.annotate(_lbl(lbl), (xi, yi), xytext=(xi + lim_max * 0.04, yi + lim_max * 0.04),
                        fontsize=7.5, arrowprops=dict(arrowstyle="->", color="0.5", lw=0.6))

    cb = ax.get_figure().colorbar(sc, ax=ax, shrink=0.78, pad=0.02)
    cb.set_label("cell count per perturbation", fontsize=9)

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("# DE genes — pdex (cell-level Wilcoxon)", fontsize=11)
    ax.set_ylabel("# DE genes — pyDESeq2 (pseudobulk)", fontsize=11)
    fdr_t = cfg["fdr_threshold"]
    lfc_t = cfg["lfc_threshold"]
    ax.set_title(
        f"DE gene count per perturbation: pyDESeq2 vs pdex\n"
        f"FDR < {fdr_t},  |LFC| ≥ {lfc_t}  •  dot size ∝ cell count  •  "
        "J = Jaccard(pdex, pyDESeq2 DE-gene sets)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
    ax.grid(alpha=0.2, lw=0.5)


# ---------------------------------------------------------------------------
# correlation matrix (perturbation × perturbation signature similarity)
# ---------------------------------------------------------------------------

def _complete_case_genes_ov(de_by: dict, perts: list[str]) -> set[str]:
    """Genes with finite LFC in EVERY method × perturbation combination."""
    common: set[str] | None = None
    for de in de_by.values():
        sub = (de.filter(pl.col("target").is_in(perts) & pl.col("log2_fold_change").is_finite())
                 .group_by("feature")
                 .agg(pl.col("target").n_unique().alias("n_perts")))
        full = set(sub.filter(pl.col("n_perts") >= len(perts))["feature"].to_list())
        common = full if common is None else common.intersection(full)
    return common or set()


def _union_de_genes_ranked_ov(de_by: dict, fdr_t: float, lfc_t: float,
                               perts: list[str]) -> list[str]:
    """All union-DE complete-case genes ranked by cross-method LFC variance (desc).

    Union membership: gene DE (FDR+LFC thresholds) in any method × perturbation.
    Complete-case: gene has finite LFC in EVERY method × perturbation (both panels use
    identical columns — the per-method NaN-drop in _build_lfc_matrix becomes a no-op).
    Ranking: LFC variance pooled across ALL methods × perturbations — symmetric across
    methods. Note: union membership itself depends on both methods' FDR calling.
    Returns the full ranked list; callers slice [:n] for any cap size.
    """
    union: set[str] = set()
    for de in de_by.values():
        sig = de.filter((pl.col("fdr") <= fdr_t) & (pl.col("log2_fold_change").abs() >= lfc_t))
        union.update(sig["feature"].to_list())
    union.intersection_update(_complete_case_genes_ov(de_by, perts))
    if not union:
        return []
    gene_set = union
    var_df = (pl.concat([
                  de.filter(pl.col("feature").is_in(gene_set) &
                            pl.col("log2_fold_change").is_finite())
                    .select(["feature", "log2_fold_change"])
                  for de in de_by.values()
              ])
              .group_by("feature")
              .agg(pl.col("log2_fold_change").var().alias("v")))
    vmap = dict(zip(var_df["feature"].to_list(), var_df["v"].to_list()))
    return sorted(union, key=lambda g: -(vmap.get(g) or 0.0))


def _build_lfc_matrix(de: pl.DataFrame, perts: list[str], genes: list[str]) -> np.ndarray:
    """Build a (n_perts × n_genes) LFC matrix.

    When genes are pre-filtered to the complete-case set (via _union_de_genes_ranked_ov),
    all columns are finite and the keep-mask is a no-op. The mask is retained as a safety
    net so that any residual NaN silently drops rather than propagates into corrcoef.
    """
    gi = {g: i for i, g in enumerate(genes)}
    M = np.full((len(perts), len(genes)), np.nan)
    for r, p in enumerate(perts):
        sub = de.filter(pl.col("target") == p)
        for fe, lv in zip(sub["feature"].to_list(), sub["log2_fold_change"].to_numpy().astype(float)):
            if fe in gi:
                M[r, gi[fe]] = lv
    keep = ~np.isnan(M).any(axis=0)
    return M[:, keep]


def _pert_corr(de: pl.DataFrame, perts: list[str], genes: list[str]) -> np.ndarray:
    """Perturbation × perturbation Spearman correlation of LFC signatures over `genes`."""
    from scipy.stats import rankdata
    M = _build_lfc_matrix(de, perts, genes)
    R = np.vstack([rankdata(row) for row in M])
    return np.corrcoef(R)


def _pert_corr_pearson(de: pl.DataFrame, perts: list[str], genes: list[str]) -> np.ndarray:
    """Perturbation × perturbation Pearson correlation of LFC signatures over `genes`."""
    M = _build_lfc_matrix(de, perts, genes)
    return np.corrcoef(M)


def _plot_corr_matrices(de_by: dict, cfg: dict, out_png: str,
                        corr_fn, metric_label: str,
                        genes: list[str],
                        gene_set_label: str = "") -> str | None:
    """Plot perturbation × perturbation correlation matrices, one panel per backend.

    Both backends use exactly the ``genes`` list — caller is responsible for supplying
    a complete-case list (finite LFC in every method × perturbation) so both panels
    are feature-comparable.
    """
    if len(genes) < 5:
        log.warning("Correlation matrix %s skipped: only %d genes", out_png, len(genes))
        return None
    fdr_t, lfc_t, ctrl = cfg["fdr_threshold"], cfg["lfc_threshold"], cfg["control_pert"]
    methods = [m for m in ("pdex", "pydeseq2") if m in de_by]
    common = set.intersection(*[set(de["target"].unique().to_list()) for de in de_by.values()])
    perts = [p for p in common if p != ctrl]
    nsig = _count_sig(de_by.get("pydeseq2", de_by[methods[0]]), fdr_t, lfc_t)
    perts = sorted(perts, key=lambda p: -nsig.get(p, 0))

    fig, axes = plt.subplots(1, len(methods), figsize=(6.8 * len(methods), 6.4), squeeze=False)
    im = None
    for ax, m in zip(axes[0], methods):
        C = corr_fn(de_by[m], perts, genes)
        diagonal_mask = np.eye(len(perts), dtype=bool)
        diagonal = C[diagonal_mask]
        off = C[~diagonal_mask]
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks(range(len(perts))); ax.set_xticklabels(perts, rotation=90, fontsize=5)
        ax.set_yticks(range(len(perts))); ax.set_yticklabels(perts, fontsize=5)
        label = {"pdex": "pdex (cell-level Wilcoxon)", "pydeseq2": "pydeseq2 (pseudobulk DESeq2)"}[m]
        ax.set_title(
            f"{label}\nmean diagonal r = {np.nanmean(diagonal):.2f}; "
            f"off-diagonal r = {np.nanmean(off):.2f}",
            fontsize=10,
        )
    glabel = gene_set_label or f"{len(genes)} genes"
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02,
                 label=f"{metric_label} over {glabel}")
    fig.suptitle("Perturbation × perturbation signature correlation — diagonal = self (1); "
                 "off-diagonal = cross-perturbation similarity (dim = more specific)", fontsize=11)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="overview.py — pyDESeq2 vs pdex DE gene count scatter")
    ap.add_argument("--config", required=True, help="path to config.yaml")
    ap.add_argument("--out", default="plots/test01_overview.png",
                    help="output PNG path (relative to outdir from config)")
    ap.add_argument("--plot-only", action="store_true",
                    help="skip DE computation, use cached tables in tables/")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    with open(args.config) as fh:
        cfg: dict = yaml.safe_load(fh)

    outdir    = cfg.get("outdir", ".")
    tables_dir = os.path.join(outdir, "tables")
    plots_dir  = os.path.join(outdir, "plots")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir,  exist_ok=True)

    ds = os.path.splitext(os.path.basename(cfg["adata_path"]))[0]  # dataset tag on every output file
    pyd_full_path    = os.path.join(tables_dir, f"overview_pydeseq2_full__{ds}.csv")
    pyd_raw_path     = os.path.join(tables_dir, f"overview_pydeseq2_raw_full__{ds}.csv")
    pdx_full_path    = os.path.join(tables_dir, f"overview_pdex_full__{ds}.csv")
    pdx_raw_path     = os.path.join(tables_dir, f"overview_pdex_raw_full__{ds}.csv")
    cell_counts_path = os.path.join(tables_dir, f"overview_cell_counts__{ds}.csv")
    cache_meta_path  = os.path.join(tables_dir, f"overview_cache_meta__{ds}.yaml")
    expected_meta = {
        "format_version": OVERVIEW_CACHE_VERSION,
        "cpm_normalization": "per_cell_then_mean",
        "min_cpm": 5.0,
        "pert_col": cfg["pert_col"],
        "control_pert": cfg["control_pert"],
        "counts_layer": cfg.get("counts_layer"),
    }

    # ------------------------------------------------------------------ #
    # Full-data DE
    # ------------------------------------------------------------------ #
    de_pyd_raw: pl.DataFrame | None = None
    de_pdx_raw: pl.DataFrame | None = None  # unfiltered pdex (for MA plot grey dots)

    if args.plot_only:
        log.info("--plot-only: loading cached DE tables")
        required_cache_paths = [
            pyd_full_path, pyd_raw_path, pdx_full_path, pdx_raw_path,
            cell_counts_path, cache_meta_path,
        ]
        missing = [path for path in required_cache_paths if not os.path.exists(path)]
        if missing:
            raise ValueError(
                "--plot-only requires a complete versioned Overview cache; missing "
                f"{missing}. Rerun without --plot-only to rebuild all DE tables."
            )
        with open(cache_meta_path) as fh:
            cache_meta = yaml.safe_load(fh) or {}
        mismatches = {
            key: (cache_meta.get(key), value)
            for key, value in expected_meta.items()
            if cache_meta.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Stale or unsupported Overview cache metadata in {cache_meta_path}: "
                f"{mismatches}. Rerun without --plot-only to rebuild both backends."
            )
        de_pyd = pl.read_csv(pyd_full_path)
        de_pdx = pl.read_csv(pdx_full_path)
        de_pyd_raw = pl.read_csv(pyd_raw_path)
        de_pdx_raw = pl.read_csv(pdx_raw_path)
        cc_df  = pl.read_csv(cell_counts_path)
        cell_counts = dict(zip(cc_df["perturbation"].to_list(), cc_df["n_cells"].to_list()))
    else:
        log.info("Loading adata: %s", cfg["adata_path"])
        adata_raw = ad.read_h5ad(cfg["adata_path"])

        # Cell counts per perturbation
        pc = cfg["pert_col"]
        cell_counts = {grp: int((adata_raw.obs[pc] == grp).sum())
                       for grp in adata_raw.obs[pc].unique()}
        pl.DataFrame({"perturbation": list(cell_counts.keys()),
                      "n_cells":      list(cell_counts.values())}
                     ).write_csv(cell_counts_path)

        log.info("Building log-normalised copy for pdex …")
        adata_norm = _make_norm_copy(adata_raw)

        log.info("Running pydeseq2 (full data) …")
        de_pyd_raw = _run_de(adata_raw, cfg, "pydeseq2")
        de_pyd_raw.write_csv(pyd_raw_path)
        log.info("Applying shared CPM filter to pydeseq2 …")
        de_pyd = _cpm_filter_multi(de_pyd_raw, adata_raw, cfg["pert_col"], cfg["control_pert"], cfg)
        de_pyd.write_csv(pyd_full_path)
        log.info("  saved → %s  (raw → %s)", pyd_full_path, pyd_raw_path)

        log.info("Running pdex (full data) …")
        de_pdx_raw = _run_de(adata_norm, cfg, "pdex")
        de_pdx_raw.write_csv(pdx_raw_path)
        log.info("Applying shared CPM filter to pdex …")
        de_pdx = _cpm_filter_multi(de_pdx_raw, adata_raw, cfg["pert_col"], cfg["control_pert"], cfg)
        de_pdx.write_csv(pdx_full_path)
        log.info("  saved → %s  (raw → %s)", pdx_full_path, pdx_raw_path)
        with open(cache_meta_path, "w") as fh:
            yaml.safe_dump(expected_meta, fh, sort_keys=True)
        log.info("  saved cache metadata → %s", cache_meta_path)

    # ------------------------------------------------------------------ #
    # Count significant genes per perturbation
    # ------------------------------------------------------------------ #
    fdr_t = cfg["fdr_threshold"]
    lfc_t = cfg["lfc_threshold"]
    sets_pyd = _sig_sets(de_pyd, fdr_t, lfc_t)
    sets_pdx = _sig_sets(de_pdx, fdr_t, lfc_t)
    n_sig_pyd = {p: len(s) for p, s in sets_pyd.items()}
    n_sig_pdx = {p: len(s) for p, s in sets_pdx.items()}
    jaccard = _jaccard(sets_pdx, sets_pyd)   # DE-gene-set overlap between backends, per perturbation
    ctrl = cfg["control_pert"]
    pl.DataFrame({"perturbation": [p for p in jaccard if p != ctrl],
                  "n_sig_pdex":   [n_sig_pdx.get(p, 0) for p in jaccard if p != ctrl],
                  "n_sig_pydeseq2": [n_sig_pyd.get(p, 0) for p in jaccard if p != ctrl],
                  "jaccard":      [jaccard[p] for p in jaccard if p != ctrl]}
                 ).sort("jaccard").write_csv(os.path.join(tables_dir, f"overview_jaccard__{ds}.csv"))

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(8, 8))
    _plot_scatter(n_sig_pyd, n_sig_pdx, cell_counts, cfg, ax, jaccard=jaccard)

    outpath = args.out
    if not os.path.isabs(outpath):
        outpath = os.path.join(outdir, outpath)
    root, ext = os.path.splitext(outpath)   # append dataset tag before the extension
    outpath = f"{root}__{ds}{ext}"
    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", outpath)

    # ------------------------------------------------------------------ #
    # Correlation matrices (perturbation × perturbation, one panel per backend)
    # ------------------------------------------------------------------ #
    # Gene selection: union-DE complete-case, ranked by cross-method LFC variance.
    # Both backends (pdex, pydeseq2) use the SAME gene list in every plot so their
    # correlation coefficients are feature-comparable.
    # Union membership depends on both methods' FDR; ranking is symmetric (pooled).
    de_by = {"pdex": de_pdx, "pydeseq2": de_pyd}
    common_perts = set.intersection(*[set(de["target"].unique().to_list()) for de in de_by.values()])
    corr_perts = [p for p in common_perts if p != ctrl]
    ranked_de_ov = _union_de_genes_ranked_ov(de_by, fdr_t, lfc_t, corr_perts)
    log.info("Union-DE complete-case genes for correlation matrices: %d", len(ranked_de_ov))
    # All-eligible: complete-case over raw (unfiltered) DE — any gene with finite LFC in
    # all methods × perturbations, regardless of CPM eligibility or FDR significance.
    de_by_raw = None
    genes_all_ov: list[str] = []
    if de_pdx_raw is not None and de_pyd_raw is not None:
        de_by_raw = {"pdex": de_pdx_raw, "pydeseq2": de_pyd_raw}
        genes_all_ov = sorted(_complete_case_genes_ov(de_by_raw, corr_perts))
        log.info("All-eligible complete-case genes: %d", len(genes_all_ov))

    for corr_fn, metric_label, fstem in [
        (_pert_corr,         "Spearman(LFC)", "corr_matrix"),
        (_pert_corr_pearson, "Pearson(LFC)",  "corr_matrix_pearson"),
    ]:
        # Primary: uncapped complete union-DE
        if len(ranked_de_ov) >= 5:
            glabel = f"union-DE complete-case, no cap (n={len(ranked_de_ov)})"
            p = os.path.join(plots_dir, f"test01_{fstem}__{ds}.png")
            _plot_corr_matrices(de_by, cfg, p, corr_fn, metric_label, ranked_de_ov,
                                gene_set_label=glabel)
            log.info("Saved → %s", p)
        # Sensitivity caps: 400 / 2000 / 4000, ranked by cross-method LFC variance.
        # Skip when cap ≥ full set (would duplicate the primary).
        for cap in (400, 2000, 4000):
            if cap >= len(ranked_de_ov):
                log.info("Sensitivity cap=%d skipped (cap ≥ full union-DE n=%d)", cap, len(ranked_de_ov))
                continue
            capped = ranked_de_ov[:cap]
            if len(capped) < 5:
                continue
            glabel = f"union-DE top-{cap} by cross-method LFC variance (n={len(capped)})"
            p = os.path.join(plots_dir, f"test01_{fstem}_top{cap}__{ds}.png")
            _plot_corr_matrices(de_by, cfg, p, corr_fn, metric_label, capped,
                                gene_set_label=glabel)
            log.info("Saved sensitivity cap=%d → %s", cap, p)
        # All-eligible reference matrix: unfiltered DE, no CPM or FDR gate
        if de_by_raw is not None and len(genes_all_ov) >= 5:
            glabel = f"all eligible genes — complete-case, no FDR filter (n={len(genes_all_ov)})"
            p = os.path.join(plots_dir, f"test01_{fstem}_all_genes__{ds}.png")
            _plot_corr_matrices(de_by_raw, cfg, p, corr_fn, metric_label, genes_all_ov,
                                gene_set_label=glabel)
            log.info("Saved all-eligible → %s", p)

    # ------------------------------------------------------------------ #
    # MA scatter: mean expression vs LFC for most / median / least cells
    # ------------------------------------------------------------------ #
    pyd_ma_cache = os.path.join(tables_dir, f"overview_ma_pydeseq2__{ds}.csv")
    pdx_ma_cache = os.path.join(tables_dir, f"overview_ma_pdex__{ds}.csv")

    picks = _pick_representative_perts(de_pyd, de_pdx, cell_counts, ctrl)
    if not picks:
        log.warning("Fewer than 3 perturbations found — skipping MA scatter plots.")
    else:
        if args.plot_only:
            if os.path.exists(pyd_ma_cache) and os.path.exists(pdx_ma_cache):
                ma_pyd = pl.read_csv(pyd_ma_cache)
                ma_pdx = pl.read_csv(pdx_ma_cache)
            else:
                log.info("MA tables not cached; loading adata to compute mean expression (DE skipped) …")
                _adata_raw = ad.read_h5ad(cfg["adata_path"])
                _adata_norm = _make_norm_copy(_adata_raw)
                ma_pyd, ma_pdx = _build_ma_tables(
                    _adata_raw, _adata_norm, de_pyd, de_pdx,
                    list(picks.values()), cfg, tables_dir, ds,
                    de_pdx_raw=de_pdx_raw,
                )
        else:
            ma_pyd, ma_pdx = _build_ma_tables(
                adata_raw, adata_norm, de_pyd, de_pdx,
                list(picks.values()), cfg, tables_dir, ds,
                de_pdx_raw=de_pdx_raw,
            )

        if ma_pyd is not None:
            p = os.path.join(plots_dir, f"test01_ma_pydeseq2__{ds}.png")
            _plot_ma_panels(ma_pyd, picks, fdr_t, lfc_t, "pydeseq2",
                            "mean raw count (pert cells)", p)
            log.info("Saved → %s", p)
        if ma_pdx is not None:
            p = os.path.join(plots_dir, f"test01_ma_pdex__{ds}.png")
            _plot_ma_panels(ma_pdx, picks, fdr_t, lfc_t, "pdex",
                            "mean raw count (pert cells)", p)
            log.info("Saved → %s", p)


if __name__ == "__main__":
    main()
