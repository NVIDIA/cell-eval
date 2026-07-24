#!/usr/bin/env python
"""Test 6 — target-gene knockdown recovery: pdex vs pyDESeq2.

For each perturbation, check whether the targeted gene is detected as DE
in the expected (knock-down) direction. Produces:
  - <outdir>/test6_knockdown_recovery_crossmethod__<dataset>.csv
  - <outdir>/test6_knockdown_recovery_crossmethod__<dataset>.md
  - <outdir>/test6_knockdown_recovery_crossmethod__<dataset>.png  (matplotlib table)

Run:
  python knockdown_recovery.py --adata /path/to.h5ad --pert-col gene \
      --control non-targeting --replicate-col batch --outdir .
"""
from __future__ import annotations

import argparse
import os

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp

from de_backends import build_de_frame, de_method_label, write_resolved_config

# ------------------------------------------------------------------ #
# DE computation
# ------------------------------------------------------------------ #

def _cpm_filter_multi(de_frame: pl.DataFrame, adata: ad.AnnData,
                      groupby: str, ctrl_label: str, counts_layer,
                      *, min_cpm: float = 5.0) -> pl.DataFrame:
    """Per-perturbation CPM eligibility filter applied identically to both pdex and pyDESeq2.

    Removes genes with mean CPM < min_cpm in BOTH perturbation AND control cells.
    Uses raw counts from counts_layer when available; auto-detects log-normalised X and
    applies expm1; otherwise uses X directly as raw counts.
    """
    obs_g = adata.obs[groupby].astype(str).to_numpy()
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


def _run_de(adata: ad.AnnData, method: str, pert_col: str, control: str,
            replicate_col: str | None, counts_layer: str | None,
            non_parametric_engine: str = "pdex", num_threads: int = 8) -> pd.DataFrame:
    if method == "pdex":
        a = adata.copy()
        if counts_layer and counts_layer in a.layers:
            a.X = a.layers[counts_layer]
        import scanpy as sc
        if _looks_raw(a.X):
            sc.pp.normalize_total(a, inplace=True)
            sc.pp.log1p(a)
        result = build_de_frame(
            mode="real", adata=a, control_pert=control, pert_col=pert_col,
            num_threads=num_threads, allow_discrete=False, de_method="pdex",
            de_kwargs={"engine": non_parametric_engine}, counts_layer=None,
        )
        # Apply shared CPM filter using raw counts from counts_layer (or adata directly).
        result = _cpm_filter_multi(result, adata, pert_col, control, counts_layer)
        return result.to_pandas()
    else:
        result = build_de_frame(
            mode="real", adata=adata, control_pert=control, pert_col=pert_col,
            num_threads=num_threads, allow_discrete=True, de_method="pydeseq2",
            de_kwargs=None, counts_layer=counts_layer, replicate_col=replicate_col,
        )
        # Apply shared CPM filter identically to pydeseq2.
        result = _cpm_filter_multi(result, adata, pert_col, control, counts_layer)
        return result.to_pandas()


def _looks_raw(X) -> bool:
    """Robust raw-integer detection: X looks like raw counts if (a sample of) its
    finite values are all (close to) integers. Matches the shared de_helpers heuristic."""
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        return False
    if data.size > 5_000_000:
        rng = np.random.default_rng(0)
        data = data[rng.integers(0, data.size, 5_000_000)]
    return bool(np.allclose(data, np.rint(data)))


# ------------------------------------------------------------------ #
# Table build
# ------------------------------------------------------------------ #

def _target_cpm(adata: ad.AnnData, pert_col: str, control: str,
                target_genes: list[str], counts_layer: str | None) -> dict[str, tuple[float, float]]:
    """Return {gene: (cpm_pert, cpm_ctrl)} — mean CPM of the target gene in perturbed vs
    control cells (consistent with the 5 CPM pdex filter threshold).
    Uses counts_layer when present, else adata.X (raw counts expected)."""
    X = adata.layers[counts_layer] if (counts_layer and counts_layer in adata.layers) else adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsr().astype(float)

    totals = np.asarray(X.sum(axis=1)).ravel().astype(float)
    obs_g = adata.obs[pert_col].astype(str).to_numpy()
    var_names = list(adata.var_names)
    g2i = {g: i for i, g in enumerate(var_names)}
    ctrl_mask = obs_g == control

    result: dict[str, tuple[float, float]] = {}
    for tgt in target_genes:
        gi = g2i.get(tgt)
        if gi is None:
            result[tgt] = (np.nan, np.nan)
            continue
        pert_mask = obs_g == tgt
        if not pert_mask.any() or not ctrl_mask.any():
            result[tgt] = (np.nan, np.nan)
            continue
        gene_col = np.asarray(X[:, gi].todense()).ravel().astype(float)
        cell_cpm = np.divide(
            gene_col * 1e6, totals, out=np.zeros_like(gene_col), where=totals > 0,
        )
        cpm_pert = float(np.mean(cell_cpm[pert_mask]))
        cpm_ctrl = float(np.mean(cell_cpm[ctrl_mask]))
        result[tgt] = (cpm_pert, cpm_ctrl)
    return result


def build_table(adata: ad.AnnData, de: dict[str, pd.DataFrame],
                pert_col: str, control: str, fdr_thr: float, lfc_thr: float = 0.1,
                counts_layer: str | None = None) -> pd.DataFrame:
    genes = set(map(str, adata.var_names))
    perts = sorted(p for p in adata.obs[pert_col].astype(str).unique() if p != control)
    stat_cols = ["lfc", "pvalue", "fdr", "rank_pvalue", "rank_fdr",
                 "rank_signed_lfc", "rank_abs_lfc", "sign_lfc"]
    methods = list(de.keys())

    cpm_map = _target_cpm(adata, pert_col, control, perts, counts_layer)

    rows = []
    for p in perts:
        cpm_pert, cpm_ctrl = cpm_map.get(p, (np.nan, np.nan))
        row: dict = {"perturbation": p, "target_in_var": p in genes,
                     "cpm_pert": cpm_pert, "cpm_ctrl": cpm_ctrl}
        for m in methods:
            d = de[m][de[m]["target"].astype(str) == p].copy()
            sig_mask = (d["fdr"] <= fdr_thr) & (d["log2_fold_change"].abs() >= lfc_thr)
            row[f"n_de_genes_{m}"] = int(sig_mask.sum())
            if not len(d) or p not in genes:
                for c in stat_cols:
                    row[f"{c}_{m}"] = np.nan
                continue
            d["_rk_p"]    = d["p_value"].rank(method="min")
            d["_rk_fdr"]  = d["fdr"].rank(method="min")
            d["_rk_slfc"] = d["log2_fold_change"].rank(method="min")
            d["_rk_alfc"] = d["log2_fold_change"].abs().rank(method="min", ascending=False)
            r = d[d["feature"].astype(str) == p]
            if not len(r):
                for c in stat_cols:
                    row[f"{c}_{m}"] = np.nan
                continue
            r = r.iloc[0]
            lfc = float(r["log2_fold_change"])
            row[f"lfc_{m}"]            = lfc
            row[f"pvalue_{m}"]         = float(r["p_value"])
            row[f"fdr_{m}"]            = float(r["fdr"])
            row[f"rank_pvalue_{m}"]    = int(r["_rk_p"])
            row[f"rank_fdr_{m}"]       = int(r["_rk_fdr"])
            row[f"rank_signed_lfc_{m}"]= int(r["_rk_slfc"])
            row[f"rank_abs_lfc_{m}"]   = int(r["_rk_alfc"])
            row[f"sign_lfc_{m}"]       = int(np.sign(lfc))
        rows.append(row)

    df = pd.DataFrame(rows)
    order = ["perturbation", "target_in_var", "cpm_pert", "cpm_ctrl"]
    for m in methods:
        order.append(f"n_de_genes_{m}")
    for c in stat_cols:
        for m in methods:
            order.append(f"{c}_{m}")
    df = df[[c for c in order if c in df.columns]]
    return df


# ------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------ #

def _fmt(v):
    if pd.isna(v):
        return "NA"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def plot_table(df: pd.DataFrame, out_png: str, dataset: str,
               methods: list[str], fdr_thr: float,
               non_parametric_engine: str = "pdex") -> None:
    # Build display-friendly column headers (two-line: stat + [method])
    _CPM_LABELS = {"cpm_pert": "cpm\n[pert]", "cpm_ctrl": "cpm\n[ctrl]"}

    def col_header(col: str) -> str:
        if col in _CPM_LABELS:
            return _CPM_LABELS[col]
        for m in methods:
            if col.endswith(f"_{m}"):
                stat = col[: -(len(m) + 1)]
                display = de_method_label(
                    m, non_parametric_engine=non_parametric_engine
                )
                return f"{stat}\n[{display}]"
        return col

    disp_cols = [col_header(c) for c in df.columns]

    # Format cell values
    cell_data = [[_fmt(v) for v in row] for _, row in df.iterrows()]

    # Highlight rows where methods most disagree (sign differs or one method misses)
    def _highlight(row: pd.Series) -> bool:
        if len(methods) < 2:
            return False
        m0, m1 = methods[0], methods[1]
        fdr0 = row.get(f"fdr_{m0}", np.nan)
        fdr1 = row.get(f"fdr_{m1}", np.nan)
        sig0 = pd.notna(fdr0) and fdr0 <= fdr_thr
        sig1 = pd.notna(fdr1) and fdr1 <= fdr_thr
        if sig0 != sig1:
            return True
        sign0 = row.get(f"sign_lfc_{m0}", np.nan)
        sign1 = row.get(f"sign_lfc_{m1}", np.nan)
        if pd.notna(sign0) and pd.notna(sign1) and sign0 != sign1:
            return True
        return False

    highlight = [_highlight(row) for _, row in df.iterrows()]

    ncols = len(df.columns)
    nrows = len(df)
    fig_w = max(22, ncols * 1.1)
    fig_h = max(4, nrows * 0.38 + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    n_perts = len(df)
    ax.set_title(
        f"Test 6 — target-gene knockdown recovery: "
        f"{' vs '.join(de_method_label(m, non_parametric_engine=non_parametric_engine) for m in methods)}"
        f"  ({dataset}, {n_perts} perturbations)",
        fontsize=11, pad=14,
    )

    tbl = ax.table(
        cellText=cell_data,
        colLabels=disp_cols,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.auto_set_column_width(list(range(ncols)))

    # Style header
    for j in range(ncols):
        cell = tbl[0, j]
        cell.set_facecolor("#d0d8e8")
        cell.set_text_props(fontweight="bold", fontsize=6.5)

    # Style data rows
    for i, is_hl in enumerate(highlight):
        for j in range(ncols):
            cell = tbl[i + 1, j]
            if is_hl:
                cell.set_facecolor("#ffe8a0")
            elif i % 2 == 0:
                cell.set_facecolor("#f8f8f8")
            else:
                cell.set_facecolor("#ffffff")

    plt.tight_layout(pad=1.2)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"plot: {os.path.abspath(out_png)}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adata",           required=True)
    ap.add_argument("--methods",         default="pdex,pydeseq2",
                    help="comma-sep DE backends (default: pdex,pydeseq2)")
    ap.add_argument("--pert-col",        default="gene")
    ap.add_argument("--control",         default="non-targeting")
    ap.add_argument("--replicate-col",   default="batch",
                    help="pseudobulk replicate column for pydeseq2")
    ap.add_argument("--counts-layer",    default=None,
                    help="raw-counts layer; auto-detects 'counts' if present")
    ap.add_argument("--fdr",             type=float, default=0.05)
    ap.add_argument("--lfc",             type=float, default=0.1,
                    help="minimum |LFC| threshold for counting DE genes (default: 0.1)")
    ap.add_argument("--non-parametric-engine", choices=("pdex", "rsc"), default="pdex",
                    help="non-parametric engine; pdex uses Arc pdex and rsc uses RAPIDS GPU Wilcoxon")
    ap.add_argument("--threads", type=int, default=8,
                    help="CPU threads for the shared multi-contrast DE model")
    ap.add_argument("--outdir",          default=".")
    ap.add_argument(
        "--expression-state", choices=("raw_counts", "log1p_normalized"), default="",
        help="user-confirmed state of adata.X; recorded in the resolved YAML",
    )
    ap.add_argument(
        "--run-root", default="",
        help="confirmed run root for configs/ and logs/ (defaults to --outdir)",
    )
    a = ap.parse_args()
    if a.threads < 1:
        ap.error("--threads must be at least 1")

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    dataset = os.path.splitext(os.path.basename(a.adata))[0]

    adata = ad.read_h5ad(a.adata)
    counts_layer = a.counts_layer
    if counts_layer is None and "counts" in adata.layers:
        counts_layer = "counts"
    print(f"loaded {a.adata}: {adata.n_obs} cells × {adata.n_vars} genes  (methods={methods})")
    a.run_root = os.path.abspath(os.path.expanduser(a.run_root or a.outdir))
    write_resolved_config(
        run_root=a.run_root,
        workflow="de_test6_knockdown_recovery",
        dataset=dataset,
        resolved={
            "arguments": vars(a),
            "methods": methods,
            "effective_counts_layer": counts_layer or "X",
            "results_outdir": os.path.abspath(a.outdir),
        },
    )

    de: dict[str, pd.DataFrame] = {}
    # run pydeseq2 first (needs raw counts, before pdex normalises in-place)
    for m in [x for x in methods if x != "pdex"] + (["pdex"] if "pdex" in methods else []):
        print(f"[t6] {m} DE …")
        de[m] = _run_de(
            adata, m, a.pert_col, a.control, a.replicate_col, counts_layer,
            non_parametric_engine=a.non_parametric_engine, num_threads=a.threads,
        )

    df = build_table(adata, de, a.pert_col, a.control, a.fdr, lfc_thr=a.lfc, counts_layer=counts_layer)
    if "pdex" in methods:
        df.insert(0, "non_parametric_engine", a.non_parametric_engine)

    os.makedirs(a.outdir, exist_ok=True)
    base = os.path.join(a.outdir, f"test6_knockdown_recovery_crossmethod__{dataset}")
    df.to_csv(base + ".csv", index=False)

    disp = df.copy()
    for c in disp.columns:
        if disp[c].dtype.kind == "f":
            disp[c] = disp[c].map(lambda x: f"{x:.4g}" if pd.notna(x) else "NA")
    disp.to_markdown(base + ".md", index=False)

    plot_table(
        df, base + ".png", dataset=dataset, methods=methods, fdr_thr=a.fdr,
        non_parametric_engine=a.non_parametric_engine,
    )

    # Recovery summary
    print(f"\n[t6] target-gene recovery summary (n={len(df)} perts):")
    for m in methods:
        det  = int((df[f"fdr_{m}"]  <= a.fdr).sum()) if f"fdr_{m}"  in df.columns else 0
        dirn = int((df[f"lfc_{m}"]  < 0    ).sum()) if f"lfc_{m}"  in df.columns else 0
        print(f"  {m:9s}: detected(FDR<{a.fdr}) {det}/{len(df)} | LFC<0 {dirn}/{len(df)}")
    print(f"wrote: {base}.csv  |  {base}.md  |  {base}.png")


if __name__ == "__main__":
    main()
