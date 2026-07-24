#!/usr/bin/env python
"""Test-2 control-control null diagnostics — pdex vs pydeseq2 on one figure.

Reuses the SHARED Test-2 helpers from the metaskill runner (`stratified_split`, `run_de`,
`null_metrics`, `lambda_gc`, `maybe_normalize`) so the control-control split null exactly matches the
real Test 2 — then computes it for BOTH backends on the SAME A/B splits (same seed) and renders a
single 3-panel comparison figure:

  Panel 1 — QQ: expected vs observed −log10(p), one curve per method, grey y=x diagonal, 95% beta
            envelope. Points above the diagonal ⇒ anti-conservative (inflated); below ⇒ conservative.
  Panel 2 — p-value histogram: 20 bins, density (Uniform = 1.0 flat), two semi-transparent hists.
  Panel 3 — λ_GC stability: boxplot + strip of per-split λ_GC per method; dashed lines at 1.0 / 0.9 / 1.1.

Key message it makes visible: pdex is typically anti-conservative (λ_GC ≳ 1, QQ curves up) while
pydeseq2 is over-conservative (λ_GC < 1, QQ curves down) — neither is perfectly calibrated, but for
opposite reasons.

Run: `python control_null_diagnostics.py --adata <h5ad> --methods pdex,pydeseq2 ...`
"""
from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import os
import sys

import anndata as ad
import numpy as np
import polars as pl

from de_backends import de_method_label, write_resolved_config

_RT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "de_helpers.py")

# colorblind-safe (Okabe-Ito / seaborn "colorblind"): pdex = blue, pydeseq2 = orange
METHOD_COLOR = {"pdex": "#0173B2", "pydeseq2": "#DE8F05", "wilcoxon": "#0173B2"}
_T2_WORKER_STATE = {}


def _load_runner():
    """Import the shared metaskill runner as a module (reuse its exact Test-2 helpers)."""
    spec = importlib.util.spec_from_file_location("rt_shared", _RT)
    rt = importlib.util.module_from_spec(spec)
    sys.modules["rt_shared"] = rt
    spec.loader.exec_module(rt)
    return rt


def _fit_control_split(r, ctrl_ad, cfg, rt):
    s = ctrl_ad.copy()
    s.obs["_arm"] = rt.stratified_split(
        ctrl_ad.obs, cfg["block_cols"], cfg["seed"] + 100 + r,
    )
    try:
        de = rt.run_de(s, cfg, groupby="_arm", reference="B")
    except Exception as exc:  # noqa: BLE001
        return r, None, str(exc)
    p = de["p_value"].to_numpy().astype(float)
    p = p[np.isfinite(p)]
    return r, {
        "p": p,
        "lambda": rt.lambda_gc(de["p_value"].to_numpy().astype(float)),
        "de": de.select(["feature", "log2_fold_change", "fdr"]),
    }, None


def _fit_control_split_worker(r):
    return _fit_control_split(r, **_T2_WORKER_STATE)


def control_null_pvalues(
    adata, cfg, rt, *, n_resamples=10, seed=0, resample_workers=1,
):
    """Control-control split null for one backend.

    Cells are split 50/50 within each block (plate/batch) so arm A and arm B each receive
    the same number of cells from every replicate. Returns (pooled_pvalues, lambda_gc_per_split, de_long).
    """
    import polars as pl
    rt.maybe_normalize(adata, cfg)  # pdex expects log-norm; no-op for pydeseq2
    pc, mcg, ctrl = cfg["pert_col"], cfg["min_cells_per_group"], cfg["control_pert"]
    ctrl_ad = adata[adata.obs[pc].astype(str) == ctrl]
    if ctrl_ad.n_obs < 2 * mcg:
        raise SystemExit(f"too few control cells ({ctrl_ad.n_obs}) for 2×min_cells_per_group={2 * mcg}")

    tasks = range(n_resamples)
    if cfg["de_method"] == "pydeseq2" and resample_workers > 1:
        _T2_WORKER_STATE.clear()
        _T2_WORKER_STATE.update(ctrl_ad=ctrl_ad, cfg=cfg, rt=rt)
        pool = mp.get_context("fork").Pool(processes=resample_workers)
        results = pool.imap(_fit_control_split_worker, tasks, chunksize=1)
    else:
        pool = None
        results = (_fit_control_split(r, ctrl_ad, cfg, rt) for r in tasks)

    pooled, lambdas, de_frames = [], [], []
    try:
        for r, fitted, error in results:
            if fitted is None:
                print(f"  split {r}: DE failed ({error})", flush=True)
                continue
            pooled.append(fitted["p"])
            lambdas.append(fitted["lambda"])
            de_frames.append(
                fitted["de"].with_columns(pl.lit(r).alias("split"))
            )
            print(f"  split {r}: {fitted['p'].size} genes  "
                  f"λ_GC={fitted['lambda']:.3f}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if not pooled:
        raise SystemExit("no valid control split produced p-values")
    return np.concatenate(pooled), np.array(lambdas, dtype=float), pl.concat(de_frames)


def _qq_points(p, *, max_points=3000):
    """(expected, observed) −log10(p) for a QQ curve, log-subsampled so the significant tail stays dense."""
    p = np.sort(p[np.isfinite(p)])
    p = np.clip(p, 1e-300, 1.0)
    n = p.size
    ranks = np.arange(1, n + 1)
    exp = -np.log10((ranks - 0.5) / n)
    obs = -np.log10(p)
    if n > max_points:
        idx = np.unique(np.round(np.logspace(0, np.log10(n), max_points)).astype(int)) - 1
        idx = idx[(idx >= 0) & (idx < n)]
        exp, obs = exp[idx], obs[idx]
    return exp, obs, n


def plot_test2_diagnostics(
    pvalues_pdex,
    pvalues_pydeseq2,
    lambda_gc_per_split_pdex,
    lambda_gc_per_split_pydeseq2,
    *,
    alpha=0.05,
    n_resamples=10,
    output_path="test_2_pvalue_diagnostics.png",
    non_parametric_engine="pdex",
):
    """Single 3-panel figure comparing pdex vs pydeseq2 on the control-control null.

    Inputs are already-pooled arrays (as returned by `control_null_pvalues`):
      pvalues_{pdex,pydeseq2}           — pooled genes × splits p-values (flattened)
      lambda_gc_per_split_{pdex,pydeseq2} — one λ_GC per split (len == n_resamples)

    Panels: (1) QQ overlaid with 95% beta envelope, (2) p-value density histogram, (3) λ_GC boxplot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    pdex_label = de_method_label(
        "pdex", non_parametric_engine=non_parametric_engine
    )
    pydeseq2_label = de_method_label("pydeseq2")
    methods = [
        (pdex_label, np.asarray(pvalues_pdex, float), np.asarray(lambda_gc_per_split_pdex, float)),
        (pydeseq2_label, np.asarray(pvalues_pydeseq2, float), np.asarray(lambda_gc_per_split_pydeseq2, float)),
    ]
    method_color = {
        pdex_label: METHOD_COLOR["pdex"],
        pydeseq2_label: METHOD_COLOR["pydeseq2"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [1.25, 1.15, 0.9], "wspace": 0.32})

    # ── Panel 1: QQ, one curve per method ────────────────────────────────
    ax = axes[0]
    # 95% envelope from the larger pooled n (uniform order stats ~ Beta(i, n-i+1))
    n_env = max(m[1][np.isfinite(m[1])].size for m in methods)
    r_env = np.unique(np.round(np.logspace(0, np.log10(n_env), 400)).astype(int))
    r_env = r_env[(r_env >= 1) & (r_env <= n_env)]
    exp_env = -np.log10((r_env - 0.5) / n_env)
    lo = -np.log10(stats.beta.ppf(0.975, r_env, n_env - r_env + 1))
    hi = -np.log10(stats.beta.ppf(0.025, r_env, n_env - r_env + 1))
    ax.fill_between(exp_env, lo, hi, color="0.87", zorder=0, label="95% null envelope")
    m_val = float(exp_env.max())
    for name, pv, lam in methods:
        exp, obs, n = _qq_points(pv)
        if n < 8:
            continue
        m_val = max(m_val, float(obs.max()))
        lam_mean = float(np.nanmean(lam)) if lam.size else float("nan")
        ax.plot(exp, obs, lw=1.4, alpha=0.9, color=method_color[name],
                label=f"{name}  (λ_GC={lam_mean:.2f})")
    ax.plot([0, m_val], [0, m_val], "--", color="0.4", lw=1, zorder=1, label="y = x (calibrated)")
    ax.set_xlabel("expected −log₁₀(p)  (Uniform)")
    ax.set_ylabel("observed −log₁₀(p)")
    ax.set_title("(1) QQ: points should hug the diagonal\nabove = anti-conservative · below = conservative",
                 fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left")

    # ── Panel 2: p-value histogram (density) ─────────────────────────────
    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    for name, pv, _ in methods:
        p = pv[np.isfinite(pv)]
        ax.hist(p, bins=bins, density=True, alpha=0.5, color=method_color[name],
                label=name, edgecolor="white", linewidth=0.3)
    ax.axhline(1.0, color="black", ls="--", lw=1.2, label="Uniform[0,1] (density = 1)")
    ax.set_xlabel("p-value")
    ax.set_ylabel("density")
    ax.set_title("(2) Histogram: should be flat at 1.0\nleft spike = inflated · right skew = conservative",
                 fontsize=9)
    ax.legend(fontsize=7.5)

    # ── Panel 3: λ_GC stability across splits ────────────────────────────
    ax = axes[2]
    rng = np.random.default_rng(42)
    positions = range(1, len(methods) + 1)
    box_data, box_colors, labels = [], [], []
    for name, _, lam in methods:
        lam = lam[np.isfinite(lam)]
        box_data.append(lam)
        box_colors.append(method_color[name])
        labels.append(name)
    bp = ax.boxplot(box_data, positions=list(positions), widths=0.5, patch_artist=True,
                    medianprops=dict(color="black", lw=1.6),
                    whiskerprops=dict(color="#555"), capprops=dict(color="#555"),
                    flierprops=dict(marker="", alpha=0))
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
    for pos, lam, c in zip(positions, box_data, box_colors):
        jitter = rng.uniform(-0.13, 0.13, size=lam.size)
        ax.scatter(pos + jitter, lam, s=30, color=c, edgecolor="white", linewidth=0.4, alpha=0.9, zorder=3)
    ax.axhline(1.0, color="#D62728", ls="--", lw=1.3, label="calibrated (λ=1.0)")
    ax.axhline(0.9, color="0.6", ls=":", lw=1)
    ax.axhline(1.1, color="0.6", ls=":", lw=1, label="±0.1 band")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel("λ_GC")
    ax.set_title(f"(3) λ_GC: should be near 1.0\n(one point per split, n={n_resamples} splits)", fontsize=9)
    ax.legend(fontsize=7.5, loc="best")

    fig.suptitle(
                 f"Test 2 — control-control null (stratified split): "
                 f"{pdex_label} vs {pydeseq2_label}",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_test2_lfc_agreement(
    de_pdex, de_pydeseq2, output_path, *, fdr=0.05, lfc=0.1,
    non_parametric_engine="pdex",
):
    """Per-gene LFC agreement on the control-control null: pdex log2FC (x) vs pydeseq2 log2FC (y).

    `de_pdex` / `de_pydeseq2` are the per-gene, per-split `de_long` frames from `control_null_pvalues`
    (columns feature, log2_fold_change, fdr, split). Because both backends ran on the SAME A/B split for
    each seed, each (feature, split) is paired across backends. Every point is one gene in one null
    split; a point is 'DE' for a backend when FDR ≤ `fdr` and |log2FC| ≥ `lfc`. Since this is a true
    null, points should cluster at the origin; any DE calls are false positives — the plot shows whether
    the two backends' false positives coincide (points off-origin on the diagonal) or are independent
    noise (points on the axes)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import polars as pl
    from scipy import stats
    pdex_label = de_method_label(
        "pdex", non_parametric_engine=non_parametric_engine
    )
    pydeseq2_label = de_method_label("pydeseq2")

    j = (de_pdex.rename({"log2_fold_change": "lfc_pdex", "fdr": "fdr_pdex"})
         .join(de_pydeseq2.rename({"log2_fold_change": "lfc_pyd", "fdr": "fdr_pyd"}),
               on=["feature", "split"], how="inner"))
    if j.height == 0:
        print("LFC agreement: no shared (feature, split) rows — skipping")
        return None
    xp = j["lfc_pdex"].to_numpy().astype(float)
    yp = j["lfc_pyd"].to_numpy().astype(float)
    sig_pdex = (j["fdr_pdex"].to_numpy() <= fdr) & (np.abs(xp) >= lfc)
    sig_pyd = (j["fdr_pyd"].to_numpy() <= fdr) & (np.abs(yp) >= lfc)
    ok = np.isfinite(xp) & np.isfinite(yp)
    both = sig_pdex & sig_pyd & ok
    only_pdex = sig_pdex & ~sig_pyd & ok
    only_pyd = sig_pyd & ~sig_pdex & ok
    neither = ~sig_pdex & ~sig_pyd & ok

    fig, ax = plt.subplots(figsize=(7.2, 7))
    lim = float(np.nanpercentile(np.abs(np.concatenate([xp[ok], yp[ok]])), 99.8)) if ok.any() else 1.0
    lim = max(lim, 0.3)
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.plot([-lim, lim], [-lim, lim], "--", color="0.5", lw=0.9, zorder=1, label="y = x")
    ax.scatter(xp[neither], yp[neither], s=5, color="0.75", alpha=0.35, linewidths=0,
               label=f"DE in neither (n={int(neither.sum())})", zorder=2)
    ax.scatter(xp[only_pdex], yp[only_pdex], s=22, color="#0173B2", alpha=0.85, edgecolor="white",
               linewidth=0.3, label=f"DE in {pdex_label} only (n={int(only_pdex.sum())})", zorder=4)
    ax.scatter(xp[only_pyd], yp[only_pyd], s=22, color="#DE8F05", alpha=0.85, edgecolor="white",
               linewidth=0.3, label=f"DE in {pydeseq2_label} only (n={int(only_pyd.sum())})", zorder=4)
    ax.scatter(xp[both], yp[both], s=34, color="#029E73", alpha=0.9, edgecolor="black",
               linewidth=0.4, label=f"DE in both (n={int(both.sum())})", zorder=5)

    rho_all = float(stats.spearmanr(xp[ok], yp[ok]).statistic) if ok.sum() >= 5 else float("nan")
    de_any = (sig_pdex | sig_pyd) & ok
    rho_de = float(stats.spearmanr(xp[de_any], yp[de_any]).statistic) if de_any.sum() >= 5 else float("nan")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"{pdex_label} log2FC (control-control null)")
    ax.set_ylabel(f"{pydeseq2_label} log2FC (control-control null)")
    ax.set_title("Test 2 — per-gene LFC agreement on the control-control null\n"
                 f"pooled over splits · Spearman ρ (all genes)={rho_all:.2f}"
                 + (f", ρ (DE in either)={rho_de:.2f}" if np.isfinite(rho_de) else "")
                 + f"\nDE call = FDR ≤ {fdr:g} AND |log2FC| ≥ {lfc:g}"
                 + "\n(true null ⇒ cloud at origin; off-origin points are false positives)", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True, help="input perturbation .h5ad path")
    ap.add_argument("--methods", default="pdex,pydeseq2",
                    help="comma-sep DE backends to compute + compare on the null")
    ap.add_argument("--pert-col", default="gene")
    ap.add_argument("--control", default="non-targeting")
    ap.add_argument("--replicate-col", default="batch")
    ap.add_argument("--block-cols", default="batch")
    ap.add_argument("--counts-layer", default=None, help="raw-counts layer (auto-uses 'counts' if present)")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--lfc", type=float, default=0.1)
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--n-resamples", type=int, default=10)
    ap.add_argument("--threads", type=int, default=8,
                    help="CPU threads supplied to each DE fit")
    ap.add_argument("--resample-workers", type=int, default=1,
                    help="parallel control splits for CPU PyDESeq2; use --threads 1")
    ap.add_argument("--non-parametric-engine", choices=("pdex", "rsc"), default="pdex",
                    help="non-parametric engine; pdex uses Arc pdex and rsc uses RAPIDS GPU Wilcoxon")
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
    if a.threads < 1 or a.resample_workers < 1:
        ap.error("--threads and --resample-workers must be at least 1")
    available_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    )
    worker_cap = max(1, available_cpus // a.threads)
    if a.resample_workers > worker_cap:
        print(f"[safety] capping --resample-workers from {a.resample_workers} "
              f"to {worker_cap} ({available_cpus} CPUs, {a.threads} threads/fit)")
        a.resample_workers = worker_cap
    rt = _load_runner()
    methods = [m for m in a.methods.split(",") if m]
    # run pydeseq2 (needs raw .X) BEFORE pdex (maybe_normalize log-transforms .X in place)
    order = [m for m in methods if m != "pdex"] + (["pdex"] if "pdex" in methods else [])

    adata = ad.read_h5ad(a.adata)
    counts_layer = a.counts_layer
    if counts_layer is None and "counts" in adata.layers:
        counts_layer = "counts"
    print(f"loaded {a.adata}: {adata.n_obs} cells × {adata.n_vars} genes  (methods={methods})")
    dataset = os.path.splitext(os.path.basename(a.adata))[0]
    a.run_root = os.path.abspath(os.path.expanduser(a.run_root or a.outdir))
    write_resolved_config(
        run_root=a.run_root,
        workflow="de_test2_control_null",
        dataset=dataset,
        resolved={
            "arguments": vars(a),
            "methods": methods,
            "effective_counts_layer": counts_layer or "X",
            "results_outdir": os.path.abspath(a.outdir),
        },
    )

    def cfg_for(m):
        return {"pert_col": a.pert_col, "control_pert": a.control, "replicate_col": a.replicate_col,
                "block_cols": [c for c in a.block_cols.split(",") if c], "de_method": m,
                "allow_discrete": m == "pydeseq2", "normalize_if_raw": m == "pdex",
                "non_parametric_engine": a.non_parametric_engine,
                "counts_layer": counts_layer, "min_cells_per_group": a.min_cells,
                "fdr_threshold": a.fdr, "lfc_threshold": a.lfc, "seed": a.seed,
                "n_resamples": a.n_resamples, "num_threads": a.threads}

    os.makedirs(a.outdir, exist_ok=True)
    os.makedirs(os.path.join(a.outdir, "tables"), exist_ok=True)
    res, de_long = {}, {}
    for m in order:
        print(f"\n=== {m} control-control null ({a.n_resamples} splits, seed={a.seed}) ===")
        pooled, lambdas, de = control_null_pvalues(adata, cfg_for(m), rt,
                                                   n_resamples=a.n_resamples, seed=a.seed,
                                                   resample_workers=(
                                                       a.resample_workers if m == "pydeseq2" else 1
                                                   ))
        res[m] = (pooled, lambdas)
        de_long[m] = de
        engine = a.non_parametric_engine if m == "pdex" else "cpu"
        table_path = os.path.join(
            a.outdir, "tables", f"test2_lfc_vectors_{m}__"
            f"{os.path.splitext(os.path.basename(a.adata))[0]}.parquet",
        )
        de.with_columns(
            pl.lit(m).alias("method"), pl.lit(engine).alias("engine")
        ).write_parquet(table_path)
        print(f"  LFC vectors: {os.path.abspath(table_path)}")
        print(f"  pooled λ_GC = {rt.lambda_gc(pooled):.3f}  ·  per-split λ_GC = "
              f"{np.array2string(lambdas, precision=2)}")

    plots_dir = os.path.join(a.outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_png = os.path.join(plots_dir, f"test_2_pvalue_diagnostics__{dataset}.png")
    p_pdex, l_pdex = res.get("pdex", (np.array([]), np.array([])))
    p_pyd, l_pyd = res.get("pydeseq2", (np.array([]), np.array([])))
    plot_test2_diagnostics(p_pdex, p_pyd, l_pdex, l_pyd,
                           alpha=a.fdr, n_resamples=a.n_resamples,
                           output_path=out_png,
                           non_parametric_engine=a.non_parametric_engine)
    print("\nTest-2 diagnostics:", os.path.abspath(out_png))

    # per-gene LFC agreement on the null: pdex log2FC (x) vs pydeseq2 log2FC (y)
    if "pdex" in de_long and "pydeseq2" in de_long:
        lfc_png = os.path.join(plots_dir, f"test_2_lfc_agreement__{dataset}.png")
        plot_test2_lfc_agreement(de_long["pdex"], de_long["pydeseq2"], lfc_png,
                                 fdr=a.fdr, lfc=a.lfc,
                                 non_parametric_engine=a.non_parametric_engine)
        print("Test-2 LFC agreement:", os.path.abspath(lfc_png))


if __name__ == "__main__":
    main()
