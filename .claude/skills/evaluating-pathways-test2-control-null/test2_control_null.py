#!/usr/bin/env python
"""Pathway Test 2: repeated control-control null calibration."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pathway_utils as pu


METHOD_COLOR = {"pdex_mwu": "#0173B2", "ols": "#DE8F05"}
METHOD_LABEL = {"pdex_mwu": "pdex MWU", "ols": "OLS"}


def _qq_points(p_values: np.ndarray, max_points: int = 3000) -> tuple[np.ndarray, np.ndarray, int]:
    """Return log-tail-subsampled expected and observed -log10(p)."""
    p = np.sort(np.asarray(p_values, float)[np.isfinite(p_values)])
    p = np.clip(p, 1e-300, 1.0)
    n = p.size
    ranks = np.arange(1, n + 1)
    expected = -np.log10((ranks - 0.5) / n)
    observed = -np.log10(p)
    if n > max_points:
        idx = np.unique(np.round(np.logspace(0, np.log10(n), max_points)).astype(int)) - 1
        idx = idx[(idx >= 0) & (idx < n)]
        expected, observed = expected[idx], observed[idx]
    return expected, observed, n


def plot_diagnostics(detail: pd.DataFrame, summary: pd.DataFrame, path: str, alpha: float, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    method_order = list(dict.fromkeys(detail["method"].astype(str)))
    methods = [
        (
            method,
            detail.loc[detail.method == method, "p_value"].to_numpy(float),
            summary.loc[summary.method == method, "lambda_gc"].to_numpy(float),
        )
        for method in method_order
    ]
    fig, axes = plt.subplots(
        1, 3, figsize=(14, 5), layout="constrained",
        gridspec_kw={"width_ratios": [1.25, 1.15, 0.9]},
    )

    ax = axes[0]
    n_env = max(np.isfinite(values).sum() for _, values, _ in methods)
    ranks = np.unique(np.round(np.logspace(0, np.log10(n_env), 400)).astype(int))
    ranks = ranks[(ranks >= 1) & (ranks <= n_env)]
    expected_envelope = -np.log10((ranks - 0.5) / n_env)
    lower = -np.log10(stats.beta.ppf(0.975, ranks, n_env - ranks + 1))
    upper = -np.log10(stats.beta.ppf(0.025, ranks, n_env - ranks + 1))
    ax.fill_between(
        expected_envelope, lower, upper, color="0.87", zorder=0,
        label="95% null envelope",
    )
    limit = float(expected_envelope.max())
    for method, p_values, lambdas in methods:
        expected, observed, n_values = _qq_points(p_values)
        if n_values < 8:
            continue
        limit = max(limit, float(observed.max()))
        mean_lambda = float(np.nanmean(lambdas)) if lambdas.size else float("nan")
        ax.plot(
            expected, observed, lw=1.4, alpha=0.9,
            color=METHOD_COLOR.get(method, "0.3"),
            label=f"{METHOD_LABEL.get(method, method)} (lambda_GC={mean_lambda:.2f})",
        )
    ax.plot([0, limit], [0, limit], "--", color="0.4", lw=1, label="y = x (calibrated)")
    ax.set_xlabel("expected -log10(p) (Uniform)")
    ax.set_ylabel("observed -log10(p)")
    ax.set_title(
        "(1) QQ: points should hug the diagonal\n"
        "above = anti-conservative; below = conservative",
        fontsize=9,
    )
    ax.legend(fontsize=7.5, loc="upper left")

    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    for method, p_values, _ in methods:
        finite = p_values[np.isfinite(p_values)]
        ax.hist(
            finite, bins=bins, density=True, alpha=0.5,
            color=METHOD_COLOR.get(method, "0.3"),
            label=METHOD_LABEL.get(method, method), edgecolor="white", linewidth=0.3,
        )
    ax.axhline(1.0, color="black", ls="--", lw=1.2, label="Uniform[0,1] (density = 1)")
    ax.set_xlabel("p-value")
    ax.set_ylabel("density")
    ax.set_title(
        "(2) Histogram: should be flat at 1.0\n"
        "left spike = inflated; right skew = conservative",
        fontsize=9,
    )
    ax.legend(fontsize=7.5)

    ax = axes[2]
    rng = np.random.default_rng(42)
    positions = list(range(1, len(methods) + 1))
    box_data = [values[np.isfinite(values)] for _, _, values in methods]
    colors = [METHOD_COLOR.get(method, "0.3") for method, _, _ in methods]
    box = ax.boxplot(
        box_data, positions=positions, widths=0.5, patch_artist=True,
        medianprops={"color": "black", "lw": 1.6},
        whiskerprops={"color": "#555"}, capprops={"color": "#555"},
        flierprops={"marker": "", "alpha": 0},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    for position, values, color in zip(positions, box_data, colors):
        jitter = rng.uniform(-0.13, 0.13, size=values.size)
        ax.scatter(
            position + jitter, values, s=30, color=color, edgecolor="white",
            linewidth=0.4, alpha=0.9, zorder=3,
        )
    ax.axhline(1.0, color="#D62728", ls="--", lw=1.3, label="calibrated (lambda=1.0)")
    ax.axhline(0.9, color="0.6", ls=":", lw=1)
    ax.axhline(1.1, color="0.6", ls=":", lw=1, label="+/-0.1 band")
    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_LABEL.get(method, method) for method, _, _ in methods])
    ax.set_ylabel("lambda_GC")
    ax.set_title(
        f"(3) lambda_GC: should be near 1.0\n"
        f"(one point per split, n={summary.repeat.nunique()} splits)",
        fontsize=9,
    )
    ax.legend(fontsize=7.5, loc="best")

    fig.suptitle(f"{title}: OLS vs pdex MWU (nominal alpha={alpha:g})", fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_effect_agreement(detail: pd.DataFrame, path: str, alpha: float, title: str) -> str | None:
    """Plot paired native pathway effects for OLS versus pdex on the null."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    if not {"ols", "pdex_mwu"}.issubset(set(detail.method)):
        return None
    ols = detail[detail.method == "ols"].rename(
        columns={"effect": "effect_ols", "fdr": "fdr_ols"}
    )[["repeat", "target", "program", "effect_ols", "fdr_ols"]]
    pdex = detail[detail.method == "pdex_mwu"].rename(
        columns={"effect": "effect_pdex", "fdr": "fdr_pdex"}
    )[["repeat", "target", "program", "effect_pdex", "fdr_pdex"]]
    paired = ols.merge(pdex, on=["repeat", "target", "program"], how="inner")
    if paired.empty:
        return None

    x = paired.effect_ols.to_numpy(float)
    y = paired.effect_pdex.to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    sig_ols = paired.fdr_ols.to_numpy(float) <= alpha
    sig_pdex = paired.fdr_pdex.to_numpy(float) <= alpha
    both = sig_ols & sig_pdex & ok
    only_ols = sig_ols & ~sig_pdex & ok
    only_pdex = sig_pdex & ~sig_ols & ok
    neither = ~sig_ols & ~sig_pdex & ok

    rho_all = float(stats.spearmanr(x[ok], y[ok]).statistic) if ok.sum() >= 5 else float("nan")
    called = (sig_ols | sig_pdex) & ok
    rho_called = (
        float(stats.spearmanr(x[called], y[called]).statistic)
        if called.sum() >= 5 else float("nan")
    )
    x_limit = max(float(np.nanpercentile(np.abs(x[ok]), 99.8)), 0.05)
    y_limit = max(float(np.nanpercentile(np.abs(y[ok]), 99.8)), 0.05)

    fig, ax = plt.subplots(figsize=(7.2, 7))
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.scatter(
        x[neither], y[neither], s=8, color="0.75", alpha=0.4, linewidths=0,
        label=f"FDR-called by neither (n={int(neither.sum())})", zorder=2,
    )
    ax.scatter(
        x[only_ols], y[only_ols], s=26, color=METHOD_COLOR["ols"], alpha=0.85,
        edgecolor="white", linewidth=0.3,
        label=f"OLS only (n={int(only_ols.sum())})", zorder=4,
    )
    ax.scatter(
        x[only_pdex], y[only_pdex], s=26, color=METHOD_COLOR["pdex_mwu"], alpha=0.85,
        edgecolor="white", linewidth=0.3,
        label=f"pdex MWU only (n={int(only_pdex.sum())})", zorder=4,
    )
    ax.scatter(
        x[both], y[both], s=36, color="#029E73", alpha=0.9,
        edgecolor="black", linewidth=0.4,
        label=f"both (n={int(both.sum())})", zorder=5,
    )
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlabel("OLS coefficient (pseudo-perturbation - control)")
    ax.set_ylabel("pdex MWU rank-biserial effect")
    subtitle = f"Spearman rho (all programs)={rho_all:.2f}"
    if np.isfinite(rho_called):
        subtitle += f", rho (FDR-called by either)={rho_called:.2f}"
    ax.set_title(
        f"{title}: native-effect agreement on the control-control null\n"
        f"pooled over splits; {subtitle}\n"
        f"call = FDR <= {alpha:g}; colored points are false positives\n"
        "axes use different native scales, so a y=x line is not meaningful",
        fontsize=9.5,
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", required=True)
    parser.add_argument("--programs", required=True)
    parser.add_argument(
        "--bioconcord-root",
        default=os.environ.get("BIOCONCORD_ROOT", ""),
        help="official ArcInstitute/bioconcord checkout (or set BIOCONCORD_ROOT)",
    )
    parser.add_argument("--outdir", default=".")
    parser.add_argument(
        "--expression-state", choices=("raw_counts", "log1p_normalized"), default="",
        help="user-confirmed state of adata.X; recorded in the resolved YAML",
    )
    parser.add_argument(
        "--run-root", default="",
        help="confirmed run root for configs/ and logs/ (defaults to --outdir)",
    )
    parser.add_argument("--pert-col", default="gene")
    parser.add_argument("--control", default="non-targeting")
    parser.add_argument("--block-cols", default="batch")
    parser.add_argument("--score-layer", default="X")
    parser.add_argument("--min-genes", type=int, default=5)
    parser.add_argument("--ctrl-size", type=int, default=50)
    parser.add_argument("--n-bins", type=int, default=25)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--min-cells-per-arm", type=int, default=20)
    parser.add_argument("--methods", default="ols,pdex_mwu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pu.configure_bioconcord_root(args.bioconcord_root)
    pu.validate_probability(args.alpha, "--alpha")
    pu.validate_positive_int(args.min_cells_per_arm, "--min-cells-per-arm")
    pu.validate_positive_int(args.n_repeats, "--n-repeats")

    pu.clear_output_prefix(args.outdir, "pathways_test2_")
    plots, tables = pu.prepare_output(args.outdir)
    dataset = pu.dataset_name(args.adata)
    args.run_root = os.path.abspath(os.path.expanduser(args.run_root or args.outdir))
    pu.write_resolved_config(
        run_root=args.run_root,
        workflow="pathways_test2_control_null",
        dataset=dataset,
        resolved={
            "arguments": vars(args),
            "methods": pu.method_list(args.methods),
            "results_outdir": os.path.abspath(args.outdir),
        },
    )
    scored = pu.load_and_score(
        args.adata, args.programs, score_layer=args.score_layer, min_genes=args.min_genes,
        ctrl_size=args.ctrl_size, n_bins=args.n_bins,
    )
    labels = scored.adata.obs[args.pert_col].astype(str).to_numpy()
    ctrl_idx = np.flatnonzero(labels == args.control)
    blocks = pu.composite_blocks(
        scored.adata.obs, [x.strip() for x in args.block_cols.split(",") if x.strip()]
    )
    methods = pu.method_list(args.methods)
    detail: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for repeat in range(args.n_repeats):
        left, right = pu.stratified_half(ctrl_idx, blocks, np.random.default_rng(args.seed + repeat))
        if min(left.size, right.size) < args.min_cells_per_arm:
            raise ValueError("A control split fell below --min-cells-per-arm")
        idx = np.concatenate([left, right])
        split_labels = np.asarray([args.control] * left.size + ["pseudo_perturbation"] * right.size)
        results = pu.run_methods(
            scored.scores[idx], split_labels, args.control, scored.program_labels,
            methods=methods, threads=args.threads,
        )
        for method, frame in results.items():
            tagged = frame.copy()
            tagged["repeat"] = repeat
            detail.append(tagged)
            summaries.append(
                {
                    "repeat": repeat,
                    "method": method,
                    "lambda_gc": pu.lambda_gc(frame.p_value),
                    "fraction_p_below_alpha": float((frame.p_value <= args.alpha).mean()),
                    "fraction_fdr_below_alpha": float((frame.fdr <= args.alpha).mean()),
                    "median_p": float(frame.p_value.median()),
                    "mean_abs_descriptive_difference": float(frame.mean_difference.abs().mean()),
                }
            )
    detail_df = pd.concat(detail, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    detail_df.to_csv(os.path.join(tables, f"pathways_test2_null_results__{dataset}.csv"), index=False)
    summary_df.to_csv(os.path.join(tables, f"pathways_test2_null_summary__{dataset}.csv"), index=False)
    plot_diagnostics(
        detail_df, summary_df,
        os.path.join(plots, f"pathways_test2_null_diagnostics__{dataset}.png"),
        args.alpha, f"Pathway Test 2 — {dataset}",
    )
    if set(methods) == set(pu.METHODS):
        plot_effect_agreement(
            detail_df,
            os.path.join(plots, f"pathways_test2_effect_agreement__{dataset}.png"),
            args.alpha,
            f"Pathway Test 2 — {dataset}",
        )
    pu.save_program_legend(scored.programs, os.path.join(tables, f"pathways_legend__{dataset}.csv"))
    pu.save_run_metadata(os.path.join(args.outdir, f"pathways_test2_metadata__{dataset}.json"), scored, args)


if __name__ == "__main__":
    main()
