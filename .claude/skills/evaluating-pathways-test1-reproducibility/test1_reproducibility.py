#!/usr/bin/env python
"""Pathway Test 1: perturbation split-half reproducibility for OLS and pdex-MWU."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pathway_utils as pu


def safe_pearson(left, right) -> float:
    """Finite, nonconstant Pearson correlation matching safe_spearman guards."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    keep = np.isfinite(left) & np.isfinite(right)
    if (
        keep.sum() < 3
        or np.unique(left[keep]).size < 2
        or np.unique(right[keep]).size < 2
    ):
        return float("nan")
    return float(np.corrcoef(left[keep], right[keep])[0, 1])


def diagonal_offdiagonal_means(matrix: np.ndarray) -> tuple[float, float]:
    """Return finite means for matching and nonmatching target pairs."""
    matrix = np.asarray(matrix, dtype=float)
    diagonal_mask = np.eye(matrix.shape[0], dtype=bool)
    return (
        float(np.nanmean(matrix[diagonal_mask])),
        float(np.nanmean(matrix[~diagonal_mask])),
    )


def split_metrics(left: pd.DataFrame, right: pd.DataFrame, fdr: float) -> pd.DataFrame:
    rows = []
    for target in sorted(set(left.target) & set(right.target)):
        a = left[left.target == target].set_index("program")
        b = right[right.target == target].set_index("program")
        programs = sorted(set(a.index) & set(b.index))
        av = a.loc[programs, "effect"].to_numpy(float)
        bv = b.loc[programs, "effect"].to_numpy(float)
        sa = set(a.index[a.fdr <= fdr])
        sb = set(b.index[b.fdr <= fdr])
        active = np.asarray([(p in sa or p in sb) for p in programs])
        rows.append(
            {
                "target": target,
                "n_cells_total": int(a.n_target.iloc[0] + b.n_target.iloc[0]),
                "effect_spearman": pu.safe_spearman(av, bv),
                "sig_jaccard": pu.jaccard(sa, sb),
                "direction_agreement_sig_union": (
                    float(np.mean(np.sign(av[active]) == np.sign(bv[active]))) if active.any() else np.nan
                ),
                "n_sig_a": len(sa),
                "n_sig_b": len(sb),
            }
        )
    return pd.DataFrame(rows)


def plot_target_corr_matrix(
    repeat_results: list[dict[str, dict[str, pd.DataFrame]]],
    path: str,
    title: str,
    unit: str = "perturbation",
    sort_by_diagonal: bool = True,
    group_separator: str | None = None,
) -> None:
    """Plot the mean split-A × split-B target correlation map.

    Spearman and Pearson maps are calculated independently for every repeat,
    then averaged cell by cell. Colors encode repeat-wise Spearman; panel
    titles report mean diagonal and off-diagonal values for both metrics. Every
    matrix cell uses all aligned pathway coefficients; FDR calls do not select
    the correlation feature set.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not repeat_results:
        raise ValueError("At least one repeat is required for a correlation map")
    methods = list(repeat_results[0]["A"])
    fig, axes = plt.subplots(
        1, len(methods), figsize=(7.2 * len(methods), 6.8), squeeze=False
    )
    matrix_archive = {
        "methods": np.asarray(methods, dtype=str),
    }
    for ax, method in zip(axes[0], methods):
        target_sets = []
        program_sets = []
        for result in repeat_results:
            left = pu.effect_matrix(result["A"][method])
            right = pu.effect_matrix(result["B"][method])
            target_sets.append(set(left.index) & set(right.index))
            program_sets.append(set(left.columns) & set(right.columns))
        targets = sorted(set.intersection(*target_sets))
        programs = sorted(set.intersection(*program_sets))

        repeat_corrs = []
        repeat_pearsons = []
        cell_counts = {target: [] for target in targets}
        for result in repeat_results:
            left_frame = result["A"][method]
            right_frame = result["B"][method]
            left = pu.effect_matrix(left_frame).loc[targets, programs]
            right = pu.effect_matrix(right_frame).loc[targets, programs]
            spearman_matrix = np.full((len(targets), len(targets)), np.nan)
            pearson_matrix = np.full((len(targets), len(targets)), np.nan)
            for i, target_a in enumerate(targets):
                for j, target_b in enumerate(targets):
                    effect_a = left.loc[target_a].to_numpy(float)
                    effect_b = right.loc[target_b].to_numpy(float)
                    spearman_matrix[i, j] = pu.safe_spearman(effect_a, effect_b)
                    pearson_matrix[i, j] = safe_pearson(effect_a, effect_b)
            repeat_corrs.append(spearman_matrix)
            repeat_pearsons.append(pearson_matrix)
            n_left = left_frame.groupby("target")["n_target"].first()
            n_right = right_frame.groupby("target")["n_target"].first()
            for target in targets:
                cell_counts[target].append(float(n_left[target] + n_right[target]))
        corr = np.nanmean(np.stack(repeat_corrs), axis=0)
        pearson_corr = np.nanmean(np.stack(repeat_pearsons), axis=0)
        diagonal = np.diag(corr)
        order = (
            np.argsort(np.where(np.isfinite(diagonal), diagonal, np.inf))
            if sort_by_diagonal
            else np.arange(len(targets))
        )
        corr = corr[np.ix_(order, order)]
        pearson_corr = pearson_corr[np.ix_(order, order)]
        ordered_targets = [targets[i] for i in order]

        sns.heatmap(
            corr,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            xticklabels=ordered_targets,
            yticklabels=ordered_targets,
            cbar_kws={
                "label": "Mean repeat-wise Spearman over all pathway coefficients",
                "shrink": 0.75,
            },
            ax=ax,
        )
        for i, target in enumerate(ordered_targets):
            n_cells = int(round(float(np.mean(cell_counts[target]))))
            ax.text(
                i + 0.5,
                i + 0.5,
                str(n_cells),
                ha="center",
                va="center",
                fontsize=5,
                color="white",
                fontweight="bold",
            )
        if group_separator:
            groups = [target.split(group_separator, 1)[0] for target in ordered_targets]
            for boundary in range(1, len(groups)):
                if groups[boundary] != groups[boundary - 1]:
                    ax.axhline(boundary, color="black", linewidth=1.1)
                    ax.axvline(boundary, color="black", linewidth=1.1)
        spearman_diagonal, spearman_offdiagonal = diagonal_offdiagonal_means(corr)
        pearson_diagonal, pearson_offdiagonal = diagonal_offdiagonal_means(pearson_corr)
        matrix_archive[f"targets__{method}"] = np.asarray(ordered_targets, dtype=str)
        matrix_archive[f"spearman__{method}"] = corr
        matrix_archive[f"pearson__{method}"] = pearson_corr
        ax.set_title(
            f"{method}\n"
            f"Spearman: diag={spearman_diagonal:.2f}, off={spearman_offdiagonal:.2f}; "
            f"Pearson: diag={pearson_diagonal:.2f}, off={pearson_offdiagonal:.2f}",
            fontsize=9,
        )
        ax.set_xlabel(f"split B {unit}")
        ax.set_ylabel(f"split A {unit}")
        ax.tick_params(axis="x", labelsize=6, rotation=90)
        ax.tick_params(axis="y", labelsize=6, rotation=0)

    fig.suptitle(
        f"{title}\n"
        f"diagonal = within-{unit} reproducibility (cell count annotated); "
        f"off-diagonal = cross-{unit} similarity; all aligned pathway coefficients used",
        fontsize=11,
    )
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    matrix_path = os.path.splitext(path)[0] + ".npz"
    np.savez_compressed(matrix_path, **matrix_archive)
    print(f"Saved correlation values: {matrix_path}")


def average_split_effects(
    repeat_results: list[dict[str, dict[str, pd.DataFrame]]],
    methods: tuple[str, ...],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Average each arm's target-by-program effects over all repeats."""
    averaged: dict[str, dict[str, pd.DataFrame]] = {"A": {}, "B": {}}
    for arm in ("A", "B"):
        for method in methods:
            combined = pd.concat(
                [result[arm][method] for result in repeat_results], ignore_index=True
            )
            frame = (
                combined.groupby(["target", "program"], as_index=False)
                .agg(
                    effect=("effect", "mean"),
                    n_target=("n_target", "mean"),
                    n_reference=("n_reference", "mean"),
                )
            )
            frame.insert(0, "method", method)
            averaged[arm][method] = frame
    return averaged


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
    parser.add_argument("--pert-col", default="gene")
    parser.add_argument("--control", default="non-targeting")
    parser.add_argument("--block-cols", default="batch")
    parser.add_argument("--score-layer", default="X")
    parser.add_argument("--min-genes", type=int, default=5)
    parser.add_argument("--ctrl-size", type=int, default=50)
    parser.add_argument("--n-bins", type=int, default=25)
    parser.add_argument("--min-cells-per-arm", type=int, default=20)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--methods", default="ols,pdex_mwu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pu.configure_bioconcord_root(args.bioconcord_root)
    pu.validate_probability(args.fdr, "--fdr")
    pu.validate_positive_int(args.min_cells_per_arm, "--min-cells-per-arm")
    pu.validate_positive_int(args.n_repeats, "--n-repeats")

    pu.clear_output_prefix(args.outdir, "pathways_test1_")
    plots, tables = pu.prepare_output(args.outdir)
    dataset = pu.dataset_name(args.adata)
    scored = pu.load_and_score(
        args.adata, args.programs, score_layer=args.score_layer, min_genes=args.min_genes,
        ctrl_size=args.ctrl_size, n_bins=args.n_bins,
    )
    labels = scored.adata.obs[args.pert_col].astype(str).to_numpy()
    blocks = pu.composite_blocks(
        scored.adata.obs, [x.strip() for x in args.block_cols.split(",") if x.strip()]
    )
    methods = pu.method_list(args.methods)
    details: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    cross_method: list[pd.DataFrame] = []
    repeat_results: list[dict[str, dict[str, pd.DataFrame]]] = []

    for repeat in range(args.n_repeats):
        idx_a, idx_b, _ = pu.split_groups(
            labels, blocks, reference=args.control,
            min_cells_per_arm=args.min_cells_per_arm, seed=args.seed + repeat,
        )
        split_results = {}
        for arm, idx in (("A", idx_a), ("B", idx_b)):
            result = pu.run_methods(
                scored.scores[idx], labels[idx], args.control, scored.program_labels,
                methods=methods, threads=args.threads,
            )
            split_results[arm] = result
            for method, frame in result.items():
                tagged = frame.copy()
                tagged["repeat"] = repeat
                tagged["arm"] = arm
                details.append(tagged)
            if set(methods) == set(pu.METHODS):
                comparison = pu.compare_methods(result, fdr=args.fdr)
                comparison["repeat"] = repeat
                comparison["arm"] = arm
                cross_method.append(comparison)
        repeat_results.append(split_results)
        for method in methods:
            metric = split_metrics(split_results["A"][method], split_results["B"][method], args.fdr)
            metric["method"] = method
            metric["repeat"] = repeat
            metrics.append(metric)

    detail_df = pd.concat(details, ignore_index=True)
    metric_df = pd.concat(metrics, ignore_index=True)
    detail_df.to_csv(os.path.join(tables, f"pathways_test1_results__{dataset}.csv"), index=False)
    metric_df.to_csv(os.path.join(tables, f"pathways_test1_reproducibility__{dataset}.csv"), index=False)
    if cross_method:
        pd.concat(cross_method, ignore_index=True).to_csv(
            os.path.join(tables, f"pathways_test1_crossmethod__{dataset}.csv"), index=False
        )
    mean_results = average_split_effects(repeat_results, methods)
    plot_target_corr_matrix(
        repeat_results,
        os.path.join(plots, f"pathways_test1_corr_matrix_mean__{dataset}.png"),
        f"Pathway Test 1 mean across {args.n_repeats} repeats — {dataset}",
    )
    for method in methods:
        method_results = [
            {arm: {method: result[arm][method]} for arm in ("A", "B")}
            for result in repeat_results
        ]
        plot_target_corr_matrix(
            method_results,
            os.path.join(
                plots, f"pathways_test1_corr_matrix_mean_{method}__{dataset}.png"
            ),
            f"Pathway Test 1 mean across {args.n_repeats} repeats — {dataset} — {method}",
        )
    for arm in ("A", "B"):
        pu.plot_effect_heatmaps(
            mean_results[arm],
            os.path.join(plots, f"pathways_test1_mean_arm{arm.lower()}__{dataset}.png"),
            f"Pathway Test 1 mean of {args.n_repeats} repeats, arm {arm} — {dataset}",
        )
    pu.save_program_legend(scored.programs, os.path.join(tables, f"pathways_legend__{dataset}.csv"))
    pu.save_run_metadata(os.path.join(args.outdir, f"pathways_test1_metadata__{dataset}.json"), scored, args)


if __name__ == "__main__":
    main()
