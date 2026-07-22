#!/usr/bin/env python
"""Pathway Test 1: perturbation split-half reproducibility for OLS and pdex-MWU."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pathway_utils as pu


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


def fdr_filename_label(threshold: float) -> str:
    """Return a stable filename label such as ``fdr05`` for 0.05."""
    percent = f"{100 * threshold:g}".replace(".", "p")
    if "p" not in percent:
        percent = percent.zfill(2)
    return f"fdr{percent}"


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
    pu.plot_target_corr_matrix(
        repeat_results,
        os.path.join(plots, f"pathways_test1_corr_matrix_mean__{dataset}.png"),
        f"Pathway Test 1 mean across {args.n_repeats} repeats — {dataset}",
    )
    for method in methods:
        pu.plot_target_corr_matrix(
            repeat_results,
            os.path.join(
                plots, f"pathways_test1_corr_matrix_mean_{method}__{dataset}.png"
            ),
            f"Pathway Test 1 mean across {args.n_repeats} repeats — {dataset} — {method}",
            methods_to_plot=(method,),
        )
    fdr_label = fdr_filename_label(args.fdr)
    pu.plot_target_corr_matrix(
        repeat_results,
        os.path.join(
            plots, f"pathways_test1_corr_matrix_mean_{fdr_label}__{dataset}.png"
        ),
        f"Pathway Test 1 mean across {args.n_repeats} repeats — {dataset} — "
        f"shared FDR <= {args.fdr:g} union",
        fdr_threshold=args.fdr,
    )
    for method in methods:
        pu.plot_target_corr_matrix(
            repeat_results,
            os.path.join(
                plots,
                f"pathways_test1_corr_matrix_mean_{method}_{fdr_label}__{dataset}.png",
            ),
            f"Pathway Test 1 mean across {args.n_repeats} repeats — {dataset} — "
            f"{method} — shared FDR <= {args.fdr:g} union",
            methods_to_plot=(method,),
            fdr_threshold=args.fdr,
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
