#!/usr/bin/env python
"""Pathway Test 4: same-guide split-half reproducibility for both methods."""
from __future__ import annotations

import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd
import pathway_utils as pu


def select_powered_guides(
    eligible: pd.DataFrame,
    *,
    max_guides: int,
    max_guides_per_gene: int,
) -> list[str]:
    """Select independently powered guides, optionally applying explicit caps."""
    if eligible.empty:
        return []
    ranked = eligible.sort_values(
        ["n_cells", "gene", "guide"],
        ascending=[False, True, True],
        kind="stable",
    ).copy()
    if max_guides_per_gene:
        within_gene_rank = ranked.groupby("gene", sort=False).cumcount()
        ranked = ranked[within_gene_rank < max_guides_per_gene]
    if max_guides:
        ranked = ranked.head(max_guides)
    return ranked.guide.astype(str).tolist()


def load_benchmark_subset(args) -> tuple[ad.AnnData, list[str]]:
    """Load independently powered guides plus a reproducible control sample."""
    backed = ad.read_h5ad(args.adata, backed="r")
    obs = backed.obs
    perturbations = obs[args.pert_col].astype(str).to_numpy()
    guides = obs[args.sgrna_col].astype(str).to_numpy()
    valid = ~np.isin(np.char.lower(guides.astype(str)), ["", "nan", "none", "na"])
    noncontrol = (perturbations != args.control) & valid
    guide_frame = pd.DataFrame(
        {"guide": guides[noncontrol], "gene": perturbations[noncontrol]}
    )
    counts = guide_frame.guide.value_counts()
    counts = counts[counts >= 2 * args.min_cells_per_arm]
    guide_gene = (
        guide_frame[guide_frame.guide.isin(counts.index)]
        .groupby("guide").gene
        .agg(lambda values: str(values.mode().iloc[0]))
    )
    eligible = pd.DataFrame(
        {
            "guide": counts.index.astype(str),
            "n_cells": counts.to_numpy(int),
            "gene": guide_gene.loc[counts.index].to_numpy(str),
        }
    )
    selected_guides = select_powered_guides(
        eligible,
        max_guides=args.max_guides,
        max_guides_per_gene=args.max_guides_per_gene,
    )
    if not selected_guides:
        backed.file.close()
        raise ValueError(
            "No non-control guide has at least "
            f"{2 * args.min_cells_per_arm} cells required for both split arms"
        )

    rng = np.random.default_rng(args.seed)
    control_idx = np.flatnonzero(perturbations == args.control)
    if args.max_control and len(control_idx) > args.max_control:
        control_idx = np.sort(rng.choice(control_idx, size=args.max_control, replace=False))
    guide_idx = np.flatnonzero(noncontrol & np.isin(guides, selected_guides))
    selected_idx = np.sort(np.concatenate([control_idx, guide_idx]))
    subset = backed[selected_idx].to_memory()
    backed.file.close()
    if args.gene_symbol_col:
        subset = pu.use_gene_symbol_var_names(subset, args.gene_symbol_col)
    if args.normalize_raw:
        if args.score_layer not in ("", "X"):
            raise ValueError("--normalize-raw requires --score-layer X")
        subset = pu.normalize_total_log1p(subset)
    return subset, selected_guides


def compact_guide_labels(
    perturbations: np.ndarray,
    guides: np.ndarray,
    control: str,
) -> dict[str, str]:
    """Map raw guide constructs to compact GENE.gN labels for plots only."""
    guide_to_gene: dict[str, str] = {}
    for guide in sorted(set(guides[perturbations != control])):
        values = pd.Series(
            perturbations[(guides == guide) & (perturbations != control)]
        )
        if not values.empty:
            guide_to_gene[guide] = str(values.mode().iloc[0])
    by_gene: dict[str, list[str]] = {}
    for guide, gene in guide_to_gene.items():
        by_gene.setdefault(gene, []).append(guide)
    return {
        guide: f"{gene} | g{index}"
        for gene in sorted(by_gene)
        for index, guide in enumerate(sorted(by_gene[gene]), start=1)
    }


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
    parser.add_argument(
        "--expression-state", choices=("raw_counts", "log1p_normalized"), default="",
        help="user-confirmed state of adata.X; recorded in the resolved YAML",
    )
    parser.add_argument(
        "--run-root", default="",
        help="confirmed run root for configs/ and logs/ (defaults to --outdir)",
    )
    parser.add_argument("--pert-col", default="gene")
    parser.add_argument("--sgrna-col", required=True)
    parser.add_argument("--control", default="non-targeting")
    parser.add_argument("--block-cols", default="batch")
    parser.add_argument("--score-layer", default="X")
    parser.add_argument("--gene-symbol-col", default="")
    parser.add_argument("--min-genes", type=int, default=5)
    parser.add_argument("--ctrl-size", type=int, default=50)
    parser.add_argument("--n-bins", type=int, default=25)
    parser.add_argument("--min-cells-per-arm", type=int, default=20)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--max-guides", type=int, default=0)
    parser.add_argument("--max-guides-per-gene", type=int, default=0)
    parser.add_argument("--max-control", type=int, default=1200)
    parser.add_argument("--normalize-raw", action="store_true")
    parser.add_argument("--methods", default="ols,pdex_mwu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pu.configure_bioconcord_root(args.bioconcord_root)
    pu.validate_probability(args.fdr, "--fdr")
    pu.validate_positive_int(args.min_cells_per_arm, "--min-cells-per-arm")
    pu.validate_positive_int(args.n_repeats, "--n-repeats")
    if args.max_guides < 0:
        raise ValueError("--max-guides cannot be negative")
    if args.max_guides_per_gene < 0:
        raise ValueError("--max-guides-per-gene cannot be negative")
    if args.max_control < 0:
        raise ValueError("--max-control cannot be negative")

    pu.clear_output_prefix(args.outdir, "pathways_test4_")
    plots, tables = pu.prepare_output(args.outdir)
    dataset = pu.dataset_name(args.adata)
    args.run_root = os.path.abspath(os.path.expanduser(args.run_root or args.outdir))
    pu.write_resolved_config(
        run_root=args.run_root,
        workflow="pathways_test4_guide_reproducibility",
        dataset=dataset,
        resolved={
            "arguments": vars(args),
            "methods": pu.method_list(args.methods),
            "results_outdir": os.path.abspath(args.outdir),
        },
    )
    subset, selected_guides = load_benchmark_subset(args)
    scored = pu.score_anndata(
        subset, args.programs, score_layer=args.score_layer, min_genes=args.min_genes,
        ctrl_size=args.ctrl_size, n_bins=args.n_bins,
    )
    if args.normalize_raw:
        scored.score_source = "raw X subset -> total-normalized to 10000 -> log1p"
    obs = scored.adata.obs
    pert = obs[args.pert_col].astype(str).to_numpy()
    guide = obs[args.sgrna_col].astype(str).to_numpy()
    guide_display = compact_guide_labels(pert, guide, args.control)
    valid_guide = ~np.isin(np.char.lower(guide.astype(str)), ["", "nan", "none", "na"])
    keep = (pert == args.control) | valid_guide
    guide_labels = np.where(pert == args.control, args.control, guide)
    blocks = pu.composite_blocks(obs, [x.strip() for x in args.block_cols.split(",") if x.strip()])
    methods = pu.method_list(args.methods)

    details: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    cross_method: list[pd.DataFrame] = []
    repeat_results: list[dict[str, dict[str, pd.DataFrame]]] = []
    for repeat in range(args.n_repeats):
        idx_a_local, idx_b_local, _ = pu.split_groups(
            guide_labels[keep], blocks[keep], reference=args.control,
            min_cells_per_arm=args.min_cells_per_arm, seed=args.seed + repeat,
        )
        kept_global = np.flatnonzero(keep)
        idx_a, idx_b = kept_global[idx_a_local], kept_global[idx_b_local]
        split_results = {}
        for arm, idx in (("A", idx_a), ("B", idx_b)):
            result = pu.run_methods(
                scored.scores[idx], guide_labels[idx], args.control, scored.program_labels,
                methods=methods, threads=args.threads,
            )
            split_results[arm] = result
            for method, frame in result.items():
                tagged = frame.copy()
                tagged["repeat"] = repeat
                tagged["arm"] = arm
                details.append(tagged)
            if set(methods) == set(pu.METHODS):
                comp = pu.compare_methods(result, fdr=args.fdr)
                comp["repeat"] = repeat
                comp["arm"] = arm
                cross_method.append(comp)
        repeat_results.append(split_results)
        for method in methods:
            metric = pu.split_metrics(
                split_results["A"][method], split_results["B"][method], args.fdr
            )
            metric["method"] = method
            metric["repeat"] = repeat
            metrics.append(metric)

    detail_df = pd.concat(details, ignore_index=True)
    metric_df = pd.concat(metrics, ignore_index=True)
    detail_df.to_csv(os.path.join(tables, f"pathways_test4_results__{dataset}.csv"), index=False)
    metric_df.to_csv(os.path.join(tables, f"pathways_test4_reproducibility__{dataset}.csv"), index=False)
    if cross_method:
        pd.concat(cross_method, ignore_index=True).to_csv(
            os.path.join(tables, f"pathways_test4_crossmethod__{dataset}.csv"), index=False
        )
    guide_counts = pd.Series(guide[pert != args.control]).value_counts()
    pd.DataFrame(
        [
            {
                "raw_guide": raw_guide,
                "target_gene": plot_label.rsplit(" | g", 1)[0],
                "plot_label": plot_label,
                "n_cells": int(guide_counts.get(raw_guide, 0)),
            }
            for raw_guide, plot_label in sorted(guide_display.items())
        ]
    ).to_csv(
        os.path.join(tables, f"pathways_test4_guide_map__{dataset}.csv"),
        index=False,
    )
    plot_repeat_results = []
    for result in repeat_results:
        display_result = {}
        for arm in ("A", "B"):
            display_result[arm] = {}
            for method in methods:
                frame = result[arm][method].copy()
                frame["target"] = frame.target.map(guide_display).fillna(frame.target)
                display_result[arm][method] = frame
        plot_repeat_results.append(display_result)
    mean_results = pu.average_split_effects(plot_repeat_results, methods)
    pu.plot_target_corr_matrix(
        plot_repeat_results,
        os.path.join(plots, f"pathways_test4_corr_matrix_mean__{dataset}.png"),
        f"Pathway Test 4 mean across {args.n_repeats} repeats — {dataset}",
        unit="guide",
        sort_by_diagonal=False,
        group_separator=" | g",
    )
    for method in methods:
        pu.plot_target_corr_matrix(
            plot_repeat_results,
            os.path.join(
                plots, f"pathways_test4_corr_matrix_mean_{method}__{dataset}.png"
            ),
            f"Pathway Test 4 mean across {args.n_repeats} repeats — "
            f"{dataset} — {method}",
            unit="guide",
            sort_by_diagonal=False,
            group_separator=" | g",
            methods_to_plot=(method,),
        )
    fdr_label = fdr_filename_label(args.fdr)
    pu.plot_target_corr_matrix(
        plot_repeat_results,
        os.path.join(
            plots,
            f"pathways_test4_corr_matrix_mean_{fdr_label}__{dataset}.png",
        ),
        f"Pathway Test 4 mean across {args.n_repeats} repeats — "
        f"{dataset} — shared FDR <= {args.fdr:g} union",
        unit="guide",
        sort_by_diagonal=False,
        group_separator=" | g",
        fdr_threshold=args.fdr,
        emit_distribution_boxplots=True,
    )
    for method in methods:
        pu.plot_target_corr_matrix(
            plot_repeat_results,
            os.path.join(
                plots,
                f"pathways_test4_corr_matrix_mean_{method}_{fdr_label}"
                f"__{dataset}.png",
            ),
            f"Pathway Test 4 mean across {args.n_repeats} repeats — "
            f"{dataset} — {method} — shared FDR <= {args.fdr:g} union",
            unit="guide",
            sort_by_diagonal=False,
            group_separator=" | g",
            methods_to_plot=(method,),
            fdr_threshold=args.fdr,
        )
    for arm in ("A", "B"):
        pu.plot_effect_heatmaps(
            mean_results[arm],
            os.path.join(plots, f"pathways_test4_mean_arm{arm.lower()}__{dataset}.png"),
            f"Pathway Test 4 mean of {args.n_repeats} repeats, arm {arm} — {dataset}",
            group_separator=" | g",
            unit_label="guide",
        )
    pu.save_program_legend(scored.programs, os.path.join(tables, f"pathways_legend__{dataset}.csv"))
    pu.save_run_metadata(
        os.path.join(args.outdir, f"pathways_test4_metadata__{dataset}.json"),
        scored,
        args,
        extra={
            "selected_guides": selected_guides,
            "plot_guide_labels": guide_display,
        },
    )


if __name__ == "__main__":
    main()
