#!/usr/bin/env python
"""Run both pathway inference methods on the full perturbation dataset."""

from __future__ import annotations

import argparse
import os

import pathway_utils as pu


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
    parser.add_argument(
        "--umi-col",
        default="UMI_count",
        help="per-cell UMI-count column used for perturbation-level QC plots",
    )
    parser.add_argument(
        "--score-layer", default="X", help="log-normalized expression layer or X"
    )
    parser.add_argument(
        "--counts-layer",
        default="counts",
        help="raw-count layer used only for expression-versus-effect plots",
    )
    parser.add_argument("--min-genes", type=int, default=5)
    parser.add_argument("--ctrl-size", type=int, default=50)
    parser.add_argument("--n-bins", type=int, default=25)
    parser.add_argument("--methods", default="ols,pdex_mwu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--fdr", type=float, default=0.05)
    args = parser.parse_args()
    pu.configure_bioconcord_root(args.bioconcord_root)
    pu.validate_probability(args.fdr, "--fdr")
    pu.validate_positive_int(args.min_genes, "--min-genes")

    pu.clear_output_prefix(args.outdir, "pathways_overview_")
    pu.clear_output_prefix(args.outdir, "pathways_run_metadata__")
    plots, tables = pu.prepare_output(args.outdir)
    dataset = pu.dataset_name(args.adata)
    args.run_root = os.path.abspath(os.path.expanduser(args.run_root or args.outdir))
    pu.write_resolved_config(
        run_root=args.run_root,
        workflow="pathways_overview",
        dataset=dataset,
        resolved={
            "arguments": vars(args),
            "methods": pu.method_list(args.methods),
            "results_outdir": os.path.abspath(args.outdir),
        },
    )
    scored = pu.load_and_score(
        args.adata,
        args.programs,
        score_layer=args.score_layer,
        min_genes=args.min_genes,
        ctrl_size=args.ctrl_size,
        n_bins=args.n_bins,
    )
    if args.pert_col not in scored.adata.obs:
        raise ValueError(f"{args.pert_col!r} is absent from adata.obs")
    labels = scored.adata.obs[args.pert_col].astype(str).to_numpy()
    methods = pu.method_list(args.methods)
    results = pu.run_methods(
        scored.scores,
        labels,
        args.control,
        scored.program_labels,
        methods=methods,
        threads=args.threads,
    )
    for method, frame in results.items():
        frame.to_csv(
            os.path.join(tables, f"pathways_overview_{method}__{dataset}.csv"),
            index=False,
        )

    targets = sorted(set().union(*(set(frame["target"]) for frame in results.values())))
    covariates = pu.perturbation_covariates(
        scored.adata.obs,
        pert_col=args.pert_col,
        umi_col=args.umi_col,
        targets=targets,
    )
    covariates.to_csv(
        os.path.join(
            tables, f"pathways_overview_perturbation_covariates__{dataset}.csv"
        ),
        index=False,
    )
    expression_effects = pu.pathway_expression_effects(
        scored.adata,
        results,
        scored.programs,
        pert_col=args.pert_col,
        counts_layer=args.counts_layer,
        covariates=covariates,
        fdr=args.fdr,
    )
    expression_effects.to_csv(
        os.path.join(
            tables, f"pathways_overview_expression_effects__{dataset}.csv"
        ),
        index=False,
    )
    for method in methods:
        pu.plot_expression_effect_clouds(
            expression_effects,
            method=method,
            path=os.path.join(
                plots,
                f"pathways_overview_expression_effects_{method}__{dataset}.png",
            ),
            title=f"Pathway expression versus effect — {dataset}",
            fdr=args.fdr,
        )

    pu.plot_effect_heatmaps(
        results,
        os.path.join(plots, f"pathways_overview_effects__{dataset}.png"),
        f"Pathway effects by perturbation — {dataset}",
    )
    pu.plot_effect_distributions_by_covariate(
        results,
        covariates,
        order_by="n_cells",
        path=os.path.join(
            plots, f"pathways_overview_effects_by_cell_count__{dataset}.png"
        ),
        title=f"Pathway effects ordered by perturbation cell count — {dataset}",
    )
    pu.plot_effect_distributions_by_covariate(
        results,
        covariates,
        order_by="median_umi_per_cell",
        path=os.path.join(
            plots, f"pathways_overview_effects_by_umi_count__{dataset}.png"
        ),
        title=f"Pathway effects ordered by median UMI count — {dataset}",
    )
    if set(methods) == set(pu.METHODS):
        summary = pu.compare_methods(results, fdr=args.fdr)
        summary.to_csv(
            os.path.join(tables, f"pathways_overview_comparison__{dataset}.csv"),
            index=False,
        )
        pu.plot_method_count_scatter(
            summary,
            os.path.join(plots, f"pathways_overview_method_comparison__{dataset}.png"),
            f"Significant pathway calls — {dataset} (FDR ≤ {args.fdr:g})",
        )

    pu.save_program_legend(
        scored.programs, os.path.join(tables, f"pathways_legend__{dataset}.csv")
    )
    pu.save_run_metadata(
        os.path.join(args.outdir, f"pathways_run_metadata__{dataset}.json"),
        scored,
        args,
    )


if __name__ == "__main__":
    main()
