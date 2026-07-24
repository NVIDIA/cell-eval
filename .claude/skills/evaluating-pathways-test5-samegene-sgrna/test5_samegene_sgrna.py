#!/usr/bin/env python
"""Pathway Test 5: same-gene versus unrelated-guide pathway concordance."""
from __future__ import annotations

import argparse
import itertools
import os

import anndata as ad
import numpy as np
import pandas as pd
import pathway_utils as pu
from scipy import stats

def load_samegene_subset(args) -> tuple[ad.AnnData, list[str], list[str]]:
    """Select powered same-gene guides and controls before pathway scoring."""
    backed = ad.read_h5ad(args.adata, backed="r")
    obs = backed.obs
    targets = obs[args.pert_col].astype(str).to_numpy()
    guides = obs[args.sgrna_col].astype(str).to_numpy()
    valid = ~np.isin(np.char.lower(guides.astype(str)), ["", "nan", "none", "na"])
    noncontrol = (targets != args.control) & valid
    counts = pd.Series(guides[noncontrol]).value_counts()
    eligible = set(counts[counts >= args.min_cells].index.astype(str))

    guide_to_gene: dict[str, str] = {}
    for guide in sorted(eligible):
        values = pd.Series(targets[noncontrol & (guides == guide)])
        if not values.empty:
            guide_to_gene[guide] = str(values.mode().iloc[0])
    by_gene: dict[str, list[str]] = {}
    for guide, gene in guide_to_gene.items():
        by_gene.setdefault(gene, []).append(guide)
    candidate_genes = [gene for gene, members in by_gene.items() if len(members) >= args.min_guides]
    candidate_genes.sort(
        key=lambda gene: (
            -len(by_gene[gene]),
            -sum(int(counts[guide]) for guide in by_gene[gene]),
            gene,
        )
    )
    selected_genes = candidate_genes[: args.max_genes] if args.max_genes else candidate_genes
    selected_guides = sorted(
        guide for gene in selected_genes for guide in by_gene[gene]
    )
    if not selected_guides:
        backed.file.close()
        raise ValueError("No target gene has the requested number of powered guides")

    rng = np.random.default_rng(args.seed)
    control_idx = np.flatnonzero(targets == args.control)
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
    return subset, selected_genes, selected_guides


def guide_order(guide_to_gene: dict[str, str]) -> tuple[list[str], list[str], list[int]]:
    """Return gene-grouped guides, compact display labels, and block boundaries."""
    by_gene: dict[str, list[str]] = {}
    for guide, gene in guide_to_gene.items():
        by_gene.setdefault(gene, []).append(guide)
    guides: list[str] = []
    labels: list[str] = []
    boundaries: list[int] = []
    for gene in sorted(by_gene):
        members = sorted(by_gene[gene])
        for index, guide in enumerate(members, start=1):
            guides.append(guide)
            labels.append(f"{gene} | g{index}")
        boundaries.append(len(guides))
    return guides, labels, boundaries[:-1]


def plot_guide_heatmap(
    frame: pd.DataFrame,
    guide_to_gene: dict[str, str],
    path: str,
    title: str,
    method: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    guides, labels, boundaries = guide_order(guide_to_gene)
    matrix = pu.effect_matrix(frame).loc[guides]
    programs = matrix.mean(axis=0).sort_values().index
    matrix = matrix.loc[:, programs]
    vmax = 1.0 if method == "pdex_mwu" else float(np.nanmax(np.abs(matrix.to_numpy()))) or 1.0
    fig, ax = plt.subplots(
        figsize=(max(12, matrix.shape[1] * 0.16), max(6, matrix.shape[0] * 0.32))
    )
    sns.heatmap(
        matrix,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        yticklabels=labels,
        cbar_kws={"label": "native pathway effect"},
        ax=ax,
    )
    for boundary in boundaries:
        ax.axhline(boundary, color="black", lw=1.2)
    ax.set_title(title)
    ax.set_xlabel("pathway (ordered by mean guide effect)")
    ax.set_ylabel("guide (grouped by target gene)")
    ax.tick_params(axis="x", labelsize=5, rotation=90)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_guide_corr_matrix(
    results: dict[str, pd.DataFrame],
    guide_to_gene: dict[str, str],
    path: str,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    guides, labels, boundaries = guide_order(guide_to_gene)
    fig, axes = plt.subplots(
        1, len(results), figsize=(7.2 * len(results), 6.8), squeeze=False
    )
    for ax, (method, frame) in zip(axes[0], results.items()):
        matrix = pu.effect_matrix(frame).loc[guides]
        corr = np.asarray(
            [
                [pu.safe_spearman(matrix.loc[left], matrix.loc[right]) for right in guides]
                for left in guides
            ],
            dtype=float,
        )
        np.fill_diagonal(corr, 1.0)
        diagonal_mask = np.eye(len(guides), dtype=bool)
        mean_diagonal = float(np.nanmean(corr[diagonal_mask]))
        mean_offdiagonal = float(np.nanmean(corr[~diagonal_mask]))
        within = []
        cross = []
        for i, left in enumerate(guides):
            for j in range(i + 1, len(guides)):
                bucket = within if guide_to_gene[left] == guide_to_gene[guides[j]] else cross
                bucket.append(corr[i, j])
        sns.heatmap(
            corr,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Spearman(native pathway effects)", "shrink": 0.75},
            ax=ax,
        )
        for boundary in boundaries:
            ax.axhline(boundary, color="black", lw=1.2)
            ax.axvline(boundary, color="black", lw=1.2)
        ax.set_title(
            f"{method}: diagonal rho={mean_diagonal:.2f}; off-diagonal rho={mean_offdiagonal:.2f}\n"
            f"within-gene off-diagonal rho={float(np.nanmean(within)):.2f}; "
            f"cross-gene off-diagonal rho={float(np.nanmean(cross)):.2f}"
        )
        ax.set_xlabel("guide")
        ax.set_ylabel("guide")
        ax.tick_params(axis="x", labelsize=5, rotation=90)
        ax.tick_params(axis="y", labelsize=5, rotation=0)
    fig.suptitle(title)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_gene_zooms(
    results: dict[str, pd.DataFrame],
    guide_to_gene: dict[str, str],
    counts: pd.Series,
    outdir: str,
    dataset: str,
    fdr: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    by_gene: dict[str, list[str]] = {}
    for guide, gene in guide_to_gene.items():
        by_gene.setdefault(gene, []).append(guide)
    ordered_guides, ordered_labels, _ = guide_order(guide_to_gene)
    display_label = dict(zip(ordered_guides, ordered_labels))
    for page, gene in enumerate(sorted(by_gene), start=1):
        guides = sorted(by_gene[gene], key=lambda guide: (-int(counts[guide]), guide))
        if len(guides) < 2:
            continue
        left, right = guides[:2]
        fig, axes = plt.subplots(len(results), 2, figsize=(12, 3.8 * len(results)), squeeze=False)
        for row, (method, frame) in enumerate(results.items()):
            indexed = frame.set_index(["target", "program"])
            programs = sorted(set(frame.program))
            left_frame = indexed.loc[left].loc[programs]
            right_frame = indexed.loc[right].loc[programs]
            active = (left_frame.fdr <= fdr) | (right_frame.fdr <= fdr)
            rho = pu.safe_spearman(left_frame.effect, right_frame.effect)
            axes[row, 0].scatter(
                left_frame.effect,
                right_frame.effect,
                c=np.where(active, "#1f4e79", "#bdbdbd"),
                s=18,
                alpha=0.8,
                linewidths=0,
            )
            axes[row, 0].axhline(0, color="0.75", lw=0.7)
            axes[row, 0].axvline(0, color="0.75", lw=0.7)
            axes[row, 0].set(
                xlabel=f"{display_label[left]} native effect",
                ylabel=f"{display_label[right]} native effect",
                title=f"{method}: two most-powered guides rho={rho:.2f}",
            )
            effect_matrix = pu.effect_matrix(frame).loc[guides]
            program_order = effect_matrix.mean(axis=0).abs().sort_values(ascending=False).index
            effect_matrix = effect_matrix.loc[:, program_order]
            vmax = 1.0 if method == "pdex_mwu" else float(np.nanmax(np.abs(effect_matrix.to_numpy()))) or 1.0
            sns.heatmap(
                effect_matrix,
                cmap="RdBu_r",
                center=0,
                vmin=-vmax,
                vmax=vmax,
                yticklabels=[display_label[guide] for guide in guides],
                xticklabels=False,
                cbar_kws={"label": "native effect", "shrink": 0.7},
                ax=axes[row, 1],
            )
            axes[row, 1].set(title=f"{method}: all {gene} guides", xlabel="pathways")
        fig.suptitle(f"Pathway Test 5 — {gene} same-gene guide concordance — {dataset}")
        fig.tight_layout()
        fig.savefig(
            os.path.join(outdir, f"pathways_test5_zoom__{dataset}_{page:02d}.png"),
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_pairs(pairs: pd.DataFrame, summary: pd.DataFrame, path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.violinplot(pairs, x="pair_type", y="effect_spearman", hue="method", cut=0, inner="quart", ax=axes[0])
    sns.barplot(summary, x="method", y="auc_same_gt_unrelated", ax=axes[1])
    axes[0].set(title="Guide-pair pathway concordance", ylabel="Spearman across programs")
    axes[1].axhline(0.5, color="0.5", linestyle="--")
    axes[1].set(title="Separation effect size", ylabel="AUC = P(same-gene > unrelated)", ylim=(0, 1))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument("--pert-col", default="gene", help="target-gene column")
    parser.add_argument("--sgrna-col", required=True)
    parser.add_argument("--control", default="non-targeting")
    parser.add_argument("--score-layer", default="X")
    parser.add_argument("--gene-symbol-col", default="")
    parser.add_argument("--min-genes", type=int, default=5)
    parser.add_argument("--ctrl-size", type=int, default=50)
    parser.add_argument("--n-bins", type=int, default=25)
    parser.add_argument("--min-cells", type=int, default=20)
    parser.add_argument("--min-guides", type=int, default=2)
    parser.add_argument("--max-genes", type=int, default=0)
    parser.add_argument("--max-control", type=int, default=1500)
    parser.add_argument("--normalize-raw", action="store_true")
    parser.add_argument("--background-multiplier", type=int, default=10)
    parser.add_argument("--methods", default="ols,pdex_mwu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pu.configure_bioconcord_root(args.bioconcord_root)
    pu.validate_probability(args.fdr, "--fdr")
    pu.validate_positive_int(args.min_cells, "--min-cells")
    if args.min_guides < 2:
        raise ValueError("--min-guides must be at least 2")
    if args.max_genes < 0 or args.max_genes == 1:
        raise ValueError("--max-genes must be 0 (unlimited) or at least 2")
    if args.max_control < 0:
        raise ValueError("--max-control cannot be negative")
    if args.background_multiplier < 0:
        raise ValueError("--background-multiplier cannot be negative")

    pu.clear_output_prefix(args.outdir, "pathways_test5_")
    plots, tables = pu.prepare_output(args.outdir)
    dataset = pu.dataset_name(args.adata)
    args.run_root = os.path.abspath(os.path.expanduser(args.run_root or args.outdir))
    pu.write_resolved_config(
        run_root=args.run_root,
        workflow="pathways_test5_samegene_sgrna",
        dataset=dataset,
        resolved={
            "arguments": vars(args),
            "methods": pu.method_list(args.methods),
            "results_outdir": os.path.abspath(args.outdir),
        },
    )
    subset, selected_genes, selected_guides = load_samegene_subset(args)
    scored = pu.score_anndata(
        subset, args.programs, score_layer=args.score_layer, min_genes=args.min_genes,
        ctrl_size=args.ctrl_size, n_bins=args.n_bins,
    )
    if args.normalize_raw:
        scored.score_source = "raw X subset -> total-normalized to 10000 -> log1p"
    obs = scored.adata.obs
    targets = obs[args.pert_col].astype(str).to_numpy()
    guides = obs[args.sgrna_col].astype(str).to_numpy()
    valid = ~np.isin(np.char.lower(guides.astype(str)), ["", "nan", "none", "na"])
    counts = pd.Series(guides[(targets != args.control) & valid]).value_counts()
    eligible = set(counts[counts >= args.min_cells].index.astype(str))
    keep = (targets == args.control) | np.isin(guides, list(eligible))
    group_labels = np.where(targets == args.control, args.control, guides)

    guide_to_gene = {}
    for guide in sorted(eligible):
        values = pd.Series(targets[(guides == guide) & (targets != args.control)])
        if not values.empty:
            guide_to_gene[guide] = str(values.mode().iloc[0])
    by_gene: dict[str, list[str]] = {}
    for guide, gene in guide_to_gene.items():
        by_gene.setdefault(gene, []).append(guide)
    same_pairs = sorted(
        (a, b) for members in by_gene.values() if len(members) >= 2
        for a, b in itertools.combinations(sorted(members), 2)
    )
    if not same_pairs:
        raise ValueError("No target gene has at least two eligible guides")
    unrelated = [
        (a, b) for a, b in itertools.combinations(sorted(guide_to_gene), 2)
        if guide_to_gene[a] != guide_to_gene[b]
    ]
    if not unrelated:
        raise ValueError(
            "Test 5 requires powered guides from at least two target genes "
            "to construct the unrelated-guide background"
        )
    rng = np.random.default_rng(args.seed)
    max_background = min(len(unrelated), max(len(same_pairs), len(same_pairs) * args.background_multiplier))
    if len(unrelated) > max_background:
        chosen = rng.choice(len(unrelated), size=max_background, replace=False)
        unrelated = [unrelated[i] for i in chosen]

    methods = pu.method_list(args.methods)
    results = pu.run_methods(
        scored.scores[keep], group_labels[keep], args.control, scored.program_labels,
        methods=methods, threads=args.threads,
    )
    pair_rows: list[dict] = []
    summary_rows: list[dict] = []
    for method, frame in results.items():
        matrix = pu.effect_matrix(frame)
        sig = {guide: pu.significant_programs(frame, guide, args.fdr) for guide in matrix.index}
        for pair_type, pairs in (("same_gene", same_pairs), ("unrelated", unrelated)):
            for guide_a, guide_b in pairs:
                if guide_a not in matrix.index or guide_b not in matrix.index:
                    continue
                pair_rows.append(
                    {
                        "method": method,
                        "pair_type": pair_type,
                        "guide_a": guide_a,
                        "guide_b": guide_b,
                        "gene_a": guide_to_gene[guide_a],
                        "gene_b": guide_to_gene[guide_b],
                        "effect_spearman": pu.safe_spearman(matrix.loc[guide_a], matrix.loc[guide_b]),
                        "sig_jaccard": pu.jaccard(sig[guide_a], sig[guide_b]),
                        "n_cells_a": int(counts.get(guide_a, 0)),
                        "n_cells_b": int(counts.get(guide_b, 0)),
                    }
                )
        method_pairs = pd.DataFrame([row for row in pair_rows if row["method"] == method])
        same = method_pairs.loc[method_pairs.pair_type == "same_gene", "effect_spearman"].dropna().to_numpy()
        background = method_pairs.loc[method_pairs.pair_type == "unrelated", "effect_spearman"].dropna().to_numpy()
        if same.size == 0 or background.size == 0:
            raise ValueError(
                f"Test 5 requires finite same-gene and unrelated Spearman samples; "
                f"method={method}, n_same={same.size}, n_unrelated={background.size}"
            )
        u, p_value = stats.mannwhitneyu(same, background, alternative="greater")
        summary_rows.append(
            {
                "method": method,
                "n_same_gene_pairs": len(same),
                "n_unrelated_pairs": len(background),
                "median_same_gene_rho": float(np.median(same)),
                "median_unrelated_rho": float(np.median(background)),
                "auc_same_gt_unrelated": float(u / (len(same) * len(background))),
                "mannwhitney_p_value": float(p_value),
            }
        )

    pair_df = pd.DataFrame(pair_rows)
    summary_df = pd.DataFrame(summary_rows)
    pair_df.to_csv(os.path.join(tables, f"pathways_test5_pairs__{dataset}.csv"), index=False)
    summary_df.to_csv(os.path.join(tables, f"pathways_test5_summary__{dataset}.csv"), index=False)
    pd.concat(results.values(), ignore_index=True).to_csv(
        os.path.join(tables, f"pathways_test5_results__{dataset}.csv"), index=False
    )
    ordered_guides, plot_labels, _ = guide_order(guide_to_gene)
    pd.DataFrame(
        {
            "raw_guide": ordered_guides,
            "target_gene": [guide_to_gene[guide] for guide in ordered_guides],
            "plot_label": plot_labels,
            "n_cells": [int(counts[guide]) for guide in ordered_guides],
        }
    ).to_csv(
        os.path.join(tables, f"pathways_test5_guide_map__{dataset}.csv"),
        index=False,
    )
    plot_pairs(pair_df, summary_df, os.path.join(plots, f"pathways_test5_samegene__{dataset}.png"), f"Pathway Test 5 — {dataset}")
    for method, frame in results.items():
        plot_guide_heatmap(
            frame,
            guide_to_gene,
            os.path.join(plots, f"pathways_test5_guide_heatmap_{method}__{dataset}.png"),
            f"Pathway Test 5 — {method} guide effects grouped by gene — {dataset}",
            method,
        )
    plot_guide_corr_matrix(
        results,
        guide_to_gene,
        os.path.join(plots, f"pathways_test5_corr_matrix__{dataset}.png"),
        f"Pathway Test 5 — same-gene guide correlation blocks — {dataset}",
    )
    plot_gene_zooms(results, guide_to_gene, counts, plots, dataset, args.fdr)
    pu.save_program_legend(scored.programs, os.path.join(tables, f"pathways_legend__{dataset}.csv"))
    pu.save_run_metadata(
        os.path.join(args.outdir, f"pathways_test5_metadata__{dataset}.json"),
        scored,
        args,
        extra={
            "selected_genes": selected_genes,
            "selected_guides": selected_guides,
            "pairwise_inference_caveat": (
                "Guide pairs share guides and are not independent; the Mann-Whitney "
                "p-value is descriptive rather than a gene-level confirmatory test."
            ),
        },
    )


if __name__ == "__main__":
    main()
