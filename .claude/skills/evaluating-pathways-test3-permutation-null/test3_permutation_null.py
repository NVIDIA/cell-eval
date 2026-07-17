#!/usr/bin/env python
"""Pathway Test 3: global and within-block permutation nulls versus real signal."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pathway_utils as pu


def counts(frame: pd.DataFrame, fdr: float) -> pd.DataFrame:
    output = frame.groupby("target", as_index=False).agg(
        n_programs=("program", "size"),
        n_sig=("fdr", lambda x: int((x <= fdr).sum())),
        n_cells=("n_target", "first"),
    )
    return output


def permutation_method_summary(results: pd.DataFrame, fdr: float) -> pd.DataFrame:
    """Summarize paired OLS/pdex null calls for each target across permutations."""
    permuted = results[results.state == "permuted"]
    rows: list[dict] = []
    for (permutation, target), group in permuted.groupby(["permutation", "target"]):
        by_method = {method: part for method, part in group.groupby("method")}
        if not {"ols", "pdex_mwu"}.issubset(by_method):
            continue
        ols = by_method["ols"]
        pdex = by_method["pdex_mwu"]
        sig_ols = set(ols.loc[ols.fdr <= fdr, "program"].astype(str))
        sig_pdex = set(pdex.loc[pdex.fdr <= fdr, "program"].astype(str))
        rows.append(
            {
                "permutation": int(permutation),
                "target": str(target),
                "n_sig_ols": len(sig_ols),
                "n_sig_pdex_mwu": len(sig_pdex),
                "sig_jaccard": pu.jaccard(sig_ols, sig_pdex),
                "n_cells": int(ols.n_target.iloc[0]),
            }
        )
    paired = pd.DataFrame(rows)
    if paired.empty:
        return paired
    return paired.groupby("target", as_index=False).agg(
        n_sig_ols=("n_sig_ols", "mean"),
        n_sig_pdex_mwu=("n_sig_pdex_mwu", "mean"),
        sig_jaccard=("sig_jaccard", "mean"),
        n_cells=("n_cells", "first"),
    )


def plot_method_comparison(
    summary: pd.DataFrame, path: str, title: str, fdr: float, shuffle_mode: str
) -> None:
    """Plot mean permuted significant-program counts for pdex versus OLS."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if summary.empty:
        return
    x = summary.n_sig_pdex_mwu.to_numpy(float)
    y = summary.n_sig_ols.to_numpy(float)
    cells = summary.n_cells.to_numpy(float)
    labels = summary.target.astype(str).tolist()
    jaccard = summary.sig_jaccard.to_numpy(float)
    cell_range = max(float(cells.max() - cells.min()), 1.0)
    sizes = 45 + 320 * (cells - cells.min()) / cell_range
    limit = max(float(np.nanmax(x)), float(np.nanmax(y)), 1.0) * 1.3 + 0.2
    norm = Normalize(vmin=float(cells.min()), vmax=float(cells.max()))
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.fill_between(
        [0, limit], [0, 0], [0, limit], color="#dce8f5", alpha=0.35, zorder=0
    )
    ax.fill_between(
        [0, limit], [0, limit], [limit, limit], color="#f5ece0", alpha=0.35,
        zorder=0,
    )
    ax.plot([0, limit], [0, limit], "--", color="#777777", lw=1.2, zorder=1)
    ax.scatter(
        x, y, s=sizes, c=cells, cmap=cmap, norm=norm, alpha=0.85,
        linewidths=0.5, edgecolors="white", zorder=3,
    )

    order = np.lexsort((np.asarray(labels), y, x))
    sides = np.empty(len(labels), dtype=int)
    sides[order] = np.where(np.arange(len(labels)) % 2 == 0, -1, 1)
    for side in (-1, 1):
        indices = np.flatnonzero(sides == side)
        indices = indices[np.argsort(y[indices])]
        label_y = np.linspace(limit * 0.05, limit * 0.95, max(len(indices), 1))
        label_x = -limit * 0.34 if side < 0 else limit * 1.34
        for position, i in zip(label_y, indices):
            ax.annotate(
                f"{labels[i]}\nmean J={jaccard[i]:.2f}",
                xy=(x[i], y[i]), xytext=(label_x, position), textcoords="data",
                ha="left" if side < 0 else "right", va="center", fontsize=6.3,
                arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.65},
                bbox={
                    "boxstyle": "round,pad=0.12", "fc": "white", "ec": "none",
                    "alpha": 0.75,
                },
            )

    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.03)
    colorbar.set_label("cell count per shuffled perturbation group")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], ls="--", color="#777777", lw=1.2, label="equal calling"),
            plt.Rectangle((0, 0), 1, 1, fc="#f5ece0", alpha=0.6, label="OLS calls more"),
            plt.Rectangle((0, 0), 1, 1, fc="#dce8f5", alpha=0.6, label="pdex MWU calls more"),
        ],
        fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3,
    )
    ax.set_xlim(-limit * 0.42, limit * 1.42)
    ax.set_ylim(-limit * 0.04, limit)
    ax.set_xlabel("mean significant programs under permutation: pdex MWU")
    ax.set_ylabel("mean significant programs under permutation: OLS")
    null_label = "within-block" if shuffle_mode == "within" else "global"
    ax.set_title(
        f"{title}\n{null_label} shuffled-label null; FDR <= {fdr:g}; "
        "calibrated methods approach the origin\n"
        "labels show mean called-set Jaccard across permutations",
        fontsize=9,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mean_permutation_corr(results: pd.DataFrame, method: str) -> tuple[list[str], np.ndarray]:
    """Average target-by-target native-effect Spearman maps across permutations."""
    part = results[(results.state == "permuted") & (results.method == method)]
    permutations = sorted(part.permutation.unique())
    if not permutations:
        return [], np.empty((0, 0))
    target_sets = [
        set(part.loc[part.permutation == permutation, "target"].astype(str))
        for permutation in permutations
    ]
    program_sets = [
        set(part.loc[part.permutation == permutation, "program"].astype(str))
        for permutation in permutations
    ]
    targets = sorted(set.intersection(*target_sets))
    programs = sorted(set.intersection(*program_sets))
    matrices = []
    for permutation in permutations:
        frame = part[part.permutation == permutation]
        effects = frame.pivot(index="target", columns="program", values="effect").loc[
            targets, programs
        ]
        corr = np.asarray(
            [
                [pu.safe_spearman(effects.loc[left], effects.loc[right]) for right in targets]
                for left in targets
            ],
            dtype=float,
        )
        np.fill_diagonal(corr, 1.0)
        matrices.append(corr)
    return targets, np.nanmean(np.stack(matrices), axis=0)


def plot_corr_matrix(
    targets: list[str], matrix: np.ndarray, path: str, title: str, method: str
) -> None:
    """Plot a method-specific mean permutation-null effect correlation matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(targets) < 2:
        return
    diagonal_mask = np.eye(len(targets), dtype=bool)
    diagonal = matrix[diagonal_mask]
    off_diagonal = matrix[~diagonal_mask]
    mean_diagonal = float(np.nanmean(diagonal))
    mean_off = float(np.nanmean(off_diagonal))
    size = max(7, len(targets) * 0.22 + 1.5)
    fig, ax = plt.subplots(figsize=(size, size), layout="constrained")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(
        targets, rotation=90, fontsize=max(4, min(7, 120 // len(targets)))
    )
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=max(4, min(7, 120 // len(targets))))
    ax.set_xlabel("fake perturbation (shuffled labels)")
    ax.set_ylabel("fake perturbation (shuffled labels)")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("mean permutation-wise Spearman(native pathway effects)")
    label = "OLS" if method == "ols" else "pdex MWU"
    ax.set_title(
        f"{title}\n{label} (mean diagonal rho = {mean_diagonal:.3f}; "
        f"off-diagonal rho = {mean_off:.3f})\n"
        "calibrated permutation null: clean diagonal, near-zero off-diagonal",
        fontsize=9,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def permute_labels(
    labels: np.ndarray, blocks: np.ndarray, seed: int, shuffle_mode: str
) -> np.ndarray:
    """Permute complete labels globally or within blocks and verify invariants."""
    labels = np.asarray(labels).astype(str)
    if shuffle_mode == "within":
        shuffled = pu.permute_within_blocks(labels, blocks, seed)
    elif shuffle_mode == "global":
        shuffled = np.random.default_rng(seed).permutation(labels)
    else:
        raise ValueError(f"Unsupported shuffle mode: {shuffle_mode!r}")

    if not np.array_equal(np.sort(shuffled), np.sort(labels)):
        raise RuntimeError(f"{shuffle_mode} shuffle changed the global label histogram")
    if shuffle_mode == "within":
        blocks = np.asarray(blocks)
        for block in np.unique(blocks):
            idx = np.flatnonzero(blocks == block)
            if not np.array_equal(np.sort(shuffled[idx]), np.sort(labels[idx])):
                raise RuntimeError(
                    f"Within-block shuffle changed the label histogram for block {block!r}"
                )
    return shuffled


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
    parser.add_argument("--n-permutations", type=int, default=5)
    parser.add_argument(
        "--shuffle-mode", choices=("global", "within", "both"), default="both",
        help="Shuffle labels globally, within block strata, or run both nulls",
    )
    parser.add_argument("--methods", default="ols,pdex_mwu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pu.configure_bioconcord_root(args.bioconcord_root)
    pu.validate_probability(args.fdr, "--fdr")
    if args.n_permutations != 5:
        parser.error("Pathway Test 3 fixes --n-permutations at 5")

    pu.clear_output_prefix(args.outdir, "pathways_test3_")
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
    shuffle_modes = (
        ("global", "within") if args.shuffle_mode == "both" else (args.shuffle_mode,)
    )
    real_results = pu.run_methods(
        scored.scores, labels, args.control, scored.program_labels,
        methods=methods, threads=args.threads,
    )
    full_results: list[pd.DataFrame] = []
    real_count_rows: list[pd.DataFrame] = []
    permuted_count_rows: list[pd.DataFrame] = []
    for method, frame in real_results.items():
        tagged = frame.copy()
        tagged["state"] = "real"
        tagged["permutation"] = -1
        tagged["shuffle_mode"] = "real"
        full_results.append(tagged)
        c = counts(frame, args.fdr)
        c["method"] = method
        c["state"] = "real"
        c["permutation"] = -1
        real_count_rows.append(c)

    for shuffle_mode in shuffle_modes:
        for permutation in range(args.n_permutations):
            shuffled = permute_labels(
                labels, blocks, args.seed + permutation, shuffle_mode
            )
            perm_results = pu.run_methods(
                scored.scores, shuffled, args.control, scored.program_labels,
                methods=methods, threads=args.threads,
            )
            for method, frame in perm_results.items():
                tagged = frame.copy()
                tagged["state"] = "permuted"
                tagged["permutation"] = permutation
                tagged["shuffle_mode"] = shuffle_mode
                full_results.append(tagged)
                c = counts(frame, args.fdr)
                c["method"] = method
                c["state"] = "permuted"
                c["permutation"] = permutation
                c["shuffle_mode"] = shuffle_mode
                permuted_count_rows.append(c)

    result_df = pd.concat(full_results, ignore_index=True)
    permuted_count_df = pd.concat(permuted_count_rows, ignore_index=True)
    real_count_df = pd.concat(real_count_rows, ignore_index=True)
    null = permuted_count_df.groupby(
        ["shuffle_mode", "method", "target"], as_index=False
    ).agg(
        null_mean=("n_sig", "mean"),
        null_sd=("n_sig", "std"),
        null_q95=("n_sig", lambda x: float(np.quantile(x, 0.95))),
    )
    separation_rows = []
    for shuffle_mode in shuffle_modes:
        mode_real = real_count_df.copy()
        mode_real["shuffle_mode"] = shuffle_mode
        mode_real = mode_real.merge(
            null[null.shuffle_mode == shuffle_mode],
            on=["shuffle_mode", "method", "target"], how="left",
        )
        mode_real["separation_z"] = (
            (mode_real.n_sig - mode_real.null_mean)
            / mode_real.null_sd.replace(0, np.nan)
        )
        separation_rows.append(mode_real)
    separation_df = pd.concat(separation_rows, ignore_index=True)
    real_for_counts = separation_df[
        [
            "target", "n_programs", "n_sig", "n_cells", "method", "state",
            "permutation", "shuffle_mode",
        ]
    ]
    count_df = pd.concat([permuted_count_df, real_for_counts], ignore_index=True)
    count_df = count_df.merge(
        separation_df[["shuffle_mode", "method", "target", "separation_z"]],
        on=["shuffle_mode", "method", "target"], how="left",
    )
    result_df.to_csv(os.path.join(tables, f"pathways_test3_results__{dataset}.csv"), index=False)
    count_df.to_csv(os.path.join(tables, f"pathways_test3_counts__{dataset}.csv"), index=False)
    separation_df.to_csv(
        os.path.join(tables, f"pathways_test3_separation__{dataset}.csv"), index=False
    )
    method_summaries = []
    for shuffle_mode in shuffle_modes:
        mode_results = result_df[
            (result_df.state == "permuted")
            & (result_df.shuffle_mode == shuffle_mode)
        ]
        if set(methods) == set(pu.METHODS):
            method_summary = permutation_method_summary(mode_results, args.fdr)
            method_summary.insert(0, "shuffle_mode", shuffle_mode)
            method_summaries.append(method_summary)
            plot_method_comparison(
                method_summary,
                os.path.join(
                    plots,
                    f"pathways_test3_shuffle_call_comparison__{shuffle_mode}__{dataset}.png",
                ),
                f"Pathway Test 3 — {dataset}",
                args.fdr,
                shuffle_mode,
            )
        for method in methods:
            targets, matrix = mean_permutation_corr(mode_results, method)
            plot_corr_matrix(
                targets,
                matrix,
                os.path.join(
                    plots,
                    f"pathways_test3_corr_matrix__{shuffle_mode}__{method}__{dataset}.png",
                ),
                f"Pathway Test 3 {shuffle_mode} shuffle, mean across "
                f"{args.n_permutations} permutations — {dataset}",
                method,
            )
    if method_summaries:
        pd.concat(method_summaries, ignore_index=True).to_csv(
            os.path.join(tables, f"pathways_test3_method_comparison__{dataset}.csv"),
            index=False,
        )
    pu.save_program_legend(scored.programs, os.path.join(tables, f"pathways_legend__{dataset}.csv"))
    pu.save_run_metadata(os.path.join(args.outdir, f"pathways_test3_metadata__{dataset}.json"), scored, args)


if __name__ == "__main__":
    main()
