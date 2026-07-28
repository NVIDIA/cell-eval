#!/usr/bin/env python
"""Pathway Test 0: raw-UMI pathway injection and count calibration curves."""

from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import pathway_utils as pu
import scipy.sparse as sp
from scipy import stats

LOGGER = logging.getLogger("pathway_test0")
_MWU_EQUIVALENCE_VALIDATED = False


def _configure_logging(path: str) -> None:
    """Log progress to the terminal and a persistent per-run file."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    terminal = logging.StreamHandler(sys.stdout)
    terminal.setFormatter(formatter)
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(terminal)
    LOGGER.addHandler(file_handler)

    def _log_uncaught(exc_type, exc_value, exc_traceback) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        LOGGER.critical(
            "uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = _log_uncaught


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _validated_counts(matrix):
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Raw counts must be finite and nonnegative")
    # Keep one compact CSR baseline per repeat.  Dense H5AD inputs otherwise
    # force every independent injection to copy the full cells-by-genes array
    # and rebuild the same sparse structure during normalization.
    return sp.csr_matrix(matrix, dtype=np.float64, copy=True)


def _inject_pathway_counts(
    base_counts,
    *,
    first_injected_row: int,
    gene_indices: np.ndarray,
    delta_log2fc: float,
) -> np.ndarray | sp.csr_matrix:
    """Multiply injected-arm pathway-gene UMIs by 2**delta and round."""
    multiplier = 2.0**delta_log2fc
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError(f"Invalid log2FC delta: {delta_log2fc}")
    if sp.issparse(base_counts):
        selected = base_counts[first_injected_row:, :][:, gene_indices].tocoo()
        updated = np.rint(selected.data * multiplier)
        difference = updated - selected.data
        changed = difference != 0
        update = sp.csr_matrix(
            (
                difference[changed],
                (
                    selected.row[changed] + first_injected_row,
                    gene_indices[selected.col[changed]],
                ),
            ),
            shape=base_counts.shape,
            dtype=np.float64,
        )
        counts = (base_counts + update).tocsr()
        counts.eliminate_zeros()
        return counts
    counts = base_counts.copy()
    selected = np.ix_(
        np.arange(first_injected_row, counts.shape[0]),
        gene_indices,
    )
    counts[selected] = np.rint(counts[selected] * multiplier)
    return counts


def _score_trial_counts(
    counts,
    *,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    gene_sets: dict[str, list[str]],
    program_labels: list[str],
    normalization_target: float,
    score_jobs: int,
    validate_against_bioconcord: bool = False,
) -> np.ndarray:
    """Normalize/log1p counts and compute exact bioconcord-equivalent scores."""
    matrix = sp.csr_matrix(counts, dtype=np.float64, copy=False)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    if (totals <= 0).any():
        raise ValueError("Cannot normalize cells with zero total UMI count")
    matrix = matrix.multiply((normalization_target / totals)[:, None]).tocsr()
    matrix.data = np.log1p(matrix.data)
    scores = _vectorized_bioconcord_scores(matrix, var.index, gene_sets)
    if validate_against_bioconcord:
        original = ad.AnnData(X=matrix.copy(), obs=obs.copy(), var=var.copy())
        bc = pu._bioconcord_module()
        bc.score_all_programs(original, gene_sets, n_jobs=score_jobs)
        expected = original.obs[program_labels].to_numpy(dtype=np.float64)
        maximum_difference = float(np.max(np.abs(scores - expected)))
        if maximum_difference > 1e-7:
            raise RuntimeError(
                "Vectorized pathway scores disagree with bioconcord: "
                f"max_abs_difference={maximum_difference:g}"
            )
        LOGGER.info(
            "validated vectorized scorer against bioconcord: "
            "max_abs_difference=%g",
            maximum_difference,
        )
    return scores


def _vectorized_pdex_mwu(
    scores: np.ndarray,
    labels: np.ndarray,
    reference: str,
    program_labels: list[str],
) -> pd.DataFrame:
    """Compute pdex's two-sided asymptotic MWU result without setup overhead."""
    values = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(labels).astype(str)
    reference_values = values[groups == reference]
    rows: list[pd.DataFrame] = []
    for target in sorted(set(groups) - {reference}):
        target_values = values[groups == target]
        statistic, p_value = stats.mannwhitneyu(
            target_values,
            reference_values,
            axis=0,
            alternative="two-sided",
            method="asymptotic",
            use_continuity=True,
        )
        denominator = float(len(target_values) * len(reference_values))
        frame = pd.DataFrame(
            {
                "method": "pdex_mwu",
                "target": target,
                "program": program_labels,
                "effect": 2.0 * np.asarray(statistic, float) / denominator - 1.0,
                "mean_difference": (
                    target_values.mean(axis=0) - reference_values.mean(axis=0)
                ),
                "statistic": np.asarray(statistic, float),
                "standard_error": np.nan,
                "p_value": np.asarray(p_value, float),
                "n_target": len(target_values),
                "n_reference": len(reference_values),
            }
        )
        frame["fdr"] = stats.false_discovery_control(
            frame["p_value"].to_numpy(float),
            method="bh",
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _run_methods_quiet(
    scores: np.ndarray,
    labels: np.ndarray,
    reference: str,
    program_labels: list[str],
    *,
    methods: list[str] | tuple[str, ...],
    threads: int,
) -> dict[str, pd.DataFrame]:
    """Run exact pathway methods while avoiding repeated pdex setup per trial.

    Test 0 evaluates thousands of two-group trials with the same 97 columns.
    Arc pdex's process/progress setup dominates each tiny fit, so after one
    strict equivalence check we use SciPy's vectorized implementation of the
    same two-sided asymptotic Mann–Whitney calculation.
    """
    global _MWU_EQUIVALENCE_VALIDATED
    output: dict[str, pd.DataFrame] = {}
    if "ols" in methods:
        output["ols"] = pu.run_ols(scores, labels, reference, program_labels)
    if "pdex_mwu" in methods:
        fast = _vectorized_pdex_mwu(scores, labels, reference, program_labels)
        if not _MWU_EQUIVALENCE_VALIDATED:
            with open(os.devnull, "w", encoding="utf-8") as sink:
                with contextlib.redirect_stderr(sink):
                    upstream = pu.run_pdex_mwu(
                        scores,
                        labels,
                        reference,
                        program_labels,
                        threads=threads,
                    )
            columns = [
                "effect",
                "mean_difference",
                "statistic",
                "p_value",
                "fdr",
            ]
            left = fast.sort_values(["target", "program"]).reset_index(drop=True)
            right = upstream.sort_values(["target", "program"]).reset_index(drop=True)
            maximum_difference = max(
                float(
                    np.nanmax(
                        np.abs(
                            left[column].to_numpy(float)
                            - right[column].to_numpy(float)
                        )
                    )
                )
                for column in columns
            )
            if maximum_difference > 1e-10:
                raise RuntimeError(
                    "Vectorized MWU disagrees with Arc pdex: "
                    f"max_abs_difference={maximum_difference:g}"
                )
            LOGGER.info(
                "validated vectorized MWU against Arc pdex: "
                "max_abs_difference=%g",
                maximum_difference,
            )
            _MWU_EQUIVALENCE_VALIDATED = True
        output["pdex_mwu"] = fast
    return output


def _vectorized_bioconcord_scores(
    expression,
    var_names,
    gene_sets: dict[str, list[str]],
) -> np.ndarray:
    """Vectorize bioconcord's fixed score_genes_standalone algorithm."""
    var_names = np.asarray(var_names).astype(str)
    positions = {gene: index for index, gene in enumerate(var_names)}
    if sp.issparse(expression):
        gene_means = np.asarray(expression.mean(axis=0)).ravel()
    else:
        gene_means = np.asarray(expression).mean(axis=0)
    bins = pd.qcut(gene_means, 25, labels=False, duplicates="drop")

    bin_members = {
        value: np.flatnonzero(bins == value)
        for value in np.unique(bins)
        if not pd.isna(value)
    }
    weights = np.zeros((len(var_names), len(gene_sets)), dtype=np.float64)
    for program_index, genes in enumerate(gene_sets.values()):
        target_indices = np.asarray([positions[gene] for gene in genes], dtype=int)
        rng = np.random.default_rng(42)
        control_indices: list[int] = []
        for target_index in target_indices:
            same_bin = bin_members.get(
                bins[target_index],
                np.empty(0, dtype=int),
            )
            same_bin = same_bin[same_bin != target_index]
            if same_bin.size:
                selected = rng.choice(
                    same_bin,
                    size=min(50, same_bin.size),
                    replace=False,
                )
                control_indices.extend(selected.tolist())
        controls = np.unique(control_indices)
        np.add.at(
            weights[:, program_index],
            target_indices,
            1.0 / len(target_indices),
        )
        if controls.size:
            np.add.at(
                weights[:, program_index],
                controls,
                -1.0 / len(controls),
            )
    scores = expression @ weights
    return scores.toarray() if sp.issparse(scores) else np.asarray(scores)


def _plot(pathway_summary: pd.DataFrame, path: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator

    methods = [
        method for method in pu.METHODS if method in set(pathway_summary["method"])
    ]
    programs = sorted(pathway_summary["injected_program"].astype(str).unique())
    deltas = sorted(pathway_summary["delta_log2fc"].unique())
    n_repeats = int(pathway_summary["n_repeats"].max())
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=max(len(programs) - 1, 1))
    metrics = (
        ("tp_count", f"TP calls across {n_repeats} repeats"),
        ("fp_count", f"FP calls across {n_repeats} repeats"),
    )

    fig, axes = plt.subplots(
        len(methods),
        2,
        figsize=(13, 4.5 * len(methods)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for row, method in enumerate(methods):
        method_frame = pathway_summary[pathway_summary["method"] == method]
        for column, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, column]
            for program_index, program in enumerate(programs):
                line = method_frame[
                    method_frame["injected_program"] == program
                ].sort_values("delta_log2fc")
                ax.plot(
                    line["delta_log2fc"],
                    line[metric],
                    color=cmap(norm(program_index)),
                    linewidth=0.9,
                    alpha=0.5,
                )

            distribution = method_frame.groupby("delta_log2fc")[metric]
            lower = distribution.quantile(0.1).reindex(deltas).to_numpy(float)
            median = distribution.median().reindex(deltas).to_numpy(float)
            upper = distribution.quantile(0.9).reindex(deltas).to_numpy(float)
            ax.fill_between(
                deltas,
                lower,
                upper,
                color="0.35",
                alpha=0.15,
                label="pathway 10–90% tube",
                zorder=0,
            )
            ax.plot(
                deltas,
                median,
                color="black",
                linewidth=2,
                marker="o",
                markersize=4,
                label="pathway median",
                zorder=4,
            )
            ax.set_ylabel(ylabel)
            ax.set_xlabel("injected pathway log2 fold-change (delta)")
            ax.grid(axis="y", color="0.88", linewidth=0.7)
            if metric == "tp_count":
                ax.set_ylim(-0.5, n_repeats + 0.5)
                ax.set_yticks(range(n_repeats + 1))
            else:
                maximum = max(1, int(np.ceil(method_frame[metric].max())))
                ax.set_ylim(-0.5, maximum + 0.5)
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if row == 0:
                ax.set_title(
                    "Injected-pathway true-positive count"
                    if metric == "tp_count"
                    else "Untouched-pathway false-positive count"
                )
            if row == len(methods) - 1 and column == 0:
                ax.legend(loc="upper left", fontsize=8)
        axes[row, 0].text(
            -0.13,
            0.5,
            "bioconcord OLS" if method == "ols" else "pdex MWU",
            rotation=90,
            va="center",
            ha="center",
            transform=axes[row, 0].transAxes,
            fontsize=11,
            fontweight="bold",
        )

    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.ravel().tolist(),
        pad=0.02,
        fraction=0.025,
    )
    tick_indices = np.unique(
        np.linspace(0, len(programs) - 1, min(8, len(programs))).astype(int)
    )
    colorbar.set_ticks(tick_indices)
    colorbar.set_ticklabels([programs[index] for index in tick_indices])
    colorbar.set_label("injected pathway (gradient; selected labels shown)")
    fig.suptitle(
        f"{title}\nOne pathway injected per trial; gene UMI multiplier = 2^delta"
    )
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
    parser.add_argument("--pert-col", default="gene")
    parser.add_argument("--control", default="non-targeting")
    parser.add_argument("--block-cols", default="batch")
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--normalization-target", type=float, default=1e4)
    parser.add_argument("--score-jobs", type=int, default=1)
    parser.add_argument("--min-genes", type=int, default=5)
    parser.add_argument("--deltas", default="0,0.5,1,2")
    parser.add_argument(
        "--pathways",
        default="",
        help="comma-separated pathway labels for a smoke test; default tests every pathway",
    )
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--min-cells-per-arm", type=int, default=20)
    parser.add_argument("--methods", default="ols,pdex_mwu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-file",
        default="",
        help=(
            "persistent progress log; default is "
            "<outdir>/pathways_test0_run__<dataset>.log"
        ),
    )
    args = parser.parse_args()
    pu.configure_bioconcord_root(args.bioconcord_root)
    pu.validate_probability(args.fdr, "--fdr")
    pu.validate_positive_int(args.min_genes, "--min-genes")
    pu.validate_positive_int(args.min_cells_per_arm, "--min-cells-per-arm")
    pu.validate_positive_int(args.n_repeats, "--n-repeats")
    pu.validate_positive_int(args.score_jobs, "--score-jobs")
    if not np.isfinite(args.normalization_target) or args.normalization_target <= 0:
        raise ValueError("--normalization-target must be finite and positive")

    pu.clear_output_prefix(args.outdir, "pathways_test0_")
    plots, tables = pu.prepare_output(args.outdir)
    dataset = pu.dataset_name(args.adata)
    args.run_root = os.path.abspath(os.path.expanduser(args.run_root or args.outdir))
    pu.write_resolved_config(
        run_root=args.run_root,
        workflow="pathways_test0_injection",
        dataset=dataset,
        resolved={
            "arguments": vars(args),
            "methods": pu.method_list(args.methods),
            "results_outdir": os.path.abspath(args.outdir),
        },
    )
    log_file = args.log_file or os.path.join(
        args.outdir, f"pathways_test0_run__{dataset}.log"
    )
    _configure_logging(log_file)
    run_started = time.perf_counter()
    LOGGER.info("starting pathway Test 0")
    LOGGER.info("log_file=%s", os.path.abspath(log_file))
    LOGGER.info("arguments=%s", vars(args))
    adata = ad.read_h5ad(args.adata)
    if not adata.var_names.is_unique:
        raise ValueError("Pathway Test 0 requires unique adata.var_names")
    if args.pert_col not in adata.obs:
        raise ValueError(f"{args.pert_col!r} is absent from adata.obs")
    if args.counts_layer in ("", "X"):
        counts_matrix = adata.X
        counts_source = "X"
    else:
        if args.counts_layer not in adata.layers:
            raise ValueError(
                f"Counts layer {args.counts_layer!r} is absent; "
                f"available={list(adata.layers)}"
            )
        counts_matrix = adata.layers[args.counts_layer]
        counts_source = f"layers[{args.counts_layer!r}]"
    programs = pu.load_programs(
        args.programs,
        adata.var_names,
        min_genes=args.min_genes,
    )
    program_labels = list(programs)
    gene_sets = {program: value["genes"] for program, value in programs.items()}
    duplicated_gene_sets = [
        program for program, genes in gene_sets.items() if len(genes) != len(set(genes))
    ]
    if duplicated_gene_sets:
        raise ValueError(
            "Pathway Test 0 requires unique genes within every program; duplicates in "
            f"{duplicated_gene_sets}"
        )
    scored = pu.ScoredData(
        adata=adata,
        scores=np.empty((adata.n_obs, 0), dtype=np.float64),
        programs=programs,
        program_labels=program_labels,
        score_source=(
            f"{counts_source} -> normalize_total({args.normalization_target:g}) "
            "-> log1p -> bioconcord score_all_programs per trial"
        ),
    )
    labels = adata.obs[args.pert_col].astype(str).to_numpy()
    ctrl_idx = np.flatnonzero(labels == args.control)
    if ctrl_idx.size < 2 * args.min_cells_per_arm:
        raise ValueError("Too few control cells for the requested split threshold")
    block_cols = [x.strip() for x in args.block_cols.split(",") if x.strip()]
    blocks = pu.composite_blocks(adata.obs, block_cols)
    methods = pu.method_list(args.methods)
    deltas = list(
        dict.fromkeys(float(value) for value in args.deltas.split(",") if value.strip())
    )
    if not deltas:
        raise ValueError("--deltas must contain at least one value")
    pathway_indices = np.arange(len(program_labels))
    requested = list(
        dict.fromkeys(
            value.strip() for value in args.pathways.split(",") if value.strip()
        )
    )
    if requested:
        unknown = sorted(set(requested) - set(program_labels))
        if unknown:
            raise ValueError(f"Unknown --pathways labels: {unknown}")
        pathway_indices = np.asarray(
            [program_labels.index(label) for label in requested], dtype=int
        )
    var_positions = pd.Series(
        np.arange(adata.n_vars, dtype=int),
        index=adata.var_names.astype(str),
    )
    pathway_gene_indices = {
        program: var_positions.loc[gene_sets[program]].to_numpy(dtype=int)
        for program in program_labels
    }
    total_trials = args.n_repeats * len(pathway_indices) * len(deltas)
    completed_trials = 0
    LOGGER.info(
        "workload: cells=%d genes=%d scored_pathways=%d injected_pathways=%d "
        "repeats=%d deltas=%d methods=%s total_trials=%d",
        adata.n_obs,
        adata.n_vars,
        len(program_labels),
        len(pathway_indices),
        args.n_repeats,
        len(deltas),
        ",".join(methods),
        total_trials,
    )

    detail: list[pd.DataFrame] = []
    trial_summary: list[dict] = []
    for repeat in range(args.n_repeats):
        rng = np.random.default_rng(args.seed + repeat)
        left, right = pu.stratified_half(ctrl_idx, blocks, rng)
        if min(left.size, right.size) < args.min_cells_per_arm:
            raise ValueError(
                "A stratified control split fell below --min-cells-per-arm"
            )
        combined = np.concatenate([left, right])
        test_labels = np.asarray([args.control] * left.size + ["injected"] * right.size)
        trial_obs = adata.obs.iloc[combined].copy()
        base_counts = _validated_counts(counts_matrix[combined])
        base_scores = _score_trial_counts(
            base_counts,
            obs=trial_obs,
            var=adata.var,
            gene_sets=gene_sets,
            program_labels=program_labels,
            normalization_target=args.normalization_target,
            score_jobs=args.score_jobs,
            validate_against_bioconcord=repeat == 0,
        )
        baseline_results = _run_methods_quiet(
            base_scores,
            test_labels,
            args.control,
            program_labels,
            methods=methods,
            threads=args.threads,
        )
        LOGGER.info(
            "repeat %d/%d: testing %d pathways",
            repeat + 1,
            args.n_repeats,
            len(pathway_indices),
        )
        pathway_order = rng.permutation(pathway_indices)
        pathway_umi_before = {
            program_labels[pathway_index]: float(
                base_counts[
                    left.size :,
                    pathway_gene_indices[program_labels[pathway_index]],
                ].sum()
            )
            for pathway_index in pathway_order
        }
        trial_specs = [
            (pathway_number, int(pathway_index), float(delta))
            for pathway_number, pathway_index in enumerate(pathway_order, start=1)
            for delta in deltas
        ]

        def run_trial(spec):
            pathway_number, pathway_index, delta = spec
            injected_program = program_labels[pathway_index]
            umi_before = pathway_umi_before[injected_program]
            if delta == 0:
                results = baseline_results
                umi_after = umi_before
            else:
                gene_indices = pathway_gene_indices[injected_program]
                injected_counts = _inject_pathway_counts(
                    base_counts,
                    first_injected_row=left.size,
                    gene_indices=gene_indices,
                    delta_log2fc=delta,
                )
                injected_scores = _score_trial_counts(
                    injected_counts,
                    obs=trial_obs,
                    var=adata.var,
                    gene_sets=gene_sets,
                    program_labels=program_labels,
                    normalization_target=args.normalization_target,
                    score_jobs=1,
                )
                results = _run_methods_quiet(
                    injected_scores,
                    test_labels,
                    args.control,
                    program_labels,
                    methods=methods,
                    threads=args.threads,
                )
                umi_after = float(
                    injected_counts[left.size :, gene_indices].sum()
                )
            return (
                pathway_number,
                injected_program,
                delta,
                umi_before,
                umi_after,
                results,
            )

        with ThreadPoolExecutor(max_workers=args.score_jobs) as executor:
            for (
                pathway_number,
                injected_program,
                delta,
                umi_before,
                umi_after,
                results,
            ) in executor.map(run_trial, trial_specs):
                if delta == deltas[0] and (
                    pathway_number == 1 or pathway_number % 10 == 0
                ):
                    LOGGER.info(
                        "pathway %d/%d in repeat %d/%d: %s",
                        pathway_number,
                        len(pathway_indices),
                        repeat + 1,
                        args.n_repeats,
                        injected_program,
                    )
                observed_multiplier = (
                    umi_after / umi_before if umi_before > 0 else np.nan
                )
                for method, frame in results.items():
                    frame = frame.copy()
                    frame["repeat"] = repeat
                    frame["delta_log2fc"] = delta
                    frame["umi_multiplier"] = 2.0**delta
                    frame["pathway_gene_umi_before"] = umi_before
                    frame["pathway_gene_umi_after"] = umi_after
                    frame["observed_umi_multiplier"] = observed_multiplier
                    frame["injected_program"] = injected_program
                    frame["is_injected"] = frame["program"] == injected_program
                    frame["called"] = frame["fdr"] <= args.fdr
                    detail.append(frame)
                    injected = frame[frame["is_injected"]]
                    untouched = frame[~frame["is_injected"]]
                    trial_summary.append(
                        {
                            "repeat": repeat,
                            "delta_log2fc": delta,
                            "umi_multiplier": 2.0**delta,
                            "pathway_gene_umi_before": umi_before,
                            "pathway_gene_umi_after": umi_after,
                            "observed_umi_multiplier": observed_multiplier,
                            "method": method,
                            "injected_program": injected_program,
                            "tp_called": int(injected["called"].iloc[0]),
                            "false_positive_count": int(untouched["called"].sum()),
                            "n_untouched": len(untouched),
                        }
                    )
                completed_trials += 1
                elapsed = time.perf_counter() - run_started
                trials_per_second = completed_trials / elapsed
                eta_seconds = (
                    (total_trials - completed_trials) / trials_per_second
                    if trials_per_second > 0
                    else float("nan")
                )
                LOGGER.info(
                    "trial %d/%d (%.1f%%): repeat=%d/%d pathway=%d/%d "
                    "program=%s delta=%g elapsed=%s eta=%s",
                    completed_trials,
                    total_trials,
                    100.0 * completed_trials / total_trials,
                    repeat + 1,
                    args.n_repeats,
                    pathway_number,
                    len(pathway_indices),
                    injected_program,
                    delta,
                    _format_duration(elapsed),
                    _format_duration(eta_seconds),
                )

    detail_df = pd.concat(detail, ignore_index=True)
    trial_df = pd.DataFrame(trial_summary)
    trial_sizes = trial_df.groupby(
        ["injected_program", "delta_log2fc", "method"]
    ).agg(n_rows=("repeat", "size"), n_unique_repeats=("repeat", "nunique"))
    if not (
        (trial_sizes.n_rows == args.n_repeats)
        & (trial_sizes.n_unique_repeats == args.n_repeats)
    ).all():
        raise RuntimeError("A Test 0 pathway/delta/method group has incomplete repeats")
    zero = trial_df[np.isclose(trial_df.delta_log2fc, 0.0, rtol=0, atol=0)]
    if not zero.empty:
        if not np.allclose(
            zero.pathway_gene_umi_after,
            zero.pathway_gene_umi_before,
            rtol=0,
            atol=0,
        ):
            raise RuntimeError("Delta 0 changed raw pathway-gene UMI totals")
        nonzero = zero.pathway_gene_umi_before > 0
        if nonzero.any() and not np.allclose(
            zero.loc[nonzero, "observed_umi_multiplier"], 1.0, rtol=0, atol=0
        ):
            raise RuntimeError("Delta 0 did not produce an observed UMI multiplier of 1")
    fourfold = trial_df[np.isclose(trial_df.delta_log2fc, 2.0, rtol=0, atol=0)]
    nonzero = fourfold.pathway_gene_umi_before > 0
    if nonzero.any() and not np.allclose(
        fourfold.loc[nonzero, "pathway_gene_umi_after"],
        4.0 * fourfold.loc[nonzero, "pathway_gene_umi_before"],
        rtol=0,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Delta 2 did not produce an exact 4x raw-UMI multiplier; "
            "verify that --counts-layer contains integer UMI counts"
        )
    pathway_summary = trial_df.groupby(
        ["injected_program", "delta_log2fc", "method"], as_index=False
    ).agg(
        umi_multiplier=("umi_multiplier", "first"),
        observed_umi_multiplier_mean=("observed_umi_multiplier", "mean"),
        n_repeats=("repeat", "size"),
        tp_count=("tp_called", "sum"),
        fp_count=("false_positive_count", "sum"),
        fp_count_median_per_repeat=("false_positive_count", "median"),
    )
    aggregate = pathway_summary.groupby(["delta_log2fc", "method"], as_index=False).agg(
        n_pathways=("injected_program", "size"),
        tp_count_median=("tp_count", "median"),
        tp_count_q10=("tp_count", lambda values: values.quantile(0.1)),
        tp_count_q90=("tp_count", lambda values: values.quantile(0.9)),
        fp_count_median=("fp_count", "median"),
        fp_count_q10=("fp_count", lambda values: values.quantile(0.1)),
        fp_count_q90=("fp_count", lambda values: values.quantile(0.9)),
    )
    detail_df.to_csv(
        os.path.join(tables, f"pathways_test0_per_program__{dataset}.csv"),
        index=False,
    )
    trial_df.to_csv(
        os.path.join(tables, f"pathways_test0_per_repeat__{dataset}.csv"),
        index=False,
    )
    pathway_summary.to_csv(
        os.path.join(tables, f"pathways_test0_per_pathway__{dataset}.csv"),
        index=False,
    )
    aggregate.to_csv(
        os.path.join(tables, f"pathways_test0_summary__{dataset}.csv"),
        index=False,
    )
    _plot(
        pathway_summary,
        os.path.join(plots, f"pathways_test0_calibration__{dataset}.png"),
        f"Pathway Test 0 — {dataset}",
    )
    pu.save_program_legend(
        scored.programs, os.path.join(tables, f"pathways_legend__{dataset}.csv")
    )
    pu.save_run_metadata(
        os.path.join(args.outdir, f"pathways_test0_metadata__{dataset}.json"),
        scored,
        args,
        extra={
            "injection_space": "raw pathway-gene UMI counts before normalization",
            "injection_unit": "raw pathway-gene UMIs multiplied by 2**delta and rounded",
            "injection_design": "one pathway per trial; every pathway repeated independently",
            "reported_outcomes": "TP and untouched-pathway FP counts, never rates",
            "plot_tube": "10th-90th percentile across pathway-specific counts",
            "rescoring": (
                "normalize_total, log1p, then bioconcord-equivalent scoring for all "
                "pathways after each nonzero injection"
            ),
            "scoring_optimization": (
                "vectorized exact bioconcord score_genes_standalone algorithm; "
                "validated against the original scorer on the first baseline split"
            ),
        },
    )
    LOGGER.info(
        "completed pathway Test 0: trials=%d elapsed=%s outdir=%s",
        completed_trials,
        _format_duration(time.perf_counter() - run_started),
        os.path.abspath(args.outdir),
    )


if __name__ == "__main__":
    main()
