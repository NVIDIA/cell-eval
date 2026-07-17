"""Self-contained pathway scoring, inference, comparison, and plotting utilities.

The feature engineering is intentionally fixed before inference:

1. score every cell with the bioconcord module-score algorithm;
2. reuse that cell-by-program matrix for every split/permutation;
3. compare reference-coded bioconcord OLS with pdex Mann-Whitney U.

An identical copy of this module is bundled in each standalone pathway skill.
Keep those flat copies byte-identical when changing the statistical contract.
"""

from __future__ import annotations

import json
import importlib.util
import os
import sys
import warnings
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats


METHODS = ("ols", "pdex_mwu")
BIOCONCORD_MODULE_PATH = os.path.join(
    "Src", "bioconcord", "testGeneProgramsConcordance.py"
)
_BIOCONCORD_ROOT: str | None = None
_BIOCONCORD_MODULE = None


class ScoredData:
    def __init__(
        self,
        adata: ad.AnnData,
        scores: np.ndarray,
        programs: dict,
        program_labels: list[str],
        score_source: str,
    ) -> None:
        self.adata = adata
        self.scores = scores
        self.programs = programs
        self.program_labels = program_labels
        self.score_source = score_source


def load_programs(csv_path: str, var_names: Sequence[str], min_genes: int = 5) -> dict:
    """Load one representative gene set per cluster from a user-supplied CSV."""
    frame = pd.read_csv(csv_path)
    required = {"cluster", "representative", "genes"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Program CSV is missing columns: {sorted(missing)}")

    var_set = set(map(str, var_names))
    programs: dict[str, dict] = {}
    for cluster, group in frame.groupby("cluster", sort=True):
        candidates: list[tuple[int, str, list[str]]] = []
        for row in group.itertuples(index=False):
            genes = [g for g in str(row.genes).split() if g in var_set]
            candidates.append((len(genes), str(row.representative), genes))
        n_present, name, genes = max(candidates, key=lambda x: x[0])
        if n_present >= min_genes:
            label = f"C{int(cluster):03d}"
            programs[label] = {"full_name": name, "genes": genes}
    if not programs:
        raise ValueError(f"No programs retain at least {min_genes} genes")
    return programs


def expression_matrix(adata: ad.AnnData, layer: str | None):
    """Return the matrix used by bioconcord-style scoring.

    ``layer=None`` or ``layer='X'`` uses ``adata.X``. The caller is responsible
    for supplying log-normalized expression, as bioconcord itself does.
    """
    if layer in (None, "", "X"):
        matrix = adata.X
        source = "X"
    else:
        if layer not in adata.layers:
            raise ValueError(
                f"Layer {layer!r} not found; available={list(adata.layers)}"
            )
        matrix = adata.layers[layer]
        source = f"layers[{layer!r}]"
    if matrix is None:
        raise ValueError("The selected expression matrix is None")
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix)
    if values.size and not np.isfinite(values).all():
        raise ValueError("The scoring matrix contains NaN or infinite values")
    return matrix, source


def normalize_total_log1p(adata: ad.AnnData, target_sum: float = 10_000.0) -> ad.AnnData:
    """Return a copy whose nonnegative raw-count X is total-normalized and log1p."""
    output = adata.copy()
    matrix = output.X
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix)
    if values.size and (not np.isfinite(values).all() or np.min(values) < 0):
        raise ValueError("Raw-count normalization requires finite nonnegative values")
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    if np.any(totals <= 0):
        raise ValueError("Raw-count normalization found a cell with zero total counts")
    factors = target_sum / totals
    if sp.issparse(matrix):
        normalized = sp.diags(factors) @ matrix.astype(np.float32)
        normalized = normalized.tocsr()
        normalized.data = np.log1p(normalized.data)
    else:
        normalized = np.asarray(matrix, dtype=np.float32) * factors[:, None]
        normalized = np.log1p(normalized)
    output.X = normalized
    return output


def use_gene_symbol_var_names(adata: ad.AnnData, column: str) -> ad.AnnData:
    """Return a copy indexed by a documented gene-symbol column."""
    if column not in adata.var:
        raise ValueError(
            f"Gene-symbol column {column!r} not found; available={list(adata.var.columns)}"
        )
    symbols = adata.var[column].astype(str)
    invalid = symbols.str.lower().isin(["", "nan", "none", "na"])
    if invalid.any():
        raise ValueError(f"Gene-symbol column {column!r} contains missing labels")
    output = adata.copy()
    output.var["source_var_name"] = output.var_names.astype(str)
    output.var_names = symbols.to_numpy()
    output.var_names_make_unique()
    return output


def configure_bioconcord_root(path: str | None) -> None:
    """Select an official ArcInstitute/bioconcord source checkout."""
    global _BIOCONCORD_ROOT, _BIOCONCORD_MODULE
    _BIOCONCORD_ROOT = None
    _BIOCONCORD_MODULE = None
    if not path:
        return
    root = os.path.abspath(os.path.expanduser(path))
    module_path = os.path.join(root, BIOCONCORD_MODULE_PATH)
    if not os.path.isfile(module_path):
        raise FileNotFoundError(
            f"--bioconcord-root does not contain {BIOCONCORD_MODULE_PATH}: {root}"
        )
    _BIOCONCORD_ROOT = root


def _bioconcord_roots() -> list[str]:
    candidates = [_BIOCONCORD_ROOT, os.environ.get("BIOCONCORD_ROOT")]
    cwd = os.path.abspath(os.getcwd())
    candidates.extend([cwd, os.path.join(cwd, "bioconcord")])

    location = os.path.abspath(os.path.dirname(__file__))
    while True:
        candidates.extend([location, os.path.join(location, "bioconcord")])
        parent = os.path.dirname(location)
        if parent == location:
            break
        location = parent

    roots: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        root = os.path.abspath(os.path.expanduser(candidate))
        if root not in roots:
            roots.append(root)
    return roots


def _bioconcord_module():
    """Load scoring and OLS code from an official Bioconcord checkout."""
    global _BIOCONCORD_MODULE
    if _BIOCONCORD_MODULE is not None:
        return _BIOCONCORD_MODULE

    for root in _bioconcord_roots():
        module_path = os.path.join(root, BIOCONCORD_MODULE_PATH)
        if not os.path.isfile(module_path):
            continue
        if root not in sys.path:
            sys.path.insert(0, root)
        spec = importlib.util.spec_from_file_location(
            "bioconcord_official_testGeneProgramsConcordance", module_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load official Bioconcord source: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Current pandas emits boolean dummy columns. Combined with the float
        # intercept, statsmodels otherwise receives an object-dtype design.
        # Keep the official regression implementation and normalize only the
        # return dtype of the statsmodels helper it already calls.
        official_add_constant = module.add_constant

        def add_constant_float(*args, **kwargs):
            return official_add_constant(*args, **kwargs).astype(float)

        module.add_constant = add_constant_float
        _BIOCONCORD_MODULE = module
        return module

    raise ImportError(
        "Official ArcInstitute/bioconcord checkout not found. Clone "
        "https://github.com/ArcInstitute/bioconcord, then pass "
        "--bioconcord-root /path/to/bioconcord or set BIOCONCORD_ROOT."
    )


def load_and_score(
    adata_path: str,
    programs_path: str,
    *,
    score_layer: str | None = None,
    min_genes: int = 5,
    ctrl_size: int = 50,
    n_bins: int = 25,
    seed: int = 42,
    score_jobs: int = 1,
) -> ScoredData:
    adata = ad.read_h5ad(adata_path)
    return score_anndata(
        adata,
        programs_path,
        score_layer=score_layer,
        min_genes=min_genes,
        ctrl_size=ctrl_size,
        n_bins=n_bins,
        seed=seed,
        score_jobs=score_jobs,
    )


def score_anndata(
    adata: ad.AnnData,
    programs_path: str,
    *,
    score_layer: str | None = None,
    min_genes: int = 5,
    ctrl_size: int = 50,
    n_bins: int = 25,
    seed: int = 42,
    score_jobs: int = 1,
) -> ScoredData:
    """Score an already loaded or prefiltered AnnData with Bioconcord."""
    if (ctrl_size, n_bins, seed) != (50, 25, 42):
        raise ValueError(
            "Exact bioconcord scoring fixes ctrl_size=50, n_bins=25, and random_state=42"
        )
    matrix, source = expression_matrix(adata, score_layer)
    programs = load_programs(programs_path, adata.var_names, min_genes=min_genes)
    # bioconcord only reads adata.X. For an explicitly selected layer, expose
    # that layer as X on a copy, then call its public scoring helper unchanged.
    score_adata = adata if source == "X" else adata.copy()
    if source != "X":
        score_adata.X = matrix.copy() if hasattr(matrix, "copy") else matrix
    gene_sets = {label: value["genes"] for label, value in programs.items()}
    bc = _bioconcord_module()
    bc.score_all_programs(score_adata, gene_sets, n_jobs=score_jobs)
    labels = list(gene_sets)
    scores = score_adata.obs[labels].to_numpy(dtype=np.float64)
    return ScoredData(adata, scores, programs, labels, source)


def validate_groups(labels: Sequence[str], reference: str) -> np.ndarray:
    values = np.asarray(labels).astype(str)
    if reference not in set(values):
        raise ValueError(f"Reference {reference!r} is absent")
    if len(set(values) - {reference}) == 0:
        raise ValueError("No non-reference group is present")
    return values


def _bh_by_target(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["fdr"] = np.nan
    for target, idx in frame.groupby("target", sort=False).groups.items():
        frame.loc[idx, "fdr"] = stats.false_discovery_control(
            frame.loc[idx, "p_value"].to_numpy(float), method="bh"
        )
    return frame


def run_ols(
    scores: np.ndarray,
    labels: Sequence[str],
    reference: str,
    program_labels: Sequence[str],
) -> pd.DataFrame:
    """Call bioconcord's reference-coded program regression unchanged.

    Current bioconcord adds a constant: ``const`` is the reference mean and
    every non-reference coefficient is the perturbation-minus-reference
    contrast. The intercept is not emitted as a pathway test row.
    """
    labels = validate_groups(labels, reference)
    y = np.asarray(scores, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] != labels.size:
        raise ValueError("scores must be cells x programs and align with labels")
    obs = pd.DataFrame({"pathway_group": labels})
    for i, program in enumerate(program_labels):
        obs[str(program)] = y[:, i]
    score_adata = ad.AnnData(
        X=np.zeros((len(labels), 1), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["placeholder"]),
    )
    bc = _bioconcord_module()
    wide = bc.run_program_regression(
        score_adata,
        perturbationsColumn="pathway_group",
        referenceLevel=reference,
        pathways=list(program_labels),
    )
    ref_mean = y[labels == reference].mean(axis=0)
    df_resid = len(labels) - len(set(labels))
    rows: list[dict] = []
    targets = [str(value) for value in wide.index if str(value) != "const"]
    for target in targets:
        target_mask = labels == target
        for i, program in enumerate(program_labels):
            coefficient = float(wide.loc[target, f"{program}_coef"])
            p_value = float(wide.loc[target, f"{program}_pval"])
            statistic = float(
                np.sign(coefficient)
                * stats.t.isf(max(p_value, np.finfo(float).tiny) / 2, df=df_resid)
            )
            standard_error = (
                abs(coefficient / statistic)
                if np.isfinite(statistic) and statistic != 0
                else np.nan
            )
            rows.append(
                {
                    "method": "ols",
                    "target": target,
                    "program": program,
                    "effect": coefficient,
                    "mean_difference": float(y[target_mask, i].mean() - ref_mean[i]),
                    "statistic": statistic,
                    "standard_error": standard_error,
                    "p_value": p_value,
                    "n_target": int(target_mask.sum()),
                    "n_reference": int(np.sum(labels == reference)),
                }
            )
    return _bh_by_target(pd.DataFrame(rows))


def run_pdex_mwu(
    scores: np.ndarray,
    labels: Sequence[str],
    reference: str,
    program_labels: Sequence[str],
    *,
    threads: int = 0,
) -> pd.DataFrame:
    """Call pdex's Mann-Whitney implementation on the pathway-score matrix."""
    from pdex import pdex

    labels = validate_groups(labels, reference)
    y = np.asarray(scores, dtype=np.float64)
    score_adata = ad.AnnData(
        X=y,
        obs=pd.DataFrame(
            {"pathway_group": labels}, index=[f"cell_{i}" for i in range(len(labels))]
        ),
        var=pd.DataFrame(index=pd.Index(list(program_labels), dtype=str)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        result = pdex(
            score_adata,
            groupby="pathway_group",
            mode="ref",
            reference=reference,
            threads=threads,
            is_log1p=False,
            geometric_mean=False,
        ).to_pandas()

    result = result.rename(columns={"feature": "program"})
    result["target"] = result["target"].astype(str)
    result["program"] = result["program"].astype(str)
    target_mean = result["target_mean"].to_numpy(float)
    ref_mean = result["ref_mean"].to_numpy(float)
    denominator = result["target_membership"].to_numpy(float) * result[
        "ref_membership"
    ].to_numpy(float)
    rank_biserial = 2.0 * result["statistic"].to_numpy(float) / denominator - 1.0
    return pd.DataFrame(
        {
            "method": "pdex_mwu",
            "target": result["target"],
            "program": result["program"],
            "effect": rank_biserial,
            "mean_difference": target_mean - ref_mean,
            "statistic": result["statistic"].to_numpy(float),
            "standard_error": np.nan,
            "p_value": result["p_value"].to_numpy(float),
            "n_target": result["target_membership"].to_numpy(int),
            "n_reference": result["ref_membership"].to_numpy(int),
            "fdr": result["fdr"].to_numpy(float),
        }
    )


def run_methods(
    scores: np.ndarray,
    labels: Sequence[str],
    reference: str,
    program_labels: Sequence[str],
    *,
    methods: Sequence[str] = METHODS,
    threads: int = 0,
) -> dict[str, pd.DataFrame]:
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    output: dict[str, pd.DataFrame] = {}
    if "ols" in methods:
        output["ols"] = run_ols(scores, labels, reference, program_labels)
    if "pdex_mwu" in methods:
        output["pdex_mwu"] = run_pdex_mwu(
            scores, labels, reference, program_labels, threads=threads
        )
    return output


def effect_matrix(frame: pd.DataFrame, value: str = "effect") -> pd.DataFrame:
    return frame.pivot(index="target", columns="program", values=value).sort_index()


def significant_programs(
    frame: pd.DataFrame, target: str, fdr: float = 0.05
) -> set[str]:
    part = frame[(frame["target"] == target) & (frame["fdr"] <= fdr)]
    return set(part["program"].astype(str))


def jaccard(left: set, right: set) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.unique(x[keep]).size < 2 or np.unique(y[keep]).size < 2:
        return float("nan")
    return float(stats.spearmanr(x[keep], y[keep]).statistic)


def safe_pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Finite, nonconstant Pearson correlation matching safe_spearman guards."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.unique(x[keep]).size < 2 or np.unique(y[keep]).size < 2:
        return float("nan")
    return float(np.corrcoef(x[keep], y[keep])[0, 1])


def diagonal_offdiagonal_means(matrix: np.ndarray) -> tuple[float, float]:
    """Return finite means for matching and nonmatching target pairs."""
    matrix = np.asarray(matrix, dtype=float)
    diagonal_mask = np.eye(matrix.shape[0], dtype=bool)
    return (
        float(np.nanmean(matrix[diagonal_mask])),
        float(np.nanmean(matrix[~diagonal_mask])),
    )


def split_metrics(
    left: pd.DataFrame, right: pd.DataFrame, fdr: float
) -> pd.DataFrame:
    """Summarize split-half agreement for targets shared by both arms."""
    rows = []
    for target in sorted(set(left.target) & set(right.target)):
        a = left[left.target == target].set_index("program")
        b = right[right.target == target].set_index("program")
        programs = sorted(set(a.index) & set(b.index))
        effects_a = a.loc[programs, "effect"].to_numpy(float)
        effects_b = b.loc[programs, "effect"].to_numpy(float)
        significant_a = set(a.index[a.fdr <= fdr])
        significant_b = set(b.index[b.fdr <= fdr])
        active = np.asarray(
            [program in significant_a or program in significant_b for program in programs]
        )
        rows.append(
            {
                "target": target,
                "n_cells_total": int(a.n_target.iloc[0] + b.n_target.iloc[0]),
                "effect_spearman": safe_spearman(effects_a, effects_b),
                "sig_jaccard": jaccard(significant_a, significant_b),
                "direction_agreement_sig_union": (
                    float(
                        np.mean(
                            np.sign(effects_a[active]) == np.sign(effects_b[active])
                        )
                    )
                    if active.any()
                    else np.nan
                ),
                "n_sig_a": len(significant_a),
                "n_sig_b": len(significant_b),
            }
        )
    return pd.DataFrame(rows)


def average_split_effects(
    repeat_results: list[dict[str, dict[str, pd.DataFrame]]],
    methods: Sequence[str],
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


def plot_target_corr_matrix(
    repeat_results: list[dict[str, dict[str, pd.DataFrame]]],
    path: str,
    title: str,
    unit: str = "perturbation",
    sort_by_diagonal: bool = True,
    group_separator: str | None = None,
) -> None:
    """Plot repeat-mean A-by-B correlations over all aligned pathway effects."""
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
    matrix_archive = {"methods": np.asarray(methods, dtype=str)}
    for ax, method in zip(axes[0], methods):
        target_sets = []
        program_sets = []
        for result in repeat_results:
            left = effect_matrix(result["A"][method])
            right = effect_matrix(result["B"][method])
            target_sets.append(set(left.index) & set(right.index))
            program_sets.append(set(left.columns) & set(right.columns))
        targets = sorted(set.intersection(*target_sets))
        programs = sorted(set.intersection(*program_sets))
        repeat_spearman = []
        repeat_pearson = []
        cell_counts = {target: [] for target in targets}
        for result in repeat_results:
            left_frame = result["A"][method]
            right_frame = result["B"][method]
            left = effect_matrix(left_frame).loc[targets, programs]
            right = effect_matrix(right_frame).loc[targets, programs]
            spearman_matrix = np.full((len(targets), len(targets)), np.nan)
            pearson_matrix = np.full((len(targets), len(targets)), np.nan)
            for i, target_a in enumerate(targets):
                for j, target_b in enumerate(targets):
                    effects_a = left.loc[target_a].to_numpy(float)
                    effects_b = right.loc[target_b].to_numpy(float)
                    spearman_matrix[i, j] = safe_spearman(effects_a, effects_b)
                    pearson_matrix[i, j] = safe_pearson(effects_a, effects_b)
            repeat_spearman.append(spearman_matrix)
            repeat_pearson.append(pearson_matrix)
            n_left = left_frame.groupby("target")["n_target"].first()
            n_right = right_frame.groupby("target")["n_target"].first()
            for target in targets:
                cell_counts[target].append(float(n_left[target] + n_right[target]))
        with np.errstate(invalid="ignore"):
            spearman = np.nanmean(np.stack(repeat_spearman), axis=0)
            pearson = np.nanmean(np.stack(repeat_pearson), axis=0)
        diagonal = np.diag(spearman)
        order = (
            np.argsort(np.where(np.isfinite(diagonal), diagonal, np.inf))
            if sort_by_diagonal
            else np.arange(len(targets))
        )
        spearman = spearman[np.ix_(order, order)]
        pearson = pearson[np.ix_(order, order)]
        ordered_targets = [targets[index] for index in order]
        sns.heatmap(
            spearman,
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
        spearman_diagonal, spearman_offdiagonal = diagonal_offdiagonal_means(
            spearman
        )
        pearson_diagonal, pearson_offdiagonal = diagonal_offdiagonal_means(pearson)
        matrix_archive[f"targets__{method}"] = np.asarray(
            ordered_targets, dtype=str
        )
        matrix_archive[f"spearman__{method}"] = spearman
        matrix_archive[f"pearson__{method}"] = pearson
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


def compare_methods(
    results: dict[str, pd.DataFrame], fdr: float = 0.05
) -> pd.DataFrame:
    if set(results) != set(METHODS):
        raise ValueError("Cross-method comparison requires ols and pdex_mwu")
    ols = results["ols"]
    mwu = results["pdex_mwu"]
    targets = sorted(set(ols.target) & set(mwu.target))
    rows = []
    for target in targets:
        left = ols[ols.target == target].set_index("program")
        right = mwu[mwu.target == target].set_index("program")
        programs = sorted(set(left.index) & set(right.index))
        sig_left = significant_programs(ols, target, fdr)
        sig_right = significant_programs(mwu, target, fdr)
        a = left.loc[programs, "effect"].to_numpy(float)
        b = right.loc[programs, "effect"].to_numpy(float)
        active = np.asarray([(p in sig_left or p in sig_right) for p in programs])
        rows.append(
            {
                "target": target,
                "n_cells": int(left["n_target"].iloc[0]),
                "n_sig_ols": len(sig_left),
                "n_sig_pdex_mwu": len(sig_right),
                "sig_jaccard": jaccard(sig_left, sig_right),
                "effect_spearman": safe_spearman(a, b),
                "direction_agreement_sig_union": (
                    float(np.mean(np.sign(a[active]) == np.sign(b[active])))
                    if active.any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def perturbation_covariates(
    obs: pd.DataFrame,
    *,
    pert_col: str,
    umi_col: str,
    targets: Sequence[str],
) -> pd.DataFrame:
    """Summarize cell and UMI counts for the tested perturbations."""
    missing = [column for column in (pert_col, umi_col) if column not in obs]
    if missing:
        raise ValueError(f"Perturbation covariate columns missing from obs: {missing}")

    frame = pd.DataFrame(
        {
            "target": obs[pert_col].astype(str).to_numpy(),
            "umi_count": pd.to_numeric(obs[umi_col], errors="coerce").to_numpy(),
        }
    )
    if not np.isfinite(frame["umi_count"]).all():
        raise ValueError(f"{umi_col!r} contains missing or non-finite values")

    summary = (
        frame.groupby("target", sort=False)["umi_count"]
        .agg(
            n_cells="size",
            median_umi_per_cell="median",
            mean_umi_per_cell="mean",
            total_umi="sum",
        )
        .reset_index()
    )
    target_order = list(map(str, targets))
    summary = summary[summary["target"].isin(target_order)].copy()
    absent = sorted(set(target_order) - set(summary["target"]))
    if absent:
        raise ValueError(f"Tested perturbations absent from obs: {absent}")
    return summary.sort_values("target").reset_index(drop=True)


def raw_count_matrix(adata: ad.AnnData, layer: str | None):
    """Return and validate the raw-count matrix used only for abundance plots."""
    if layer in (None, "", "X"):
        matrix = adata.X
        source = "X"
    else:
        if layer not in adata.layers:
            raise ValueError(
                f"Count layer {layer!r} not found; available={list(adata.layers)}"
            )
        matrix = adata.layers[layer]
        source = f"layers[{layer!r}]"
    if matrix is None:
        raise ValueError("The selected count matrix is None")
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix)
    if values.size and (not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError("The count matrix must be finite and nonnegative")
    return matrix, source


def representative_perturbations(covariates: pd.DataFrame) -> pd.DataFrame:
    """Select distinct least-, median-, and most-cell perturbations."""
    ordered = covariates.sort_values(["n_cells", "target"]).reset_index(drop=True)
    if len(ordered) < 3:
        raise ValueError(
            "Expression-versus-effect plots require at least three perturbations"
        )
    positions = (0, len(ordered) // 2, len(ordered) - 1)
    roles = ("least_cells", "median_cells", "most_cells")
    selected = ordered.iloc[list(positions)][["target", "n_cells"]].copy()
    selected.insert(0, "selection_role", roles)
    return selected.reset_index(drop=True)


def pathway_expression_effects(
    adata: ad.AnnData,
    results: dict[str, pd.DataFrame],
    programs: dict,
    *,
    pert_col: str,
    counts_layer: str | None,
    covariates: pd.DataFrame,
    fdr: float,
) -> pd.DataFrame:
    """Pair raw pathway abundance with native effects for three perturbations.

    Raw abundance is the total UMI count summed over the Cartesian product of
    a perturbation's cells and the retained genes in one pathway. It is not
    normalized by cell count or pathway size.
    """
    if pert_col not in adata.obs:
        raise ValueError(f"{pert_col!r} is absent from adata.obs")
    counts, count_source = raw_count_matrix(adata, counts_layer)
    labels = adata.obs[pert_col].astype(str).to_numpy()
    selected = representative_perturbations(covariates)
    var_positions = pd.Series(
        np.arange(adata.n_vars, dtype=int), index=adata.var_names.astype(str)
    )
    gene_indices = {
        program: var_positions.loc[definition["genes"]].to_numpy(dtype=int)
        for program, definition in programs.items()
    }

    abundance_rows: list[dict] = []
    for selected_row in selected.itertuples(index=False):
        target = str(selected_row.target)
        target_rows = np.flatnonzero(labels == target)
        if target_rows.size != int(selected_row.n_cells):
            raise ValueError(
                f"Cell-count mismatch for {target}: "
                f"obs={target_rows.size}, summary={selected_row.n_cells}"
            )
        target_counts = counts[target_rows]
        for program, indices in gene_indices.items():
            pathway_umi_sum = float(np.asarray(target_counts[:, indices].sum()).item())
            abundance_rows.append(
                {
                    "selection_role": selected_row.selection_role,
                    "target": target,
                    "n_cells": int(target_rows.size),
                    "program": program,
                    "n_program_genes": int(len(indices)),
                    "pathway_umi_sum": pathway_umi_sum,
                    "pathway_umi_sum_plus_one": pathway_umi_sum + 1.0,
                    "count_source": count_source,
                }
            )
    abundance = pd.DataFrame(abundance_rows)

    output: list[pd.DataFrame] = []
    keys = ["target", "program"]
    for method, frame in results.items():
        native = frame[
            ["target", "program", "effect", "mean_difference", "p_value", "fdr"]
        ].copy()
        native["target"] = native["target"].astype(str)
        native["program"] = native["program"].astype(str)
        combined = abundance.merge(native, on=keys, how="left", validate="one_to_one")
        if combined[["effect", "p_value", "fdr"]].isna().any().any():
            raise ValueError(
                f"Missing native pathway results in expression plot for {method}"
            )
        combined.insert(0, "method", method)
        combined["called"] = combined["fdr"] <= fdr
        output.append(combined)
    return pd.concat(output, ignore_index=True)


def composite_blocks(obs: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    columns = [c for c in columns if c]
    if not columns:
        return np.repeat("all", len(obs))
    missing = [c for c in columns if c not in obs.columns]
    if missing:
        raise ValueError(f"Block columns missing from obs: {missing}")
    return obs[columns].astype(str).agg("||".join, axis=1).to_numpy()


def stratified_half(
    indices: np.ndarray, blocks: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    if indices.size < 2:
        return indices.copy(), np.empty(0, dtype=int)
    local_blocks = np.asarray(blocks)[indices]
    left: list[int] = []
    right: list[int] = []
    singleton_toggle = bool(rng.integers(0, 2))
    for block in np.unique(local_blocks):
        members = rng.permutation(indices[local_blocks == block])
        if members.size == 1:
            (left if singleton_toggle else right).append(int(members[0]))
            singleton_toggle = not singleton_toggle
            continue
        cut = members.size // 2
        left.extend(members[:cut].tolist())
        right.extend(members[cut:].tolist())
    if not left or not right:
        members = rng.permutation(indices)
        cut = members.size // 2
        left, right = members[:cut].tolist(), members[cut:].tolist()
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int)


def split_groups(
    labels: Sequence[str],
    blocks: Sequence[str],
    *,
    reference: str,
    min_cells_per_arm: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = np.asarray(labels).astype(str)
    blocks = np.asarray(blocks)
    rng = np.random.default_rng(seed)
    groups = [reference] + sorted(set(labels) - {reference})
    left_all: list[int] = []
    right_all: list[int] = []
    kept: list[str] = []
    for group in groups:
        idx = np.flatnonzero(labels == group)
        left, right = stratified_half(idx, blocks, rng)
        if min(len(left), len(right)) < min_cells_per_arm:
            if group == reference:
                raise ValueError(
                    f"Reference {reference!r} has fewer than {min_cells_per_arm} cells per split arm"
                )
            continue
        kept.append(group)
        left_all.extend(left.tolist())
        right_all.extend(right.tolist())
    if kept == [reference]:
        raise ValueError("No non-reference group passed the split cell threshold")
    return np.asarray(left_all), np.asarray(right_all), kept


def permute_within_blocks(
    labels: Sequence[str], blocks: Sequence[str], seed: int
) -> np.ndarray:
    labels = np.asarray(labels).astype(str)
    blocks = np.asarray(blocks)
    rng = np.random.default_rng(seed)
    output = labels.copy()
    for block in np.unique(blocks):
        idx = np.flatnonzero(blocks == block)
        output[idx] = rng.permutation(output[idx])
    return output


def lambda_gc(p_values: Sequence[float]) -> float:
    p = np.asarray(p_values, float)
    p = np.clip(p[np.isfinite(p)], np.finfo(float).tiny, 1.0)
    if not p.size:
        return float("nan")
    observed = stats.chi2.isf(p, df=1)
    return float(np.median(observed) / stats.chi2.ppf(0.5, df=1))


def method_list(value: str) -> tuple[str, ...]:
    methods = tuple(dict.fromkeys(x.strip() for x in value.split(",") if x.strip()))
    unknown = set(methods) - set(METHODS)
    if unknown or not methods:
        raise ValueError(f"--methods must use {METHODS}; received {methods}")
    return methods


def validate_probability(value: float, name: str) -> None:
    """Require a finite probability or threshold in the closed unit interval."""
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between 0 and 1; received {value}")


def validate_positive_int(value: int, name: str) -> None:
    """Require a strictly positive integer-valued CLI argument."""
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer; received {value}")


def save_program_legend(programs: dict, path: str) -> None:
    rows = [
        {
            "program": key,
            "full_name": value["full_name"],
            "n_genes_in_adata": len(value["genes"]),
        }
        for key, value in programs.items()
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def save_run_metadata(
    path: str, scored: ScoredData, args, extra: dict | None = None
) -> None:
    payload = {
        "score_formula": "mean(program genes) - mean(expression-bin-matched background genes)",
        "score_source": scored.score_source,
        "n_cells": int(scored.adata.n_obs),
        "n_genes": int(scored.adata.n_vars),
        "n_programs": len(scored.program_labels),
        "methods": {
            "ols": "bioconcord run_program_regression unchanged (with intercept); effect=perturbation-minus-reference coefficient",
            "pdex_mwu": "pdex Mann-Whitney U; effect=rank-biserial correlation",
        },
        "fdr_scope": "Benjamini-Hochberg across programs within each target",
        "postprocessing_note": "OLS BH-FDR is added after the unchanged bioconcord regression; mean_difference is retained as a cross-check of the fitted contrast",
        "arguments": vars(args),
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def prepare_output(outdir: str) -> tuple[str, str]:
    plots = os.path.join(outdir, "plots")
    tables = os.path.join(outdir, "tables")
    os.makedirs(plots, exist_ok=True)
    os.makedirs(tables, exist_ok=True)
    return plots, tables


def clear_output_prefix(outdir: str, prefix: str) -> None:
    """Remove files owned by one workflow before reusing its output directory."""
    for directory in (outdir, os.path.join(outdir, "plots"), os.path.join(outdir, "tables")):
        if not os.path.isdir(directory):
            continue
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name.startswith(prefix):
                os.remove(entry.path)


def dataset_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def plot_effect_heatmaps(
    results: dict[str, pd.DataFrame],
    path: str,
    title: str,
    *,
    group_separator: str | None = None,
    unit_label: str = "target",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    methods = list(results)
    matrices = {method: effect_matrix(results[method]) for method in methods}
    limits = {
        method: (
            1.0
            if method == "pdex_mwu"
            else float(np.nanmax(np.abs(matrix.to_numpy()))) or 1.0
        )
        for method, matrix in matrices.items()
    }
    fig, axes = plt.subplots(
        len(methods),
        1,
        figsize=(
            max(12, len(next(iter(matrices.values())).columns) * 0.16),
            5 * len(methods),
        ),
        squeeze=False,
    )
    for ax, method in zip(axes[:, 0], methods):
        matrix = matrices[method]
        vmax = limits[method]
        sns.heatmap(
            matrix,
            cmap="RdBu_r",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            yticklabels=True,
            ax=ax,
            cbar_kws={"label": "effect"},
        )
        ax.set_title(
            f"{method}: "
            f"{'bioconcord coefficient (pert - reference)' if method == 'ols' else 'rank-biserial'}"
        )
        ax.tick_params(axis="x", labelsize=6, rotation=90)
        ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
        ax.set_yticklabels(matrix.index, fontsize=7, rotation=0)
        ax.set_ylabel(unit_label)
        if group_separator:
            groups = [str(target).split(group_separator, 1)[0] for target in matrix.index]
            for boundary in range(1, len(groups)):
                if groups[boundary] != groups[boundary - 1]:
                    ax.axhline(boundary, color="black", linewidth=1.1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_effect_distributions_by_covariate(
    results: dict[str, pd.DataFrame],
    covariates: pd.DataFrame,
    *,
    order_by: str,
    path: str,
    title: str,
) -> None:
    """Plot each perturbation's pathway effects as a boxplot plus pathway dots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _format_count(value: float) -> str:
        return f"{value:,.1f}".rstrip("0").rstrip(".")

    specifications = {
        "n_cells": {
            "tick": lambda row: f"{row.target} (n={int(row.n_cells)})",
            "axis": "Perturbation (ordered by cell count; parentheses show n cells)",
        },
        "median_umi_per_cell": {
            "tick": lambda row: (
                f"{row.target} ({_format_count(row.median_umi_per_cell)})"
            ),
            "axis": (
                "Perturbation (ordered by median UMI count; parentheses show "
                "median UMI per cell)"
            ),
        },
    }
    if order_by not in specifications:
        raise ValueError(f"Unsupported effect-distribution ordering: {order_by!r}")

    ordered = covariates.sort_values([order_by, "target"]).reset_index(drop=True)
    targets = ordered["target"].astype(str).tolist()
    positions = np.arange(1, len(targets) + 1, dtype=float)
    tick_labels = [
        specifications[order_by]["tick"](row) for row in ordered.itertuples()
    ]
    method_styles = {
        "ols": ("#4C72B0", "bioconcord OLS coefficient"),
        "pdex_mwu": ("#DD8452", "pdex MWU rank-biserial effect"),
    }

    fig, axes = plt.subplots(
        len(results),
        1,
        figsize=(max(14, 0.58 * len(targets)), 4.5 * len(results)),
        sharex=True,
        squeeze=False,
    )
    rng = np.random.default_rng(42)
    for ax, (method, frame) in zip(axes[:, 0], results.items()):
        matrix = effect_matrix(frame).reindex(targets)
        distributions = [matrix.loc[target].to_numpy(dtype=float) for target in targets]
        color, effect_label = method_styles.get(method, ("#777777", method))
        boxes = ax.boxplot(
            distributions,
            positions=positions,
            widths=0.56,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.1},
            whiskerprops={"color": color, "linewidth": 1.0},
            capprops={"color": color, "linewidth": 1.0},
        )
        for box in boxes["boxes"]:
            box.set(facecolor=color, edgecolor=color, alpha=0.22, linewidth=1.0)
        for position, values in zip(positions, distributions):
            finite = np.isfinite(values)
            jitter = rng.uniform(-0.2, 0.2, finite.sum())
            ax.scatter(
                position + jitter,
                values[finite],
                s=9,
                color=color,
                edgecolors="none",
                alpha=0.5,
                zorder=3,
            )
        ax.axhline(0, color="0.35", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_ylabel("pathway effect")
        ax.set_title(effect_label, fontsize=11)
        ax.grid(axis="y", color="0.9", linewidth=0.7)

    axes[-1, 0].set_xticks(positions)
    axes[-1, 0].set_xticklabels(tick_labels, rotation=90, fontsize=7)
    axes[-1, 0].set_xlabel(specifications[order_by]["axis"])
    fig.suptitle(f"{title}\nBoxes summarize pathways; each dot is one pathway")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_expression_effect_clouds(
    frame: pd.DataFrame,
    *,
    method: str,
    path: str,
    title: str,
    fdr: float,
) -> None:
    """Plot pathway abundance versus native effect for three perturbations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_frame = frame[frame["method"] == method].copy()
    if method_frame.empty:
        raise ValueError(f"No expression-effect rows are available for {method!r}")
    role_order = ("least_cells", "median_cells", "most_cells")
    role_titles = {
        "least_cells": "least cells",
        "median_cells": "median cells",
        "most_cells": "most cells",
    }
    effect_labels = {
        "ols": "bioconcord OLS coefficient",
        "pdex_mwu": "pdex MWU rank-biserial effect",
    }
    effects = method_frame["effect"].to_numpy(float)
    if method == "pdex_mwu":
        effect_limit = 1.0
    else:
        effect_limit = float(np.nanmax(np.abs(effects))) if effects.size else 1.0
        effect_limit = effect_limit or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True, squeeze=False)
    for ax, role in zip(axes[0], role_order):
        part = method_frame[method_frame["selection_role"] == role]
        if part.empty:
            raise ValueError(f"Missing representative perturbation role {role!r}")
        target = str(part["target"].iloc[0])
        n_cells = int(part["n_cells"].iloc[0])
        not_called = part[~part["called"]]
        called = part[part["called"]]
        ax.scatter(
            not_called["pathway_umi_sum_plus_one"],
            not_called["effect"],
            s=20,
            color="#A9A9A9",
            edgecolors="none",
            alpha=0.55,
            label=f"not called (n={len(not_called)})",
            zorder=2,
        )
        ax.scatter(
            called["pathway_umi_sum_plus_one"],
            called["effect"],
            s=28,
            color="#EF4B55",
            edgecolors="none",
            alpha=0.9,
            label=f"FDR-called (n={len(called)})",
            zorder=3,
        )
        ax.axhline(0, color="0.45", linestyle="--", linewidth=0.9, zorder=1)
        ax.set_xscale("log")
        ax.set_ylim(-1.05 * effect_limit, 1.05 * effect_limit)
        ax.set_title(f"{target}\n{role_titles[role]} (n={n_cells})")
        ax.set_xlabel("total raw UMI summed across pathway genes and cells + 1")
        ax.grid(color="0.9", linewidth=0.6, alpha=0.7)
        ax.legend(loc="best", fontsize=8)
    axes[0, 0].set_ylabel(effect_labels.get(method, f"{method} effect"))
    fig.suptitle(
        f"{method} — {title}\n"
        f"gray = not called · red = BH-FDR ≤ {fdr:g} · each point = one pathway",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_method_count_scatter(summary: pd.DataFrame, path: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8))
    size_scale = 0.75
    sizes = summary["n_cells"].to_numpy(float) * size_scale
    scatter = ax.scatter(
        summary["n_sig_pdex_mwu"],
        summary["n_sig_ols"],
        s=sizes,
        c=summary["n_cells"],
        cmap="viridis",
        alpha=0.8,
    )
    limit = max(summary["n_sig_pdex_mwu"].max(), summary["n_sig_ols"].max(), 1)
    ax.plot([0, limit], [0, limit], "--", color="0.5")
    labels = [f"{row.target}  J={row.sig_jaccard:.2f}" for row in summary.itertuples()]
    x = summary["n_sig_pdex_mwu"].to_numpy(float)
    y = summary["n_sig_ols"].to_numpy(float)
    # Alternate labels to the left and right, then repel them vertically on each
    # side. This keeps labels deterministic and readable without an optional
    # third-party text-layout dependency.
    order = np.lexsort((x, y))
    sides = np.empty(len(summary), dtype=int)
    sides[order] = np.where(np.arange(len(summary)) % 2 == 0, -1, 1)

    def _spread_y(indices: np.ndarray, min_gap: float = 2.1) -> dict[int, float]:
        sorted_indices = indices[np.argsort(y[indices])]
        positions = y[sorted_indices].copy()
        for j in range(1, len(positions)):
            positions[j] = max(positions[j], positions[j - 1] + min_gap)
        top = limit * 1.1
        if len(positions) and positions[-1] > top:
            positions -= positions[-1] - top
            for j in range(len(positions) - 2, -1, -1):
                positions[j] = min(positions[j], positions[j + 1] - min_gap)
        return dict(zip(sorted_indices, positions))

    label_y = {}
    for side in (-1, 1):
        label_y.update(_spread_y(np.flatnonzero(sides == side)))

    for i, (xi, yi, label) in enumerate(zip(x, y, labels)):
        side = sides[i]
        ax.annotate(
            label,
            (xi, yi),
            xytext=(xi + side * 2.6, label_y[i]),
            textcoords="data",
            ha="right" if side < 0 else "left",
            va="center",
            fontsize=6.5,
            arrowprops={"arrowstyle": "->", "color": "0.45", "lw": 0.6},
        )
    ax.set(
        xlabel="significant programs: pdex MWU",
        ylabel="significant programs: OLS",
        title=f"{title}\nLabels: J = Jaccard similarity of called-pathway sets",
    )
    ax.set_xlim(-limit * 0.05, limit * 1.15)
    ax.set_ylim(-limit * 0.05, limit * 1.15)
    fig.colorbar(scatter, ax=ax, label="cell count per perturbation")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
