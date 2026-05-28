import logging
import os
from importlib import import_module
from itertools import combinations
from typing import Any, Literal, Mapping, Sequence, cast

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
from pdex import pdex

from ._types import DEComparison, DEResults

logger = logging.getLogger(__name__)

DEMethod = Literal["pdex", "pydeseq2"]
KNOWN_DE_METHODS: tuple[DEMethod, ...] = ("pdex", "pydeseq2")

DEInput = pl.DataFrame | pd.DataFrame | str | None
DEInputByMethod = DEInput | Mapping[str, DEInput]


def normalize_de_methods(
    de_method: str | Sequence[str] = "pdex",
    de_methods: str | Sequence[str] | None = None,
) -> tuple[DEMethod, ...]:
    """Normalize a single method or method list into a validated tuple."""
    raw_methods = de_methods if de_methods is not None else de_method
    if isinstance(raw_methods, str):
        methods = [method.strip() for method in raw_methods.split(",")]
    else:
        methods = [str(method).strip() for method in raw_methods]

    normalized: list[DEMethod] = []
    for method in methods:
        if not method:
            continue
        if method not in KNOWN_DE_METHODS:
            raise ValueError(
                f"Unknown DE method '{method}'. Expected one of: {', '.join(KNOWN_DE_METHODS)}"
            )
        if method not in normalized:
            normalized.append(cast(DEMethod, method))

    if not normalized:
        raise ValueError("At least one DE method must be provided")
    return tuple(normalized)


def kwargs_for_de_method(
    de_kwargs: Mapping[str, Any] | None,
    method: DEMethod,
) -> dict[str, Any]:
    """Return kwargs for one DE backend.

    ``de_kwargs`` may either be a flat backend kwargs dictionary, or a nested
    dictionary keyed by backend name.
    """
    if not de_kwargs:
        return {}
    if any(key in KNOWN_DE_METHODS for key in de_kwargs):
        method_kwargs = de_kwargs.get(method, {})
        if method_kwargs is None:
            return {}
        if not isinstance(method_kwargs, Mapping):
            raise TypeError(f"de_kwargs['{method}'] must be a mapping")
        return dict(method_kwargs)
    return dict(de_kwargs)


def de_input_for_method(
    de_input: DEInputByMethod,
    method: DEMethod,
    *,
    multiple_methods: bool,
    label: str,
) -> DEInput:
    """Resolve an optional precomputed DE input for a backend."""
    if isinstance(de_input, Mapping):
        de_input_by_method = cast(Mapping[str, DEInput], de_input)
        return de_input_by_method.get(method)
    if multiple_methods and de_input is not None:
        raise ValueError(
            f"{label} must be a mapping keyed by DE method when running multiple methods"
        )
    return de_input


def _sanitize_path_component(value: str | None) -> str | None:
    return value.replace("/", "-") if value is not None else None


def write_de_frame(
    frame: pl.DataFrame,
    *,
    mode: Literal["pred", "real"],
    outdir: str | None,
    prefix: str | None,
    de_method: DEMethod,
    include_method_in_filename: bool,
) -> None:
    if outdir is None:
        return

    safe_prefix = _sanitize_path_component(prefix)
    parts = []
    if safe_prefix:
        parts.append(safe_prefix)
    if include_method_in_filename:
        parts.append(de_method)
    parts.extend([mode, "de"])
    pathname = "_".join(parts) + ".csv"

    logger.info(f"Writing {mode} {de_method} DE results to: {pathname}")
    frame.write_csv(os.path.join(outdir, pathname))


def build_de_frame(
    *,
    mode: Literal["pred", "real"],
    de_path: DEInput = None,
    adata: ad.AnnData | None = None,
    control_pert: str,
    pert_col: str,
    num_threads: int,
    allow_discrete: bool,
    de_method: DEMethod,
    de_kwargs: Mapping[str, Any] | None = None,
    counts_layer: str | None = None,
    replicate_col: str | None = None,
) -> pl.DataFrame:
    """Load precomputed DE results or compute them with the selected backend."""
    if de_path is not None:
        if de_kwargs:
            logger.warning("DE backend kwargs are ignored when reading DE results")
        return load_de_frame(de_path, mode=mode)

    if adata is None:
        raise ValueError("adata must be provided if de_path is not provided")

    logger.info(f"Computing DE for {mode} data using {de_method}")
    match de_method:
        case "pdex":
            return compute_pdex_de(
                adata=adata,
                reference=control_pert,
                groupby=pert_col,
                threads=num_threads,
                allow_discrete=allow_discrete,
                pdex_kwargs=dict(de_kwargs or {}),
            )
        case "pydeseq2":
            return compute_pydeseq2_de(
                adata=adata,
                reference=control_pert,
                groupby=pert_col,
                threads=num_threads,
                counts_layer=counts_layer,
                replicate_col=replicate_col,
                pydeseq2_kwargs=dict(de_kwargs or {}),
            )
        case _:
            raise ValueError(f"Unknown DE method: {de_method}")


def load_de_frame(
    de_path: pl.DataFrame | pd.DataFrame | str, *, mode: str
) -> pl.DataFrame:
    if isinstance(de_path, str):
        logger.info(f"Reading {mode} DE results from {de_path}")
        return pl.read_csv(
            de_path,
            schema_overrides={
                "target": pl.Utf8,
                "feature": pl.Utf8,
            },
        )
    if isinstance(de_path, pl.DataFrame):
        return de_path
    if isinstance(de_path, pd.DataFrame):
        return pl.from_pandas(de_path)
    raise TypeError(f"Unexpected type for de_path: {type(de_path)}")


def _build_pdex_kwargs(
    reference: str,
    groupby: str,
    threads: int,
    allow_discrete: bool,
    pdex_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pdex_kwargs = pdex_kwargs or {}
    if "reference" not in pdex_kwargs:
        pdex_kwargs["reference"] = reference
    if "groupby" not in pdex_kwargs:
        pdex_kwargs["groupby"] = groupby
    if "threads" not in pdex_kwargs:
        pdex_kwargs["threads"] = threads
    if "is_log1p" not in pdex_kwargs:
        pdex_kwargs["is_log1p"] = not allow_discrete
    return pdex_kwargs


def compute_pdex_de(
    *,
    adata: ad.AnnData,
    reference: str,
    groupby: str,
    threads: int,
    allow_discrete: bool,
    pdex_kwargs: dict[str, Any] | None = None,
) -> pl.DataFrame:
    pdex_kwargs = _build_pdex_kwargs(
        reference=reference,
        groupby=groupby,
        threads=threads,
        allow_discrete=allow_discrete,
        pdex_kwargs=pdex_kwargs or {},
    )
    logger.info(f"Using the following pdex kwargs: {pdex_kwargs}")
    frame = pdex(
        adata=adata,
        mode="ref",
        **pdex_kwargs,
    )
    return cast(pl.DataFrame, frame)


def compute_pydeseq2_de(
    *,
    adata: ad.AnnData,
    reference: str,
    groupby: str,
    threads: int,
    counts_layer: str | None = None,
    replicate_col: str | None = None,
    pydeseq2_kwargs: dict[str, Any] | None = None,
) -> pl.DataFrame:
    try:
        DeseqDataSet = getattr(import_module("pydeseq2.dds"), "DeseqDataSet")
        DeseqStats = getattr(import_module("pydeseq2.ds"), "DeseqStats")
    except ImportError as error:
        raise ImportError(
            "The pydeseq2 backend requires the optional PyDESeq2 dependency. "
            "Install it with `pip install cell-eval[pydeseq2]` or "
            "`uv pip install 'cell-eval[pydeseq2]'`."
        ) from error

    kwargs = dict(pydeseq2_kwargs or {})
    min_total_count = int(kwargs.pop("min_total_count", 0))
    fill_filtered = bool(kwargs.pop("fill_filtered", True))
    dds_kwargs = dict(kwargs.pop("dds_kwargs", {}))
    stats_kwargs = dict(kwargs.pop("stats_kwargs", {}))

    if replicate_col is None:
        raise ValueError(
            "PyDESeq2 requires replicate_col so cell-eval can create pseudobulk "
            "samples. Provide an obs column such as 'batch', 'sample', or 'donor'."
        )

    dds_kwargs.setdefault("design", f"~{groupby}")
    dds_kwargs.setdefault("refit_cooks", True)
    dds_kwargs.setdefault("quiet", True)
    dds_kwargs.setdefault("low_memory", True)
    dds_kwargs.setdefault("n_cpus", threads)
    dds_kwargs.update(kwargs)

    stats_kwargs.setdefault("quiet", True)
    stats_kwargs.setdefault("n_cpus", threads)

    counts, metadata = build_pydeseq2_inputs(
        adata=adata,
        groupby=groupby,
        reference=reference,
        counts_layer=counts_layer,
        replicate_col=replicate_col,
        min_total_count=min_total_count,
    )

    logger.info(
        "Running PyDESeq2 with %d pseudobulk samples and %d genes",
        counts.shape[0],
        counts.shape[1],
    )
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        **dds_kwargs,
    )
    dds.deseq2()

    frames: list[pl.DataFrame] = []
    perturbations = sorted(
        value for value in metadata[groupby].astype(str).unique() if value != reference
    )
    for pert in perturbations:
        logger.info("Running PyDESeq2 contrast %s vs %s", pert, reference)
        stats = DeseqStats(
            dds,
            contrast=[groupby, pert, reference],
            **stats_kwargs,
        )
        stats.summary()
        frames.append(
            normalize_pydeseq2_results(
                stats.results_df,
                target=pert,
                fill_filtered=fill_filtered,
            )
        )

    if not frames:
        raise ValueError("PyDESeq2 did not produce any perturbation contrasts")
    return pl.concat(frames)


def build_pydeseq2_inputs(
    *,
    adata: ad.AnnData,
    groupby: str,
    reference: str,
    counts_layer: str | None,
    replicate_col: str,
    min_total_count: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build pseudobulk count and metadata frames for PyDESeq2."""
    if groupby not in adata.obs.columns:
        raise ValueError(f"Column '{groupby}' not found in adata.obs")
    if replicate_col not in adata.obs.columns:
        raise ValueError(f"Column '{replicate_col}' not found in adata.obs")
    if reference not in set(adata.obs[groupby].astype(str)):
        raise ValueError(f"Reference '{reference}' not found in adata.obs['{groupby}']")

    matrix = _get_count_matrix(adata, counts_layer=counts_layer)
    obs = pd.DataFrame(
        {
            groupby: adata.obs[groupby].astype(str).to_numpy(),
            replicate_col: adata.obs[replicate_col].astype(str).to_numpy(),
        },
        index=adata.obs_names,
    )

    sample_counts: list[np.ndarray] = []
    sample_metadata: list[dict[str, str]] = []
    # Create one pseudobulk sample per replicate/perturbation by summing raw
    # counts across all cells in that group.
    for key, row_index in obs.groupby(
        [replicate_col, groupby], sort=True, observed=True
    ).indices.items():
        replicate, pert = cast(tuple[Any, Any], key)
        counts = _sum_count_rows(matrix, np.asarray(row_index, dtype=np.int64))
        counts = _validate_counts(counts, context=f"{replicate}/{pert}")
        sample_id = f"{replicate}__{pert}"
        sample_counts.append(counts)
        sample_metadata.append(
            {
                "sample": sample_id,
                replicate_col: str(replicate),
                groupby: str(pert),
            }
        )

    counts_array = np.vstack(sample_counts)
    gene_names = np.asarray(adata.var_names, dtype=str)
    if min_total_count > 0:
        keep_genes = counts_array.sum(axis=0) >= min_total_count
        counts_array = counts_array[:, keep_genes]
        gene_names = gene_names[keep_genes]
        if counts_array.shape[1] == 0:
            raise ValueError(
                f"No genes remain after min_total_count={min_total_count} filtering"
            )

    counts_df = pd.DataFrame(
        counts_array,
        index=pd.Index([item["sample"] for item in sample_metadata], dtype=str),
        columns=gene_names,
    )
    metadata = pd.DataFrame(sample_metadata).set_index("sample")

    _validate_pydeseq2_design(metadata, groupby=groupby, reference=reference)
    return counts_df, metadata


def normalize_pydeseq2_results(
    results_df: pd.DataFrame,
    *,
    target: str,
    fill_filtered: bool = True,
) -> pl.DataFrame:
    """Convert a PyDESeq2 result table to the canonical cell-eval DE schema."""
    frame = results_df.copy()
    frame.index.name = "feature"
    frame = frame.reset_index()
    result = (
        pl.from_pandas(frame)
        .rename(
            {
                "log2FoldChange": "log2_fold_change",
                "pvalue": "p_value",
                "padj": "fdr",
            }
        )
        .with_columns(pl.lit(target).alias("target"))
    )
    if fill_filtered:
        result = result.with_columns(
            pl.col("p_value").fill_null(1.0),
            pl.col("fdr").fill_null(1.0),
        )
    return result.select(["target", "feature", "log2_fold_change", "p_value", "fdr"])


def _get_count_matrix(adata: ad.AnnData, *, counts_layer: str | None):
    if counts_layer is None:
        if adata.X is None:
            raise ValueError("adata.X is None")
        return adata.X
    if counts_layer not in adata.layers:
        raise ValueError(
            f"counts_layer '{counts_layer}' not found in adata.layers: {list(adata.layers.keys())}"
        )
    return adata.layers[counts_layer]


def _sum_count_rows(matrix: Any, row_indices: np.ndarray) -> np.ndarray:
    subset = matrix[row_indices, :]
    if sp.issparse(subset):
        return np.asarray(subset.sum(axis=0)).ravel()
    return np.asarray(subset).sum(axis=0)


def _validate_counts(counts: np.ndarray, *, context: str) -> np.ndarray:
    if not np.all(np.isfinite(counts)):
        raise ValueError(f"PyDESeq2 counts contain NaN or infinite values in {context}")
    if np.any(counts < 0):
        raise ValueError(f"PyDESeq2 counts contain negative values in {context}")
    rounded = np.rint(counts)
    if not np.allclose(counts, rounded, atol=1e-6):
        raise ValueError(
            "PyDESeq2 requires raw integer counts. "
            f"Found fractional counts in {context}; provide a raw counts layer "
            "or run with raw-count .X and allow_discrete=True."
        )
    return rounded.astype(np.int64, copy=False)


def _validate_pydeseq2_design(
    metadata: pd.DataFrame,
    *,
    groupby: str,
    reference: str,
) -> None:
    perts = set(metadata[groupby].astype(str))
    if reference not in perts:
        raise ValueError(
            f"Reference '{reference}' missing after pseudobulk aggregation"
        )
    non_reference = perts - {reference}
    if not non_reference:
        raise ValueError("No non-reference perturbations remain for PyDESeq2")
    counts = metadata[groupby].value_counts()
    missing_replication = counts[counts < 2]
    if not missing_replication.empty:
        logger.warning(
            "Some PyDESeq2 conditions have fewer than two pseudobulk samples: %s",
            missing_replication.to_dict(),
        )


def compare_de_backends(
    comparisons: Mapping[str, DEComparison],
    *,
    fdr_threshold: float = 0.05,
    top_ks: Sequence[int] = (50, 100, 200),
) -> pl.DataFrame:
    """Compare DE result tables across backend pairs for real and pred data."""
    rows: list[dict[str, Any]] = []
    for left_method, right_method in combinations(comparisons.keys(), 2):
        left = comparisons[left_method]
        right = comparisons[right_method]
        rows.extend(
            _compare_de_result_pair(
                left.real,
                right.real,
                dataset="real",
                left_method=left_method,
                right_method=right_method,
                fdr_threshold=fdr_threshold,
                top_ks=top_ks,
            )
        )
        rows.extend(
            _compare_de_result_pair(
                left.pred,
                right.pred,
                dataset="pred",
                left_method=left_method,
                right_method=right_method,
                fdr_threshold=fdr_threshold,
                top_ks=top_ks,
            )
        )

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _compare_de_result_pair(
    left: DEResults,
    right: DEResults,
    *,
    dataset: Literal["real", "pred"],
    left_method: str,
    right_method: str,
    fdr_threshold: float,
    top_ks: Sequence[int],
) -> list[dict[str, Any]]:
    target_col = left.target_col
    feature_col = left.feature_col
    left_lfc = left.log2_fold_change_col
    right_lfc = f"{right.log2_fold_change_col}_right"
    left_fdr = left.fdr_col
    right_fdr = f"{right.fdr_col}_right"

    joined = left.data.join(
        right.data,
        on=[target_col, feature_col],
        suffix="_right",
        how="inner",
    )

    rows: list[dict[str, Any]] = []
    for pert in sorted(set(left.get_perts()) & set(right.get_perts())):
        pert_joined = joined.filter(pl.col(target_col) == pert)
        left_pert = left.data.filter(pl.col(target_col) == pert)
        right_pert = right.data.filter(pl.col(target_col) == pert)

        sig_left = set(
            left_pert.filter(pl.col(left.fdr_col) < fdr_threshold)[
                feature_col
            ].to_list()
        )
        sig_right = set(
            right_pert.filter(pl.col(right.fdr_col) < fdr_threshold)[
                feature_col
            ].to_list()
        )
        sig_union = sig_left | sig_right
        row: dict[str, Any] = {
            "dataset": dataset,
            "method_left": left_method,
            "method_right": right_method,
            "target": pert,
            "n_common_genes": pert_joined.height,
            "n_sig_left": len(sig_left),
            "n_sig_right": len(sig_right),
            "sig_jaccard": (
                len(sig_left & sig_right) / len(sig_union) if sig_union else 1.0
            ),
        }

        finite_lfc_joined = _filter_finite_lfc(pert_joined, left_lfc, right_lfc)
        row["n_finite_lfc"] = finite_lfc_joined.height
        if finite_lfc_joined.height >= 2:
            row["lfc_spearman"] = finite_lfc_joined.select(
                pl.corr(
                    pl.col(left_lfc).cast(pl.Float64),
                    pl.col(right_lfc).cast(pl.Float64),
                    method="spearman",
                )
            ).item()
            row["lfc_pearson"] = finite_lfc_joined.select(
                pl.corr(
                    pl.col(left_lfc).cast(pl.Float64),
                    pl.col(right_lfc).cast(pl.Float64),
                    method="pearson",
                )
            ).item()
        else:
            row["lfc_spearman"] = float("nan")
            row["lfc_pearson"] = float("nan")

        sig_joined = pert_joined.filter(
            (pl.col(left_fdr) < fdr_threshold) | (pl.col(right_fdr) < fdr_threshold)
        )
        finite_sig_joined = _filter_finite_lfc(sig_joined, left_lfc, right_lfc)
        row["direction_match_sig_union"] = (
            finite_sig_joined.select(
                (pl.col(left_lfc).sign() == pl.col(right_lfc).sign()).mean()
            ).item()
            if finite_sig_joined.height
            else float("nan")
        )

        for k in top_ks:
            left_top = _top_gene_set(
                left_pert, feature_col, left.abs_log2_fold_change_col, k
            )
            right_top = _top_gene_set(
                right_pert, feature_col, right.abs_log2_fold_change_col, k
            )
            denom = min(k, len(left_top), len(right_top))
            row[f"top_{k}_overlap"] = (
                len(left_top & right_top) / denom if denom else float("nan")
            )

        rows.append(row)
    return rows


def _filter_finite_lfc(
    frame: pl.DataFrame,
    left_lfc_col: str,
    right_lfc_col: str,
) -> pl.DataFrame:
    return frame.filter(
        pl.col(left_lfc_col).cast(pl.Float64).is_finite().fill_null(False)
        & pl.col(right_lfc_col).cast(pl.Float64).is_finite().fill_null(False)
    )


def _top_gene_set(
    frame: pl.DataFrame,
    feature_col: str,
    abs_lfc_col: str,
    k: int,
) -> set[str]:
    return set(
        frame.sort(abs_lfc_col, descending=True)
        .head(k)[feature_col]
        .cast(pl.Utf8)
        .to_list()
    )
