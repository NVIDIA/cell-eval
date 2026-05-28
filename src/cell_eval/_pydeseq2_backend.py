import logging
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Mapping, cast

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp

logger = logging.getLogger(__name__)

_PYDESEQ2_KWARG_KEYS = frozenset(
    {
        "min_total_count",
        "fill_filtered",
        "dds_kwargs",
        "stats_kwargs",
    }
)


@dataclass(frozen=True)
class PyDESeq2Options:
    min_total_count: int
    fill_filtered: bool
    dds_kwargs: dict[str, Any]
    stats_kwargs: dict[str, Any]


def compute_pydeseq2_de(
    *,
    adata: ad.AnnData,
    reference: str,
    groupby: str,
    threads: int,
    counts_layer: str | None = None,
    replicate_col: str | None = None,
    pydeseq2_kwargs: Mapping[str, Any] | None = None,
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

    options = _parse_pydeseq2_kwargs(pydeseq2_kwargs)

    if replicate_col is None:
        raise ValueError(
            "PyDESeq2 requires replicate_col so cell-eval can create pseudobulk "
            "samples. Provide an obs column identifying the independent sample unit, "
            "such as 'donor', 'sample', or an experimental batch."
        )

    dds_kwargs = dict(options.dds_kwargs)
    dds_kwargs.setdefault("design", f"~{groupby}")
    dds_kwargs.setdefault("refit_cooks", True)
    dds_kwargs.setdefault("quiet", True)
    dds_kwargs.setdefault("low_memory", True)
    dds_kwargs.setdefault("n_cpus", threads)

    stats_kwargs = dict(options.stats_kwargs)
    stats_kwargs.setdefault("quiet", True)
    stats_kwargs.setdefault("n_cpus", threads)

    counts, metadata = build_pydeseq2_inputs(
        adata=adata,
        groupby=groupby,
        reference=reference,
        counts_layer=counts_layer,
        replicate_col=replicate_col,
        min_total_count=options.min_total_count,
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
                fill_filtered=options.fill_filtered,
            )
        )

    if not frames:
        raise ValueError("PyDESeq2 did not produce any perturbation contrasts")
    return pl.concat(frames)


def _parse_pydeseq2_kwargs(
    pydeseq2_kwargs: Mapping[str, Any] | None,
) -> PyDESeq2Options:
    raw_kwargs = dict(pydeseq2_kwargs or {})
    unknown_keys = sorted(set(raw_kwargs) - _PYDESEQ2_KWARG_KEYS)
    if unknown_keys:
        expected = ", ".join(sorted(_PYDESEQ2_KWARG_KEYS))
        received = ", ".join(unknown_keys)
        raise ValueError(
            f"Unknown PyDESeq2 option(s): {received}. "
            f"Expected top-level keys: {expected}. "
            "Pass PyDESeq2 constructor options under 'dds_kwargs' and "
            "DeseqStats options under 'stats_kwargs'."
        )

    dds_kwargs = _mapping_option(raw_kwargs, "dds_kwargs")
    stats_kwargs = _mapping_option(raw_kwargs, "stats_kwargs")

    return PyDESeq2Options(
        min_total_count=int(raw_kwargs.get("min_total_count", 0)),
        fill_filtered=bool(raw_kwargs.get("fill_filtered", True)),
        dds_kwargs=dds_kwargs,
        stats_kwargs=stats_kwargs,
    )


def _mapping_option(raw_kwargs: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = raw_kwargs.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"pydeseq2_kwargs['{key}'] must be a mapping")
    return dict(value)


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
    _validate_obs_columns(adata, columns=[groupby, replicate_col])

    pert_values = _obs_str_values(adata, groupby)
    replicate_values = _obs_str_values(adata, replicate_col)
    if reference not in set(pert_values):
        raise ValueError(f"Reference '{reference}' not found in adata.obs['{groupby}']")

    matrix = _get_count_matrix(adata, counts_layer=counts_layer)
    obs = pd.DataFrame(
        {
            groupby: pert_values,
            replicate_col: replicate_values,
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
        context = f"{replicate}/{pert}"
        counts = _sum_count_rows(
            matrix,
            np.asarray(row_index, dtype=np.int64),
            context=context,
        )
        counts = _validate_counts(counts, context=context)
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


def _validate_obs_columns(adata: ad.AnnData, *, columns: list[str]) -> None:
    for column in columns:
        if column not in adata.obs.columns:
            raise ValueError(f"Column '{column}' not found in adata.obs")

        values = adata.obs[column]
        if values.isna().any():
            raise ValueError(f"Column '{column}' contains missing values")

        string_values = values.astype(str)
        if (string_values == "").any():
            raise ValueError(f"Column '{column}' contains empty values")


def _obs_str_values(adata: ad.AnnData, column: str) -> np.ndarray:
    return adata.obs[column].astype(str).to_numpy()


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


def _sum_count_rows(
    matrix: Any,
    row_indices: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    subset = matrix[row_indices, :]
    _validate_count_values(subset, context=context)
    if sp.issparse(subset):
        return np.asarray(subset.sum(axis=0)).ravel()
    return np.asarray(subset).sum(axis=0)


def _validate_count_values(values: Any, *, context: str) -> None:
    if sp.issparse(values):
        values = values.data
    else:
        values = np.asarray(values)
    _validate_counts(np.asarray(values), context=context)


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
