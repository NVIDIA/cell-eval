"""Self-contained DE adapters for the robustness-evaluation skills.

The statistical implementations come directly from the upstream ``pdex`` and
``pydeseq2`` packages.  This module only performs input preparation, optional
RAPIDS dispatch, and conversion to the shared DE result schema used by the
skills.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
import re
from typing import Any, Literal, Mapping, cast

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import yaml
import scipy.sparse as sp
from pdex import pdex

logger = logging.getLogger(__name__)

DEMethod = Literal["pdex", "pydeseq2"]
DEInput = pl.DataFrame | pd.DataFrame | str | None

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


def _yaml_safe(value: Any) -> Any:
    """Convert resolved runtime values to objects accepted by ``safe_dump``."""
    if isinstance(value, Mapping):
        return {str(key): _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_resolved_config(
    *,
    run_root: str,
    workflow: str,
    dataset: str,
    resolved: Mapping[str, Any],
) -> str:
    """Create run folders and save one immutable, timestamped YAML snapshot."""
    root = Path(run_root).expanduser().resolve()
    config_dir = root / "configs"
    (root / "logs").mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc)
    stamp = created.strftime("%Y%m%dT%H%M%S.%fZ")

    def slug(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
        return cleaned or "run"

    path = config_dir / f"{stamp}__{slug(workflow)}__{slug(dataset)}.yaml"
    payload = {
        "workflow": workflow,
        "dataset": dataset,
        "created_utc": created.isoformat(),
        "run_root": str(root),
        "resolved_config": _yaml_safe(resolved),
    }
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    print(f"Resolved configuration: {path}", flush=True)
    return str(path)


def de_method_label(
    method: str,
    *,
    non_parametric_engine: str = "pdex",
    verbose: bool = False,
) -> str:
    """Return a presentation label without changing stable result-schema keys."""
    if method in {"pdex", "wilcoxon"}:
        if non_parametric_engine == "rsc":
            return "RSC (RAPIDS GPU Wilcoxon)" if verbose else "RSC"
        return "pdex (Arc cell-level Wilcoxon)" if verbose else "pdex"
    if method == "pydeseq2":
        return "PyDESeq2 (pseudobulk DESeq2)" if verbose else "PyDESeq2"
    return str(method)


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
    """Load precomputed DE results or compute them with an upstream backend."""
    if de_path is not None:
        if de_kwargs:
            logger.warning("DE backend kwargs are ignored when reading DE results")
        return load_de_frame(de_path, mode=mode)
    if adata is None:
        raise ValueError("adata must be provided if de_path is not provided")

    logger.info("Computing DE for %s data using %s", mode, de_method)
    if de_method == "pdex":
        return compute_pdex_de(
            adata=adata,
            reference=control_pert,
            groupby=pert_col,
            threads=num_threads,
            allow_discrete=allow_discrete,
            pdex_kwargs=dict(de_kwargs or {}),
        )
    if de_method == "pydeseq2":
        return compute_pydeseq2_de(
            adata=adata,
            reference=control_pert,
            groupby=pert_col,
            threads=num_threads,
            counts_layer=counts_layer,
            replicate_col=replicate_col,
            pydeseq2_kwargs=de_kwargs,
        )
    raise ValueError(f"Unknown DE method: {de_method}")


def load_de_frame(de_path: DEInput, *, mode: str) -> pl.DataFrame:
    if isinstance(de_path, str):
        logger.info("Reading %s DE results from %s", mode, de_path)
        return pl.read_csv(
            de_path,
            schema_overrides={"target": pl.Utf8, "feature": pl.Utf8},
        )
    if isinstance(de_path, pl.DataFrame):
        return de_path
    if isinstance(de_path, pd.DataFrame):
        return pl.from_pandas(de_path)
    raise TypeError(f"Unexpected type for de_path: {type(de_path)}")


def _build_pdex_kwargs(
    *,
    reference: str,
    groupby: str,
    threads: int,
    allow_discrete: bool,
    pdex_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    options = dict(pdex_kwargs or {})
    options.setdefault("reference", reference)
    options.setdefault("groupby", groupby)
    options.setdefault("threads", threads)
    options.setdefault("is_log1p", not allow_discrete)
    return options


def compute_pdex_de(
    *,
    adata: ad.AnnData,
    reference: str,
    groupby: str,
    threads: int,
    allow_discrete: bool,
    pdex_kwargs: Mapping[str, Any] | None = None,
) -> pl.DataFrame:
    """Run Arc pdex directly, or its optional RAPIDS Wilcoxon counterpart."""
    options = dict(pdex_kwargs or {})
    engine = str(options.pop("engine", "pdex"))
    if engine == "rsc":
        return _compute_rsc_de(
            adata=adata,
            reference=reference,
            groupby=groupby,
        )
    if engine != "pdex":
        raise ValueError(
            f"Unknown non-parametric engine {engine!r}; expected 'pdex' or 'rsc'"
        )

    options = _build_pdex_kwargs(
        reference=reference,
        groupby=groupby,
        threads=threads,
        allow_discrete=allow_discrete,
        pdex_kwargs=options,
    )
    logger.info("Using Arc pdex with options: %s", options)
    return cast(pl.DataFrame, pdex(adata=adata, mode="ref", **options))


def _looks_raw_integer(matrix: Any) -> bool:
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix).ravel()
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return False
    if values.size > 5_000_000:
        rng = np.random.default_rng(0)
        values = values[rng.integers(0, values.size, 5_000_000)]
    return bool(np.allclose(values, np.rint(values)))


def _compute_rsc_de(
    *,
    adata: ad.AnnData,
    reference: str,
    groupby: str,
) -> pl.DataFrame:
    try:
        import rapids_singlecell as rsc
        import scanpy as sc
    except ImportError as error:
        raise ImportError(
            "non-parametric engine 'rsc' requires rapids-singlecell, scanpy, "
            "and a CUDA-capable CuPy installation"
        ) from error

    if groupby not in adata.obs:
        raise ValueError(f"groupby column {groupby!r} not found in adata.obs")
    labels = adata.obs[groupby].astype(str).to_numpy()
    levels = sorted(set(labels.tolist()))
    if reference not in levels:
        raise ValueError(f"Reference {reference!r} not found in adata.obs[{groupby!r}]")
    groups = [level for level in levels if level != reference]
    if not groups:
        raise ValueError("RSC did not find a non-reference group")

    work = adata.copy()
    work.obs[groupby] = pd.Categorical(
        labels,
        categories=[reference, *groups],
        ordered=False,
    )
    if _looks_raw_integer(work.X):
        sc.pp.normalize_total(work, inplace=True)
        sc.pp.log1p(work)
    rsc.get.anndata_to_GPU(work)
    key = "_skill_rsc_pdex"
    rsc.tl.rank_genes_groups(
        work,
        groupby=groupby,
        groups=groups,
        reference=reference,
        method="wilcoxon",
        n_genes=work.n_vars,
        tie_correct=True,
        use_raw=False,
        key_added=key,
    )

    result = work.uns[key]
    frames = []
    for group in groups:
        frames.append(
            pl.DataFrame(
                {
                    "target": [group] * work.n_vars,
                    "feature": np.asarray(result["names"][group], dtype=str),
                    "log2_fold_change": np.asarray(
                        result["logfoldchanges"][group],
                        dtype=float,
                    ),
                    "p_value": np.asarray(result["pvals"][group], dtype=float),
                    "fdr": np.asarray(result["pvals_adj"][group], dtype=float),
                }
            )
        )
    return pl.concat(frames, how="vertical")


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
    """Pseudobulk cells by replicate, then run scverse PyDESeq2 directly."""
    try:
        DeseqDataSet = getattr(import_module("pydeseq2.dds"), "DeseqDataSet")
        DeseqStats = getattr(import_module("pydeseq2.ds"), "DeseqStats")
    except ImportError as error:
        raise ImportError(
            "The pydeseq2 backend requires the upstream PyDESeq2 package. "
            "Install it with `pip install pydeseq2`."
        ) from error

    options = _parse_pydeseq2_kwargs(pydeseq2_kwargs)
    if replicate_col is None:
        raise ValueError(
            "PyDESeq2 requires replicate_col so the skill can create pseudobulk "
            "samples (for example donor, sample, plate, or experimental batch)."
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
        "Running upstream PyDESeq2 with %d pseudobulk samples and %d genes",
        counts.shape[0],
        counts.shape[1],
    )
    dds = DeseqDataSet(counts=counts, metadata=metadata, **dds_kwargs)
    dds.deseq2()

    frames: list[pl.DataFrame] = []
    perturbations = sorted(
        value
        for value in metadata[groupby].astype(str).unique()
        if value != reference
    )
    for perturbation in perturbations:
        logger.info(
            "Running PyDESeq2 contrast %s vs %s",
            perturbation,
            reference,
        )
        stats = DeseqStats(
            dds,
            contrast=[groupby, perturbation, reference],
            **stats_kwargs,
        )
        stats.summary()
        frames.append(
            normalize_pydeseq2_results(
                stats.results_df,
                target=perturbation,
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
            "Pass constructor options under 'dds_kwargs' and "
            "DeseqStats options under 'stats_kwargs'."
        )
    return PyDESeq2Options(
        min_total_count=int(raw_kwargs.get("min_total_count", 0)),
        fill_filtered=bool(raw_kwargs.get("fill_filtered", True)),
        dds_kwargs=_mapping_option(raw_kwargs, "dds_kwargs"),
        stats_kwargs=_mapping_option(raw_kwargs, "stats_kwargs"),
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
    """Build one raw-count pseudobulk sample per replicate and condition."""
    _validate_obs_columns(adata, columns=[groupby, replicate_col])
    perturbations = _obs_str_values(adata, groupby)
    replicates = _obs_str_values(adata, replicate_col)
    if reference not in set(perturbations):
        raise ValueError(f"Reference '{reference}' not found in adata.obs['{groupby}']")

    matrix = _get_count_matrix(adata, counts_layer=counts_layer)
    obs = pd.DataFrame(
        {groupby: perturbations, replicate_col: replicates},
        index=adata.obs_names,
    )
    sample_counts: list[np.ndarray] = []
    sample_metadata: list[dict[str, str]] = []
    for key, row_indices in obs.groupby(
        [replicate_col, groupby],
        sort=True,
        observed=True,
    ).indices.items():
        replicate, perturbation = cast(tuple[Any, Any], key)
        context = f"{replicate}/{perturbation}"
        counts = _sum_count_rows(
            matrix,
            np.asarray(row_indices, dtype=np.int64),
            context=context,
        )
        counts = _validate_counts(counts, context=context)
        sample_id = f"{replicate}__{perturbation}"
        sample_counts.append(counts)
        sample_metadata.append(
            {
                "sample": sample_id,
                replicate_col: str(replicate),
                groupby: str(perturbation),
            }
        )

    counts_array = np.vstack(sample_counts)
    gene_names = np.asarray(adata.var_names, dtype=str)
    if min_total_count > 0:
        keep = counts_array.sum(axis=0) >= min_total_count
        counts_array = counts_array[:, keep]
        gene_names = gene_names[keep]
        if counts_array.shape[1] == 0:
            raise ValueError(
                f"No genes remain after min_total_count={min_total_count} filtering"
            )

    counts_df = pd.DataFrame(
        counts_array,
        index=pd.Index(
            [item["sample"] for item in sample_metadata],
            dtype=str,
        ),
        columns=gene_names,
    )
    metadata = pd.DataFrame(sample_metadata).set_index("sample")
    _validate_pydeseq2_design(
        metadata,
        groupby=groupby,
        reference=reference,
    )
    return counts_df, metadata


def normalize_pydeseq2_results(
    results_df: pd.DataFrame,
    *,
    target: str,
    fill_filtered: bool = True,
) -> pl.DataFrame:
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
    return result.select(
        ["target", "feature", "log2_fold_change", "p_value", "fdr"]
    )


def _validate_obs_columns(adata: ad.AnnData, *, columns: list[str]) -> None:
    for column in columns:
        if column not in adata.obs.columns:
            raise ValueError(f"Column '{column}' not found in adata.obs")
        values = adata.obs[column]
        if values.isna().any():
            raise ValueError(f"Column '{column}' contains missing values")
        if (values.astype(str) == "").any():
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
            f"counts_layer '{counts_layer}' not found in adata.layers: "
            f"{list(adata.layers.keys())}"
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
    data = values.data if sp.issparse(values) else np.asarray(values)
    _validate_counts(np.asarray(data), context=context)


def _validate_counts(counts: np.ndarray, *, context: str) -> np.ndarray:
    if not np.all(np.isfinite(counts)):
        raise ValueError(f"PyDESeq2 counts contain NaN or infinite values in {context}")
    if np.any(counts < 0):
        raise ValueError(f"PyDESeq2 counts contain negative values in {context}")
    rounded = np.rint(counts)
    if not np.allclose(counts, rounded, atol=1e-6):
        raise ValueError(
            "PyDESeq2 requires raw integer counts. "
            f"Found fractional counts in {context}; provide a raw counts layer."
        )
    return rounded.astype(np.int64, copy=False)


def _validate_pydeseq2_design(
    metadata: pd.DataFrame,
    *,
    groupby: str,
    reference: str,
) -> None:
    conditions = set(metadata[groupby].astype(str))
    if reference not in conditions:
        raise ValueError(
            f"Reference '{reference}' missing after pseudobulk aggregation"
        )
    if not conditions - {reference}:
        raise ValueError("No non-reference perturbations remain for PyDESeq2")
    counts = metadata[groupby].value_counts()
    missing_replication = counts[counts < 2]
    if not missing_replication.empty:
        logger.warning(
            "Some PyDESeq2 conditions have fewer than two pseudobulk samples: %s",
            missing_replication.to_dict(),
        )
