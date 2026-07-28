"""Self-contained DE adapters for the robustness-evaluation skills.

The statistical implementations come directly from the upstream ``pdex`` and
``pydeseq2`` packages.  This module only performs input preparation, optional
RAPIDS dispatch, and conversion to the shared DE result schema used by the
skills.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import multiprocessing
import os
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
        "fit_strategy",
        "max_joint_samples",
        "workers",
        "checkpoint_dir",
        "resume",
        "continue_on_error",
        "result_mode",
    }
)


@dataclass(frozen=True)
class PyDESeq2Options:
    min_total_count: int
    fill_filtered: bool
    dds_kwargs: dict[str, Any]
    stats_kwargs: dict[str, Any]
    fit_strategy: str
    max_joint_samples: int
    workers: int
    checkpoint_dir: str | None
    resume: bool
    continue_on_error: bool
    result_mode: str


_PAIRWISE_ADATA: ad.AnnData | None = None
_PAIRWISE_SETTINGS: dict[str, Any] | None = None
_PAIRWISE_CONTROL_COUNTS: pd.DataFrame | None = None
_PAIRWISE_CONTROL_METADATA: pd.DataFrame | None = None


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
    result_mode = str(options.pop("result_mode", "all"))
    copy_input = bool(options.pop("copy", True))
    if result_mode not in {"all", "target_only"}:
        raise ValueError("pdex result_mode must be 'all' or 'target_only'")
    if engine == "rsc":
        result = _compute_rsc_de(
            adata=adata,
            reference=reference,
            groupby=groupby,
            copy_input=copy_input,
        )
    elif engine != "pdex":
        raise ValueError(
            f"Unknown non-parametric engine {engine!r}; expected 'pdex' or 'rsc'"
        )
    else:
        options = _build_pdex_kwargs(
            reference=reference,
            groupby=groupby,
            threads=threads,
            allow_discrete=allow_discrete,
            pdex_kwargs=options,
        )
        logger.info("Using Arc pdex with options: %s", options)
        result = cast(pl.DataFrame, pdex(adata=adata, mode="ref", **options))
    if result_mode == "target_only":
        result = result.filter(
            pl.col("feature").cast(pl.Utf8) == pl.col("target").cast(pl.Utf8)
        )
    return result


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
    copy_input: bool = True,
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
    group_sizes = pd.Series(labels, copy=False).value_counts()
    if int(group_sizes.get(reference, 0)) < 2:
        raise ValueError(
            f"Reference {reference!r} has fewer than two cells and cannot support "
            "RSC Wilcoxon differential expression"
        )
    dropped_groups = [
        level
        for level in levels
        if level != reference and int(group_sizes.get(level, 0)) < 2
    ]
    if dropped_groups:
        logger.warning(
            "RSC omitted %d target group(s) with fewer than two cells: %s",
            len(dropped_groups),
            ", ".join(dropped_groups),
        )
    groups = [
        level
        for level in levels
        if level != reference and int(group_sizes.get(level, 0)) >= 2
    ]
    if not groups:
        raise ValueError("RSC did not find a non-reference group")

    if dropped_groups:
        supported = np.isin(labels, [reference, *groups])
        work = adata[supported].copy()
        labels = labels[supported]
    else:
        work = adata.copy() if copy_input else adata
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
    """Pseudobulk cells by replicate, then run scverse PyDESeq2 directly.

    Large multi-condition inputs automatically use independent target-versus-
    reference fits.  This avoids materialising one enormous samples × genes
    DESeq2 object, reuses the reference pseudobulks, bounds worker memory, and
    optionally checkpoints every completed contrast.
    """
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

    perturbations = sorted(
        value
        for value in set(_obs_str_values(adata, groupby).tolist())
        if value != reference
    )
    if not perturbations:
        raise ValueError("PyDESeq2 did not find any non-reference perturbations")
    sample_count = (
        pd.DataFrame(
            {
                groupby: _obs_str_values(adata, groupby),
                replicate_col: _obs_str_values(adata, replicate_col),
            }
        )
        .drop_duplicates()
        .shape[0]
    )
    strategy = options.fit_strategy
    if strategy == "auto":
        strategy = (
            "pairwise"
            if sample_count > options.max_joint_samples
            else "joint"
        )
    if strategy == "pairwise":
        logger.info(
            "Using pairwise PyDESeq2 fits for %d contrasts (%d joint "
            "pseudobulk samples would exceed max_joint_samples=%d)",
            len(perturbations),
            sample_count,
            options.max_joint_samples,
        )
        return _compute_pydeseq2_pairwise(
            adata=adata,
            reference=reference,
            groupby=groupby,
            threads=threads,
            counts_layer=counts_layer,
            replicate_col=replicate_col,
            perturbations=perturbations,
            options=options,
            dds_kwargs=dds_kwargs,
            stats_kwargs=stats_kwargs,
        )

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
    frames = _fit_pydeseq2_counts(
        counts=counts,
        metadata=metadata,
        groupby=groupby,
        reference=reference,
        perturbations=perturbations,
        fill_filtered=options.fill_filtered,
        dds_kwargs=dds_kwargs,
        stats_kwargs=stats_kwargs,
        DeseqDataSet=DeseqDataSet,
        DeseqStats=DeseqStats,
    )
    if not frames:
        raise ValueError("PyDESeq2 did not produce any perturbation contrasts")
    result = pl.concat(frames)
    if options.result_mode == "target_only":
        result = result.filter(
            pl.col("feature").cast(pl.Utf8) == pl.col("target").cast(pl.Utf8)
        )
    return result


def _fit_pydeseq2_counts(
    *,
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    groupby: str,
    reference: str,
    perturbations: list[str],
    fill_filtered: bool,
    dds_kwargs: Mapping[str, Any],
    stats_kwargs: Mapping[str, Any],
    DeseqDataSet: Any,
    DeseqStats: Any,
) -> list[pl.DataFrame]:
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        **dict(dds_kwargs),
    )
    dds.deseq2()
    frames: list[pl.DataFrame] = []
    for perturbation in perturbations:
        logger.info("Running PyDESeq2 contrast %s vs %s", perturbation, reference)
        stats = DeseqStats(
            dds,
            contrast=[groupby, perturbation, reference],
            **dict(stats_kwargs),
        )
        stats.summary()
        frames.append(
            normalize_pydeseq2_results(
                stats.results_df,
                target=perturbation,
                fill_filtered=fill_filtered,
            )
        )
        del stats
    del dds
    gc.collect()
    return frames


def _condition_pseudobulks(
    *,
    adata: ad.AnnData,
    condition: str,
    groupby: str,
    replicate_col: str,
    counts_layer: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = _obs_str_values(adata, groupby)
    replicates = _obs_str_values(adata, replicate_col)
    matrix = _get_count_matrix(adata, counts_layer=counts_layer)
    condition_rows = np.flatnonzero(labels == condition)
    if condition_rows.size == 0:
        raise ValueError(f"Condition {condition!r} has no cells")

    rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, str]] = []
    for replicate in sorted(set(replicates[condition_rows].tolist())):
        indices = condition_rows[replicates[condition_rows] == replicate]
        context = f"{replicate}/{condition}"
        counts = _sum_count_rows(matrix, indices, context=context)
        rows.append(_validate_counts(counts, context=context))
        metadata_rows.append(
            {
                "sample": f"{replicate}__{condition}",
                replicate_col: str(replicate),
                groupby: str(condition),
            }
        )
    frame = pd.DataFrame(
        np.vstack(rows),
        index=pd.Index([item["sample"] for item in metadata_rows], dtype=str),
        columns=np.asarray(adata.var_names, dtype=str),
    )
    metadata = pd.DataFrame(metadata_rows).set_index("sample")
    return frame, metadata


def _pairwise_counts_for_target(target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if (
        _PAIRWISE_ADATA is None
        or _PAIRWISE_SETTINGS is None
        or _PAIRWISE_CONTROL_COUNTS is None
        or _PAIRWISE_CONTROL_METADATA is None
    ):
        raise RuntimeError("Pairwise PyDESeq2 worker state is not initialized")
    settings = _PAIRWISE_SETTINGS
    target_counts, target_metadata = _condition_pseudobulks(
        adata=_PAIRWISE_ADATA,
        condition=target,
        groupby=settings["groupby"],
        replicate_col=settings["replicate_col"],
        counts_layer=settings["counts_layer"],
    )
    counts = pd.concat([_PAIRWISE_CONTROL_COUNTS, target_counts], axis=0)
    metadata = pd.concat(
        [_PAIRWISE_CONTROL_METADATA, target_metadata],
        axis=0,
    )
    min_total_count = int(settings["min_total_count"])
    if min_total_count > 0:
        keep = counts.sum(axis=0).to_numpy() >= min_total_count
        counts = counts.loc[:, keep]
        if counts.shape[1] == 0:
            raise ValueError(
                f"{target}: no genes remain after "
                f"min_total_count={min_total_count}"
            )
    _validate_pydeseq2_design(
        metadata,
        groupby=settings["groupby"],
        reference=settings["reference"],
    )
    return counts, metadata


def _pairwise_pydeseq2_worker(target: str) -> tuple[str, pl.DataFrame]:
    if _PAIRWISE_SETTINGS is None:
        raise RuntimeError("Pairwise PyDESeq2 worker state is not initialized")
    DeseqDataSet = getattr(import_module("pydeseq2.dds"), "DeseqDataSet")
    DeseqStats = getattr(import_module("pydeseq2.ds"), "DeseqStats")
    counts, metadata = _pairwise_counts_for_target(target)
    frames = _fit_pydeseq2_counts(
        counts=counts,
        metadata=metadata,
        groupby=_PAIRWISE_SETTINGS["groupby"],
        reference=_PAIRWISE_SETTINGS["reference"],
        perturbations=[target],
        fill_filtered=bool(_PAIRWISE_SETTINGS["fill_filtered"]),
        dds_kwargs=_PAIRWISE_SETTINGS["dds_kwargs"],
        stats_kwargs=_PAIRWISE_SETTINGS["stats_kwargs"],
        DeseqDataSet=DeseqDataSet,
        DeseqStats=DeseqStats,
    )
    frame = frames[0]
    if _PAIRWISE_SETTINGS["result_mode"] == "target_only":
        frame = frame.filter(pl.col("feature").cast(pl.Utf8) == target)
    return target, frame


def _spawned_pairwise_pydeseq2_worker(
    payload: tuple[
        str,
        pd.DataFrame,
        pd.DataFrame,
        str,
        str,
        bool,
        dict[str, Any],
        dict[str, Any],
        str,
    ],
) -> tuple[str, pl.DataFrame | None, str | None]:
    """Fit one self-contained contrast in a clean spawned process.

    Passing only the already-pseudobulked pair avoids pickling the full
    AnnData and avoids inheriting BLAS/Numba locks from a forked parent.
    """
    (
        target,
        counts,
        metadata,
        groupby,
        reference,
        fill_filtered,
        dds_kwargs,
        stats_kwargs,
        result_mode,
    ) = payload
    try:
        DeseqDataSet = getattr(import_module("pydeseq2.dds"), "DeseqDataSet")
        DeseqStats = getattr(import_module("pydeseq2.ds"), "DeseqStats")
        frame = _fit_pydeseq2_counts(
            counts=counts,
            metadata=metadata,
            groupby=groupby,
            reference=reference,
            perturbations=[target],
            fill_filtered=fill_filtered,
            dds_kwargs=dds_kwargs,
            stats_kwargs=stats_kwargs,
            DeseqDataSet=DeseqDataSet,
            DeseqStats=DeseqStats,
        )[0]
        if result_mode == "target_only":
            frame = frame.filter(pl.col("feature").cast(pl.Utf8) == target)
        return target, frame, None
    except Exception as error:  # noqa: BLE001
        return target, None, f"{type(error).__name__}: {error}"


def _spawned_pairwise_payloads(
    targets: list[str],
) -> Any:
    if _PAIRWISE_SETTINGS is None:
        raise RuntimeError("Pairwise PyDESeq2 worker state is not initialized")
    settings = _PAIRWISE_SETTINGS
    for target in targets:
        counts, metadata = _pairwise_counts_for_target(target)
        yield (
            target,
            counts,
            metadata,
            settings["groupby"],
            settings["reference"],
            bool(settings["fill_filtered"]),
            dict(settings["dds_kwargs"]),
            dict(settings["stats_kwargs"]),
            str(settings["result_mode"]),
        )


def _available_memory_bytes() -> int | None:
    try:
        fields: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                fields[key] = int(value.strip().split()[0]) * 1024
        return fields.get("MemAvailable")
    except (OSError, ValueError):
        return None


def hardware_worker_limit(
    *,
    requested: int,
    threads_per_worker: int,
    worker_memory_bytes: int,
    max_auto_workers: int = 16,
    memory_fraction: float = 0.60,
) -> int:
    """Resolve a CPU- and RAM-safe process count.

    ``requested=0`` enables automatic selection. Explicit requests are still
    capped so a portable skill command cannot oversubscribe a smaller host.
    """
    if requested < 0:
        raise ValueError("requested workers must be at least 0")
    if threads_per_worker < 1:
        raise ValueError("threads_per_worker must be at least 1")
    if worker_memory_bytes < 1:
        raise ValueError("worker_memory_bytes must be positive")
    cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    cpu_cap = max(1, cpus // threads_per_worker)
    available = _available_memory_bytes()
    memory_cap = (
        max(1, int(available * memory_fraction) // worker_memory_bytes)
        if available is not None
        else 1
    )
    desired = min(max_auto_workers, cpu_cap) if requested == 0 else requested
    selected = max(1, min(desired, cpu_cap, memory_cap))
    logger.info(
        "Hardware worker plan: requested=%d, selected=%d, cpu_cap=%d, "
        "memory_cap=%d, threads/worker=%d, memory/worker=%.2f GiB",
        requested,
        selected,
        cpu_cap,
        memory_cap,
        threads_per_worker,
        worker_memory_bytes / 1024**3,
    )
    return selected


def _safe_pairwise_workers(
    *,
    requested: int,
    threads: int,
    adata: ad.AnnData,
    groupby: str,
    reference: str,
) -> int:
    labels = _obs_str_values(adata, groupby)
    control_cells = int(np.count_nonzero(labels == reference))
    largest_target = max(
        (int(np.count_nonzero(labels == value)) for value in set(labels) - {reference}),
        default=0,
    )
    itemsize = np.dtype(getattr(adata.X, "dtype", np.float64)).itemsize
    pair_bytes = max(
        512 * 1024**2,
        int((control_cells + largest_target) * adata.n_vars * itemsize * 2.5),
    )
    workers = hardware_worker_limit(
        requested=requested,
        threads_per_worker=threads,
        worker_memory_bytes=pair_bytes,
        max_auto_workers=16,
        memory_fraction=0.60,
    )
    return workers


def _checkpoint_path(checkpoint_dir: Path, target: str) -> Path:
    digest = hashlib.sha256(target.encode("utf-8")).hexdigest()[:20]
    return checkpoint_dir / f"contrast_{digest}.parquet"


def _prepare_checkpoint_dir(
    *,
    checkpoint_dir: str | None,
    adata: ad.AnnData,
    groupby: str,
    reference: str,
    replicate_col: str,
    counts_layer: str | None,
    options: PyDESeq2Options,
) -> Path | None:
    if not checkpoint_dir:
        return None
    path = Path(checkpoint_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    source = str(getattr(adata, "filename", "") or "")
    source_stat: dict[str, int] = {}
    if source and os.path.exists(source):
        stat = os.stat(source)
        source_stat = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    obs_identity = pd.DataFrame(
        {
            "group": _obs_str_values(adata, groupby),
            "replicate": _obs_str_values(adata, replicate_col),
        },
        index=adata.obs_names.astype(str),
    )
    obs_hash = hashlib.sha256(
        pd.util.hash_pandas_object(
            obs_identity,
            index=True,
            categorize=True,
        ).to_numpy(dtype=np.uint64).tobytes()
    ).hexdigest()
    signature = {
        "format_version": 2,
        "source": source,
        "source_stat": source_stat,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obs_group_replicate_hash": obs_hash,
        "var_hash": hashlib.sha256(
            "\0".join(map(str, adata.var_names)).encode("utf-8")
        ).hexdigest(),
        "groupby": groupby,
        "reference": reference,
        "replicate_col": replicate_col,
        "counts_layer": counts_layer,
        "min_total_count": options.min_total_count,
        "fill_filtered": options.fill_filtered,
        "result_mode": options.result_mode,
        "dds_kwargs": _yaml_safe(options.dds_kwargs),
        "stats_kwargs": _yaml_safe(options.stats_kwargs),
    }
    metadata_path = path / "metadata.yaml"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as handle:
            cached = yaml.safe_load(handle) or {}
        if cached != signature:
            raise ValueError(
                f"Stale PyDESeq2 checkpoint metadata in {metadata_path}; "
                "choose a new checkpoint directory or remove the stale cache"
            )
    else:
        temporary = path / f".metadata.{os.getpid()}.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(signature, handle, sort_keys=True)
        os.replace(temporary, metadata_path)
    return path


def _write_contrast_checkpoint(
    checkpoint_dir: Path,
    *,
    target: str,
    frame: pl.DataFrame,
) -> None:
    path = _checkpoint_path(checkpoint_dir, target)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    os.replace(temporary, path)


def _read_contrast_checkpoint(
    checkpoint_dir: Path,
    *,
    target: str,
) -> pl.DataFrame | None:
    path = _checkpoint_path(checkpoint_dir, target)
    if not path.exists():
        return None
    frame = pl.read_parquet(path)
    if (
        frame.height > 0
        and set(frame["target"].cast(pl.Utf8).unique().to_list()) != {target}
    ):
        raise ValueError(f"Invalid PyDESeq2 contrast checkpoint: {path}")
    return frame


def _compute_pydeseq2_pairwise(
    *,
    adata: ad.AnnData,
    reference: str,
    groupby: str,
    threads: int,
    counts_layer: str | None,
    replicate_col: str,
    perturbations: list[str],
    options: PyDESeq2Options,
    dds_kwargs: Mapping[str, Any],
    stats_kwargs: Mapping[str, Any],
) -> pl.DataFrame:
    global _PAIRWISE_ADATA
    global _PAIRWISE_SETTINGS
    global _PAIRWISE_CONTROL_COUNTS
    global _PAIRWISE_CONTROL_METADATA

    _PAIRWISE_ADATA = adata
    _PAIRWISE_CONTROL_COUNTS, _PAIRWISE_CONTROL_METADATA = (
        _condition_pseudobulks(
            adata=adata,
            condition=reference,
            groupby=groupby,
            replicate_col=replicate_col,
            counts_layer=counts_layer,
        )
    )
    _PAIRWISE_SETTINGS = {
        "groupby": groupby,
        "reference": reference,
        "replicate_col": replicate_col,
        "counts_layer": counts_layer,
        "min_total_count": options.min_total_count,
        "fill_filtered": options.fill_filtered,
        "dds_kwargs": dict(dds_kwargs),
        "stats_kwargs": dict(stats_kwargs),
        "result_mode": options.result_mode,
    }
    checkpoint_dir = _prepare_checkpoint_dir(
        checkpoint_dir=options.checkpoint_dir,
        adata=adata,
        groupby=groupby,
        reference=reference,
        replicate_col=replicate_col,
        counts_layer=counts_layer,
        options=options,
    )
    completed: dict[str, pl.DataFrame] = {}
    pending: list[str] = []
    for target in perturbations:
        cached = (
            _read_contrast_checkpoint(checkpoint_dir, target=target)
            if checkpoint_dir is not None and options.resume
            else None
        )
        if cached is None:
            pending.append(target)
        else:
            completed[target] = cached
    if completed:
        logger.info(
            "Resumed %d/%d PyDESeq2 contrasts from %s",
            len(completed),
            len(perturbations),
            checkpoint_dir,
        )

    workers = _safe_pairwise_workers(
        requested=options.workers,
        threads=threads,
        adata=adata,
        groupby=groupby,
        reference=reference,
    )
    failures: dict[str, str] = {}

    def record(target: str, frame: pl.DataFrame) -> None:
        completed[target] = frame
        if checkpoint_dir is not None:
            _write_contrast_checkpoint(
                checkpoint_dir,
                target=target,
                frame=frame,
            )
        logger.info(
            "Completed pairwise PyDESeq2 contrast %d/%d: %s",
            len(completed),
            len(perturbations),
            target,
        )

    if workers > 1 and pending:
        # Forking an imported NumPy/SciPy process can inherit locked native
        # mutexes and leave every worker sleeping forever. Spawn clean
        # interpreters and send only bounded pseudobulk pairs instead.
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            processes=min(workers, len(pending)),
            maxtasksperchild=16,
        ) as pool:
            fitted = pool.imap_unordered(
                _spawned_pairwise_pydeseq2_worker,
                _spawned_pairwise_payloads(pending),
                chunksize=1,
            )
            for target, frame, error in fitted:
                if error is None and frame is not None:
                    record(target, frame)
                    continue
                failures[target] = error or "unknown worker failure"
                logger.error(
                    "Pairwise PyDESeq2 contrast failed for %s: %s",
                    target,
                    failures[target],
                )
                if not options.continue_on_error:
                    pool.terminate()
                    raise RuntimeError(
                        f"Pairwise PyDESeq2 contrast failed for {target}: "
                        f"{failures[target]}"
                    )
    else:
        for target in pending:
            try:
                completed_target, frame = _pairwise_pydeseq2_worker(target)
                record(completed_target, frame)
            except Exception as error:
                failures[target] = repr(error)
                logger.exception(
                    "Pairwise PyDESeq2 contrast failed for %s",
                    target,
                )
                if not options.continue_on_error:
                    raise

    if checkpoint_dir is not None and failures:
        failure_path = checkpoint_dir / "failures.yaml"
        temporary = failure_path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(failures, handle, sort_keys=True)
        os.replace(temporary, failure_path)
    if failures:
        logger.warning(
            "PyDESeq2 completed %d/%d contrasts; %d failed%s",
            len(completed),
            len(perturbations),
            len(failures),
            f" (see {checkpoint_dir / 'failures.yaml'})"
            if checkpoint_dir is not None
            else "",
        )
    if not completed:
        raise RuntimeError(
            f"All {len(perturbations)} pairwise PyDESeq2 contrasts failed"
        )
    return pl.concat([completed[target] for target in perturbations if target in completed])


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
    fit_strategy = str(raw_kwargs.get("fit_strategy", "auto"))
    if fit_strategy not in {"auto", "joint", "pairwise"}:
        raise ValueError(
            "pydeseq2 fit_strategy must be 'auto', 'joint', or 'pairwise'"
        )
    max_joint_samples = int(raw_kwargs.get("max_joint_samples", 4096))
    workers = int(raw_kwargs.get("workers", 1))
    if max_joint_samples < 2:
        raise ValueError("pydeseq2 max_joint_samples must be at least 2")
    if workers < 0:
        raise ValueError("pydeseq2 workers must be at least 0")
    result_mode = str(raw_kwargs.get("result_mode", "all"))
    if result_mode not in {"all", "target_only"}:
        raise ValueError("pydeseq2 result_mode must be 'all' or 'target_only'")
    return PyDESeq2Options(
        min_total_count=int(raw_kwargs.get("min_total_count", 0)),
        fill_filtered=bool(raw_kwargs.get("fill_filtered", True)),
        dds_kwargs=_mapping_option(raw_kwargs, "dds_kwargs"),
        stats_kwargs=_mapping_option(raw_kwargs, "stats_kwargs"),
        fit_strategy=fit_strategy,
        max_joint_samples=max_joint_samples,
        workers=workers,
        checkpoint_dir=(
            str(raw_kwargs["checkpoint_dir"])
            if raw_kwargs.get("checkpoint_dir")
            else None
        ),
        resume=bool(raw_kwargs.get("resume", True)),
        continue_on_error=bool(raw_kwargs.get("continue_on_error", True)),
        result_mode=result_mode,
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
