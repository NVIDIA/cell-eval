import logging
import multiprocessing as mp
import os
from typing import Any, Literal

import anndata as ad
import pandas as pd
import polars as pl
import scanpy as sc

from cell_eval.utils import guess_is_lognorm

from ._de_backends import (
    DEInputByMethod,
    DEMethod,
    _build_pdex_kwargs,
    build_de_frame,
    compare_de_backends,
    de_input_for_method,
    kwargs_for_de_method,
    normalize_de_methods,
    write_de_frame,
)
from ._pipeline import MetricPipeline
from ._types import PerturbationAnndataPair, initialize_de_comparison
from .utils import _cast_float16_to_float32

logger = logging.getLogger(__name__)


def _available_cpus() -> int:
    """Return CPUs the current process is allowed to use.

    Uses ``os.sched_getaffinity`` on Linux so SLURM/cgroup/taskset limits are
    respected; falls back to ``mp.cpu_count`` on macOS/Windows where that API
    is unavailable (those platforms typically run locally without cgroup caps).
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return mp.cpu_count()


class MetricsEvaluator:
    """
    Evaluates benchmarking metrics of a predicted and real anndata object.

    Arguments
    =========

    adata_pred: ad.AnnData | str
        Predicted anndata object or path to anndata object.
    adata_real: ad.AnnData | str
        Real anndata object or path to anndata object.
    de_pred: pl.DataFrame | str | None = None
        Predicted differential expression results or path to differential expression results.
        If `None`, differential expression will be computed using parallel_differential_expression
    de_real: pl.DataFrame | str | None = None
        Real differential expression results or path to differential expression results.
        If `None`, differential expression will be computed using parallel_differential_expression
    control_pert: str = "non-targeting"
        Control perturbation name.
    pert_col: str = "target"
        Perturbation column name.
    num_threads: int = -1
        Number of threads for parallel differential expression.
    outdir: str = "./cell-eval-outdir"
        Output directory.
    allow_discrete: bool = False
        Allow discrete data.
    prefix: str | None = None
        Prefix for output files.
    pdex_kwargs: dict[str, Any] | None = None
        Keyword arguments for parallel_differential_expression.
        These will overwrite arguments passed to MetricsEvaluator.__init__ if they conflict.
    de_method: str = "pdex"
        Differential-expression backend to use. One of "pdex" or "pydeseq2".
    de_methods: list[str] | str | None = None
        Optional list/comma-separated list of DE backends to run in one evaluation.
        If provided, this overrides de_method.
    de_kwargs: dict[str, Any] | None = None
        Keyword arguments for DE backends. For multi-backend runs, this may be
        nested by backend name, e.g. {"pdex": {...}, "pydeseq2": {...}}.
    counts_layer: str | None = None
        AnnData layer containing raw integer counts for PyDESeq2. If omitted,
        PyDESeq2 uses .X and validates it contains raw integer counts.
    replicate_col: str | None = None
        AnnData obs column used to create PyDESeq2 pseudobulk replicates.
    """

    def __init__(
        self,
        adata_pred: ad.AnnData | str,
        adata_real: ad.AnnData | str,
        de_pred: DEInputByMethod = None,
        de_real: DEInputByMethod = None,
        control_pert: str = "non-targeting",
        pert_col: str = "target",
        num_threads: int = -1,
        outdir: str = "./cell-eval-outdir",
        allow_discrete: bool = False,
        prefix: str | None = None,
        pdex_kwargs: dict[str, Any] | None = None,
        de_method: str | list[str] = "pdex",
        de_methods: str | list[str] | None = None,
        de_kwargs: dict[str, Any] | None = None,
        counts_layer: str | None = None,
        replicate_col: str | None = None,
        skip_de: bool = False,
    ):
        # Enable a global string cache for categorical columns
        pl.enable_string_cache()

        normalized_de_methods = normalize_de_methods(de_method, de_methods)
        if pdex_kwargs is not None:
            if de_kwargs is not None:
                raise ValueError("Use either pdex_kwargs or de_kwargs, not both")
            de_kwargs = {"pdex": pdex_kwargs}

        if num_threads == -1:
            num_threads = _available_cpus()

        if os.path.exists(outdir):
            logger.warning(
                f"Output directory {outdir} already exists, potential overwrite occurring"
            )
        os.makedirs(outdir, exist_ok=True)

        self.anndata_pair = _build_anndata_pair(
            real=adata_real,
            pred=adata_pred,
            control_pert=control_pert,
            pert_col=pert_col,
            allow_discrete=allow_discrete,
        )

        if skip_de:
            self.de_comparison = None
            self.de_comparisons: dict[str, Any] = {}
            self.de_backend_comparison = pl.DataFrame()
        else:
            self.de_comparisons = _build_de_comparisons(
                anndata_pair=self.anndata_pair,
                de_pred=de_pred,
                de_real=de_real,
                num_threads=num_threads,
                allow_discrete=allow_discrete,
                outdir=outdir,
                prefix=prefix,
                de_methods=normalized_de_methods,
                de_kwargs=de_kwargs or {},
                counts_layer=counts_layer,
                replicate_col=replicate_col,
            )
            self.de_comparison = next(iter(self.de_comparisons.values()))
            self.de_backend_comparison = compare_de_backends(self.de_comparisons)
            if self.de_backend_comparison.height > 0:
                safe_prefix = prefix.replace("/", "-") if prefix is not None else None
                basename = (
                    f"{safe_prefix}_de_backend_comparison.csv"
                    if safe_prefix
                    else "de_backend_comparison.csv"
                )
                outpath = os.path.join(outdir, basename)
                logger.info(f"Writing DE backend comparison to {outpath}")
                self.de_backend_comparison.write_csv(outpath)

        self.outdir = outdir
        self.prefix = prefix

    def compute(
        self,
        profile: Literal["full", "vcc", "minimal", "de", "anndata"] = "full",
        metric_configs: dict[str, dict[str, Any]] | None = None,
        skip_metrics: list[str] | None = None,
        basename: str = "results.csv",
        write_csv: bool = True,
        break_on_error: bool = False,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        if len(self.de_comparisons) > 1:
            return self._compute_multi_de_method(
                profile=profile,
                metric_configs=metric_configs,
                skip_metrics=skip_metrics,
                basename=basename,
                write_csv=write_csv,
                break_on_error=break_on_error,
            )

        pipeline = MetricPipeline(
            profile=profile,
            metric_configs=metric_configs,
            break_on_error=break_on_error,
        )
        if skip_metrics is not None:
            pipeline.skip_metrics(skip_metrics)
        pipeline.compute_de_metrics(self.de_comparison)
        pipeline.compute_anndata_metrics(self.anndata_pair)
        results = pipeline.get_results()
        agg_results = pipeline.get_agg_results()

        if write_csv:
            if self.prefix is not None:
                self.prefix = self.prefix.replace(
                    "/", "-"
                )  # some prefixes (e.g. HepG2/C3A) may have slashes in them
            if basename is not None:
                basename = basename.replace(
                    "/", "-"
                )  # some basenames (e.g. HepG2/C3A_results.csv) may have slashes in them
            outpath = os.path.join(
                self.outdir,
                f"{self.prefix}_{basename}" if self.prefix else basename,
            )
            agg_outpath = os.path.join(
                self.outdir,
                f"{self.prefix}_agg_{basename}" if self.prefix else f"agg_{basename}",
            )

            logger.info(f"Writing perturbation level metrics to {outpath}")
            results.write_csv(outpath)

            logger.info(f"Writing aggregate metrics to {agg_outpath}")
            agg_results.write_csv(agg_outpath)

        return results, agg_results

    def _compute_multi_de_method(
        self,
        profile: Literal["full", "vcc", "minimal", "de", "anndata"] = "full",
        metric_configs: dict[str, dict[str, Any]] | None = None,
        skip_metrics: list[str] | None = None,
        basename: str = "results.csv",
        write_csv: bool = True,
        break_on_error: bool = False,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        anndata_pipeline = MetricPipeline(
            profile=profile,
            metric_configs=metric_configs,
            break_on_error=break_on_error,
        )
        if skip_metrics is not None:
            anndata_pipeline.skip_metrics(skip_metrics)
        anndata_pipeline.compute_anndata_metrics(self.anndata_pair)
        anndata_results = anndata_pipeline.get_results()

        results_by_method: list[pl.DataFrame] = []
        agg_by_method: list[pl.DataFrame] = []
        for de_method, de_comparison in self.de_comparisons.items():
            de_pipeline = MetricPipeline(
                profile=profile,
                metric_configs=metric_configs,
                break_on_error=break_on_error,
            )
            if skip_metrics is not None:
                de_pipeline.skip_metrics(skip_metrics)
            de_pipeline.compute_de_metrics(de_comparison)
            de_results = de_pipeline.get_results()
            results = _merge_metric_results(de_results, anndata_results)
            if not results.is_empty():
                results = results.with_columns(pl.lit(de_method).alias("de_method"))
                results = results.select("de_method", pl.all().exclude("de_method"))

            agg_results = _describe_metric_results(results)
            if not agg_results.is_empty():
                agg_results = agg_results.with_columns(
                    pl.lit(de_method).alias("de_method")
                )
                agg_results = agg_results.select(
                    "de_method", pl.all().exclude("de_method")
                )

            results_by_method.append(results)
            agg_by_method.append(agg_results)

        results = pl.concat(results_by_method, how="diagonal")
        agg_results = pl.concat(agg_by_method, how="diagonal")

        if write_csv:
            if self.prefix is not None:
                self.prefix = self.prefix.replace("/", "-")
            if basename is not None:
                basename = basename.replace("/", "-")
            outpath = os.path.join(
                self.outdir,
                f"{self.prefix}_{basename}" if self.prefix else basename,
            )
            agg_outpath = os.path.join(
                self.outdir,
                f"{self.prefix}_agg_{basename}" if self.prefix else f"agg_{basename}",
            )

            logger.info(f"Writing perturbation level metrics to {outpath}")
            results.write_csv(outpath)

            logger.info(f"Writing aggregate metrics to {agg_outpath}")
            agg_results.write_csv(agg_outpath)

        return results, agg_results


def _build_anndata_pair(
    real: ad.AnnData | str,
    pred: ad.AnnData | str,
    control_pert: str,
    pert_col: str,
    allow_discrete: bool = False,
):
    if isinstance(real, str):
        logger.info(f"Reading real anndata from {real}")
        real = ad.read_h5ad(real)
    if isinstance(pred, str):
        logger.info(f"Reading pred anndata from {pred}")
        pred = ad.read_h5ad(pred)

    # Cast float16 to float32 since NUMBA (used by pdex) does not support float16
    _cast_float16_to_float32(real, which="real")
    _cast_float16_to_float32(pred, which="pred")

    # Validate that the input is normalized and log-transformed
    _convert_to_normlog(real, which="real", allow_discrete=allow_discrete)
    _convert_to_normlog(pred, which="pred", allow_discrete=allow_discrete)

    # Build the anndata pair
    return PerturbationAnndataPair(
        real=real, pred=pred, control_pert=control_pert, pert_col=pert_col
    )


def _convert_to_normlog(
    adata: ad.AnnData,
    which: str | None = None,
    allow_discrete: bool = False,
):
    """Performs a norm-log conversion if the input is integer data (inplace).

    Will skip if the input is not integer data.
    """
    if guess_is_lognorm(adata=adata, validate=not allow_discrete):
        logger.info(
            "Input is found to be log-normalized already - skipping transformation."
        )
        return  # Input is already log-normalized

    # User specified that they want to allow discrete data
    if allow_discrete:
        if which:
            logger.info(
                f"Discovered integer data for {which}. Configuration set to allow discrete. "
                "Make sure this is intentional."
            )
        else:
            logger.info(
                "Discovered integer data. Configuration set to allow discrete. "
                "Make sure this is intentional."
            )
        return  # proceed without conversion

    # Convert the data to norm-log
    if which:
        logger.info(f"Discovered integer data for {which}. Converting to norm-log.")
    sc.pp.normalize_total(adata=adata, inplace=True)  # normalize to median
    sc.pp.log1p(adata)  # log-transform (log1p)


def _build_de_comparison(
    anndata_pair: PerturbationAnndataPair | None = None,
    de_pred: pl.DataFrame | str | None = None,
    de_real: pl.DataFrame | str | None = None,
    num_threads: int = 1,
    allow_discrete: bool = False,
    outdir: str | None = None,
    prefix: str | None = None,
    pdex_kwargs: dict[str, Any] | None = None,
):
    return _build_de_comparisons(
        anndata_pair=anndata_pair,
        de_pred=de_pred,
        de_real=de_real,
        num_threads=num_threads,
        allow_discrete=allow_discrete,
        outdir=outdir,
        prefix=prefix,
        de_methods=("pdex",),
        de_kwargs={"pdex": pdex_kwargs or {}},
    )["pdex"]


def _build_de_comparisons(
    anndata_pair: PerturbationAnndataPair | None = None,
    de_pred: DEInputByMethod = None,
    de_real: DEInputByMethod = None,
    num_threads: int = 1,
    allow_discrete: bool = False,
    outdir: str | None = None,
    prefix: str | None = None,
    de_methods: tuple[DEMethod, ...] = ("pdex",),
    de_kwargs: dict[str, Any] | None = None,
    counts_layer: str | None = None,
    replicate_col: str | None = None,
) -> dict[str, Any]:
    comparisons = {}
    multiple_methods = len(de_methods) > 1
    for de_method in de_methods:
        real = _load_or_build_de(
            mode="real",
            de_path=de_input_for_method(
                de_real,
                de_method,
                multiple_methods=multiple_methods,
                label="de_real",
            ),
            anndata_pair=anndata_pair,
            num_threads=num_threads,
            allow_discrete=allow_discrete,
            outdir=outdir,
            prefix=prefix,
            de_method=de_method,
            de_kwargs=kwargs_for_de_method(de_kwargs, de_method),
            counts_layer=counts_layer,
            replicate_col=replicate_col,
            include_method_in_filename=multiple_methods or de_method != "pdex",
        )
        pred = _load_or_build_de(
            mode="pred",
            de_path=de_input_for_method(
                de_pred,
                de_method,
                multiple_methods=multiple_methods,
                label="de_pred",
            ),
            anndata_pair=anndata_pair,
            num_threads=num_threads,
            allow_discrete=allow_discrete,
            outdir=outdir,
            prefix=prefix,
            de_method=de_method,
            de_kwargs=kwargs_for_de_method(de_kwargs, de_method),
            counts_layer=counts_layer,
            replicate_col=replicate_col,
            include_method_in_filename=multiple_methods or de_method != "pdex",
        )
        comparisons[de_method] = initialize_de_comparison(real=real, pred=pred)
    return comparisons


def _load_or_build_de(
    mode: Literal["pred", "real"],
    de_path: pl.DataFrame | pd.DataFrame | str | None = None,
    anndata_pair: PerturbationAnndataPair | None = None,
    num_threads: int = 1,
    outdir: str | None = None,
    prefix: str | None = None,
    allow_discrete: bool = False,
    pdex_kwargs: dict[str, Any] | None = None,
    de_method: DEMethod = "pdex",
    de_kwargs: dict[str, Any] | None = None,
    counts_layer: str | None = None,
    replicate_col: str | None = None,
    include_method_in_filename: bool = False,
) -> pl.DataFrame:
    if pdex_kwargs is not None:
        if de_kwargs is not None:
            raise ValueError("Use either pdex_kwargs or de_kwargs, not both")
        de_kwargs = pdex_kwargs

    if anndata_pair is None and de_path is None:
        raise ValueError("anndata_pair must be provided if de_path is not provided")

    adata = None
    control_pert = ""
    pert_col = ""
    if anndata_pair is not None:
        adata = anndata_pair.real if mode == "real" else anndata_pair.pred
        control_pert = anndata_pair.control_pert
        pert_col = anndata_pair.pert_col

    frame = build_de_frame(
        mode=mode,
        de_path=de_path,
        adata=adata,
        control_pert=control_pert,
        pert_col=pert_col,
        num_threads=num_threads,
        allow_discrete=allow_discrete,
        de_method=de_method,
        de_kwargs=de_kwargs or {},
        counts_layer=counts_layer,
        replicate_col=replicate_col,
    )
    write_de_frame(
        frame,
        mode=mode,
        outdir=outdir if de_path is None else None,
        prefix=prefix,
        de_method=de_method,
        include_method_in_filename=include_method_in_filename,
    )
    return frame


def _merge_metric_results(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    if left.is_empty():
        return right
    if right.is_empty():
        return left
    return left.join(right, on="perturbation", how="full", coalesce=True)


def _describe_metric_results(results: pl.DataFrame) -> pl.DataFrame:
    if results.is_empty():
        return pl.DataFrame()
    metric_results = results.drop(
        [
            column
            for column in ["de_method", "perturbation"]
            if column in results.columns
        ]
    )
    if metric_results.is_empty():
        return pl.DataFrame()
    return metric_results.describe()
