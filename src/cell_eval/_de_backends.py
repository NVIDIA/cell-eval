import logging
import os
from typing import Any, Literal, Mapping, Sequence, cast

import anndata as ad
import pandas as pd
import polars as pl

from ._de_backend_comparison import compare_de_backends
from ._pdex_backend import _build_pdex_kwargs, compute_pdex_de
from ._pydeseq2_backend import (
    build_pydeseq2_inputs,
    compute_pydeseq2_de,
    normalize_pydeseq2_results,
)

logger = logging.getLogger(__name__)

DEMethod = Literal["pdex", "pydeseq2"]
KNOWN_DE_METHODS: tuple[DEMethod, ...] = ("pdex", "pydeseq2")

DEInput = pl.DataFrame | pd.DataFrame | str | None
DEInputByMethod = DEInput | Mapping[str, DEInput]

__all__ = [
    "DEInput",
    "DEInputByMethod",
    "DEMethod",
    "KNOWN_DE_METHODS",
    "_build_pdex_kwargs",
    "build_de_frame",
    "build_pydeseq2_inputs",
    "compare_de_backends",
    "de_input_for_method",
    "kwargs_for_de_method",
    "load_de_frame",
    "normalize_de_methods",
    "normalize_pydeseq2_results",
    "write_de_frame",
]


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
                f"Unknown DE method '{method}'. "
                f"Expected one of: {', '.join(KNOWN_DE_METHODS)}"
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

    ``de_kwargs`` may be a flat backend kwargs dictionary for single-backend
    runs or a dictionary keyed by backend name for multi-backend runs.
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
                pydeseq2_kwargs=de_kwargs,
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
