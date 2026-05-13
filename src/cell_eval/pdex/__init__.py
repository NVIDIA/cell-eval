"""Vendored pdex with cpu (numba) / gpu (rapids-singlecell) backends.

Backend selection is shared with the rest of cell_eval via
:mod:`cell_eval._backend` (env var ``CELL_EVAL_BACKEND``).

Accepted values: ``"cpu"`` / ``"illico"`` (numba) and ``"gpu"`` / ``"rsc"``
(rapids-singlecell). The gpu backend requires ``cupy`` and
``rapids-singlecell``; install with ``pip install cell-eval[gpu]``.
"""

from __future__ import annotations

from typing import Any, Literal

import anndata as ad
import pandas as pd
import polars as pl

from .._backend import Backend as PdexBackend
from .._backend import resolve_backend

PDEX_MODES = Literal["ref", "all", "on_target"]
DEFAULT_REFERENCE = "non-targeting"

__all__ = ["pdex", "DEFAULT_REFERENCE", "PDEX_MODES", "PdexBackend"]


def pdex(
    adata: ad.AnnData,
    groupby: str,
    mode: PDEX_MODES = "ref",
    threads: int = 0,
    is_log1p: bool | None = None,
    geometric_mean: bool = True,
    as_pandas: bool = False,
    epsilon: float = 0.0,
    *,
    backend: PdexBackend | None = None,
    **kwargs: Any,
) -> pl.DataFrame | pd.DataFrame:
    """Run parallel differential expression on either the cpu or gpu backend.

    All positional / keyword arguments other than ``backend`` are forwarded to
    the underlying backend. See ``cell_eval.pdex._illico.pdex`` for the full
    parameter docstring.
    """
    resolved = resolve_backend(backend)
    if resolved == "cpu":
        from ._illico import pdex as _impl
    else:
        try:
            from ._rsc import pdex as _impl
        except ImportError as e:
            raise ImportError(
                "The gpu pdex backend requires cupy and rapids-singlecell. "
                "Install with `pip install cell-eval[gpu]`."
            ) from e
    return _impl(
        adata,
        groupby=groupby,
        mode=mode,
        threads=threads,
        is_log1p=is_log1p,
        geometric_mean=geometric_mean,
        as_pandas=as_pandas,
        epsilon=epsilon,
        **kwargs,
    )
