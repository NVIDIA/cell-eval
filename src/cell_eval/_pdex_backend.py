import logging
from typing import Any, cast

import anndata as ad
import polars as pl
from pdex import pdex

logger = logging.getLogger(__name__)


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
