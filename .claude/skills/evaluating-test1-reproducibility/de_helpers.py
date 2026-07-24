"""Shared DE helpers used by the individual robustness test scripts.

Extracted from the former evaluating-metrics-robustness/run_template.py.
Imported at runtime via importlib (the _load_runner() pattern in each test script).
"""
from __future__ import annotations

import numpy as np
import polars as pl
import scanpy as sc
import scipy.sparse as sp
import anndata as ad
from scipy import stats

from de_backends import build_de_frame

CHI2_MEDIAN_1DF = 0.4549  # chi2.ppf(0.5, df=1)


def _looks_raw_integer(adata: ad.AnnData) -> bool:
    X = adata.X
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        return False
    if data.size > 5_000_000:
        rng = np.random.default_rng(0)
        data = data[rng.integers(0, data.size, 5_000_000)]
    return bool(np.allclose(data, np.rint(data)))


def maybe_normalize(adata: ad.AnnData, cfg: dict) -> None:
    if not cfg.get("normalize_if_raw"):
        return
    if _looks_raw_integer(adata):
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)


def run_de(adata: ad.AnnData, cfg: dict, groupby: str, reference: str) -> pl.DataFrame:
    return build_de_frame(
        mode="real",
        adata=adata,
        control_pert=reference,
        pert_col=groupby,
        num_threads=cfg["num_threads"],
        allow_discrete=cfg["allow_discrete"],
        de_method=cfg["de_method"],
        de_kwargs=({"engine": cfg.get("non_parametric_engine", "pdex")}
                   if cfg["de_method"] == "pdex" else None),
        counts_layer=cfg.get("counts_layer"),
        replicate_col=cfg.get("replicate_col"),
    )


def _sig_mask(lfc, fdr, cfg):
    return np.isfinite(fdr) & (fdr <= cfg["fdr_threshold"]) & (np.abs(lfc) >= cfg["lfc_threshold"])


def lambda_gc(p: np.ndarray) -> float:
    p = p[np.isfinite(p)]
    p = np.clip(p, 1e-300, 1.0)
    if p.size == 0:
        return float("nan")
    chi2 = stats.chi2.isf(p, df=1)
    return float(np.median(chi2) / CHI2_MEDIAN_1DF)


def stratified_split(obs, strat_cols, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols = [c for c in strat_cols if c in obs.columns]
    out = np.empty(len(obs), dtype=object)
    if cols:
        groups = obs.groupby(cols, observed=True, sort=False).indices.values()
    else:
        groups = [np.arange(len(obs))]
    for idx in groups:
        idx = np.asarray(idx, dtype=int)
        rng.shuffle(idx)
        half = idx.size // 2
        out[idx[:half]] = "A"
        out[idx[half:]] = "B"
    return out.astype(str)


def _de_two(adata, pos1, pos2, cfg, lab1="G1", lab2="G2"):
    """DE of cells at global positions pos1 vs pos2, reference=lab2."""
    sel = np.concatenate([np.asarray(pos1, int), np.asarray(pos2, int)])
    s = adata[sel].copy()
    s.obs["_g"] = np.array([lab1] * len(pos1) + [lab2] * len(pos2))
    return run_de(s, cfg, groupby="_g", reference=lab2)
