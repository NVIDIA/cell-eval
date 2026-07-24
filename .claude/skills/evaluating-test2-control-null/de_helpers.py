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


def _cpm_eligible_genes(adata: ad.AnnData, groupby: str, pert_label: str,
                         ctrl_label: str, cfg: dict, *, min_cpm: float = 5.0) -> set:
    """Return the set of genes with mean CPM ≥ min_cpm in either perturbed or control cells.

    Shared eligibility filter applied identically to both pdex and pyDESeq2.  Uses raw
    counts from cfg['counts_layer'] when available; otherwise uses expm1(adata.X) for
    log-normalised X or adata.X directly for integer-count X.
    """
    obs_g = adata.obs[groupby].astype(str).to_numpy()
    counts_layer = cfg.get("counts_layer")
    if counts_layer and counts_layer in adata.layers:
        X = adata.layers[counts_layer]
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        X = X.tocsr()
    else:
        X = adata.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        X = X.tocsr()
        sample = X.data[:min(5_000, X.data.size)] if X.data.size > 0 else np.array([0.0])
        if not np.allclose(sample, np.rint(sample)):
            X = X.copy()
            X.data = np.expm1(X.data)

    pos_p = np.where(obs_g == pert_label)[0]
    pos_c = np.where(obs_g == ctrl_label)[0]
    if pos_p.size == 0 or pos_c.size == 0:
        return set(adata.var_names)

    def mean_cpm(pos):
        group = X[pos]
        library_sizes = np.asarray(group.sum(axis=1)).ravel().astype(float)
        scales = np.divide(
            1e6, library_sizes, out=np.zeros_like(library_sizes),
            where=library_sizes > 0,
        )
        return np.asarray(group.multiply(scales[:, None]).mean(axis=0)).ravel()

    m_pert = mean_cpm(pos_p)
    m_ctrl = mean_cpm(pos_c)
    eligible_idx = np.where((m_pert >= min_cpm) | (m_ctrl >= min_cpm))[0]
    return set(adata.var_names[eligible_idx])


def run_de(adata: ad.AnnData, cfg: dict, groupby: str, reference: str) -> pl.DataFrame:
    result = build_de_frame(
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
    # Shared CPM eligibility filter — applied to both pdex and pyDESeq2 identically.
    labels = adata.obs[groupby].astype(str).unique().tolist()
    pert_label = next((lb for lb in labels if lb != reference), None)
    if pert_label is not None:
        eligible = _cpm_eligible_genes(adata, groupby, pert_label, reference, cfg)
        result = result.filter(pl.col("feature").is_in(eligible))
    return result


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
