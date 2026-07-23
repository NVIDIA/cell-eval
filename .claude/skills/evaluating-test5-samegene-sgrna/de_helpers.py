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
import pandas as pd
from scipy import stats

from cell_eval._de_backends import build_de_frame

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
    if cfg.get("non_parametric_engine") == "rsc":
        # RSC receives small guide/control subsets in _de_two. Keep the master
        # AnnData raw for CPM and normalize each subset immediately before GPU transfer.
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
        de_kwargs=None,
        counts_layer=cfg.get("counts_layer"),
        replicate_col=cfg.get("replicate_col"),
    )


def _sig_mask(lfc, fdr, cfg):
    return np.isfinite(fdr) & (fdr <= cfg["fdr_threshold"]) & (np.abs(lfc) >= cfg["lfc_threshold"])


def _compute_cpm_elig(adata, pos_target, pos_control, cfg, *, min_cpm=5.0):
    """Return genes reaching ``min_cpm`` in either cell group.

    Converting a screen-scale CSC matrix to CSR can take tens of seconds.  The
    count matrix and the invariant control-cell CPM mean are therefore cached
    on the per-method run config and reused for every guide.  Target and
    control library sizes remain cell-specific, so this is exactly the same
    per-cell CPM calculation as the uncached implementation.
    """
    cache = cfg.setdefault("_cpm_cache", {})
    source_key = (id(adata), cfg.get("counts_layer"))
    if cache.get("source_key") != source_key:
        counts_layer = cfg.get("counts_layer")
        X = (adata.layers[counts_layer]
             if counts_layer and counts_layer in adata.layers else adata.X)
        X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
        sample = X.data[:min(5_000, X.data.size)] if X.data.size else np.array([0.0])
        if not np.allclose(sample, np.rint(sample)):
            X = X.copy()
            X.data = np.expm1(X.data)
        cache.clear()
        cache.update(source_key=source_key, X=X, control_means={})

    X = cache["X"]

    def mean_cpm(pos):
        group = X[np.asarray(pos, dtype=int)]
        library_sizes = np.asarray(group.sum(axis=1)).ravel().astype(float)
        scales = np.divide(
            1e6, library_sizes, out=np.zeros_like(library_sizes),
            where=library_sizes > 0,
        )
        return np.asarray(group.multiply(scales[:, None]).mean(axis=0)).ravel()

    control_pos = np.asarray(pos_control, dtype=np.int64)
    control_key = control_pos.tobytes()
    control_means = cache["control_means"]
    if control_key not in control_means:
        control_means[control_key] = mean_cpm(control_pos)
    target_mean = mean_cpm(pos_target)
    control_mean = control_means[control_key]
    keep = (target_mean >= min_cpm) | (control_mean >= min_cpm)
    return set(adata.var_names[np.flatnonzero(keep)])


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
    s.obs["_g"] = pd.Categorical(
        [lab1] * len(pos1) + [lab2] * len(pos2), categories=[lab2, lab1]
    )
    if cfg.get("de_method") == "pdex" and cfg.get("non_parametric_engine") == "rsc":
        import rapids_singlecell as rsc

        if _looks_raw_integer(s):
            sc.pp.normalize_total(s, inplace=True)
            sc.pp.log1p(s)
        rsc.get.anndata_to_GPU(s)
        key = "_rsc_pdex"
        rsc.tl.rank_genes_groups(
            s,
            groupby="_g",
            groups=[lab1],
            reference=lab2,
            method="wilcoxon",
            n_genes=s.n_vars,
            tie_correct=True,
            use_raw=False,
            key_added=key,
        )
        result = s.uns[key]
        return pl.DataFrame({
            "target": [lab1] * s.n_vars,
            "feature": np.asarray(result["names"][lab1], dtype=str),
            "log2_fold_change": np.asarray(
                result["logfoldchanges"][lab1], dtype=float
            ),
            "p_value": np.asarray(result["pvals"][lab1], dtype=float),
            "fdr": np.asarray(result["pvals_adj"][lab1], dtype=float),
        })
    return run_de(s, cfg, groupby="_g", reference=lab2)
