"""GPU-accelerated AnnData metrics (rapids-singlecell / cupy / cuml).

Mirrors :mod:`cell_eval.metrics._anndata._cpu`. Imports of cupy / rsc / cuml
happen lazily inside the functions so cpu-only installs don't choke at import
time.

The shared-PCA edistance and the centroid-based metrics follow the patterns
established in ``run_arc_gpu_benchmark.py``.
"""

from __future__ import annotations

from logging import getLogger
from typing import Literal, Sequence, cast

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import issparse
from scipy.stats import pearsonr
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from ..._types import PerturbationAnndataPair

logger = getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers: pull stacked centroids from cell-eval's cached bulk arrays.
# --------------------------------------------------------------------------- #

def _stack_effects(
    data: PerturbationAnndataPair,
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
    """Return ``(cent_real, cent_pred, eff_real, eff_pred)`` as host numpy arrays.

    ``eff_*`` are control-subtracted (perturbation_effect).  Shapes are
    ``(n_perts_non_ctrl, n_features)`` for the effect matrices and the matching
    perturbation row arrangement for centroids. Always operates on ``adata.X``.
    """
    bulks = list(data.iter_bulk_arrays())
    cent_real = np.vstack([b.pert_real for b in bulks])
    cent_pred = np.vstack([b.pert_pred for b in bulks])
    eff_real = np.vstack([b.perturbation_effect(which="real", abs=False) for b in bulks])
    eff_pred = np.vstack([b.perturbation_effect(which="pred", abs=False) for b in bulks])
    return cent_real, cent_pred, eff_real, eff_pred


def _pert_keys(data: PerturbationAnndataPair) -> list[str]:
    return [b.key for b in data.iter_bulk_arrays()]


# --------------------------------------------------------------------------- #
# Centroid-based metrics — vectorized cupy.
# --------------------------------------------------------------------------- #

def pearson_delta(data: PerturbationAnndataPair) -> dict[str, float]:
    import cupy as cp

    _, _, eff_real, eff_pred = _stack_effects(data)
    keys = _pert_keys(data)

    er = cp.asarray(eff_real, dtype=cp.float64)
    ep = cp.asarray(eff_pred, dtype=cp.float64)
    er -= er.mean(axis=1, keepdims=True)
    ep -= ep.mean(axis=1, keepdims=True)
    num = (er * ep).sum(axis=1)
    den = cp.sqrt((er * er).sum(axis=1) * (ep * ep).sum(axis=1))
    corr = cp.asnumpy(num / den)
    return {k: float(v) for k, v in zip(keys, corr)}


def mse(data: PerturbationAnndataPair) -> dict[str, float]:
    import cupy as cp

    cent_real, cent_pred, _, _ = _stack_effects(data)
    keys = _pert_keys(data)
    cr = cp.asarray(cent_real, dtype=cp.float64)
    cp_ = cp.asarray(cent_pred, dtype=cp.float64)
    vals = cp.asnumpy(((cp_ - cr) ** 2).mean(axis=1))
    return {k: float(v) for k, v in zip(keys, vals)}


def mae(data: PerturbationAnndataPair) -> dict[str, float]:
    import cupy as cp

    cent_real, cent_pred, _, _ = _stack_effects(data)
    keys = _pert_keys(data)
    cr = cp.asarray(cent_real, dtype=cp.float64)
    cp_ = cp.asarray(cent_pred, dtype=cp.float64)
    vals = cp.asnumpy(cp.abs(cp_ - cr).mean(axis=1))
    return {k: float(v) for k, v in zip(keys, vals)}


def mse_delta(data: PerturbationAnndataPair) -> dict[str, float]:
    import cupy as cp

    _, _, eff_real, eff_pred = _stack_effects(data)
    keys = _pert_keys(data)
    er = cp.asarray(eff_real, dtype=cp.float64)
    ep = cp.asarray(eff_pred, dtype=cp.float64)
    vals = cp.asnumpy(((ep - er) ** 2).mean(axis=1))
    return {k: float(v) for k, v in zip(keys, vals)}


def mae_delta(data: PerturbationAnndataPair) -> dict[str, float]:
    import cupy as cp

    _, _, eff_real, eff_pred = _stack_effects(data)
    keys = _pert_keys(data)
    er = cp.asarray(eff_real, dtype=cp.float64)
    ep = cp.asarray(eff_pred, dtype=cp.float64)
    vals = cp.asnumpy(cp.abs(ep - er).mean(axis=1))
    return {k: float(v) for k, v in zip(keys, vals)}


def discrimination_score(
    data: PerturbationAnndataPair,
    metric: str = "l1",
    exclude_target_gene: bool = True,
) -> dict[str, float]:
    """cuml-pairwise discrimination score on stacked effect matrices."""
    import cupy as cp
    from cuml.metrics import pairwise_distances as cuml_pairwise

    _, _, eff_real, eff_pred = _stack_effects(data)
    keys = _pert_keys(data)
    perts = np.asarray(keys)

    if exclude_target_gene:
        genes = np.asarray(data.genes)
        out = {}
        # exclude_target_gene varies per pert, so we can't reuse a single
        # pairwise call — but we can still keep the per-pert distance
        # computation on the GPU.
        eff_real_d = cp.asarray(eff_real, dtype=cp.float32)
        eff_pred_d = cp.asarray(eff_pred, dtype=cp.float32)
        for p_idx, p in enumerate(perts):
            mask = cp.asarray(genes != p)
            dist = cuml_pairwise(
                eff_real_d[:, mask],
                eff_pred_d[p_idx, mask].reshape(1, -1),
                metric=metric,
            ).ravel()
            order = cp.argsort(dist)
            rank = int(cp.flatnonzero(order == p_idx)[0].get())
            out[str(p)] = 1.0 - rank / perts.size
        return out

    eff_real_d = cp.asarray(eff_real, dtype=cp.float32)
    eff_pred_d = cp.asarray(eff_pred, dtype=cp.float32)
    D = cuml_pairwise(eff_pred_d, eff_real_d, metric=metric)  # (n, n)
    sorted_idx = cp.argsort(D, axis=1)
    n = D.shape[0]
    ranks_host = np.empty(n, dtype=np.int64)
    for i in range(n):
        ranks_host[i] = int(cp.flatnonzero(sorted_idx[i] == i)[0].get())
    return {str(perts[i]): 1.0 - ranks_host[i] / n for i in range(n)}


# --------------------------------------------------------------------------- #
# Shared-PCA edistance — concat real+pred, fit one PCA on GPU, split, compute
# E-distance per side using rsc.ptg.Distance(obsm_key="X_pca").
# --------------------------------------------------------------------------- #

def edistance(
    data: PerturbationAnndataPair,
    embed_key: str | None = None,
    metric: str = "euclidean",
    n_comps: int = 50,
    **_: object,
) -> float:
    import cupy as cp
    import rapids_singlecell as rsc

    real_X_pca, pred_X_pca = _shared_features_gpu(
        data, embed_key=embed_key, n_comps=n_comps
    )

    pert_order = np.asarray(data.perts)
    e_real = _onesided_edistances_gpu(
        data.real, data.pert_col, data.control_pert, real_X_pca, pert_order
    )
    e_pred = _onesided_edistances_gpu(
        data.pred, data.pert_col, data.control_pert, pred_X_pca, pert_order
    )
    return float(pearsonr(e_real, e_pred).statistic)


def _shared_features_gpu(
    data: PerturbationAnndataPair,
    embed_key: str | None,
    n_comps: int,
):
    """Return ``(real_X, pred_X)`` as host numpy arrays sharing a basis.

    If ``embed_key`` is set, both ``obsm[embed_key]`` blocks are returned
    as-is. Otherwise builds a dask-of-cupy-sparse array directly from the two
    GPU-resident sides (one chunk per side — pdex(gpu) leaves both ``X``
    matrices on the GPU, so we can skip a 19 GB host round trip), runs
    ``rsc.pp.pca`` on the dask array, and splits the resulting embedding.

    The per-side nnz must fit in int32 for the cupyx CSR indices/indptr
    (2.15 B max). Real and pred individually clear this; only their concat
    would not. Falls back to the host-vstack + chunked-reupload path if either
    side lives off-GPU.
    """
    if embed_key is not None:
        return (
            np.asarray(data.real.obsm[embed_key]),
            np.asarray(data.pred.obsm[embed_key]),
        )

    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    import dask
    import dask.array as da
    import rapids_singlecell as rsc

    real_X = data.real.X
    pred_X = data.pred.X
    n_real = real_X.shape[0]

    def _is_gpu_sparse(X):
        return cp_sparse.issparse(X)

    INT32_NNZ_LIMIT = 2_147_483_647

    # Fast path: both sides are already cupy sparse on the GPU (typical after
    # pdex(gpu) ran), and each fits in int32 nnz — wrap each as a single dask
    # chunk in place, no host trip.
    both_gpu_sparse = _is_gpu_sparse(real_X) and _is_gpu_sparse(pred_X)
    both_chunks_fit = (
        both_gpu_sparse
        and real_X.nnz <= INT32_NNZ_LIMIT
        and pred_X.nnz <= INT32_NNZ_LIMIT
    )

    if both_chunks_fit:
        logger.info(
            "edistance(gpu): fitting shared sparse PCA on 2-chunk dask "
            f"(real {real_X.shape}/{real_X.nnz} nnz + pred {pred_X.shape}/{pred_X.nnz} nnz) "
            f"— skipping host round trip"
        )
        # cupyx sparse .astype() doesn't take copy= — gate on dtype to avoid an
        # unnecessary copy when the input is already float32.
        real_chunk = real_X if real_X.dtype == cp.float32 else real_X.astype(cp.float32)
        pred_chunk = pred_X if pred_X.dtype == cp.float32 else pred_X.astype(cp.float32)
        chunks = [
            da.from_delayed(
                dask.delayed(lambda x: x)(real_chunk),
                shape=real_chunk.shape,
                dtype=np.float32,
                meta=cp_sparse.csr_matrix((0, 0), dtype=cp.float32),
            ),
            da.from_delayed(
                dask.delayed(lambda x: x)(pred_chunk),
                shape=pred_chunk.shape,
                dtype=np.float32,
                meta=cp_sparse.csr_matrix((0, 0), dtype=cp.float32),
            ),
        ]
        X_in = da.concatenate(chunks, axis=0)
        n_components = min(n_comps, *X_in.shape)
    else:
        # Generic path: pull both sides to host, vstack, chunk back to GPU.
        from ._sparse import stack_real_pred_sparse

        merged, n_real = stack_real_pred_sparse(real_X, pred_X)
        n_components = min(n_comps, *merged.shape)
        use_dask = sp.issparse(merged) and merged.nnz > INT32_NNZ_LIMIT
        logger.info(
            "edistance(gpu): fitting shared sparse PCA on concat(real, pred): "
            f"shape={merged.shape}, nnz={getattr(merged, 'nnz', '-')}, "
            f"n_components={n_components}, dask_chunked={use_dask}"
        )
        if use_dask:
            merged = merged.astype(np.float32, copy=False)
            CHUNK_ROWS = 50_000

            def _chunk(start: int, stop: int):
                return cp_sparse.csr_matrix(merged[start:stop])

            chunks = []
            for i in range(0, merged.shape[0], CHUNK_ROWS):
                end = min(i + CHUNK_ROWS, merged.shape[0])
                chunks.append(
                    da.from_delayed(
                        dask.delayed(_chunk)(i, end),
                        shape=(end - i, merged.shape[1]),
                        dtype=np.float32,
                        meta=cp_sparse.csr_matrix((0, 0), dtype=cp.float32),
                    )
                )
            X_in = da.concatenate(chunks, axis=0)
        elif sp.issparse(merged):
            X_in = cp_sparse.csr_matrix(merged.astype(np.float32, copy=False))
        else:
            X_in = cp.asarray(merged, dtype=cp.float32)

    merged_adata = ad.AnnData(X=X_in)
    rsc.pp.pca(merged_adata, n_comps=n_components)

    X_pca = merged_adata.obsm["X_pca"]
    if hasattr(X_pca, "compute"):
        X_pca = X_pca.compute()
    if hasattr(X_pca, "get"):
        X_pca = X_pca.get()
    X_pca = np.asarray(X_pca)
    return X_pca[:n_real], X_pca[n_real:]




def _onesided_edistances_gpu(
    adata: ad.AnnData,
    pert_col: str,
    control_pert: str,
    X_pca: np.ndarray,
    pert_order: np.ndarray,
) -> np.ndarray:
    """E-distance from each pert in ``pert_order`` to control via rsc.ptg.Distance.

    Returns one value per entry in ``pert_order``, in that exact order — caller
    is responsible for ensuring ``control_pert`` is absent from it.
    """
    import rapids_singlecell as rsc

    n = adata.n_obs
    obs = adata.obs.copy()
    if not isinstance(obs[pert_col].dtype, pd.CategoricalDtype):
        obs[pert_col] = obs[pert_col].astype("category")
    lite = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32), obs=obs)
    lite.obsm["X_pca"] = X_pca

    dist = rsc.ptg.Distance(metric="edistance", obsm_key="X_pca")
    series = dist.onesided_distances(
        lite, groupby=pert_col, selected_group=control_pert
    )
    return np.asarray(
        [float(series[p]) for p in pert_order], dtype=np.float64
    )


# --------------------------------------------------------------------------- #
# ClusteringAgreement — same sweep as the cpu version, but rsc neighbors+leiden.
# --------------------------------------------------------------------------- #

class ClusteringAgreement:
    """GPU mirror of :class:`._cpu.ClusteringAgreement` using rsc."""

    def __init__(
        self,
        real_resolution: float = 1.0,
        pred_resolutions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0),
        metric: Literal["ami", "nmi", "ari"] = "ami",
        n_neighbors: int = 15,
    ) -> None:
        self.real_resolution = real_resolution
        self.pred_resolutions = pred_resolutions
        self.metric = metric
        self.n_neighbors = n_neighbors

    @staticmethod
    def _score(
        labels_real: Sequence[int],
        labels_pred: Sequence[int],
        metric: Literal["ami", "nmi", "ari"],
    ) -> float:
        if metric == "ami":
            return adjusted_mutual_info_score(labels_real, labels_pred)
        if metric == "nmi":
            return normalized_mutual_info_score(labels_real, labels_pred)
        if metric == "ari":
            return (adjusted_rand_score(labels_real, labels_pred) + 1) / 2
        raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def _centroid_ann(
        adata: ad.AnnData,
        category_key: str,
        control_pert: str,
    ) -> ad.AnnData:
        """Per-category mean centroid AnnData via ``rsc.get.aggregate``.

        Mirrors ``run_arc_gpu_benchmark.py:aggregate_mean_pseudobulk``. The
        per-group reduction runs entirely on the GPU; only the small
        ``(n_groups, n_features)`` centroid matrix lands on host.
        """
        import cupy as cp
        import cupyx.scipy.sparse as cp_sparse
        import rapids_singlecell as rsc

        if not isinstance(adata.obs[category_key].dtype, pd.CategoricalDtype):
            adata = adata.copy()
            adata.obs[category_key] = adata.obs[category_key].astype("category")

        # rsc.get.aggregate requires the source array on GPU. Move only if
        # needed — pdex(gpu) may have already done so.
        if not (cp_sparse.issparse(adata.X) or isinstance(adata.X, cp.ndarray)):
            rsc.get.anndata_to_GPU(adata)

        bulk = rsc.get.aggregate(adata, by=category_key, func="mean")
        mean_mat = bulk.X if bulk.X is not None else bulk.layers["mean"]

        if cp_sparse.issparse(mean_mat):
            mean_mat = mean_mat.toarray()
        if isinstance(mean_mat, cp.ndarray):
            mean_mat = cp.asnumpy(mean_mat)
        mean_mat = np.asarray(mean_mat, dtype=np.float64)

        adc = ad.AnnData(X=mean_mat)
        adc.obs[category_key] = np.asarray(bulk.obs_names)
        return adc[adc.obs[category_key] != control_pert]

    @staticmethod
    def _cluster_leiden_gpu(
        adata: ad.AnnData, resolution: float, key_added: str, n_neighbors: int
    ) -> None:
        import rapids_singlecell as rsc

        if key_added in adata.obs.columns:
            return
        if "neighbors" not in adata.uns:
            rsc.pp.neighbors(
                adata,
                n_neighbors=min(n_neighbors, adata.n_obs - 1),
                use_rep="X",
            )
        rsc.tl.leiden(adata, resolution=resolution, key_added=key_added)

    def __call__(self, data: PerturbationAnndataPair) -> float:
        import rapids_singlecell as rsc

        cats_sorted = sorted([c for c in data.perts if c != data.control_pert])
        ad_real = self._centroid_ann(
            data.real, data.pert_col, data.control_pert
        )
        ad_pred = self._centroid_ann(
            data.pred, data.pert_col, data.control_pert
        )

        # Materialise to plain AnnData with the embedded centroids on GPU.
        ad_real = ad_real.to_memory().copy()
        ad_pred = ad_pred.to_memory().copy()
        ad_real.X = ad_real.X.astype(np.float32, copy=False)
        ad_pred.X = ad_pred.X.astype(np.float32, copy=False)
        rsc.get.anndata_to_GPU(ad_real)
        rsc.get.anndata_to_GPU(ad_pred)

        real_key = "real_clusters"
        self._cluster_leiden_gpu(
            ad_real, self.real_resolution, real_key, self.n_neighbors
        )
        ad_real.obs = (
            cast(pd.DataFrame, ad_real.obs)
            .set_index(data.pert_col)
            .loc[cats_sorted]
        )
        real_labels = pd.Categorical(ad_real.obs[real_key])

        best_score = 0.0
        ad_pred.obs = (
            cast(pd.DataFrame, ad_pred.obs)
            .set_index(data.pert_col)
            .loc[cats_sorted]
        )
        for r in self.pred_resolutions:
            pred_key = f"pred_clusters_{r}"
            self._cluster_leiden_gpu(ad_pred, r, pred_key, self.n_neighbors)
            pred_labels = pd.Categorical(ad_pred.obs[pred_key])
            score = self._score(real_labels, pred_labels, self.metric)
            best_score = max(best_score, score)
        return float(best_score)
