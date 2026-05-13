"""Array metrics module."""

from logging import getLogger
from typing import Callable, Literal, Sequence, cast

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.metrics as skm
from scipy.sparse import issparse
from scipy.stats import pearsonr
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from ..._types import PerturbationAnndataPair

logger = getLogger(__name__)


def pearson_delta(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute Pearson correlation between mean differences from control."""
    return _generic_evaluation(
        data, pearsonr, use_delta=True, embed_key=embed_key
    )


def mse(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean squared error of each perturbation from control."""
    return _generic_evaluation(
        data, skm.mean_squared_error, use_delta=False, embed_key=embed_key
    )


def mae(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean absolute error of each perturbation from control."""
    return _generic_evaluation(
        data, skm.mean_absolute_error, use_delta=False, embed_key=embed_key
    )


def mse_delta(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean squared error of each perturbation-control delta."""
    return _generic_evaluation(
        data, skm.mean_squared_error, use_delta=True, embed_key=embed_key
    )


def mae_delta(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean absolute error of each perturbation-control delta."""
    return _generic_evaluation(
        data, skm.mean_absolute_error, use_delta=True, embed_key=embed_key
    )


def edistance(
    data: PerturbationAnndataPair,
    embed_key: str | None = None,
    metric: str = "euclidean",
    n_comps: int = 50,
    **kwargs,
) -> float:
    """Pearson correlation of per-pert E-distances on a shared PCA basis.

    The cpu and gpu paths both follow the pertpy convention: concat real+pred,
    fit one PCA, split, and compute E-distance per side in that shared space.
    Matches ``rsc.ptg.Distance(metric='edistance', obsm_key='X_pca')``.

    Parameters
    ----------
    embed_key:
        If provided, PCA is skipped and ``data.real.obsm[embed_key]`` /
        ``data.pred.obsm[embed_key]`` are used directly as the shared basis.
        Caller is responsible for ensuring the embedding is comparable across
        sides (e.g. produced by a single PCA fit on the concat).
    metric:
        Pairwise distance metric (passed to sklearn).
    n_comps:
        Number of PCA components when ``embed_key is None``.
    """
    from sklearn.decomposition import PCA

    real_X, pred_X = _shared_features(data, embed_key=embed_key, n_comps=n_comps)

    real_perts = np.asarray(data.real.obs[data.pert_col].values)
    pred_perts = np.asarray(data.pred.obs[data.pert_col].values)
    pert_order = np.asarray(data.perts)

    e_real = _edistances_to_control_cpu(
        real_X, real_perts, data.control_pert, pert_order, metric=metric, **kwargs
    )
    e_pred = _edistances_to_control_cpu(
        pred_X, pred_perts, data.control_pert, pert_order, metric=metric, **kwargs
    )

    return float(pearsonr(e_real, e_pred).statistic)


def _shared_features(
    data: PerturbationAnndataPair,
    embed_key: str | None,
    n_comps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (real_X, pred_X) on a shared feature basis.

    If ``embed_key`` is set, returns ``adata.obsm[embed_key]`` per side as-is.
    Otherwise concatenates real and pred X as sparse on host, fits scanpy's
    ``sc.pp.pca`` (which handles sparse natively via the arpack solver), and
    splits the resulting embedding. Mirrors the gpu path's sparse handling.
    """
    if embed_key is not None:
        return (
            np.asarray(data.real.obsm[embed_key]),
            np.asarray(data.pred.obsm[embed_key]),
        )

    from ._sparse import stack_real_pred_sparse

    merged, n_real = stack_real_pred_sparse(data.real.X, data.pred.X)
    n_components = min(n_comps, *merged.shape)
    logger.info(
        "edistance: fitting shared sparse PCA on concat(real, pred): "
        f"shape={merged.shape}, nnz={getattr(merged, 'nnz', '-')}, "
        f"n_components={n_components}"
    )
    merged_adata = ad.AnnData(X=merged)
    sc.pp.pca(merged_adata, n_comps=n_components)
    X_pca = np.asarray(merged_adata.obsm["X_pca"])
    return X_pca[:n_real], X_pca[n_real:]


def _edistances_to_control_cpu(
    X: np.ndarray,
    perts: np.ndarray,
    control_pert: str,
    pert_order: np.ndarray,
    metric: str = "euclidean",
    **kwargs,
) -> np.ndarray:
    """E-distance from each non-control perturbation to control, in feature space.

    Uses the **unbiased** Székely within-group estimator (upper triangle only,
    denominator ``n*(n-1)/2``) to match the pertpy / rsc convention. For
    ``metric='euclidean'`` the numba kernels vendored from pertpy never
    materialise the full pairwise matrix; other metrics fall back to sklearn.
    """
    from ._pertpy_pairwise import pairwise_distance_mean

    ctrl_mask = perts == control_pert
    if not ctrl_mask.any():
        raise ValueError(
            f"control perturbation {control_pert!r} not found in obs[pert_col]"
        )
    ctrl_X = X[ctrl_mask]
    sigma_ctrl = pairwise_distance_mean(ctrl_X, metric=metric, **kwargs)

    out = np.zeros(pert_order.size, dtype=np.float64)
    for i, p in enumerate(pert_order):
        if p == control_pert:
            continue
        pert_X = X[perts == p]
        sigma_pert = pairwise_distance_mean(pert_X, metric=metric, **kwargs)
        delta = pairwise_distance_mean(pert_X, ctrl_X, metric=metric, **kwargs)
        out[i] = 2.0 * delta - sigma_pert - sigma_ctrl
    return out


def discrimination_score(
    data: PerturbationAnndataPair,
    metric: str = "l1",
    embed_key: str | None = None,
    exclude_target_gene: bool = True,
) -> dict[str, float]:
    """Base implementation for discrimination score computation.

    Best score is 1.0 - worst score is 0.0.

    Args:
        data: PerturbationAnndataPair containing real and predicted data
        metric: Metric for distance calculation (e.g., "l1", "l2", see `scipy.metrics.pairwise.distance_metrics`)
        embed_key: Optional ``obsm`` key. When set, distances run in that
            embedding space; when ``None`` they run on raw expression.
            Ignored for l1/manhattan/cityblock — gene-space l1 is the
            interpretable form.
        exclude_target_gene: Whether to exclude target gene from calculation
            (only meaningful in gene space; ignored when ``embed_key`` is set)

    Returns:
        Dictionary mapping perturbation names to normalized ranks
    """
    if metric in {"l1", "manhattan", "cityblock"}:
        embed_key = None

    # Compute perturbation effects for all perturbations
    real_effects = np.vstack(
        [
            d.perturbation_effect(which="real", abs=False)
            for d in data.iter_bulk_arrays(embed_key=embed_key)
        ]
    )
    pred_effects = np.vstack(
        [
            d.perturbation_effect(which="pred", abs=False)
            for d in data.iter_bulk_arrays(embed_key=embed_key)
        ]
    )

    norm_ranks = {}
    for p_idx, p in enumerate(data.perts):
        # Determine which features to include in the comparison
        if exclude_target_gene and not embed_key:
            include_mask = np.flatnonzero(data.genes != p)
        else:
            # For embedding data or when not excluding target gene, use all features
            include_mask = np.ones(real_effects.shape[1], dtype=bool)

        # Compute distances to all real effects
        distances = skm.pairwise_distances(
            real_effects[
                :, include_mask
            ],  # compare to all real effects across perturbations
            pred_effects[p_idx, include_mask].reshape(
                1, -1
            ),  # select pred effect for current perturbation
            metric=metric,
        ).flatten()

        # Sort by distance (ascending - lower distance = better match)
        sorted_indices = np.argsort(distances)

        # Find rank of the correct perturbation
        p_index = np.flatnonzero(data.perts == p)[0]
        rank = np.flatnonzero(sorted_indices == p_index)[0]

        # Normalize rank by total number of perturbations
        norm_rank = rank / data.perts.size
        norm_ranks[str(p)] = 1 - norm_rank

    return norm_ranks


def _generic_evaluation(
    data: PerturbationAnndataPair,
    func: Callable[[np.ndarray, np.ndarray], float],
    use_delta: bool = False,
    embed_key: str | None = None,
) -> dict[str, float]:
    """Generic evaluation function for anndata pair."""
    res = {}
    for bulk_array in data.iter_bulk_arrays(embed_key=embed_key):
        if use_delta:
            x = bulk_array.perturbation_effect(which="pred", abs=False)
            y = bulk_array.perturbation_effect(which="real", abs=False)
        else:
            x = bulk_array.pert_pred
            y = bulk_array.pert_real

        result = func(x, y)
        if isinstance(result, tuple):
            result = result[0]

        res[bulk_array.key] = float(result)

    return res


# TODO: clean up this implementation
class ClusteringAgreement:
    """Compute clustering agreement between real and predicted perturbation centroids."""

    def __init__(
        self,
        embed_key: str | None = None,
        real_resolution: float = 1.0,
        pred_resolutions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0),
        metric: Literal["ami", "nmi", "ari"] = "ami",
        n_neighbors: int = 15,
    ) -> None:
        self.embed_key = embed_key
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
    def _cluster_leiden(
        adata: ad.AnnData,
        resolution: float,
        key_added: str,
        n_neighbors: int = 15,
    ) -> None:
        # scanpy's flavor='igraph' path requires the python-igraph package.
        # Check upfront so the user gets an actionable error instead of a
        # cryptic ImportError from deep inside scanpy.
        try:
            import igraph  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "ClusteringAgreement (cpu backend) requires `igraph` for the "
                "scanpy leiden path. Install with `pip install cell-eval[igraph]`."
            ) from e

        if key_added in adata.obs:
            return
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(
                adata, n_neighbors=min(n_neighbors, adata.n_obs - 1), use_rep="X"
            )
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=key_added,
            flavor="igraph",
            n_iterations=2,
        )

    @staticmethod
    def _centroid_ann(
        adata: ad.AnnData,
        category_key: str,
        control_pert: str,
        embed_key: str | None = None,
    ) -> ad.AnnData:
        """Per-category mean centroid AnnData via ``scanpy.get.aggregate``.

        Mirrors the gpu path (`rsc.get.aggregate`) — same API, same semantics.
        The aggregate handles sparse natively; we only pull the small
        ``(n_groups, n_features)`` mean matrix into a fresh AnnData and drop
        the control row.
        """
        if not isinstance(adata.obs[category_key].dtype, pd.CategoricalDtype):
            adata = adata.copy()
            adata.obs[category_key] = adata.obs[category_key].astype("category")

        bulk = sc.get.aggregate(
            adata, by=category_key, func="mean", obsm=embed_key
        )
        mean_mat = bulk.X if bulk.X is not None else bulk.layers["mean"]
        if issparse(mean_mat):
            mean_mat = mean_mat.toarray()  # type: ignore
        mean_mat = np.asarray(mean_mat, dtype=np.float64)

        adc = ad.AnnData(X=mean_mat)
        adc.obs[category_key] = np.asarray(bulk.obs_names)
        return adc[adc.obs[category_key] != control_pert]

    def __call__(self, data: PerturbationAnndataPair) -> float:
        cats_sorted = sorted([c for c in data.perts if c != data.control_pert])

        # 2. build centroids
        ad_real_cent = self._centroid_ann(
            adata=data.real,
            category_key=data.pert_col,
            control_pert=data.control_pert,
            embed_key=self.embed_key,
        )
        ad_pred_cent = self._centroid_ann(
            adata=data.pred,
            category_key=data.pert_col,
            control_pert=data.control_pert,
            embed_key=self.embed_key,
        )

        # 3. cluster real once
        real_key = "real_clusters"
        self._cluster_leiden(
            ad_real_cent, self.real_resolution, real_key, self.n_neighbors
        )
        ad_real_cent.obs = (
            cast(pd.DataFrame, ad_real_cent.obs)
            .set_index(data.pert_col)
            .loc[cats_sorted]
        )
        real_labels = pd.Categorical(ad_real_cent.obs[real_key])

        # 4. sweep predicted resolutions
        best_score = 0.0
        ad_pred_cent.obs = (
            cast(pd.DataFrame, ad_pred_cent.obs)
            .set_index(data.pert_col)
            .loc[cats_sorted]
        )
        for r in self.pred_resolutions:
            pred_key = f"pred_clusters_{r}"
            self._cluster_leiden(ad_pred_cent, r, pred_key, self.n_neighbors)
            pred_labels = pd.Categorical(ad_pred_cent.obs[pred_key])
            score = self._score(real_labels, pred_labels, self.metric)  # type: ignore
            best_score = max(best_score, score)

        return float(best_score)
