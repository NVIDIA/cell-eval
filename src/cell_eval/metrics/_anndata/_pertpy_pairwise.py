"""Vendored ``pairwise_distance_mean`` from pertpy (MIT-licensed).

Source: ``pertpy/tools/_distances/_distances.py`` (Lukas Heumos et al.). We
vendor the two numba kernels (``_euclidean_pairwise_mean_within`` /
``_between``) and the ``pairwise_distance_mean`` dispatcher so the cpu
edistance backend uses the same fast path pertpy uses, with no extra
runtime dependency on pertpy itself.

The numba kernels are 10-100x faster than ``sklearn.metrics.pairwise_distances(...).mean()``
because they never materialise the full pairwise matrix.
"""

from __future__ import annotations

import warnings

import numpy as np
from numba import jit, prange
from sklearn.metrics import pairwise_distances


@jit(nopython=True, cache=True)
def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute euclidean distance between two vectors."""
    dist_sq = 0.0
    for k in range(x.shape[0]):
        diff = x[k] - y[k]
        dist_sq += diff * diff
    return np.sqrt(dist_sq)


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _euclidean_pairwise_mean_within(X: np.ndarray) -> float:
    """Mean pairwise euclidean distance within a group (X to X).

    Iterates the upper triangle only (``i < j``) and divides by ``n*(n-1)/2``
    — the **unbiased** Székely estimator. Required to match pertpy / rsc
    edistance results.
    """
    n_samples = X.shape[0]
    if n_samples < 2:
        return 0.0

    total_distance = 0.0
    n_pairs = n_samples * (n_samples - 1) / 2.0

    for i in prange(n_samples):
        for j in range(i + 1, n_samples):
            total_distance += _euclidean_distance(X[i], X[j])

    return total_distance / n_pairs


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _euclidean_pairwise_mean_between(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean pairwise euclidean distance between two groups (X to Y)."""
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]

    if n_samples_X == 0 or n_samples_Y == 0:
        return 0.0

    total_distance = 0.0
    n_pairs = n_samples_X * n_samples_Y

    for i in prange(n_samples_X):
        for j in range(n_samples_Y):
            total_distance += _euclidean_distance(X[i], Y[j])

    return total_distance / n_pairs


def pairwise_distance_mean(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    metric: str = "euclidean",
    **kwargs,
) -> float:
    """Compute mean pairwise distance. Fast path for euclidean, sklearn fallback otherwise.

    If ``Y`` is ``None``, computes the within-group mean (upper triangle,
    unbiased). Otherwise computes the between-group mean over all
    ``n_X * n_Y`` cross-pairs.
    """
    if metric == "euclidean":
        if kwargs:
            warnings.warn(
                "kwargs are not used for euclidean distance.",
                UserWarning,
                stacklevel=2,
            )
        if Y is None:
            return _euclidean_pairwise_mean_within(np.ascontiguousarray(X))
        return _euclidean_pairwise_mean_between(
            np.ascontiguousarray(X), np.ascontiguousarray(Y)
        )
    if Y is None:
        # sklearn fallback — match pertpy's behaviour for non-euclidean metrics
        return float(pairwise_distances(X, X, metric=metric, **kwargs).mean())
    return float(pairwise_distances(X, Y, metric=metric, **kwargs).mean())
