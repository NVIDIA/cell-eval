"""AnnData metric dispatchers.

Each public metric here resolves a backend via :func:`cell_eval._backend.resolve_backend`
(env var ``CELL_EVAL_BACKEND``) and delegates to the cpu or gpu implementation.

``embed_key`` is exposed only on :func:`edistance` — it lets a caller plug a
precomputed joint embedding (e.g. their own PCA) into the shared-basis E-distance
path instead of having the metric fit its own PCA. Other metrics always operate
on ``adata.X``.
"""

from __future__ import annotations

from typing import Any

from ..._backend import Backend, resolve_backend
from ..._types import PerturbationAnndataPair

__all__ = [
    "pearson_delta",
    "mse",
    "mae",
    "mse_delta",
    "mae_delta",
    "discrimination_score",
    "edistance",
    "ClusteringAgreement",
]


def _impl(backend: Backend | None, name: str):
    """Lazy-import the cpu or gpu attribute named ``name``."""
    resolved = resolve_backend(backend)
    if resolved == "cpu":
        from . import _cpu as mod
    else:
        try:
            from . import _gpu as mod  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The gpu metric backend requires cupy, cuml, and rapids-singlecell. "
                "Install with `pip install cell-eval[gpu]`."
            ) from e
    return getattr(mod, name)


def pearson_delta(
    data: PerturbationAnndataPair,
    *,
    backend: Backend | None = None,
) -> dict[str, float]:
    return _impl(backend, "pearson_delta")(data)


def mse(
    data: PerturbationAnndataPair,
    *,
    backend: Backend | None = None,
) -> dict[str, float]:
    return _impl(backend, "mse")(data)


def mae(
    data: PerturbationAnndataPair,
    *,
    backend: Backend | None = None,
) -> dict[str, float]:
    return _impl(backend, "mae")(data)


def mse_delta(
    data: PerturbationAnndataPair,
    *,
    backend: Backend | None = None,
) -> dict[str, float]:
    return _impl(backend, "mse_delta")(data)


def mae_delta(
    data: PerturbationAnndataPair,
    *,
    backend: Backend | None = None,
) -> dict[str, float]:
    return _impl(backend, "mae_delta")(data)


def discrimination_score(
    data: PerturbationAnndataPair,
    metric: str = "l1",
    exclude_target_gene: bool = True,
    *,
    backend: Backend | None = None,
) -> dict[str, float]:
    return _impl(backend, "discrimination_score")(
        data,
        metric=metric,
        exclude_target_gene=exclude_target_gene,
    )


def edistance(
    data: PerturbationAnndataPair,
    embed_key: str | None = None,
    metric: str = "euclidean",
    n_comps: int = 50,
    *,
    backend: Backend | None = None,
    **kwargs: Any,
) -> float:
    return _impl(backend, "edistance")(
        data, embed_key=embed_key, metric=metric, n_comps=n_comps, **kwargs
    )


class ClusteringAgreement:
    """Dispatcher class.

    Holds the cpu/gpu kwargs and instantiates the backend-specific implementation
    on call. Backend choice is read at call time so a process-wide
    ``CELL_EVAL_BACKEND`` flip is honoured.
    """

    def __init__(
        self,
        real_resolution: float = 1.0,
        pred_resolutions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0),
        metric: str = "ami",
        n_neighbors: int = 15,
        *,
        backend: Backend | None = None,
    ) -> None:
        self._kwargs = dict(
            real_resolution=real_resolution,
            pred_resolutions=pred_resolutions,
            metric=metric,
            n_neighbors=n_neighbors,
        )
        self._backend = backend

    def __call__(self, data: PerturbationAnndataPair) -> float:
        cls = _impl(self._backend, "ClusteringAgreement")
        return cls(**self._kwargs)(data)
