"""Parity tests between the cpu and gpu metric backends.

The gpu portion is skipped if rapids-singlecell / cupy / cuml are not
importable in the current environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from cell_eval._types import PerturbationAnndataPair
from cell_eval.data import build_random_anndata
from cell_eval.metrics._anndata import (
    ClusteringAgreement,
    discrimination_score,
    edistance,
    mae,
    mae_delta,
    mse,
    mse_delta,
    pearson_delta,
)

gpu = pytest.importorskip("rapids_singlecell")
pytest.importorskip("cupy")
pytest.importorskip("cuml")


@pytest.fixture(scope="module")
def pair():
    real = build_random_anndata(n_cells=300, n_genes=60, n_perts=5, normlog=True)
    pred = build_random_anndata(
        n_cells=300, n_genes=60, n_perts=5, normlog=True, random_state=43
    )
    return PerturbationAnndataPair(
        real=real, pred=pred, pert_col="perturbation", control_pert="control"
    )


def _max_abs_diff(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) & set(b))
    return float(
        max(abs(float(a[k]) - float(b[k])) for k in keys)
    )


@pytest.mark.parametrize(
    "fn, tol",
    [
        (pearson_delta, 1e-12),
        (mse, 1e-12),
        (mae, 1e-12),
        (mse_delta, 1e-12),
        (mae_delta, 1e-12),
    ],
)
def test_centroid_metrics_parity(pair, fn, tol):
    cpu = fn(pair, backend="cpu")
    gpu_r = fn(pair, backend="gpu")
    assert _max_abs_diff(cpu, gpu_r) < tol


@pytest.mark.parametrize("metric", ["l1", "l2", "cosine"])
def test_discrimination_score_parity(pair, metric):
    cpu = discrimination_score(pair, metric=metric, backend="cpu")
    gpu_r = discrimination_score(pair, metric=metric, backend="gpu")
    # Ranks are integers so the score is exact up to float roundoff.
    assert _max_abs_diff(cpu, gpu_r) < 1e-9


def test_edistance_parity_shared_pca(pair):
    # Float32-vs-float64 PCA solvers differ in the basis but the unbiased
    # E-distance is rotation-invariant; the residual is float-precision-level.
    cpu = edistance(pair, backend="cpu", n_comps=10)
    gpu_r = edistance(pair, backend="gpu", n_comps=10)
    assert abs(cpu - gpu_r) < 5e-3


def test_clustering_agreement_runs_both_backends(pair):
    # Different leiden implementations (scanpy vs rsc) — don't assert
    # numerical equality, just that both produce a valid score in [0, 1].
    cpu = ClusteringAgreement(backend="cpu")(pair)
    gpu_r = ClusteringAgreement(backend="gpu")(pair)
    for v in (cpu, gpu_r):
        assert 0.0 <= v <= 1.0
