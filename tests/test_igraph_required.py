"""ClusteringAgreement (cpu) should fail-fast with a clear error if igraph
is not importable.
"""

from __future__ import annotations

import builtins
import sys

import pytest

from cell_eval._types import PerturbationAnndataPair
from cell_eval.data import build_random_anndata
from cell_eval.metrics._anndata._cpu import ClusteringAgreement


@pytest.fixture(scope="module")
def pair():
    real = build_random_anndata(n_cells=200, n_genes=40, n_perts=4, normlog=True)
    pred = build_random_anndata(
        n_cells=200, n_genes=40, n_perts=4, normlog=True, random_state=43
    )
    return PerturbationAnndataPair(
        real=real, pred=pred, pert_col="perturbation", control_pert="control"
    )


def test_missing_igraph_raises_with_install_hint(monkeypatch, pair):
    # Pretend igraph is not installed.
    monkeypatch.setitem(sys.modules, "igraph", None)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "igraph":
            raise ImportError("No module named 'igraph'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"pip install cell-eval\[igraph\]"):
        ClusteringAgreement()(pair)
