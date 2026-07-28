from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]


def _load_skill_module(skill: str, runner: str, module_name: str):
    skill_dir = ROOT / ".claude" / "skills" / skill
    sys.path.insert(0, str(skill_dir))
    spec = importlib.util.spec_from_file_location(
        module_name,
        skill_dir / runner,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _null_counts(*, grouped: bool = False) -> ad.AnnData:
    rng = np.random.default_rng(20260727)
    counts = rng.poisson(4, size=(320, 96)).astype(np.float32)
    if grouped:
        condition = np.repeat([f"p{i}" for i in range(4)], 80)
        batch = np.tile(
            np.repeat([f"r{i}" for i in range(8)], 10),
            4,
        )
    else:
        condition = np.repeat("non-targeting", 320)
        batch = np.repeat([f"r{i}" for i in range(8)], 40)
    result = ad.AnnData(
        X=sp.csr_matrix(counts),
        obs=pd.DataFrame(
            {"condition": condition, "batch": batch},
            index=[f"cell_{i}" for i in range(320)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(96)]),
    )
    result.layers["counts"] = result.X.copy()
    return result


def test_delta_zero_injection_false_positive_rate_remains_near_null() -> None:
    os.environ["CELL_EVAL_THREAD_GUARD"] = "1"
    module = _load_skill_module(
        "evaluating-test0-injection",
        "inject_and_count_de.py",
        "test0_statistical_oracle",
    )
    data = _null_counts()
    data.obs["gene"] = data.obs["condition"]
    aggregate, _ = module.recover_injected(
        data,
        deltas=(0.0,),
        n_genes=8,
        n_repeats=5,
        seed=1,
        pert_col="gene",
        control_label="non-targeting",
        replicate_col="batch",
        counts_layer="counts",
        fdr=0.05,
        lfc=0.1,
        num_threads=1,
        block_cols=["batch"],
        non_parametric_engine="pdex",
        methods=("pdex",),
        workers=1,
    )
    row = aggregate.row(0, named=True)
    assert 0 <= row["mean_FPR"] <= 0.10
    assert 0 <= row["mean_TPR"] <= 0.10


def test_control_control_null_is_not_grossly_inflated() -> None:
    module = _load_skill_module(
        "evaluating-test2-control-null",
        "control_null_diagnostics.py",
        "test2_statistical_oracle",
    )
    data = _null_counts()
    config = {
        "block_cols": ["batch"],
        "seed": 3,
        "pert_col": "condition",
        "min_cells_per_group": 10,
        "control_pert": "non-targeting",
        "de_method": "pdex",
        "non_parametric_engine": "pdex",
        "normalize_if_raw": True,
        "num_threads": 1,
        "allow_discrete": False,
        "counts_layer": "counts",
        "replicate_col": "batch",
        "fdr_threshold": 0.05,
        "lfc_threshold": 0.1,
    }
    pvalues, lambdas, _ = module.control_null_pvalues(
        data,
        config,
        module._load_runner(),
        n_resamples=5,
        seed=3,
        resample_workers=1,
    )
    assert 0.35 <= float(np.mean(pvalues)) <= 0.65
    assert float(np.mean(pvalues <= 0.05)) <= 0.12
    assert 0.4 <= float(np.nanmedian(lambdas)) <= 1.8


def test_permutation_null_stays_flat() -> None:
    module = _load_skill_module(
        "evaluating-test3-permutation-null",
        "shuffle_de_comparison.py",
        "test3_statistical_oracle",
    )
    raw = _null_counts(grouped=True)
    normalized = raw.copy()
    sc.pp.normalize_total(normalized)
    sc.pp.log1p(normalized)
    groups = sorted(raw.obs["condition"].astype(str).unique())
    results = module._run_shuffled_null(
        raw,
        normalized,
        groups,
        "condition",
        ["batch"],
        4,
        0.05,
        0.1,
        1,
        "counts",
        "batch",
        shuffle_mode="within",
        min_cells=10,
        gene_names=raw.var_names.astype(str).tolist(),
        methods=["pdex"],
        non_parametric_engine="pdex",
        comparison_workers=1,
    )
    de_counts = np.asarray(
        [result["n_pdex"] for result in results.values()],
        dtype=int,
    )
    lfcs = np.concatenate(
        [result["lfc_pdex"] for result in results.values()]
    )
    assert de_counts.sum() <= 10
    assert float(np.nanmedian(np.abs(lfcs))) < 0.25


def test_permutation_rsc_serializes_gpu_without_serializing_pydeseq2(
    monkeypatch,
) -> None:
    module = _load_skill_module(
        "evaluating-test3-permutation-null",
        "shuffle_de_comparison.py",
        "test3_method_scheduler",
    )
    raw = ad.AnnData(
        X=sp.csr_matrix(np.ones((8, 3), dtype=np.float32)),
        obs=pd.DataFrame(
            {
                "condition": ["p0"] * 4 + ["p1"] * 4,
                "batch": ["r0"] * 8,
            },
            index=[f"cell_{i}" for i in range(8)],
        ),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    normalized = raw.copy()
    calls: list[tuple[str, ...]] = []
    pool_sizes: list[int] = []

    def fake_comparison(task, **kwargs):
        comparison, reference = task
        selected = tuple(kwargs["methods"])
        calls.append(selected)
        n_genes = len(kwargs["gene_names"])
        result = {
            "n_pdex": 1 if "pdex" in selected else 0,
            "genes_pdex": {"g0"} if "pdex" in selected else set(),
            "n_pydx": 2 if "pydeseq2" in selected else 0,
            "genes_pydx": {"g1", "g2"} if "pydeseq2" in selected else set(),
            "n_cells": 4,
            "reference": reference,
            "n_reference_cells": 4,
            "lfc_pdex": np.full(
                n_genes, 1.0 if "pdex" in selected else np.nan
            ),
            "lfc_pydx": np.full(
                n_genes, 2.0 if "pydeseq2" in selected else np.nan
            ),
        }
        return comparison, result

    class FakePool:
        def __init__(self, *, processes, maxtasksperchild):
            pool_sizes.append(processes)
            assert maxtasksperchild == 16

        def imap(self, function, tasks, chunksize):
            assert chunksize == 1
            return map(function, tasks)

        def close(self):
            return None

        def join(self):
            return None

    class FakeContext:
        Pool = FakePool

    monkeypatch.setattr(module, "_run_one_fake_comparison", fake_comparison)
    monkeypatch.setattr(module.mp, "get_context", lambda _: FakeContext())

    results = module._run_shuffled_null(
        raw,
        normalized,
        ["p0", "p1"],
        "condition",
        ["batch"],
        7,
        0.05,
        0.1,
        1,
        None,
        "batch",
        shuffle_mode="within",
        min_cells=1,
        gene_names=["g0", "g1", "g2"],
        methods=["pdex", "pydeseq2"],
        non_parametric_engine="rsc",
        comparison_workers=4,
    )

    assert pool_sizes == [4]
    assert calls.count(("pydeseq2",)) == 2
    assert calls.count(("pdex",)) == 2
    assert len(results) == 2
    for result in results.values():
        assert result["n_pdex"] == 1
        assert result["genes_pdex"] == {"g0"}
        assert result["n_pydx"] == 2
        assert result["genes_pydx"] == {"g1", "g2"}
        np.testing.assert_array_equal(result["lfc_pdex"], np.ones(3))
        np.testing.assert_array_equal(result["lfc_pydx"], np.full(3, 2.0))
