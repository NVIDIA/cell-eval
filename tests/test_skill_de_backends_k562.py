"""Before/after accuracy checks for the skills' bundled DE adapter.

The "before" implementation is the legacy project adapter. The "after"
implementation is the self-contained adapter bundled with each robustness
skill. A small deterministic K562 essential subset keeps both upstream pdex
and PyDESeq2 checks practical while exercising real sparse counts and metadata.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import anndata as ad
import numpy as np
import polars as pl
import pytest

from cell_eval._de_backends import build_de_frame as build_de_frame_before


ROOT = Path(__file__).resolve().parents[1]
SKILL_NAMES = (
    "evaluating-test0-injection",
    "evaluating-test01-overview",
    "evaluating-test1-reproducibility",
    "evaluating-test2-control-null",
    "evaluating-test3-permutation-null",
    "evaluating-test4-guide-reproducibility",
    "evaluating-test5-samegene-sgrna",
    "evaluating-test6-knockdown-recovery",
)
ADAPTER_PATHS = tuple(
    ROOT / ".claude" / "skills" / name / "de_backends.py"
    for name in SKILL_NAMES
)
REAL_FIXTURE = ROOT / "tests" / "data" / "k562_de_fixture.h5ad"
GOLDEN_TABLE = (
    ROOT / "tests" / "data" / "de_backend_real_fixture_golden.csv"
)


def _load_bundled_adapter() -> ModuleType:
    path = ADAPTER_PATHS[0]
    spec = importlib.util.spec_from_file_location("skill_de_backends_after", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load bundled DE adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def k562_de_subset() -> ad.AnnData:
    assert REAL_FIXTURE.is_file(), "the committed real-data fixture is missing"
    subset = ad.read_h5ad(REAL_FIXTURE)
    assert subset.shape == (120, 128)
    assert {"counts", "lognorm"} <= set(subset.layers)
    assert {"condition", "replicate"} <= set(subset.obs.columns)
    return subset


def _sort_de(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.sort(["target", "feature"])


def _single_target_fixture(adata: ad.AnnData) -> ad.AnnData:
    keep = adata.obs["condition"].astype(str).isin(
        ["non-targeting", "SRRT"]
    )
    return adata[keep].copy()


def _run_bundled_real_fixture(
    adata: ad.AnnData,
    methods: tuple[str, ...],
) -> pl.DataFrame:
    adapter = _load_bundled_adapter()
    subset = _single_target_fixture(adata)
    frames: list[pl.DataFrame] = []
    for method in methods:
        kwargs = {
            "mode": "real",
            "adata": subset,
            "control_pert": "non-targeting",
            "pert_col": "condition",
            "num_threads": 1,
            "allow_discrete": method == "pydeseq2",
            "de_method": method,
            "de_kwargs": (
                {"engine": "pdex"}
                if method == "pdex"
                else {"fit_strategy": "pairwise", "workers": 1}
            ),
            "counts_layer": "counts" if method == "pydeseq2" else None,
            "replicate_col": "replicate",
        }
        frames.append(
            adapter.build_de_frame(**kwargs)
            .with_columns(pl.lit(method).alias("method"))
            .select(
                [
                    "method",
                    "target",
                    "feature",
                    "log2_fold_change",
                    "p_value",
                    "fdr",
                ]
            )
        )
    return pl.concat(frames).sort(["method", "target", "feature"])


def _assert_de_accuracy(
    before: pl.DataFrame,
    after: pl.DataFrame,
    *,
    rtol: float,
    atol: float,
) -> None:
    before = _sort_de(before)
    after = _sort_de(after)
    assert before.columns == after.columns
    assert before.select(["target", "feature"]).equals(
        after.select(["target", "feature"])
    )
    for column in ("log2_fold_change", "p_value", "fdr"):
        expected = before[column].cast(pl.Float64).to_numpy()
        actual = after[column].cast(pl.Float64).to_numpy()
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            err_msg=f"before/after mismatch in {column}",
        )

    finite = np.isfinite(before["log2_fold_change"].to_numpy()) & np.isfinite(
        after["log2_fold_change"].to_numpy()
    )
    if finite.sum() >= 2:
        before_lfc = before["log2_fold_change"].to_numpy()[finite]
        after_lfc = after["log2_fold_change"].to_numpy()[finite]
        assert np.corrcoef(before_lfc, after_lfc)[0, 1] > 0.999999999
        assert np.array_equal(np.sign(before_lfc), np.sign(after_lfc))


def test_all_skills_bundle_identical_upstream_adapters() -> None:
    contents = [path.read_bytes() for path in ADAPTER_PATHS]
    assert all(content == contents[0] for content in contents[1:])
    source = contents[0].decode()
    assert "from pdex import pdex" in source
    assert 'import_module("pydeseq2.dds")' in source
    assert "cell_eval._de_backends" not in source


def test_real_fixture_golden_schema_is_explicit() -> None:
    golden = pl.read_csv(GOLDEN_TABLE)
    assert golden.columns == [
        "method",
        "target",
        "feature",
        "log2_fold_change",
        "p_value",
        "fdr",
    ]
    assert golden.schema == {
        "method": pl.String,
        "target": pl.String,
        "feature": pl.String,
        "log2_fold_change": pl.Float64,
        "p_value": pl.Float64,
        "fdr": pl.Float64,
    }
    assert golden.height == 256
    assert golden.group_by("method").len().sort("method").to_dicts() == [
        {"method": "pdex", "len": 128},
        {"method": "pydeseq2", "len": 128},
    ]


def test_real_fixture_golden_values_with_tolerance(
    k562_de_subset: ad.AnnData,
) -> None:
    pytest.importorskip("pydeseq2")
    expected = pl.read_csv(GOLDEN_TABLE).sort(
        ["method", "target", "feature"]
    )
    actual = _run_bundled_real_fixture(
        k562_de_subset,
        ("pdex", "pydeseq2"),
    )
    assert actual.select(["method", "target", "feature"]).equals(
        expected.select(["method", "target", "feature"])
    )
    for method, rtol, atol in (
        ("pdex", 1e-12, 1e-14),
        ("pydeseq2", 1e-7, 1e-10),
    ):
        actual_method = actual.filter(pl.col("method") == method)
        expected_method = expected.filter(pl.col("method") == method)
        for column in ("log2_fold_change", "p_value", "fdr"):
            np.testing.assert_allclose(
                actual_method[column].to_numpy(),
                expected_method[column].to_numpy(),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
                err_msg=f"{method} golden mismatch in {column}",
            )


@pytest.mark.parametrize(
    "methods",
    [
        ("pdex",),
        ("pydeseq2",),
        ("pdex", "pydeseq2"),
    ],
)
def test_requested_method_subsets_have_no_phantom_rows(
    k562_de_subset: ad.AnnData,
    methods: tuple[str, ...],
) -> None:
    if "pydeseq2" in methods:
        pytest.importorskip("pydeseq2")
    observed = _run_bundled_real_fixture(k562_de_subset, methods)
    assert set(observed["method"].unique().to_list()) == set(methods)


def test_k562_pdex_accuracy_before_after(
    k562_de_subset: ad.AnnData,
) -> None:
    after_adapter = _load_bundled_adapter()
    kwargs = {
        "mode": "real",
        "adata": k562_de_subset,
        "control_pert": "non-targeting",
        "pert_col": "condition",
        "num_threads": 1,
        "allow_discrete": False,
        "de_method": "pdex",
        "de_kwargs": {"engine": "pdex"},
        "counts_layer": None,
        "replicate_col": "replicate",
    }
    before = build_de_frame_before(**kwargs)
    after = after_adapter.build_de_frame(**kwargs)
    _assert_de_accuracy(before, after, rtol=0.0, atol=0.0)


def test_k562_pydeseq2_accuracy_before_after(
    k562_de_subset: ad.AnnData,
) -> None:
    pytest.importorskip("pydeseq2")
    after_adapter = _load_bundled_adapter()
    kwargs = {
        "mode": "real",
        "adata": k562_de_subset,
        "control_pert": "non-targeting",
        "pert_col": "condition",
        "num_threads": 1,
        "allow_discrete": True,
        "de_method": "pydeseq2",
        "de_kwargs": None,
        "counts_layer": "counts",
        "replicate_col": "replicate",
    }
    before = build_de_frame_before(**kwargs)
    after = after_adapter.build_de_frame(**kwargs)
    _assert_de_accuracy(before, after, rtol=1e-10, atol=1e-12)
