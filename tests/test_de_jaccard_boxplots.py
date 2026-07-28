from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEST1_SKILL = (
    ROOT / ".claude/skills/evaluating-test1-reproducibility"
)


def _load_test1_module():
    os.environ["CELL_EVAL_THREAD_GUARD"] = "1"
    sys.path.insert(0, str(TEST1_SKILL))
    spec = importlib.util.spec_from_file_location(
        "test1_reproducibility_jaccard",
        TEST1_SKILL / "reproducibility_heatmap.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _signature(genes: np.ndarray, de_a: set[str], de_b: set[str]) -> dict:
    return {
        "rho": 0.0,
        "n_cells": 20,
        "genes": genes.copy(),
        "lfc_a": np.asarray(
            [1.0 if gene in de_a else 0.0 for gene in genes]
        ),
        "lfc_b": np.asarray(
            [1.0 if gene in de_b else 0.0 for gene in genes]
        ),
        "fdr_a": np.asarray(
            [0.01 if gene in de_a else 1.0 for gene in genes]
        ),
        "fdr_b": np.asarray(
            [0.01 if gene in de_b else 1.0 for gene in genes]
        ),
        "cpm_elig": np.ones(len(genes), dtype=bool),
    }


def test_de_jaccard_matrix_uses_split_specific_cpm_qualified_calls() -> None:
    module = _load_test1_module()
    genes = np.asarray(["g1", "g2", "g3", "g4", "g5"])
    signatures = {
        "u1": _signature(genes, {"g1", "g2"}, {"g1"}),
        "u2": _signature(genes, {"g3"}, {"g2", "g3"}),
    }
    config = {"fdr_threshold": 0.05, "lfc_threshold": 0.1}

    observed = module._de_jaccard_matrix(
        signatures, ["u1", "u2"], config
    )

    np.testing.assert_allclose(
        observed,
        np.asarray([[0.5, 1.0 / 3.0], [0.0, 0.5]], dtype=np.float32),
    )


def test_de_jaccard_empty_union_is_undefined() -> None:
    module = _load_test1_module()
    genes = np.asarray(["g1", "g2"])
    signatures = {"u1": _signature(genes, set(), set())}

    observed = module._de_jaccard_matrix(
        signatures, ["u1"],
        {"fdr_threshold": 0.05, "lfc_threshold": 0.1},
    )

    assert np.isnan(observed[0, 0])


def test_layer3_emits_all_jaccard_boxplot_artifacts(tmp_path: Path) -> None:
    module = _load_test1_module()
    genes = np.asarray(["g1", "g2", "g3", "g4", "g5"])
    signatures = {
        "u1": _signature(genes, {"g1", "g2"}, {"g1"}),
        "u2": _signature(genes, {"g3"}, {"g2", "g3"}),
    }
    config = {
        "fdr_threshold": 0.05,
        "lfc_threshold": 0.1,
        "de_method": "pdex",
        "non_parametric_engine": "pdex",
    }
    matrix_png = tmp_path / "test4_corr_matrix__synthetic.png"

    module.layer3_corr_matrix(
        {"pdex": [signatures]},
        str(matrix_png),
        config,
        genes.tolist(),
        methods=["pdex"],
        title_prefix="Test-4",
        unit="guide",
        emit_boxplots=True,
    )

    expected_stems = (
        "test4_corr_matrix_de_jaccard_diagonal_boxplot__synthetic",
        "test4_corr_matrix_de_jaccard_off_diagonal_boxplot__synthetic",
    )
    for stem in expected_stems:
        assert (tmp_path / f"{stem}.png").exists()
        assert (tmp_path / f"{stem}.csv").exists()

    archive = np.load(matrix_png.with_suffix(".npz"))
    np.testing.assert_allclose(
        archive["jaccard__pdex"],
        np.asarray([[0.5, 1.0 / 3.0], [0.0, 0.5]], dtype=np.float32),
    )
    assert np.all(archive["finite_jaccard_repeats__pdex"] == 1)


def test_pair_specific_numbers_are_invariant_to_worker_count() -> None:
    module = _load_test1_module()
    rng = np.random.default_rng(20260727)
    genes = np.asarray([f"g{i:02d}" for i in range(32)])
    signatures = {}
    for index in range(64):
        lfc_a = rng.normal(size=len(genes))
        lfc_b = lfc_a + rng.normal(scale=0.15, size=len(genes))
        de_a = set(genes[np.argsort(np.abs(lfc_a))[-12:]])
        de_b = set(genes[np.argsort(np.abs(lfc_b))[-12:]])
        signature = _signature(genes, de_a, de_b)
        signature["lfc_a"] = lfc_a
        signature["lfc_b"] = lfc_b
        signatures[f"u{index:02d}"] = signature
    units = sorted(signatures)
    config = {"fdr_threshold": 0.05, "lfc_threshold": 0.1}

    serial = module._pair_specific_repeat(
        signatures,
        units,
        config,
        workers=1,
        rows_per_task=4,
    )
    parallel = module._pair_specific_repeat(
        signatures,
        units,
        config,
        workers=4,
        rows_per_task=4,
    )

    np.testing.assert_array_equal(serial[0], parallel[0])
    for serial_array, parallel_array in zip(serial[1:], parallel[1:]):
        np.testing.assert_array_equal(serial_array, parallel_array)


def test_split_half_statistical_oracle_has_stronger_diagonal() -> None:
    module = _load_test1_module()
    rng = np.random.default_rng(11)
    genes = np.asarray([f"g{i:02d}" for i in range(48)])
    signatures = {}
    for index in range(16):
        shared = rng.normal(size=len(genes))
        signature = _signature(genes, set(genes), set(genes))
        signature["lfc_a"] = shared + rng.normal(scale=0.05, size=len(genes))
        signature["lfc_b"] = shared + rng.normal(scale=0.05, size=len(genes))
        signatures[f"u{index:02d}"] = signature
    units = sorted(signatures)

    _, spearman, _, _ = module._pair_specific_repeat(
        signatures,
        units,
        {"fdr_threshold": 0.05, "lfc_threshold": 0.1},
        workers=1,
    )
    diagonal = np.diag(spearman)
    off_diagonal = spearman[~np.eye(len(units), dtype=bool)]
    assert np.nanmedian(diagonal) > 0.9
    assert np.nanmedian(diagonal) > np.nanmedian(off_diagonal) + 0.5
