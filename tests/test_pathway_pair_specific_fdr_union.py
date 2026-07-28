from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PATHWAY_SKILL = (
    ROOT / ".claude/skills/evaluating-pathways-test1-reproducibility"
)


def _load_pathway_utils():
    spec = importlib.util.spec_from_file_location(
        "pathway_utils_fdr_union",
        PATHWAY_SKILL / "pathway_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROGRAMS = ("p1", "p2", "p3", "p4", "p5")
TARGETS = ("u1", "u2")


def _frame(
    effects: dict[str, list[float]],
    significant: dict[str, set[str]],
) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        for program, effect in zip(PROGRAMS, effects[target]):
            rows.append(
                {
                    "target": target,
                    "program": program,
                    "effect": effect,
                    "fdr": 0.01 if program in significant[target] else 0.5,
                    "n_target": 20,
                    "n_reference": 25,
                }
            )
    return pd.DataFrame(rows)


def _repeat_results():
    effects_a = {
        "u1": [1.0, 2.0, 4.0, 3.0, 5.0],
        "u2": [5.0, 1.0, 2.0, 4.0, 3.0],
    }
    effects_b = {
        "u1": [1.2, 2.2, 3.8, 2.8, 4.9],
        "u2": [4.8, 1.1, 2.2, 4.2, 3.1],
    }
    return [
        {
            "A": {
                "ols": _frame(
                    effects_a,
                    {"u1": {"p1", "p2", "p3"}, "u2": {"p1", "p3", "p5"}},
                ),
                "pdex_mwu": _frame(
                    effects_a,
                    {"u1": {"p2", "p3", "p4"}, "u2": {"p2", "p4", "p5"}},
                ),
            },
            "B": {
                "ols": _frame(
                    effects_b,
                    {"u1": {"p1", "p2", "p4"}, "u2": {"p1", "p4", "p5"}},
                ),
                "pdex_mwu": _frame(
                    effects_b,
                    {"u1": {"p2", "p4", "p5"}, "u2": {"p2", "p3", "p5"}},
                ),
            },
        }
    ]


def test_fdr_union_map_archives_pairwise_jaccard_and_all_boxplots(
    tmp_path: Path,
) -> None:
    pathway_utils = _load_pathway_utils()
    matrix_png = (
        tmp_path / "pathways_test4_corr_matrix_mean_fdr05__synthetic.png"
    )

    pathway_utils.plot_target_corr_matrix(
        _repeat_results(),
        str(matrix_png),
        "Synthetic pathway Test 4",
        unit="guide",
        sort_by_diagonal=False,
        fdr_threshold=0.05,
        emit_distribution_boxplots=True,
    )

    expected_stems = (
        "pathways_test4_corr_matrix_mean_fdr05_correlation_boxplots__synthetic",
        "pathways_test4_corr_matrix_mean_fdr05_jaccard_diagonal_boxplot__synthetic",
        "pathways_test4_corr_matrix_mean_fdr05_jaccard_off_diagonal_boxplot__synthetic",
    )
    for stem in expected_stems:
        assert (tmp_path / f"{stem}.png").exists()
        assert (tmp_path / f"{stem}.csv").exists()

    archive = np.load(matrix_png.with_suffix(".npz"))
    targets = archive["basis_targets"].tolist()
    u1 = targets.index("u1")
    assert archive["feature_count"][u1, u1] == 5
    assert np.isclose(archive["jaccard__ols"][u1, u1], 0.5)
    assert np.isclose(archive["jaccard__pdex_mwu"][u1, u1], 0.5)
    assert archive["finite_jaccard_repeats__ols"][u1, u1] == 1


def test_pathway_test4_runner_enables_fdr_union_family() -> None:
    source = (
        ROOT
        / ".claude/skills/evaluating-pathways-test4-guide-reproducibility"
        / "test4_guide_reproducibility.py"
    ).read_text()
    assert "pathways_test4_corr_matrix_mean_{fdr_label}" in source
    assert "fdr_threshold=args.fdr" in source
    assert "emit_distribution_boxplots=True" in source


def test_pathway_test1_runner_enables_fdr_union_distributions() -> None:
    source = (
        ROOT
        / ".claude/skills/evaluating-pathways-test1-reproducibility"
        / "test1_reproducibility.py"
    ).read_text()
    assert "pathways_test1_corr_matrix_mean_{fdr_label}" in source
    assert "fdr_threshold=args.fdr" in source
    assert "emit_distribution_boxplots=True" in source
