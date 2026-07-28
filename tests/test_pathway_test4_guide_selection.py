from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SKILL_DIR = (
    Path(__file__).resolve().parents[1]
    / ".claude"
    / "skills"
    / "evaluating-pathways-test4-guide-reproducibility"
)
sys.path.insert(0, str(SKILL_DIR))
try:
    SPEC = importlib.util.spec_from_file_location(
        "pathway_test4_guide_reproducibility",
        SKILL_DIR / "test4_guide_reproducibility.py",
    )
    assert SPEC is not None and SPEC.loader is not None
    MODULE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.remove(str(SKILL_DIR))


def test_select_powered_guides_accepts_one_guide_per_gene() -> None:
    eligible = pd.DataFrame(
        {
            "guide": ["guide-a", "guide-b", "guide-c"],
            "gene": ["GENE_A", "GENE_B", "GENE_C"],
            "n_cells": [40, 80, 60],
        }
    )

    selected = MODULE.select_powered_guides(
        eligible,
        max_guides=0,
        max_guides_per_gene=0,
    )

    assert selected == ["guide-b", "guide-c", "guide-a"]


def test_select_powered_guides_applies_caps_without_multi_guide_requirement() -> None:
    eligible = pd.DataFrame(
        {
            "guide": ["a-high", "a-low", "b-only"],
            "gene": ["GENE_A", "GENE_A", "GENE_B"],
            "n_cells": [100, 80, 90],
        }
    )

    selected = MODULE.select_powered_guides(
        eligible,
        max_guides=1,
        max_guides_per_gene=1,
    )

    assert selected == ["a-high"]
