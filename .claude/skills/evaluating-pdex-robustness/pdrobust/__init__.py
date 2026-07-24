"""pdrobust — robustness/calibration test battery for cell-eval's pdex backend.

pdex is the default cell-level Wilcoxon DE backend (``cell-eval run --de-methods pdex``).
See ../SKILL.md and ../reference/TEST_PLAN.md.
"""

from .harness import RobustnessConfig, load_config
from .report import write_report
from .tests import ALL_TESTS, TestResult, select_tests

__all__ = [
    "RobustnessConfig",
    "load_config",
    "write_report",
    "ALL_TESTS",
    "TestResult",
    "select_tests",
]
