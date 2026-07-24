"""pdrobust — robustness/calibration test battery for cell-eval's PyDESeq2 backend.

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
