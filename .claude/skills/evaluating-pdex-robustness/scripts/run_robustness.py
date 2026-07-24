#!/usr/bin/env python
"""Driver for the pdex-backend robustness battery.

Usage (from the cell-eval repo root):

    uv run python .claude/skills/evaluating-pdex-robustness/scripts/run_robustness.py \
        --config path/to/config.yaml

    # or run a subset of tests:
    uv run python .../run_robustness.py --config cfg.yaml --tests test_1,test_3,test_6

Writes to ``<outdir>/``: robustness_report.md, robustness_summary.json, and per-test
CSV tables under ``<outdir>/tables/``. The cell-eval run pipeline also writes its own
per-test result CSVs into ``<outdir>`` (tests 3 and 4).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the sibling pdrobust package importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdrobust import load_config, select_tests, write_report  # noqa: E402
from pdrobust.harness import load_adata  # noqa: E402
from pdrobust.tests import ALL_TESTS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="YAML or JSON RobustnessConfig file")
    ap.add_argument("--tests", default=None, help="comma-separated test ids (default: config/all)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("pdrobust")

    cfg = load_config(args.config)
    requested = args.tests.split(",") if args.tests else cfg.tests
    test_ids = select_tests(requested)
    log.info("Running tests: %s", test_ids)

    adata = load_adata(cfg)
    log.info("Loaded AnnData: %d cells x %d genes", adata.n_obs, adata.n_vars)

    results = []
    for tid in test_ids:
        log.info("=== %s ===", tid)
        try:
            res = ALL_TESTS[tid](adata, cfg)
        except Exception:
            log.exception("Test %s raised; recording as FAIL", tid)
            from pdrobust.tests import TestResult

            res = TestResult(tid, tid, "FAIL", notes="Unhandled exception (see log).")
        log.info("%s -> %s | %s", tid, res.verdict, res.headline)
        results.append(res)

    md_path = write_report(results, cfg)
    log.info("Report written to %s", md_path)
    print(f"\nReport: {md_path}")
    for r in results:
        print(f"  {r.name:8s} {r.verdict:5s} {r.title}")
    fails = [r.name for r in results if r.verdict == "FAIL"]
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
