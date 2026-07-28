from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]

DE_SKILLS = (
    ("evaluating-test0-injection", "inject_and_count_de.py"),
    ("evaluating-test01-overview", "overview.py"),
    ("evaluating-test1-reproducibility", "reproducibility_heatmap.py"),
    ("evaluating-test2-control-null", "control_null_diagnostics.py"),
    ("evaluating-test3-permutation-null", "shuffle_de_comparison.py"),
    ("evaluating-test4-guide-reproducibility", "guide_split_reproducibility.py"),
    ("evaluating-test5-samegene-sgrna", "samegene_guide_heatmap.py"),
    ("evaluating-test6-knockdown-recovery", "knockdown_recovery.py"),
)

PATHWAY_SKILLS = (
    ("evaluating-pathways-overview", "overview.py"),
    ("evaluating-pathways-test0-injection", "test0_injection.py"),
    ("evaluating-pathways-test1-reproducibility", "test1_reproducibility.py"),
    ("evaluating-pathways-test2-control-null", "test2_control_null.py"),
    ("evaluating-pathways-test3-permutation-null", "test3_permutation_null.py"),
    ("evaluating-pathways-test4-guide-reproducibility", "test4_guide_reproducibility.py"),
    ("evaluating-pathways-test5-samegene-sgrna", "test5_samegene_sgrna.py"),
)

ALL_SKILLS = DE_SKILLS + PATHWAY_SKILLS


def _skill_path(name: str, filename: str) -> Path:
    return ROOT / ".claude" / "skills" / name / filename


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_every_runner_records_confirmed_run_configuration() -> None:
    for skill, runner in ALL_SKILLS:
        source = _skill_path(skill, runner).read_text()
        assert "--run-root" in source, skill
        assert "--expression-state" in source, skill
        assert "write_resolved_config(" in source, skill


def test_every_skill_requires_preflight_confirmation_and_capture() -> None:
    required = (
        "## Mandatory preflight and run capture",
        "explicit confirmation",
        "--expression-state",
        "<run-root>/logs",
        "<run-root>/configs",
        "2>&1 | tee",
        "Every box-and-whisker plot must overlay every finite",
    )
    for skill, _ in ALL_SKILLS:
        contract = _skill_path(skill, "SKILL.md").read_text()
        for phrase in required:
            assert phrase in contract, (skill, phrase)


def test_bundled_shared_helpers_are_identical() -> None:
    de_hashes = {
        _hash(_skill_path(skill, "de_backends.py")) for skill, _ in DE_SKILLS
    }
    pathway_hashes = {
        _hash(_skill_path(skill, "pathway_utils.py"))
        for skill, _ in PATHWAY_SKILLS
    }
    memory_hashes = {
        _hash(_skill_path(skill, "pathway-methods-memory.md"))
        for skill, _ in PATHWAY_SKILLS
    }
    assert len(de_hashes) == 1
    execution_contract_hashes = {
        _hash(_skill_path(skill, "hardware-execution-contract.md"))
        for skill, _ in DE_SKILLS
    }
    watchdog_hashes = {
        _hash(_skill_path(skill, "run_with_watchdog.py"))
        for skill, _ in DE_SKILLS
    }
    assert len(pathway_hashes) == 1
    assert len(memory_hashes) == 1
    assert len(execution_contract_hashes) == 1
    assert len(watchdog_hashes) == 1


def test_resolved_yaml_writer_creates_configs_and_logs(tmp_path: Path) -> None:
    module = _load_module(
        _skill_path("evaluating-pathways-overview", "pathway_utils.py"),
        "pathway_utils_run_capture_test",
    )
    output = Path(
        module.write_resolved_config(
            run_root=str(tmp_path / "run"),
            workflow="pathway test",
            dataset="data/set",
            resolved={
                "path": Path("input.h5ad"),
                "methods": ("ols", "pdex_mwu"),
                "seed": np.int64(42),
            },
        )
    )
    assert output.parent == (tmp_path / "run" / "configs").resolve()
    assert (tmp_path / "run" / "logs").is_dir()
    payload = yaml.safe_load(output.read_text())
    assert payload["workflow"] == "pathway test"
    assert payload["dataset"] == "data/set"
    assert payload["resolved_config"]["methods"] == ["ols", "pdex_mwu"]
    assert payload["resolved_config"]["seed"] == 42


def test_dense_pathway_matrices_suppress_only_diagonal_count_text() -> None:
    source = _skill_path(
        "evaluating-pathways-overview", "pathway_utils.py"
    ).read_text()
    assert "MAX_DIAGONAL_COUNT_LABELS = 40" in source
    assert "if len(ordered_targets) <= MAX_DIAGONAL_COUNT_LABELS:" in source
    for skill, _ in PATHWAY_SKILLS:
        contract = _skill_path(skill, "SKILL.md").read_text()
        assert "more than 40" in contract
        assert "white diagonal" in contract


def test_all_boxplot_implementations_overlay_scatter_points() -> None:
    implementations = (
        _skill_path("evaluating-test0-injection", "inject_and_count_de.py"),
        _skill_path(
            "evaluating-test1-reproducibility", "reproducibility_heatmap.py"
        ),
        _skill_path(
            "evaluating-test2-control-null", "control_null_diagnostics.py"
        ),
        _skill_path("evaluating-pathways-overview", "pathway_utils.py"),
        _skill_path(
            "evaluating-pathways-test2-control-null", "test2_control_null.py"
        ),
    )
    for path in implementations:
        source = path.read_text()
        assert ".boxplot(" in source, path
        assert ".scatter(" in source, path


def test_de_engine_presentation_labels_match_selected_backend() -> None:
    module = _load_module(
        _skill_path("evaluating-test0-injection", "de_backends.py"),
        "de_backends_label_test",
    )
    assert module.de_method_label("pdex", non_parametric_engine="pdex") == "pdex"
    assert module.de_method_label("pdex", non_parametric_engine="rsc") == "RSC"
    assert "RAPIDS GPU Wilcoxon" in module.de_method_label(
        "pdex", non_parametric_engine="rsc", verbose=True
    )


def test_watchdog_records_success_and_hardware(tmp_path: Path) -> None:
    watchdog = _skill_path(
        "evaluating-test01-overview",
        "run_with_watchdog.py",
    )
    report = tmp_path / "success.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(watchdog),
            "--artifact-root",
            str(tmp_path),
            "--idle-seconds",
            "5",
            "--wall-seconds",
            "10",
            "--poll-seconds",
            "0.05",
            "--label",
            "test-success",
            "--report",
            str(report),
            "--",
            sys.executable,
            "-c",
            "print('progress', flush=True)",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(report.read_text())
    assert payload["trigger"] == "exit"
    assert payload["return_code"] == 0
    assert payload["hardware"]["cpu_affinity_count"] >= 1
    assert payload["wall_seconds"] >= 0


def test_watchdog_terminates_a_quiet_process_group(tmp_path: Path) -> None:
    watchdog = _skill_path(
        "evaluating-test01-overview",
        "run_with_watchdog.py",
    )
    report = tmp_path / "idle.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(watchdog),
            "--artifact-root",
            str(tmp_path / "no-artifacts"),
            "--idle-seconds",
            "1",
            "--wall-seconds",
            "10",
            "--poll-seconds",
            "0.05",
            "--label",
            "test-idle",
            "--report",
            str(report),
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 124
    payload = json.loads(report.read_text())
    assert payload["trigger"] == "idle_timeout"
