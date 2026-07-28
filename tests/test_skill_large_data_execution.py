from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData
from cell_eval.data import build_random_anndata


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = (
    ROOT
    / ".claude"
    / "skills"
    / "evaluating-test01-overview"
    / "de_backends.py"
)


def _load_adapter(name: str = "large_data_de_adapter") -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ADAPTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _three_condition_counts() -> AnnData:
    return AnnData(
        X=np.asarray(
            [
                [10, 2, 1],
                [12, 3, 1],
                [9, 2, 2],
                [11, 4, 1],
                [30, 2, 1],
                [32, 3, 2],
                [29, 3, 1],
                [31, 4, 2],
                [10, 20, 1],
                [11, 22, 2],
                [12, 19, 1],
                [9, 21, 2],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame(
            {
                "condition": [
                    "control",
                    "control",
                    "control",
                    "control",
                    "p1",
                    "p1",
                    "p1",
                    "p1",
                    "p2",
                    "p2",
                    "p2",
                    "p2",
                ],
                "replicate": ["r1", "r1", "r2", "r2"] * 3,
            },
            index=[f"cell_{index}" for index in range(12)],
        ),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )


def test_fixed_seed_synthetic_fixture_is_deterministic() -> None:
    first = build_random_anndata(
        n_cells=128,
        n_genes=128,
        n_perts=4,
        random_state=1729,
        as_sparse=True,
        normlog=False,
    )
    second = build_random_anndata(
        n_cells=128,
        n_genes=128,
        n_perts=4,
        random_state=1729,
        as_sparse=True,
        normlog=False,
    )
    np.testing.assert_array_equal(first.X.toarray(), second.X.toarray())
    pd.testing.assert_frame_equal(first.obs, second.obs)


class _FakeDeseqDataSet:
    fits: list[set[str]] = []

    def __init__(self, *, counts, metadata, **kwargs):
        del counts, kwargs
        self.metadata = metadata
        self.__class__.fits.append(set(metadata["condition"].astype(str)))

    def deseq2(self) -> None:
        return None


class _FakeDeseqStats:
    def __init__(self, dds, *, contrast, **kwargs):
        del dds, kwargs
        target = contrast[1]
        self.results_df = pd.DataFrame(
            {
                "baseMean": [10.0, 20.0, 30.0],
                "log2FoldChange": [
                    1.0 if target == "p1" else 0.1,
                    0.2 if target == "p1" else 1.2,
                    -0.1,
                ],
                "pvalue": [0.01, 0.2, 0.8],
                "padj": [0.03, 0.3, 0.9],
            },
            index=["g1", "g2", "g3"],
        )

    def summary(self) -> None:
        return None


def test_pairwise_pydeseq_dispatch_checkpoints_and_reuses_control(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_adapter()
    original_import = module.import_module

    def fake_import(name: str):
        if name == "pydeseq2.dds":
            return type("DDSModule", (), {"DeseqDataSet": _FakeDeseqDataSet})
        if name == "pydeseq2.ds":
            return type("DSModule", (), {"DeseqStats": _FakeDeseqStats})
        return original_import(name)

    monkeypatch.setattr(module, "import_module", fake_import)
    calls: list[str] = []
    original_condition_pseudobulks = module._condition_pseudobulks

    def counted_condition_pseudobulks(**kwargs):
        calls.append(kwargs["condition"])
        return original_condition_pseudobulks(**kwargs)

    monkeypatch.setattr(
        module,
        "_condition_pseudobulks",
        counted_condition_pseudobulks,
    )
    _FakeDeseqDataSet.fits.clear()
    checkpoint_dir = tmp_path / "contrasts"
    result = module.compute_pydeseq2_de(
        adata=_three_condition_counts(),
        reference="control",
        groupby="condition",
        threads=1,
        replicate_col="replicate",
        pydeseq2_kwargs={
            "fit_strategy": "auto",
            "max_joint_samples": 2,
            "workers": 1,
            "checkpoint_dir": str(checkpoint_dir),
        },
    )

    assert set(result["target"].to_list()) == {"p1", "p2"}
    assert result.height == 6
    assert calls.count("control") == 1
    assert calls.count("p1") == 1
    assert calls.count("p2") == 1
    assert _FakeDeseqDataSet.fits == [
        {"control", "p1"},
        {"control", "p2"},
    ]
    assert len(list(checkpoint_dir.glob("contrast_*.parquet"))) == 2

    class _MustNotRefit:
        def __init__(self, **kwargs):
            del kwargs
            raise AssertionError("a completed contrast was refit")

    def cached_import(name: str):
        if name == "pydeseq2.dds":
            return type("DDSModule", (), {"DeseqDataSet": _MustNotRefit})
        if name == "pydeseq2.ds":
            return type("DSModule", (), {"DeseqStats": _FakeDeseqStats})
        return original_import(name)

    monkeypatch.setattr(module, "import_module", cached_import)
    resumed = module.compute_pydeseq2_de(
        adata=_three_condition_counts(),
        reference="control",
        groupby="condition",
        threads=1,
        replicate_col="replicate",
        pydeseq2_kwargs={
            "fit_strategy": "pairwise",
            "workers": 1,
            "checkpoint_dir": str(checkpoint_dir),
            "resume": True,
        },
    )
    assert resumed.equals(result)

    changed_identity = _three_condition_counts()
    changed_identity.obs.loc["cell_0", "replicate"] = "different"
    with np.testing.assert_raises_regex(
        ValueError,
        "Stale PyDESeq2 checkpoint metadata",
    ):
        module.compute_pydeseq2_de(
            adata=changed_identity,
            reference="control",
            groupby="condition",
            threads=1,
            replicate_col="replicate",
            pydeseq2_kwargs={
                "fit_strategy": "pairwise",
                "workers": 1,
                "checkpoint_dir": str(checkpoint_dir),
                "resume": True,
            },
        )


def test_target_only_mode_keeps_one_result_per_contrast(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_adapter("target_only_de_adapter")
    original_import = module.import_module

    def fake_import(name: str):
        if name == "pydeseq2.dds":
            return type("DDSModule", (), {"DeseqDataSet": _FakeDeseqDataSet})
        if name == "pydeseq2.ds":
            return type("DSModule", (), {"DeseqStats": _FakeDeseqStats})
        return original_import(name)

    monkeypatch.setattr(module, "import_module", fake_import)
    result = module.compute_pydeseq2_de(
        adata=_three_condition_counts(),
        reference="control",
        groupby="condition",
        threads=1,
        replicate_col="replicate",
        pydeseq2_kwargs={
            "fit_strategy": "pairwise",
            "workers": 1,
            "checkpoint_dir": str(tmp_path / "target_only"),
            "result_mode": "target_only",
        },
    )
    # These synthetic targets are not feature names, so target-only retention
    # correctly avoids materialising any irrelevant genome-wide result rows.
    assert result.height == 0


def test_hardware_worker_limit_caps_cpu_and_memory(monkeypatch) -> None:
    module = _load_adapter("hardware_limit_de_adapter")
    monkeypatch.setattr(module, "_available_memory_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda _pid: set(range(12)))

    assert module.hardware_worker_limit(
        requested=0,
        threads_per_worker=2,
        worker_memory_bytes=2 * 1024**3,
        max_auto_workers=16,
        memory_fraction=0.5,
    ) == 2
    assert module.hardware_worker_limit(
        requested=20,
        threads_per_worker=4,
        worker_memory_bytes=512 * 1024**2,
        max_auto_workers=16,
        memory_fraction=0.5,
    ) == 3


def test_rsc_request_executes_rsc_path_and_reports_rsc(
    monkeypatch,
) -> None:
    module = _load_adapter("rsc_provenance_adapter")
    calls: list[dict[str, object]] = []

    def fake_rsc(**kwargs):
        calls.append(kwargs)
        return pl.DataFrame(
            {
                "target": ["p1"],
                "feature": ["g1"],
                "log2_fold_change": [1.0],
                "p_value": [0.01],
                "fdr": [0.02],
            }
        )

    monkeypatch.setattr(module, "_compute_rsc_de", fake_rsc)
    result = module.build_de_frame(
        mode="real",
        adata=_three_condition_counts(),
        control_pert="control",
        pert_col="condition",
        num_threads=1,
        allow_discrete=True,
        de_method="pdex",
        de_kwargs={"engine": "rsc", "copy": False},
    )

    assert result.height == 1
    assert len(calls) == 1
    assert calls[0]["copy_input"] is False
    assert module.de_method_label(
        "pdex",
        non_parametric_engine="rsc",
    ) == "RSC"


def test_rsc_omits_singleton_targets_but_keeps_supported_groups(
    monkeypatch,
) -> None:
    module = _load_adapter("rsc_singleton_adapter")
    adata = AnnData(
        X=np.asarray(
            [
                [1, 2],
                [2, 1],
                [8, 2],
                [9, 1],
                [3, 7],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame(
            {"condition": ["control", "control", "supported", "supported", "singleton"]},
            index=[f"cell_{index}" for index in range(5)],
        ),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    observed_groups: list[str] = []

    class _FakeGet:
        @staticmethod
        def anndata_to_GPU(work) -> None:
            del work

    class _FakeTl:
        @staticmethod
        def rank_genes_groups(work, *, groups, key_added, **kwargs) -> None:
            del kwargs
            observed_groups.extend(groups)
            dtype = [(group, object) for group in groups]
            names = np.empty(2, dtype=dtype)
            lfcs = np.empty(2, dtype=dtype)
            pvals = np.empty(2, dtype=dtype)
            padj = np.empty(2, dtype=dtype)
            for group in groups:
                names[group] = ["g1", "g2"]
                lfcs[group] = [1.0, -1.0]
                pvals[group] = [0.01, 0.2]
                padj[group] = [0.02, 0.2]
            work.uns[key_added] = {
                "names": names,
                "logfoldchanges": lfcs,
                "pvals": pvals,
                "pvals_adj": padj,
            }

    fake_rsc = ModuleType("rapids_singlecell")
    fake_rsc.get = _FakeGet()
    fake_rsc.tl = _FakeTl()
    fake_scanpy = ModuleType("scanpy")
    fake_scanpy.pp = type(
        "_FakePp",
        (),
        {
            "normalize_total": staticmethod(lambda work, inplace: None),
            "log1p": staticmethod(lambda work: None),
        },
    )()
    monkeypatch.setitem(sys.modules, "rapids_singlecell", fake_rsc)
    monkeypatch.setitem(sys.modules, "scanpy", fake_scanpy)

    result = module._compute_rsc_de(
        adata=adata,
        reference="control",
        groupby="condition",
    )

    assert observed_groups == ["supported"]
    assert set(result["target"].to_list()) == {"supported"}
