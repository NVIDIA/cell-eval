import numpy as np
import pandas as pd
import polars as pl
import pytest
from anndata import AnnData

from cell_eval import MetricsEvaluator, initialize_de_comparison
from cell_eval._de_backends import (
    build_pydeseq2_inputs,
    compare_de_backends,
    normalize_de_methods,
    normalize_pydeseq2_results,
)
from cell_eval._pydeseq2_backend import _parse_pydeseq2_kwargs


def _adata_for_de() -> AnnData:
    counts = np.array(
        [
            [10, 0, 3],
            [9, 1, 2],
            [30, 0, 5],
            [29, 1, 4],
            [10, 20, 1],
            [11, 19, 2],
        ],
        dtype=np.float32,
    )
    return AnnData(
        X=np.log1p(counts),
        obs=pd.DataFrame(
            {
                "target": ["control", "control", "p1", "p1", "p2", "p2"],
                "batch": ["b1", "b2", "b1", "b2", "b1", "b2"],
            },
            index=pd.Index([f"cell_{idx}" for idx in range(counts.shape[0])]),
        ),
        var=pd.DataFrame(index=pd.Index(["g1", "g2", "g3"])),
    )


def _de_frame(lfc_scale: float = 1.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "target": ["p1", "p1", "p1", "p2", "p2", "p2"],
            "feature": ["g1", "g2", "g3", "g1", "g2", "g3"],
            "log2_fold_change": [
                2.0 * lfc_scale,
                0.1,
                -0.5,
                0.2,
                2.5 * lfc_scale,
                -0.2,
            ],
            "p_value": [0.001, 0.6, 0.2, 0.5, 0.001, 0.4],
            "fdr": [0.01, 0.8, 0.3, 0.7, 0.01, 0.5],
        }
    )


def test_normalize_de_methods_accepts_comma_separated_values() -> None:
    assert normalize_de_methods("pdex,pydeseq2") == ("pdex", "pydeseq2")


def test_normalize_de_methods_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown DE method"):
        normalize_de_methods("pdex,unknown")


def test_build_pydeseq2_inputs_pseudobulks_by_replicate() -> None:
    adata = AnnData(
        X=np.array(
            [
                [1, 2],
                [3, 4],
                [10, 20],
                [30, 40],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame(
            {
                "target": ["control", "control", "p1", "p1"],
                "batch": ["b1", "b2", "b1", "b2"],
            },
            index=pd.Index(["c1", "c2", "c3", "c4"]),
        ),
        var=pd.DataFrame(index=pd.Index(["g1", "g2"])),
    )

    counts, metadata = build_pydeseq2_inputs(
        adata=adata,
        groupby="target",
        reference="control",
        counts_layer=None,
        replicate_col="batch",
    )

    assert counts.loc["b1__control"].to_list() == [1, 2]
    assert counts.loc["b2__control"].to_list() == [3, 4]
    assert counts.loc["b1__p1"].to_list() == [10, 20]
    assert counts.loc["b2__p1"].to_list() == [30, 40]
    assert metadata.loc["b1__p1", "target"] == "p1"
    assert metadata.loc["b1__p1", "batch"] == "b1"


def test_build_pydeseq2_inputs_rejects_fractional_counts() -> None:
    adata = AnnData(
        X=np.array([[1.2, 2], [3, 4]], dtype=np.float32),
        obs=pd.DataFrame(
            {
                "target": ["control", "p1"],
                "batch": ["b1", "b1"],
            },
            index=pd.Index(["c1", "c2"]),
        ),
        var=pd.DataFrame(index=pd.Index(["g1", "g2"])),
    )

    with pytest.raises(ValueError, match="raw integer counts"):
        build_pydeseq2_inputs(
            adata=adata,
            groupby="target",
            reference="control",
            counts_layer=None,
            replicate_col="batch",
        )


def test_build_pydeseq2_inputs_rejects_fractional_values_even_if_sums_are_integer() -> (
    None
):
    adata = AnnData(
        X=np.array(
            [
                [0.5, 1.5],
                [0.5, 2.5],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame(
            {
                "target": ["control", "control", "p1", "p1"],
                "batch": ["b1", "b1", "b1", "b1"],
            },
            index=pd.Index(["c1", "c2", "c3", "c4"]),
        ),
        var=pd.DataFrame(index=pd.Index(["g1", "g2"])),
    )

    with pytest.raises(ValueError, match="raw integer counts"):
        build_pydeseq2_inputs(
            adata=adata,
            groupby="target",
            reference="control",
            counts_layer=None,
            replicate_col="batch",
        )


def test_build_pydeseq2_inputs_rejects_missing_obs_values() -> None:
    adata = AnnData(
        X=np.array([[1, 2], [3, 4]], dtype=np.float32),
        obs=pd.DataFrame(
            {
                "target": ["control", None],
                "batch": ["b1", "b1"],
            },
            index=pd.Index(["c1", "c2"]),
        ),
        var=pd.DataFrame(index=pd.Index(["g1", "g2"])),
    )

    with pytest.raises(ValueError, match="target.*missing"):
        build_pydeseq2_inputs(
            adata=adata,
            groupby="target",
            reference="control",
            counts_layer=None,
            replicate_col="batch",
        )


def test_build_pydeseq2_inputs_rejects_empty_obs_values() -> None:
    adata = AnnData(
        X=np.array([[1, 2], [3, 4]], dtype=np.float32),
        obs=pd.DataFrame(
            {
                "target": ["control", "p1"],
                "batch": ["b1", ""],
            },
            index=pd.Index(["c1", "c2"]),
        ),
        var=pd.DataFrame(index=pd.Index(["g1", "g2"])),
    )

    with pytest.raises(ValueError, match="batch.*empty"):
        build_pydeseq2_inputs(
            adata=adata,
            groupby="target",
            reference="control",
            counts_layer=None,
            replicate_col="batch",
        )


def test_parse_pydeseq2_kwargs_accepts_nested_backend_options() -> None:
    options = _parse_pydeseq2_kwargs(
        {
            "min_total_count": 10,
            "fill_filtered": False,
            "dds_kwargs": {"fit_type": "mean"},
            "stats_kwargs": {"alpha": 0.1},
        }
    )

    assert options.min_total_count == 10
    assert options.fill_filtered is False
    assert options.dds_kwargs == {"fit_type": "mean"}
    assert options.stats_kwargs == {"alpha": 0.1}


def test_parse_pydeseq2_kwargs_rejects_unknown_top_level_keys() -> None:
    with pytest.raises(ValueError, match="Unknown PyDESeq2 option"):
        _parse_pydeseq2_kwargs({"fit_type": "mean"})


def test_normalize_pydeseq2_results_maps_to_cell_eval_schema() -> None:
    native = pd.DataFrame(
        {
            "baseMean": [10.0, 0.0],
            "log2FoldChange": [1.5, -0.2],
            "pvalue": [0.01, np.nan],
            "padj": [0.02, np.nan],
        },
        index=pd.Index(["g1", "g2"]),
    )

    result = normalize_pydeseq2_results(native, target="p1")

    assert result.columns == [
        "target",
        "feature",
        "log2_fold_change",
        "p_value",
        "fdr",
    ]
    assert result["target"].to_list() == ["p1", "p1"]
    assert result["feature"].to_list() == ["g1", "g2"]
    assert result["p_value"].to_list() == [0.01, 1.0]
    assert result["fdr"].to_list() == [0.02, 1.0]


def test_metrics_evaluator_can_score_multiple_precomputed_de_methods(tmp_path) -> None:
    adata = _adata_for_de()
    real_pdex = _de_frame()
    pred_pdex = _de_frame(lfc_scale=0.9)
    real_pydeseq2 = _de_frame(lfc_scale=1.1)
    pred_pydeseq2 = _de_frame(lfc_scale=0.8)

    evaluator = MetricsEvaluator(
        adata_pred=adata.copy(),
        adata_real=adata.copy(),
        de_real={"pdex": real_pdex, "pydeseq2": real_pydeseq2},
        de_pred={"pdex": pred_pdex, "pydeseq2": pred_pydeseq2},
        control_pert="control",
        pert_col="target",
        de_methods=["pdex", "pydeseq2"],
        outdir=str(tmp_path),
    )

    results, agg_results = evaluator.compute(profile="de", write_csv=False)

    assert set(results["de_method"].to_list()) == {"pdex", "pydeseq2"}
    assert set(results["perturbation"].to_list()) == {"p1", "p2"}
    assert set(agg_results["de_method"].to_list()) == {"pdex", "pydeseq2"}
    assert evaluator.de_backend_comparison.height > 0


def test_compare_de_backends_filters_non_finite_lfc_for_correlations() -> None:
    left = pl.DataFrame(
        {
            "target": ["p1", "p1", "p1", "p1"],
            "feature": ["g1", "g2", "g3", "g4"],
            "log2_fold_change": [1.0, 2.0, np.inf, 4.0],
            "p_value": [0.01, 0.01, 0.01, 0.01],
            "fdr": [0.01, 0.01, 0.01, 0.01],
        }
    )
    right = pl.DataFrame(
        {
            "target": ["p1", "p1", "p1", "p1"],
            "feature": ["g1", "g2", "g3", "g4"],
            "log2_fold_change": [1.0, 3.0, 4.0, None],
            "p_value": [0.01, 0.01, 0.01, 0.01],
            "fdr": [0.01, 0.01, 0.01, 0.01],
        }
    )

    comparison = compare_de_backends(
        {
            "pdex": initialize_de_comparison(real=left, pred=left),
            "pydeseq2": initialize_de_comparison(real=right, pred=right),
        }
    )

    assert comparison["n_common_genes"].to_list() == [4, 4]
    assert comparison["n_finite_lfc"].to_list() == [2, 2]
    assert comparison["lfc_pearson"].to_list() == [1.0, 1.0]
    assert comparison["lfc_spearman"].to_list() == [1.0, 1.0]
