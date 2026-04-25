import os
import shutil
from typing import Literal, cast

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from cell_eval import MetricsEvaluator
from cell_eval._baseline import _build_pert_baseline
from cell_eval.metrics._anndata import ClusteringAgreement
from cell_eval._types._anndata import PerturbationAnndataPair
from cell_eval.data import (
    CONTROL_VAR,
    PERT_COL,
    build_random_anndata,
    downsample_cells,
)

OUTDIR = "TEST_OUTPUT_DIRECTORY"
KNOWN_PROFILES: list[Literal["full", "vcc", "minimal", "de", "anndata"]] = [
    "full",
    "vcc",
    "minimal",
    "de",
    "anndata",
]


def test_broken_adata_mismatched_var_size():
    adata_real = build_random_anndata(normlog=False)
    adata_pred = adata_real.copy()

    # Randomly subset genes on pred
    var_mask = np.random.random(adata_real.shape[1]) < 0.8
    adata_pred = adata_pred[:, var_mask]

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            outdir=OUTDIR,
        )


def test_broken_adata_mismatched_var_ordering():
    adata_real = build_random_anndata(normlog=False)
    adata_pred = adata_real.copy()

    # Randomly subset genes on pred
    indices = np.arange(adata_real.shape[1])
    np.random.shuffle(indices)
    adata_pred = adata_pred[:, indices]

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            outdir=OUTDIR,
        )


def test_broken_adata_not_normlog():
    adata_real = build_random_anndata(normlog=False)
    adata_pred = adata_real.copy()
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
    )
    evaluator.compute(
        break_on_error=True,
    )


def test_broken_adata_not_normlog_skip_check():
    adata_real = build_random_anndata(normlog=False)
    adata_pred = adata_real.copy()
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
        allow_discrete=True,
    )
    evaluator.compute(
        break_on_error=True,
    )


def test_broken_adata_invalid_pred_scale():
    """Test that predicted data with invalid scale is rejected."""
    adata_real = build_random_anndata(normlog=True)
    adata_pred = adata_real.copy()

    # Create invalid predicted data: mix of raw counts and log1p
    adata_pred.X = np.random.uniform(
        0,
        5000,
        size=adata_pred.X.shape,  # type: ignore
    )

    with pytest.raises(ValueError, match="Invalid scale.*exceeds log1p threshold"):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            outdir=OUTDIR,
        )


def test_broken_adata_missing_pertcol_in_real():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove pert_col from adata_real
    cast(pd.DataFrame, adata_real.obs).drop(columns=[PERT_COL], inplace=True)

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            outdir=OUTDIR,
        )


def test_broken_adata_missing_pertcol_in_pred():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove pert_col from adata_pred
    cast(pd.DataFrame, adata_pred.obs).drop(columns=[PERT_COL], inplace=True)

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            outdir=OUTDIR,
        )


def test_broken_adata_missing_control_in_real():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove control_pert from adata_real
    adata_real = adata_real[adata_real.obs[PERT_COL] != CONTROL_VAR].copy()

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            outdir=OUTDIR,
        )


def test_broken_adata_missing_control_in_pred():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove control_pert from adata_pred
    adata_pred = adata_pred[adata_pred.obs[PERT_COL] != CONTROL_VAR].copy()

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            outdir=OUTDIR,
        )


def test_unknown_alternative_de_metric():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove control_pert from adata_pred
    adata_pred = adata_pred[adata_pred.obs[PERT_COL] != CONTROL_VAR].copy()

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            outdir=OUTDIR,
            de_method="unknown",  # ty: ignore[unknown-argument]
        ).compute()


def test_eval_simple():
    adata_real = build_random_anndata()
    adata_pred = downsample_cells(adata_real, fraction=0.5)
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert="control",
        pert_col="perturbation",
    )
    evaluator.compute(
        break_on_error=True,
    )


def test_eval_simple_profiles():
    adata_real = build_random_anndata()
    adata_pred = downsample_cells(adata_real, fraction=0.5)
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert="control",
        pert_col="perturbation",
    )
    for profile in KNOWN_PROFILES:
        evaluator.compute(
            profile=profile,
            break_on_error=True,
        )

    with pytest.raises(ValueError):
        evaluator.compute(
            profile="unknown",  # type: ignore
            break_on_error=True,
        )


def test_eval_missing_celltype_col():
    adata_real = build_random_anndata()
    adata_pred = downsample_cells(adata_real, fraction=0.5)

    cast(pd.DataFrame, adata_real.obs).drop(columns="celltype", inplace=True)
    cast(pd.DataFrame, adata_pred.obs).drop(columns="celltype", inplace=True)

    assert "celltype" not in adata_real.obs.columns
    assert "celltype" not in adata_pred.obs.columns

    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert="control",
        pert_col="perturbation",
    )
    evaluator.compute(
        break_on_error=True,
    )


def _reference_group_means(
    matrix: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.unique(labels)
    values = np.vstack([matrix[labels == key].mean(axis=0) for key in keys]).astype(
        np.float64
    )
    return keys, values


def test_bulk_anndata_matches_reference_for_dense_and_sparse():
    matrix = np.array(
        [
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 1.0],
            [0.0, 1.0, 5.0],
            [2.0, 3.0, 7.0],
            [4.0, 5.0, 9.0],
        ],
        dtype=np.float64,
    )
    labels = np.array(["pert_b", "control", "pert_b", "pert_a", "control"])
    obs = pd.DataFrame({PERT_COL: labels}, index=np.arange(len(labels)).astype(str))
    var = pd.DataFrame(index=["gene_0", "gene_1", "gene_2"])

    expected_keys, expected_values = _reference_group_means(matrix, labels)

    adata_dense = ad.AnnData(X=matrix.copy(), obs=obs.copy(), var=var.copy())
    dense_keys, dense_values = PerturbationAnndataPair._bulk_anndata(
        adata_dense, PERT_COL
    )

    adata_sparse = ad.AnnData(
        X=csr_matrix(matrix), obs=obs.copy(), var=var.copy()
    )
    sparse_keys, sparse_values = PerturbationAnndataPair._bulk_anndata(
        adata_sparse, PERT_COL
    )

    np.testing.assert_array_equal(dense_keys, expected_keys)
    np.testing.assert_array_equal(sparse_keys, expected_keys)
    np.testing.assert_allclose(dense_values, expected_values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sparse_values, expected_values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sparse_values, dense_values, rtol=1e-12, atol=1e-12)


def test_bulk_anndata_embed_key_matches_reference():
    obs = pd.DataFrame(
        {PERT_COL: np.array(["control", "pert_a", "pert_a", "pert_b"])},
        index=np.arange(4).astype(str),
    )
    adata = ad.AnnData(X=np.zeros((4, 2), dtype=np.float64), obs=obs)
    adata.obsm["X_test"] = np.array(
        [
            [1.0, 0.0, 2.0],
            [3.0, 1.0, 4.0],
            [5.0, 3.0, 6.0],
            [7.0, 5.0, 8.0],
        ],
        dtype=np.float64,
    )

    expected_keys, expected_values = _reference_group_means(
        adata.obsm["X_test"], obs[PERT_COL].to_numpy(str)
    )
    keys, values = PerturbationAnndataPair._bulk_anndata(
        adata, PERT_COL, embed_key="X_test"
    )

    np.testing.assert_array_equal(keys, expected_keys)
    np.testing.assert_allclose(values, expected_values, rtol=1e-12, atol=1e-12)


def test_build_pert_baseline_matches_reference_for_dense_and_sparse():
    matrix = np.array(
        [
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 1.0],
            [0.0, 1.0, 5.0],
            [2.0, 3.0, 7.0],
            [4.0, 5.0, 9.0],
        ],
        dtype=np.float64,
    )
    labels = np.array(["non-targeting", "pert_b", "pert_b", "pert_a", "non-targeting"])
    obs = pd.DataFrame({"target_gene": labels}, index=np.arange(len(labels)).astype(str))

    expected_keys, expected_matrix = _reference_group_means(matrix, labels)
    pert_mask = expected_keys != "non-targeting"
    expected_mean = expected_matrix.mean(axis=0)
    expected_delta = (expected_matrix[pert_mask] - expected_matrix[~pert_mask]).mean(axis=0)

    dense = ad.AnnData(X=matrix.copy(), obs=obs.copy())
    sparse = ad.AnnData(X=csr_matrix(matrix), obs=obs.copy())

    for adata in (dense, sparse):
        np.testing.assert_allclose(
            _build_pert_baseline(adata, as_delta=False),
            expected_mean,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            _build_pert_baseline(adata, as_delta=True),
            expected_delta,
            rtol=1e-12,
            atol=1e-12,
        )


def test_centroid_ann_matches_reference_for_dense_and_sparse():
    matrix = np.array(
        [
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 1.0],
            [0.0, 1.0, 5.0],
            [2.0, 3.0, 7.0],
            [4.0, 5.0, 9.0],
        ],
        dtype=np.float64,
    )
    labels = np.array(["pert_b", "control", "pert_b", "pert_a", "control"])
    obs = pd.DataFrame({PERT_COL: labels}, index=np.arange(len(labels)).astype(str))
    var = pd.DataFrame(index=["gene_0", "gene_1", "gene_2"])

    expected_keys, expected_values = _reference_group_means(matrix, labels)
    keep = expected_keys != "control"

    dense = ad.AnnData(X=matrix.copy(), obs=obs.copy(), var=var.copy())
    sparse = ad.AnnData(X=csr_matrix(matrix), obs=obs.copy(), var=var.copy())

    for adata in (dense, sparse):
        centroid = ClusteringAgreement._centroid_ann(
            adata,
            category_key=PERT_COL,
            control_pert="control",
        )
        np.testing.assert_array_equal(
            centroid.obs[PERT_COL].to_numpy(str),
            expected_keys[keep],
        )
        np.testing.assert_allclose(
            np.asarray(centroid.X),
            expected_values[keep],
            rtol=1e-12,
            atol=1e-12,
        )


def test_centroid_ann_embed_key_matches_reference():
    obs = pd.DataFrame(
        {PERT_COL: np.array(["control", "pert_a", "pert_a", "pert_b"])},
        index=np.arange(4).astype(str),
    )
    adata = ad.AnnData(X=np.zeros((4, 2), dtype=np.float64), obs=obs)
    adata.obsm["X_test"] = np.array(
        [
            [1.0, 0.0, 2.0],
            [3.0, 1.0, 4.0],
            [5.0, 3.0, 6.0],
            [7.0, 5.0, 8.0],
        ],
        dtype=np.float64,
    )

    expected_keys, expected_values = _reference_group_means(
        adata.obsm["X_test"], obs[PERT_COL].to_numpy(str)
    )
    keep = expected_keys != "control"

    centroid = ClusteringAgreement._centroid_ann(
        adata,
        category_key=PERT_COL,
        control_pert="control",
        embed_key="X_test",
    )

    np.testing.assert_array_equal(
        centroid.obs[PERT_COL].to_numpy(str),
        expected_keys[keep],
    )
    np.testing.assert_allclose(
        np.asarray(centroid.X),
        expected_values[keep],
        rtol=1e-12,
        atol=1e-12,
    )


def test_eval_pdex_kwargs():
    adata_real = build_random_anndata()
    adata_pred = downsample_cells(adata_real, fraction=0.5)
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert="control",
        pert_col="perturbation",
        pdex_kwargs={
            "geometric_mean": False,
        },
    )
    evaluator.compute(
        break_on_error=True,
    )


def test_eval_pdex_kwargs_duplicated():
    adata_real = build_random_anndata()
    adata_pred = downsample_cells(adata_real, fraction=0.5)
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert="control",
        pert_col="perturbation",
        pdex_kwargs={
            "geometric_mean": False,
            "threads": 4,
        },
    )
    evaluator.compute(
        break_on_error=True,
    )


def validate_expected_files(
    outdir: str, prefix: str | None = None, remove: bool = True
):
    base_real_de = "real_de.csv" if not prefix else f"{prefix}_real_de.csv"
    base_pred_de = "pred_de.csv" if not prefix else f"{prefix}_pred_de.csv"
    base_results = "results.csv" if not prefix else f"{prefix}_results.csv"
    assert os.path.exists(f"{outdir}/{base_real_de}"), (
        "Expected file for real DE results missing"
    )
    assert os.path.exists(f"{outdir}/{base_pred_de}"), (
        "Expected file for predicted DE results missing"
    )
    assert os.path.exists(f"{outdir}/{base_results}"), (
        "Expected file for results missing"
    )
    if remove:
        shutil.rmtree(outdir)


def test_eval():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
    )
    evaluator.compute(
        break_on_error=True,
    )
    validate_expected_files(OUTDIR)


def test_eval_prefix():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
        prefix="arbitrary",
    )
    evaluator.compute(
        break_on_error=True,
    )
    validate_expected_files(OUTDIR, prefix="arbitrary")


def test_minimal_eval():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
    )
    evaluator.compute(
        profile="minimal",
        break_on_error=True,
    )
    validate_expected_files(OUTDIR)


def test_eval_sparse():
    adata_real = build_random_anndata(as_sparse=True)
    adata_pred = adata_real.copy()
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
    )
    evaluator.compute(
        break_on_error=True,
    )
    validate_expected_files(OUTDIR)


def test_eval_downsampled_cells():
    adata_real = build_random_anndata()
    adata_pred = downsample_cells(adata_real, fraction=0.5)
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
    )
    evaluator.compute(
        break_on_error=True,
    )
    validate_expected_files(OUTDIR)
