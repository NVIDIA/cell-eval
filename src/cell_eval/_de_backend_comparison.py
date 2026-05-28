from itertools import combinations
from typing import Any, Literal, Mapping, Sequence

import polars as pl

from ._types import DEComparison, DEResults


def compare_de_backends(
    comparisons: Mapping[str, DEComparison],
    *,
    fdr_threshold: float = 0.05,
    top_ks: Sequence[int] = (50, 100, 200),
) -> pl.DataFrame:
    """Compare DE result tables across backend pairs for real and pred data."""
    rows: list[dict[str, Any]] = []
    for left_method, right_method in combinations(comparisons.keys(), 2):
        left = comparisons[left_method]
        right = comparisons[right_method]
        rows.extend(
            _compare_de_result_pair(
                left.real,
                right.real,
                dataset="real",
                left_method=left_method,
                right_method=right_method,
                fdr_threshold=fdr_threshold,
                top_ks=top_ks,
            )
        )
        rows.extend(
            _compare_de_result_pair(
                left.pred,
                right.pred,
                dataset="pred",
                left_method=left_method,
                right_method=right_method,
                fdr_threshold=fdr_threshold,
                top_ks=top_ks,
            )
        )

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _compare_de_result_pair(
    left: DEResults,
    right: DEResults,
    *,
    dataset: Literal["real", "pred"],
    left_method: str,
    right_method: str,
    fdr_threshold: float,
    top_ks: Sequence[int],
) -> list[dict[str, Any]]:
    target_col = left.target_col
    feature_col = left.feature_col
    left_lfc = left.log2_fold_change_col
    right_lfc = f"{right.log2_fold_change_col}_right"
    left_fdr = left.fdr_col
    right_fdr = f"{right.fdr_col}_right"

    joined = left.data.join(
        right.data,
        on=[target_col, feature_col],
        suffix="_right",
        how="inner",
    )

    rows: list[dict[str, Any]] = []
    for pert in sorted(set(left.get_perts()) & set(right.get_perts())):
        pert_joined = joined.filter(pl.col(target_col) == pert)
        left_pert = left.data.filter(pl.col(target_col) == pert)
        right_pert = right.data.filter(pl.col(target_col) == pert)

        sig_left = set(
            left_pert.filter(pl.col(left.fdr_col) < fdr_threshold)[
                feature_col
            ].to_list()
        )
        sig_right = set(
            right_pert.filter(pl.col(right.fdr_col) < fdr_threshold)[
                feature_col
            ].to_list()
        )
        sig_union = sig_left | sig_right
        row: dict[str, Any] = {
            "dataset": dataset,
            "method_left": left_method,
            "method_right": right_method,
            "target": pert,
            "n_common_genes": pert_joined.height,
            "n_sig_left": len(sig_left),
            "n_sig_right": len(sig_right),
            "sig_jaccard": (
                len(sig_left & sig_right) / len(sig_union) if sig_union else 1.0
            ),
        }

        finite_lfc_joined = _filter_finite_lfc(pert_joined, left_lfc, right_lfc)
        row["n_finite_lfc"] = finite_lfc_joined.height
        if finite_lfc_joined.height >= 2:
            row["lfc_spearman"] = finite_lfc_joined.select(
                pl.corr(
                    pl.col(left_lfc).cast(pl.Float64),
                    pl.col(right_lfc).cast(pl.Float64),
                    method="spearman",
                )
            ).item()
            row["lfc_pearson"] = finite_lfc_joined.select(
                pl.corr(
                    pl.col(left_lfc).cast(pl.Float64),
                    pl.col(right_lfc).cast(pl.Float64),
                    method="pearson",
                )
            ).item()
        else:
            row["lfc_spearman"] = float("nan")
            row["lfc_pearson"] = float("nan")

        sig_joined = pert_joined.filter(
            (pl.col(left_fdr) < fdr_threshold) | (pl.col(right_fdr) < fdr_threshold)
        )
        finite_sig_joined = _filter_finite_lfc(sig_joined, left_lfc, right_lfc)
        row["direction_match_sig_union"] = (
            finite_sig_joined.select(
                (pl.col(left_lfc).sign() == pl.col(right_lfc).sign()).mean()
            ).item()
            if finite_sig_joined.height
            else float("nan")
        )

        for k in top_ks:
            left_top = _top_gene_set(
                left_pert, feature_col, left.abs_log2_fold_change_col, k
            )
            right_top = _top_gene_set(
                right_pert, feature_col, right.abs_log2_fold_change_col, k
            )
            denom = min(k, len(left_top), len(right_top))
            row[f"top_{k}_overlap"] = (
                len(left_top & right_top) / denom if denom else float("nan")
            )

        rows.append(row)
    return rows


def _filter_finite_lfc(
    frame: pl.DataFrame,
    left_lfc_col: str,
    right_lfc_col: str,
) -> pl.DataFrame:
    return frame.filter(
        pl.col(left_lfc_col).cast(pl.Float64).is_finite().fill_null(False)
        & pl.col(right_lfc_col).cast(pl.Float64).is_finite().fill_null(False)
    )


def _top_gene_set(
    frame: pl.DataFrame,
    feature_col: str,
    abs_lfc_col: str,
    k: int,
) -> set[str]:
    return set(
        frame.sort(abs_lfc_col, descending=True)
        .head(k)[feature_col]
        .cast(pl.Utf8)
        .to_list()
    )
