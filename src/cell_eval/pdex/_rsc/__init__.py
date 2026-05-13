import logging
import warnings
from typing import Literal

import anndata as ad
import cupy as cp
import cupyx.scipy.sparse as cp_sparse
import numpy as np
import pandas as pd
import polars as pl
import rapids_singlecell as rsc
from scipy.sparse import issparse as sp_issparse
from scipy.stats import false_discovery_control

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.WARNING)

PDEX_MODES = Literal["ref", "all", "on_target"]
DEFAULT_REFERENCE = "non-targeting"

__all__ = ["pdex", "DEFAULT_REFERENCE", "PDEX_MODES"]


# ---------- GPU math helpers (all-on-GPU until the final DataFrame build) ----------

def _ensure_gpu(adata: ad.AnnData) -> None:
    """Move adata.X to GPU if it isn't already."""
    X = adata.X
    if isinstance(X, (cp.ndarray, cp_sparse.csr_matrix, cp_sparse.csc_matrix)):
        return
    rsc.get.anndata_to_GPU(adata)


def _transformed_x(X, direction: Literal["log1p", "expm1"]):
    """Copy X with log1p/expm1 applied (sparsity-preserving for sparse inputs)."""
    if cp_sparse.issparse(X):
        return X.log1p() if direction == "log1p" else X.expm1()
    return cp.log1p(X) if direction == "log1p" else cp.expm1(X)


def _fold_change_gpu(x: cp.ndarray, y: cp.ndarray, epsilon: float) -> cp.ndarray:
    return cp.log2((x + epsilon) / (y + epsilon))


def _percent_change_gpu(x: cp.ndarray, y: cp.ndarray, epsilon: float) -> cp.ndarray:
    return (x - y) / (y + epsilon)


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR via scipy (host-side — ~18k entries, negligible)."""
    return false_discovery_control(pvals)


def _group_sums_on_gpu(
    adata: ad.AnnData,
    groupby: str,
    transform: Literal["none", "log1p", "expm1"],
) -> tuple[np.ndarray, cp.ndarray]:
    """Per-group column sums in the given transform space, on GPU.

    Returns ``(group_names, cupy sum matrix (n_groups, n_genes))``.
    """
    if transform == "none":
        agg = rsc.get.aggregate(adata, by=groupby, func="sum")
    else:
        direction = "log1p" if transform == "log1p" else "expm1"
        # Swap X in place rather than writing to layers — anndata's layer
        # storage is finicky about GPU (cupy sparse) arrays.
        original_X = adata.X
        adata.X = _transformed_x(original_X, direction)
        try:
            agg = rsc.get.aggregate(adata, by=groupby, func="sum")
        finally:
            adata.X = original_X
    # rsc.get.aggregate stores single-func results in agg.layers[func]
    # (agg.X stays None). Read from the layer.
    X = agg.X if agg.X is not None else agg.layers["sum"]
    if cp_sparse.issparse(X):
        X = X.toarray()
    return np.asarray(agg.obs_names), X


def _derive_mean_gpu(sum_row: cp.ndarray, count: int, geometric_mean: bool) -> cp.ndarray:
    m = sum_row / max(count, 1)
    return cp.expm1(m) if geometric_mean else m


# ---------- public API ----------

def pdex(
    adata: ad.AnnData,
    groupby: str,
    mode: PDEX_MODES = "ref",
    threads: int = 0,  # kept for API compat; unused on GPU path
    is_log1p: bool | None = None,
    geometric_mean: bool = True,
    as_pandas: bool = False,
    epsilon: float = 0.0,
    **kwargs,
) -> pl.DataFrame | pd.DataFrame:
    """rsc-backed differential expression matching pdex's schema.

    ``statistic`` is the Mann-Whitney U statistic (rsc's ``return_u_values=True``
    path). P-values use the normal-approximation with tie correction and
    continuity correction, matching ``scipy.stats.mannwhitneyu`` and the cpu
    (numba-mwu) backend bit-for-bit up to float precision.

    All pseudobulk, fold-change, percent-change, and FDR math runs on GPU
    via cupy; only the final (group-wise) DataFrame materializes on host.
    """
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if groupby not in adata.obs.columns:
        raise ValueError(
            f"Missing column: {groupby}. Available: {', '.join(adata.obs.columns)}"
        )
    if mode not in {"ref", "all", "on_target"}:
        raise ValueError(f"Invalid mode: {mode}")

    reference = kwargs.pop("reference", DEFAULT_REFERENCE)
    gene_col = kwargs.pop("gene_col", None)
    if mode == "on_target":
        if gene_col is None:
            raise ValueError("'gene_col' is required for mode='on_target'")
        if gene_col not in adata.obs.columns:
            raise ValueError(
                f"Missing column: {gene_col}. "
                f"Available: {', '.join(adata.obs.columns)}"
            )
    if kwargs:
        warnings.warn(
            f"Unexpected keyword arguments for mode={mode!r} (ignored): "
            f"{', '.join(kwargs)}",
            UserWarning,
            stacklevel=2,
        )

    if is_log1p is None:
        X = adata.X
        is_sparse = sp_issparse(X) or cp_sparse.issparse(X)
        sample = X.data[:1_000_000] if is_sparse else X.ravel()[:1_000_000]
        sample = cp.asnumpy(sample) if isinstance(sample, cp.ndarray) else np.asarray(sample)
        is_log1p = float(np.nanmax(sample)) < 15.0
        log.warning(
            "is_log1p not specified; auto-detected %s. "
            "Pass is_log1p explicitly to suppress this message.",
            is_log1p,
        )

    # Aggregation transform so derived means land in count space correctly.
    if geometric_mean and not is_log1p:
        transform = "log1p"
    elif not geometric_mean and is_log1p:
        transform = "expm1"
    else:
        transform = "none"

    labels = pd.Categorical(adata.obs[groupby])
    labels = labels.remove_categories(
        [c for c in labels.categories if c == "" or pd.isna(c)]
    )
    unique_groups = np.asarray(labels.categories)
    if mode in {"ref", "on_target"} and reference not in unique_groups:
        raise ValueError(
            f"Missing reference: {reference}. "
            f"Available: {', '.join(unique_groups)}"
        )

    feature_names = np.asarray(adata.var_names)
    # rsc.tl.rank_genes_groups requires the groupby column to be categorical.
    if not isinstance(adata.obs[groupby].dtype, pd.CategoricalDtype):
        adata.obs[groupby] = adata.obs[groupby].astype("category")
    _ensure_gpu(adata)

    group_counts = (
        adata.obs[groupby].value_counts().reindex(unique_groups).fillna(0).astype(int)
    ).to_dict()

    # ------- on_target: subset + single Wilcoxon + single aggregate -------
    if mode == "on_target":
        pairs = adata.obs[[groupby, gene_col]].drop_duplicates().dropna()
        pairs = pairs[pairs[groupby] != reference]
        cc = pairs[groupby].value_counts()
        multi = cc[cc > 1]
        if not multi.empty:
            raise ValueError(
                f"Groups map to multiple genes in {gene_col!r}: "
                f"{', '.join(multi.index.values)}"
            )
        feature_set = set(feature_names)
        missing = ~pairs[gene_col].isin(feature_set)
        if missing.any():
            bad = ", ".join(
                f"{g} ({gene})"
                for g, gene in zip(pairs[groupby][missing], pairs[gene_col][missing])
            )
            msg = (
                f"Found {int(missing.sum())} groups with missing on-target "
                f"genes in adata.var: {bad}"
            )
            log.warning(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
            pairs = pairs[~missing]
        group_gene_map = dict(zip(pairs[groupby], pairs[gene_col]))
        target_gene_set = set(group_gene_map.values())

        var_mask = np.array(
            [g in target_gene_set for g in feature_names], dtype=bool
        )
        sub = adata[:, var_mask].copy()
        _ensure_gpu(sub)

        rsc.tl.rank_genes_groups(
            sub, groupby=groupby, reference=reference,
            method="wilcoxon", tie_correct=True, use_continuity=True,
            return_u_values=True, use_raw=False,
            key_added="de",
        )
        ug = sub.uns["de"]

        group_names, sums = _group_sums_on_gpu(sub, groupby, transform)
        sum_of = {g: sums[i] for i, g in enumerate(group_names)}
        sub_vars = np.asarray(sub.var_names)
        sub_col_of = {g: i for i, g in enumerate(sub_vars)}
        ref_n = int(group_counts[reference])
        ref_mean_all = _derive_mean_gpu(sum_of[reference], ref_n, geometric_mean)

        rows = []
        for group_name, gene_name in group_gene_map.items():
            grp_n = int(group_counts[group_name])
            col = sub_col_of[gene_name]
            tgt_vec = _derive_mean_gpu(sum_of[group_name], grp_n, geometric_mean)
            tm = float(tgt_vec[col].get())
            rm = float(ref_mean_all[col].get())
            fc = float(_fold_change_gpu(
                cp.asarray([tm]), cp.asarray([rm]), epsilon
            )[0].get())
            pc = float(_percent_change_gpu(
                cp.asarray([tm]), cp.asarray([rm]), epsilon
            )[0].get())

            names_arr = ug["names"][group_name]
            i = int(np.where(names_arr == gene_name)[0][0])
            rows.append({
                "target": group_name,
                "feature": gene_name,
                "target_mean": tm,
                "ref_mean": rm,
                "target_membership": grp_n,
                "ref_membership": ref_n,
                "fold_change": fc,
                "percent_change": pc,
                "p_value": float(np.clip(ug["pvals"][group_name][i], 0, 1)),
                "statistic": float(ug["scores"][group_name][i]),
            })
        df = pd.DataFrame(rows)
        df["fdr"] = _bh_fdr(df["p_value"].to_numpy())
        return df if as_pandas else pl.from_pandas(df)

    # ------- ref / all: one Wilcoxon + one aggregate, GPU math per group -------
    rsc.tl.rank_genes_groups(
        adata, groupby=groupby,
        reference=reference if mode == "ref" else "rest",
        method="wilcoxon", tie_correct=True, use_continuity=True,
        return_u_values=True, use_raw=False,
        key_added="de",
    )
    ug = adata.uns["de"]

    group_names, sums = _group_sums_on_gpu(adata, groupby, transform)
    sum_of = {g: sums[i] for i, g in enumerate(group_names)}
    global_sum = sums.sum(axis=0) if mode == "all" else None
    n_total = int(adata.n_obs) if mode == "all" else None

    if mode == "ref":
        ref_n_const = int(group_counts[reference])
        ref_bulk_gpu = _derive_mean_gpu(sum_of[reference], ref_n_const, geometric_mean)

    # Accumulate column arrays per group, then build ONE DataFrame at the end.
    # Per-pert pd.DataFrame construction + pd.concat dominates DE wall time on
    # large pert sets (734 perts × ~10 cols on Arc data was ~25 s of the prior
    # 47 s init).
    n_g = len(feature_names)
    iter_groups = [g for g in unique_groups if not (mode == "ref" and g == reference)]
    n_total_rows = len(iter_groups) * n_g

    targets_col = np.empty(n_total_rows, dtype=object)
    features_col = np.tile(feature_names, len(iter_groups))
    # rsc aggregator and downstream cupy ops run in float64; keep that here so
    # cpu/gpu pdex outputs agree to ~1e-9 (the prior parity headroom).
    target_mean_col = np.empty(n_total_rows, dtype=np.float64)
    ref_mean_col = np.empty(n_total_rows, dtype=np.float64)
    target_membership_col = np.empty(n_total_rows, dtype=np.int64)
    ref_membership_col = np.empty(n_total_rows, dtype=np.int64)
    fc_col = np.empty(n_total_rows, dtype=np.float64)
    pc_col = np.empty(n_total_rows, dtype=np.float64)
    p_col = np.empty(n_total_rows, dtype=np.float64)
    z_col = np.empty(n_total_rows, dtype=np.float64)
    fdr_col = np.empty(n_total_rows, dtype=np.float64)

    for i, group_name in enumerate(iter_groups):
        grp_n = int(group_counts[group_name])
        tgt_bulk = _derive_mean_gpu(sum_of[group_name], grp_n, geometric_mean)

        if mode == "ref":
            rb = ref_bulk_gpu
            ref_n = ref_n_const
        else:
            ref_n = n_total - grp_n
            rb = _derive_mean_gpu(global_sum - sum_of[group_name], ref_n, geometric_mean)

        fc_gpu = _fold_change_gpu(tgt_bulk, rb, epsilon)
        pc_gpu = _percent_change_gpu(tgt_bulk, rb, epsilon)

        # Pull MWU stats for this group; rsc sorts by score so reorder to var_names
        names_arr = ug["names"][group_name]
        name_to_pos = {n: j for j, n in enumerate(names_arr)}
        order = np.array([name_to_pos[f] for f in feature_names])
        z = ug["scores"][group_name][order]
        p = np.clip(ug["pvals"][group_name][order], 0, 1)
        fdr = _bh_fdr(p)

        s = slice(i * n_g, (i + 1) * n_g)
        targets_col[s] = group_name
        target_mean_col[s] = cp.asnumpy(tgt_bulk)
        ref_mean_col[s] = cp.asnumpy(rb)
        target_membership_col[s] = grp_n
        ref_membership_col[s] = ref_n
        fc_col[s] = cp.asnumpy(fc_gpu)
        pc_col[s] = cp.asnumpy(pc_gpu)
        p_col[s] = p
        z_col[s] = z
        fdr_col[s] = fdr

    out = pd.DataFrame({
        "target": targets_col,
        "feature": features_col,
        "target_mean": target_mean_col,
        "ref_mean": ref_mean_col,
        "target_membership": target_membership_col,
        "ref_membership": ref_membership_col,
        "fold_change": fc_col,
        "percent_change": pc_col,
        "p_value": p_col,
        "statistic": z_col,
        "fdr": fdr_col,
    })
    return out if as_pandas else pl.from_pandas(out)
