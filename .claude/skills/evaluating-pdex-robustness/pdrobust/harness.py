"""Shared machinery for driving cell-eval's pdex backend under robustness tests.

Everything here builds on the *actual* cell-eval code paths so the tests exercise
the real run pipeline, not a reimplementation:

- ``run_pdex_de``          -> thin wrapper over ``cell_eval._pdex_backend.compute_pdex_de``
                             (cell-level Wilcoxon DE; the default of ``cell-eval run``)
- ``run_pipeline``         -> drives ``cell_eval.MetricsEvaluator`` with ``de_methods=["pdex"]``
                             (this is exactly ``cell-eval run --de-methods pdex``)
- ``de_summary``           -> per-target null/effect summary of a DE frame
- ``compare_signatures``   -> reproducibility comparison between two DE frames
- ``stratified_split`` / ``permute_labels_within_blocks`` -> obs surgery for null/repro designs

All DE frames follow the canonical cell-eval schema produced by the backend:
``[target, feature, log2_fold_change, p_value, fdr]``.

pdex is a *cell-level* Wilcoxon test (NOT pseudobulk), so there is no replicate /
pseudobulk-sample requirement, and it expects **log-normalized** expression in ``.X``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy import stats

from cell_eval import MetricsEvaluator
from cell_eval._pdex_backend import compute_pdex_de
from cell_eval.utils import guess_is_lognorm

logger = logging.getLogger("pdrobust")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class RobustnessConfig:
    """Resolved configuration for a robustness run.

    Built from a YAML/JSON file (see ``reference/config.example.yaml``). Only
    ``adata_path``, ``pert_col`` and ``control_pert`` are truly required; the rest
    gate which tests can run. ``replicate_col`` is OPTIONAL for the pdex (cell-level)
    backend — when given it is used only to stratify splits/subsampling; when None,
    stratification falls back to ``block_cols`` or perturbation alone.
    """

    adata_path: str
    pert_col: str
    control_pert: str
    replicate_col: str | None = None  # optional: stratification unit only (pdex is cell-level)

    target_gene_col: str | None = None  # sgRNA -> target gene (tests 5, 6, 7)
    sgrna_col: str | None = None  # per-guide identity (tests 4, 5)

    # Optional blocking covariates used to stratify splits / permutations.
    block_cols: list[str] = field(default_factory=list)

    fdr_threshold: float = 0.05
    seed: int = 0
    num_threads: int = 1
    # pdex expects log-normalized .X. allow_discrete=False => is_log1p=True in pdex AND
    # lets the run pipeline auto-normalize raw .X for the AnnData-level metrics. Keep False
    # and let load_adata normalize raw counts once (see normalize_if_raw).
    allow_discrete: bool = False
    # When .X looks like raw integer counts, normalize_total(1e4)+log1p once at load so pdex
    # receives lognorm input. Set False to skip (e.g. .X is already lognorm or you want raw).
    normalize_if_raw: bool = True
    outdir: str = "./pdrobust-out"

    # Stress-test grid (tests 9, 10).
    downsample_grid: list[int] = field(default_factory=lambda: [25, 50, 100, 200, 500, 1000])
    # Control downsampling (test 10) typically needs a finer/larger grid than perturbed
    # downsampling — small, batch-fragmented control pseudobulks destabilize LFCs. If None,
    # falls back to downsample_grid.
    control_downsample_grid: list[int] | None = field(
        default_factory=lambda: [500, 1000, 2000, 5000]
    )
    n_repeats: int = 3
    # LFC correlation-to-full at/above which a downsample level is deemed "stable".
    stable_lfc_corr: float = 0.9
    # Warn when a downsample level keeps fewer than this many cells per group/condition.
    min_cells_per_group: int = 10

    # Cap on how many conditions/perts iterated per test (cost control). None => all.
    max_conditions: int | None = None

    # pdex backend options forwarded as de_kwargs["pdex"].
    pdex_kwargs: dict[str, Any] = field(default_factory=dict)

    # Test selection + optional curated resources.
    tests: list[str] = field(default_factory=lambda: ["all"])
    curated_targets_csv: str | None = None  # test 7
    gene_sets_json: str | None = None  # test 8 (gene -> {pathway: [genes]})

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RobustnessConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}. Known: {sorted(known)}")
        return cls(**dict(d))


def load_config(path: str) -> RobustnessConfig:
    import json
    import os

    with open(path) as fh:
        text = fh.read()
    if path.endswith((".yaml", ".yml")):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    cfg = RobustnessConfig.from_dict(data)
    os.makedirs(cfg.outdir, exist_ok=True)
    return cfg


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_adata(cfg: RobustnessConfig) -> ad.AnnData:
    logger.info("Reading AnnData from %s", cfg.adata_path)
    adata = ad.read_h5ad(cfg.adata_path)
    _require_obs(adata, cfg.pert_col)
    _require_obs(adata, cfg.replicate_col)  # optional; only checked if set
    if cfg.control_pert not in set(adata.obs[cfg.pert_col].astype(str)):
        raise ValueError(
            f"control_pert '{cfg.control_pert}' not found in obs['{cfg.pert_col}']"
        )
    # pdex expects log-normalized .X. If .X looks like raw integer counts, normalize
    # + log1p once here so every test below (and the run-pipeline) gets lognorm input.
    if cfg.normalize_if_raw:
        try:
            is_lognorm = guess_is_lognorm(adata, validate=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("guess_is_lognorm failed (%s); falling back to integer check", exc)
            is_lognorm = not _x_is_integer(adata)
        if not is_lognorm:
            logger.info(
                "X looks like RAW integer counts -> applying normalize_total(1e4)+log1p "
                "so pdex receives log-normalized input."
            )
            _normalize_lognorm(adata)
        else:
            logger.info("X looks log-normalized already; leaving it unchanged.")
    return adata


def _x_is_integer(adata: ad.AnnData) -> bool:
    X = adata.X
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        return False
    return bool(np.allclose(data[:200000], np.rint(data[:200000]), atol=1e-6))


def _normalize_lognorm(adata: ad.AnnData, target_sum: float = 1e4) -> None:
    """In-place library-size normalize to ``target_sum`` then log1p (scanpy-style)."""
    import scanpy as sc

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)


def _require_obs(adata: ad.AnnData, col: str | None) -> None:
    if col is not None and col not in adata.obs.columns:
        raise ValueError(f"Column '{col}' not in adata.obs (have: {list(adata.obs.columns)})")


# --------------------------------------------------------------------------- #
# DE backend wrappers
# --------------------------------------------------------------------------- #
def run_pdex_de(
    adata: ad.AnnData,
    cfg: RobustnessConfig,
    *,
    groupby: str | None = None,
    reference: str | None = None,
) -> pl.DataFrame:
    """Run cell-eval's pdex backend on ``adata``; return the canonical DE frame.

    Cell-level Wilcoxon contrasts of every non-reference level of ``groupby`` against
    ``reference``. pdex expects log-normalized ``.X`` (``is_log1p = not allow_discrete``).
    No replicate / pseudobulk grouping is used.
    """
    groupby = groupby or cfg.pert_col
    reference = reference or cfg.control_pert
    return compute_pdex_de(
        adata=adata,
        reference=reference,
        groupby=groupby,
        threads=cfg.num_threads,
        allow_discrete=cfg.allow_discrete,
        pdex_kwargs=cfg.pdex_kwargs or None,
    )


def run_pipeline(
    adata_real: ad.AnnData,
    adata_pred: ad.AnnData,
    cfg: RobustnessConfig,
    *,
    prefix: str,
    profile: str = "de",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run the full cell-eval run pipeline (pdex backend) on a real/pred pair.

    This is the exact code path behind ``cell-eval run --de-methods pdex``.
    Returns ``(perturbation_results, agg_results)``.
    """
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=cfg.control_pert,
        pert_col=cfg.pert_col,
        num_threads=cfg.num_threads,
        outdir=cfg.outdir,
        prefix=prefix,
        allow_discrete=cfg.allow_discrete,
        de_methods=["pdex"],
        de_kwargs={"pdex": cfg.pdex_kwargs} if cfg.pdex_kwargs else None,
    )
    return evaluator.compute(profile=profile, write_csv=True, basename="results.csv")


# --------------------------------------------------------------------------- #
# DE frame summaries / comparisons
# --------------------------------------------------------------------------- #
def de_summary(de: pl.DataFrame, fdr_threshold: float) -> pl.DataFrame:
    """Per-target null/effect summary of a DE frame.

    Columns: target, n_genes, n_sig, frac_sig, mean_lfc, median_lfc,
    mean_abs_lfc, frac_pos_lfc, ks_p_uniform (KS test of p-values vs Uniform[0,1];
    small => departure from null).
    """
    rows: list[dict[str, Any]] = []
    for target, sub in de.group_by("target"):
        tname = target[0] if isinstance(target, tuple) else target
        pvals = sub["p_value"].to_numpy().astype(float)
        lfc = sub["log2_fold_change"].to_numpy().astype(float)
        fdr = sub["fdr"].to_numpy().astype(float)
        finite = np.isfinite(lfc)
        n_genes = int(sub.height)
        n_sig = int(np.nansum(fdr < fdr_threshold))
        pv = pvals[np.isfinite(pvals)]
        ks_p = float(stats.kstest(pv, "uniform").pvalue) if pv.size >= 8 else float("nan")
        rows.append(
            {
                "target": str(tname),
                "n_genes": n_genes,
                "n_sig": n_sig,
                "frac_sig": n_sig / n_genes if n_genes else float("nan"),
                "mean_lfc": float(np.nanmean(lfc[finite])) if finite.any() else float("nan"),
                "median_lfc": float(np.nanmedian(lfc[finite])) if finite.any() else float("nan"),
                "mean_abs_lfc": float(np.nanmean(np.abs(lfc[finite]))) if finite.any() else float("nan"),
                "frac_pos_lfc": float(np.mean(lfc[finite] > 0)) if finite.any() else float("nan"),
                "ks_p_uniform": ks_p,
            }
        )
    return pl.DataFrame(rows).sort("target")


def compare_signatures(
    de_a: pl.DataFrame,
    de_b: pl.DataFrame,
    *,
    fdr_threshold: float,
    top_ks: Sequence[int] = (50, 100, 200),
) -> pl.DataFrame:
    """Reproducibility comparison of two DE frames, per shared target.

    For each target present in both frames returns: n_common_genes, n_sig_a,
    n_sig_b, sig_jaccard, lfc_pearson, lfc_spearman, direction_match (over the
    union of significant genes), and top_K_overlap (top-K by |LFC| among A's
    significant genes recovered in B's).
    """
    targets = sorted(
        set(de_a["target"].cast(str).unique().to_list())
        & set(de_b["target"].cast(str).unique().to_list())
    )
    rows: list[dict[str, Any]] = []
    for t in targets:
        a = de_a.filter(pl.col("target").cast(str) == t)
        b = de_b.filter(pl.col("target").cast(str) == t)
        joined = a.join(b, on="feature", how="inner", suffix="_b")
        if joined.height == 0:
            continue
        lfc_a = joined["log2_fold_change"].to_numpy().astype(float)
        lfc_b = joined["log2_fold_change_b"].to_numpy().astype(float)
        fdr_a = joined["fdr"].to_numpy().astype(float)
        fdr_b = joined["fdr_b"].to_numpy().astype(float)
        feat = joined["feature"].cast(str).to_numpy()

        ok = np.isfinite(lfc_a) & np.isfinite(lfc_b)
        sig_a = ok & (fdr_a < fdr_threshold)
        sig_b = ok & (fdr_b < fdr_threshold)
        union = sig_a | sig_b
        n_union = int(union.sum())

        pearson = _safe_corr(lfc_a[ok], lfc_b[ok], "pearson")
        spearman = _safe_corr(lfc_a[ok], lfc_b[ok], "spearman")
        dir_match = (
            float(np.mean(np.sign(lfc_a[union]) == np.sign(lfc_b[union]))) if n_union else float("nan")
        )
        jac = (
            float((sig_a & sig_b).sum() / n_union) if n_union else float("nan")
        )

        row: dict[str, Any] = {
            "target": t,
            "n_common_genes": int(ok.sum()),
            "n_sig_a": int(sig_a.sum()),
            "n_sig_b": int(sig_b.sum()),
            "sig_jaccard": jac,
            "lfc_pearson": pearson,
            "lfc_spearman": spearman,
            "direction_match": dir_match,
        }
        # top-K overlap by |LFC| among A's significant genes.
        a_sig_feat = feat[sig_a]
        a_sig_abs = np.abs(lfc_a[sig_a])
        order = np.argsort(-a_sig_abs)
        ranked = a_sig_feat[order]
        b_sig_set = set(feat[sig_b].tolist())
        for k in top_ks:
            keff = min(k, ranked.size)
            row[f"top_{k}_overlap"] = (
                float(np.mean([g in b_sig_set for g in ranked[:keff]])) if keff else float("nan")
            )
        rows.append(row)
    return pl.DataFrame(rows).sort("target") if rows else pl.DataFrame()


def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    if kind == "pearson":
        return float(stats.pearsonr(x, y).statistic)
    return float(stats.spearmanr(x, y).statistic)


# --------------------------------------------------------------------------- #
# obs surgery for null / reproducibility designs
# --------------------------------------------------------------------------- #
def _rng(cfg: RobustnessConfig, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng(cfg.seed + salt)


def stratified_split(
    adata: ad.AnnData,
    *,
    group_col: str,
    block_cols: Sequence[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Assign each cell to half "A" or "B", balanced within (group x blocks).

    Returns a string array aligned to ``adata.obs`` with values in {"A", "B"}.
    """
    obs = adata.obs
    keys = [group_col, *[c for c in block_cols if c in obs.columns]]
    out = np.empty(adata.n_obs, dtype=object)
    grouped = obs.groupby(keys, observed=True, sort=False) if keys else [(None, obs)]
    for _, idx in (grouped.indices.items() if hasattr(grouped, "indices") else []):
        pos = np.asarray(idx, dtype=np.int64)
        rng.shuffle(pos)
        half = pos.size // 2
        out[pos[:half]] = "A"
        out[pos[half:]] = "B"
    return out.astype(str)


def permute_labels_within_blocks(
    adata: ad.AnnData,
    *,
    pert_col: str,
    block_cols: Sequence[str],
    control_pert: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle perturbation labels among cells within each block.

    Control cells keep their label (they define the reference); only non-control
    labels are permuted, preserving label composition within each block.
    """
    obs = adata.obs.reset_index(drop=True)
    labels = obs[pert_col].astype(str).to_numpy().copy()
    is_ctrl = labels == control_pert
    blocks = [c for c in block_cols if c in obs.columns]
    if blocks:
        keyed = obs.groupby(blocks, observed=True, sort=False).indices.values()
        groups = [np.asarray(v, dtype=np.int64) for v in keyed]
    else:
        groups = [np.arange(len(labels), dtype=np.int64)]
    for pos in groups:
        pos = pos[~is_ctrl[pos]]
        if pos.size > 1:
            perm = pos.copy()
            rng.shuffle(perm)
            labels[pos] = labels[perm]
    return labels


def with_obs_column(adata: ad.AnnData, col: str, values: np.ndarray) -> ad.AnnData:
    out = adata.copy()
    out.obs[col] = values
    return out
