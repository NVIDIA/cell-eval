"""The ten robustness tests + curated-resource tests, driving the PyDESeq2 backend.

Each ``test_*`` function takes ``(adata, cfg)`` and returns a :class:`TestResult`.
Tests are grouped as in the analysis plan:

  Pure nulls            : test_1_within_split_null, test_2_control_control_null,
                          test_3_label_permutation_null
  Reproducibility ceil. : test_4_same_pert_split, test_5_same_gene_guides
  Biological positives  : test_6_target_knockdown, test_7_curated_targets,
                          test_8_pathway_recovery
  Stress tests          : test_9_cell_downsampling, test_10_control_downsampling

Verdict heuristics (thresholds at the top of this module) follow the "Expected
Behavior" sections of the plan; they are deliberately conservative and documented
in reference/TEST_PLAN.md. A verdict of WARN/FAIL is a prompt to investigate, not
proof of a bug.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from .harness import (
    RobustnessConfig,
    compare_signatures,
    de_summary,
    permute_labels_within_blocks,
    run_pipeline,
    run_pydeseq2_de,
    stratified_split,
    with_obs_column,
)

logger = logging.getLogger("pdrobust")

# --- verdict thresholds (see reference/TEST_PLAN.md) ----------------------- #
NULL_FRAC_SIG_WARN = 0.02  # >2% of genes significant under a null => suspicious
NULL_FRAC_SIG_FAIL = 0.05
NULL_ABS_MEAN_LFC_WARN = 0.10
REPRO_CORR_GOOD = 0.50  # median LFC correlation expected for a reproducibility ceiling
REPRO_OVERLAP_GOOD = 0.30
PERM_SEPARATION_MIN = 0.20  # identity-minus-permuted gap on discriminative metrics
KNOCKDOWN_DETECT_GOOD = 0.50  # fraction of targets detected as DE
KNOCKDOWN_DIR_GOOD = 0.60  # fraction in expected (negative) direction

_BACKEND_LABEL = "PyDESeq2"  # shown in QQ-plot subtitles


def _qq_plot(cfg, pvals, fname: str, *, title: str, subtitle: str) -> dict:
    """Write a uniform-null QQ-plot PNG to <outdir>/plots; return its stats + path.

    Returns {} on any failure (plotting must never fail a test).
    """
    try:
        from .plots import qq_uniform_plot

        path = os.path.join(cfg.outdir, "plots", f"{fname}.png")
        res = qq_uniform_plot(pvals, path, title=title, subtitle=subtitle)
        res["path"] = path
        return res
    except Exception as exc:
        logger.warning("QQ plot %s failed (%s)", fname, exc)
        return {}


@dataclass
class TestResult:
    name: str
    title: str
    verdict: str  # PASS | WARN | FAIL | INFO | SKIP
    headline: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    tables: dict[str, pl.DataFrame] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _non_control_perts(adata: ad.AnnData, cfg: RobustnessConfig) -> list[str]:
    vals = adata.obs[cfg.pert_col].astype(str)
    perts = sorted(set(vals) - {cfg.control_pert})
    if cfg.max_conditions is not None:
        perts = perts[: cfg.max_conditions]
    return perts


def _pert_to_gene(adata: ad.AnnData, cfg: RobustnessConfig, key_col: str) -> dict[str, str]:
    """Map each value of ``key_col`` to its modal ``target_gene_col`` value.

    Handles the common case where ``key_col`` *is* the target-gene column
    (then the map is the identity) and avoids duplicate-column selection.
    """
    assert cfg.target_gene_col is not None
    keys = adata.obs[key_col].astype(str)
    if key_col == cfg.target_gene_col:
        return {v: v for v in keys.unique()}
    df = pd.DataFrame({"_k": keys.to_numpy(), "_g": adata.obs[cfg.target_gene_col].astype(str).to_numpy()})
    out: dict[str, str] = {}
    for k, sub in df.groupby("_k", observed=True):
        out[str(k)] = sub["_g"].mode().iloc[0]
    return out


def _verdict_from_frac_sig(frac: float, abs_mean_lfc: float) -> tuple[str, str]:
    if frac >= NULL_FRAC_SIG_FAIL:
        return "FAIL", f"frac_sig={frac:.3f} exceeds null FAIL threshold {NULL_FRAC_SIG_FAIL}"
    if frac >= NULL_FRAC_SIG_WARN or abs(abs_mean_lfc) >= NULL_ABS_MEAN_LFC_WARN:
        return "WARN", f"frac_sig={frac:.3f}, |mean_lfc|={abs(abs_mean_lfc):.3f}"
    return "PASS", f"frac_sig={frac:.3f}, |mean_lfc|={abs(abs_mean_lfc):.3f} — null-like"


def _compare_two_targets(
    de: pl.DataFrame, t1: str, t2: str, *, fdr_threshold: float
) -> dict[str, float]:
    a = de.filter(pl.col("target").cast(str) == t1).rename({"target": "_a"}).with_columns(
        pl.lit("X").alias("target")
    )
    b = de.filter(pl.col("target").cast(str) == t2).rename({"target": "_b"}).with_columns(
        pl.lit("X").alias("target")
    )
    cmp = compare_signatures(
        a.select(["target", "feature", "log2_fold_change", "p_value", "fdr"]),
        b.select(["target", "feature", "log2_fold_change", "p_value", "fdr"]),
        fdr_threshold=fdr_threshold,
        top_ks=(50,),
    )
    if cmp.height == 0:
        return {}
    r = cmp.row(0, named=True)
    return {
        "lfc_pearson": r["lfc_pearson"],
        "lfc_spearman": r["lfc_spearman"],
        "direction_match": r["direction_match"],
        "sig_jaccard": r["sig_jaccard"],
    }


def _agg_metric_means(results: pl.DataFrame) -> dict[str, float]:
    """Mean of each numeric metric column in a perturbation-level results frame."""
    if results.is_empty():
        return {}
    out: dict[str, float] = {}
    for col in results.columns:
        if col in ("perturbation", "de_method"):
            continue
        try:
            out[col] = float(results[col].cast(pl.Float64).mean())  # type: ignore[arg-type]
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------- #
# Test 1 — Within-Condition Direct Split Null
# --------------------------------------------------------------------------- #
def test_1_within_split_null(adata: ad.AnnData, cfg: RobustnessConfig) -> TestResult:
    rng = np.random.default_rng(cfg.seed + 1)
    perts = _non_control_perts(adata, cfg)
    summaries: list[pl.DataFrame] = []
    pval_rows: list[pl.DataFrame] = []
    for pert in perts:
        sub = adata[adata.obs[cfg.pert_col].astype(str) == pert]
        if sub.n_obs < 8:
            continue
        ab = stratified_split(
            sub, group_col=cfg.pert_col, block_cols=[cfg.replicate_col, *cfg.block_cols], rng=rng
        )
        sub2 = with_obs_column(sub, "_ab", ab)
        if len(set(ab)) < 2:
            continue
        try:
            de = run_pydeseq2_de(sub2, cfg, groupby="_ab", reference="B")
        except Exception as exc:  # design too thin for this condition
            logger.warning("Test1: skipping %s (%s)", pert, exc)
            continue
        s = de_summary(de, cfg.fdr_threshold).with_columns(pl.lit(pert).alias("condition"))
        summaries.append(s)
        pval_rows.append(de.select(pl.col("p_value")).with_columns(pl.lit(pert).alias("condition")))
    if not summaries:
        return TestResult("test_1", "Within-Condition Direct Split Null", "SKIP",
                          notes="No condition had enough cells/replicates for an A/B split.")
    table = pl.concat(summaries)
    frac = float(table["frac_sig"].mean())
    abs_mean = float(table["mean_lfc"].abs().mean())
    verdict, why = _verdict_from_frac_sig(frac, abs_mean)

    # QQ-plot of the pooled within-condition null p-values vs Uniform[0,1].
    pvalues = pl.concat(pval_rows)
    qq = _qq_plot(cfg, pvalues["p_value"].to_numpy().astype(float), "test_1_qq",
                  title="Test 1 — Within-condition null p-value QQ",
                  subtitle=f"A/B splits of {len(summaries)} conditions, {_BACKEND_LABEL}")
    if qq:
        why += f" | QQ plot -> {qq['path']}; lambda_GC={qq['lambda_gc']:.3f}"
    return TestResult(
        "test_1", "Within-Condition Direct Split Null", verdict,
        headline={"n_conditions": len(summaries), "mean_frac_sig": frac, "mean_abs_lfc": abs_mean,
                  "median_frac_sig": float(table["frac_sig"].median()),
                  "lambda_gc": qq.get("lambda_gc", float("nan")) if qq else float("nan")},
        notes=f"A-vs-B within each condition should be null; pooled null p-values should track the "
              f"QQ diagonal (lambda_GC ~ 1). {why}",
        tables={"per_condition": table, "null_pvalues": pvalues},
    )


# --------------------------------------------------------------------------- #
# Test 2 — Control-Control Split Null
# --------------------------------------------------------------------------- #
def test_2_control_control_null(adata: ad.AnnData, cfg: RobustnessConfig) -> TestResult:
    ctrl = adata[adata.obs[cfg.pert_col].astype(str) == cfg.control_pert]
    if ctrl.n_obs < 16:
        return TestResult("test_2", "Control-Control Split Null", "SKIP",
                          notes=f"Only {ctrl.n_obs} control cells; need >=16.")
    summaries: list[pl.DataFrame] = []
    pval_rows: list[pl.DataFrame] = []
    for r in range(cfg.n_repeats):
        rng = np.random.default_rng(cfg.seed + 100 + r)
        ab = stratified_split(
            ctrl, group_col=cfg.pert_col, block_cols=[cfg.replicate_col, *cfg.block_cols], rng=rng
        )
        labels = np.where(ab == "A", "ctrl_pert", "ctrl_ref")
        sub = with_obs_column(ctrl, "_pseudo", labels.astype(str))
        if len(set(labels)) < 2:
            continue
        try:
            de = run_pydeseq2_de(sub, cfg, groupby="_pseudo", reference="ctrl_ref")
        except Exception as exc:
            logger.warning("Test2: repeat %d skipped (%s)", r, exc)
            continue
        summaries.append(de_summary(de, cfg.fdr_threshold).with_columns(pl.lit(r).alias("repeat")))
        pval_rows.append(de.select(pl.col("p_value")).with_columns(pl.lit(r).alias("repeat")))
    if not summaries:
        return TestResult("test_2", "Control-Control Split Null", "SKIP",
                          notes="Control split could not form a valid PyDESeq2 design.")
    table = pl.concat(summaries)
    frac = float(table["frac_sig"].mean())
    abs_mean = float(table["mean_lfc"].abs().mean())
    verdict, why = _verdict_from_frac_sig(frac, abs_mean)

    # QQ-plot of the pooled null p-values vs Uniform[0,1] (+ genomic inflation lambda).
    pvalues = pl.concat(pval_rows)
    qq = _qq_plot(cfg, pvalues["p_value"].to_numpy().astype(float), "test_2_qq",
                  title="Test 2 — Control-Control null p-value QQ",
                  subtitle=f"{cfg.control_pert} pseudo-split, {_BACKEND_LABEL}, {len(summaries)} repeat(s)")
    if qq:
        why += f" | QQ plot -> {qq['path']}; lambda_GC={qq['lambda_gc']:.3f}"
    return TestResult(
        "test_2", "Control-Control Split Null", verdict,
        headline={"n_repeats": len(summaries), "mean_frac_sig": frac, "mean_n_sig": float(table["n_sig"].mean()),
                  "mean_abs_lfc": abs_mean, "lambda_gc": qq.get("lambda_gc", float("nan"))},
        notes=f"Pseudo-perturbations from controls should be null; null p-values should track the "
              f"QQ diagonal (lambda_GC ~ 1). {why}",
        tables={"per_repeat": table, "null_pvalues": pvalues},
    )


# --------------------------------------------------------------------------- #
# Test 3 — Label Permutation Null (run pipeline)
# --------------------------------------------------------------------------- #
def test_3_label_permutation_null(adata: ad.AnnData, cfg: RobustnessConfig) -> TestResult:
    perts = _non_control_perts(adata, cfg)
    keep = adata[adata.obs[cfg.pert_col].astype(str).isin([*perts, cfg.control_pert])].copy()

    # Positive control: pred == real (metrics should be near-perfect).
    ident_res, _ = run_pipeline(keep, keep.copy(), cfg, prefix="test3_identity", profile="de")

    # Null: pred has perturbation labels permuted within blocks.
    rng = np.random.default_rng(cfg.seed + 3)
    perm_labels = permute_labels_within_blocks(
        keep, pert_col=cfg.pert_col, block_cols=cfg.block_cols, control_pert=cfg.control_pert, rng=rng
    )
    pred_perm = with_obs_column(keep, cfg.pert_col, perm_labels)
    perm_res, _ = run_pipeline(keep, pred_perm, cfg, prefix="test3_permuted", profile="de")

    ident = _agg_metric_means(ident_res)
    perm = _agg_metric_means(perm_res)
    # Discriminative metrics where higher == more recovered signal.
    discr = ["overlap_at_N", "de_spearman_lfc_sig", "de_spearman_sig", "de_direction_match",
             "roc_auc", "pr_auc", "de_sig_genes_recall", "precision_at_N"]
    rows = []
    seps = []
    for m in sorted(set(ident) | set(perm)):
        gap = ident.get(m, float("nan")) - perm.get(m, float("nan"))
        rows.append({"metric": m, "identity": ident.get(m), "permuted": perm.get(m), "gap": gap})
        if m in discr and np.isfinite(gap):
            seps.append(gap)
    table = pl.DataFrame(rows).sort("metric")
    mean_sep = float(np.mean(seps)) if seps else float("nan")
    if not np.isfinite(mean_sep):
        verdict = "INFO"
    elif mean_sep >= PERM_SEPARATION_MIN:
        verdict = "PASS"
    else:
        verdict = "WARN"
    return TestResult(
        "test_3", "Label Permutation Null", verdict,
        headline={"mean_separation_discriminative": mean_sep,
                  "n_discriminative_metrics": len(seps)},
        notes="Permuting labels should collapse metrics toward null; identity should max them. "
              f"Mean identity-minus-permuted gap on discriminative metrics = {mean_sep:.3f}.",
        tables={"metric_gaps": table},
    )


# --------------------------------------------------------------------------- #
# Test 4 — Same-Perturbation Split Reproducibility (run pipeline)
# --------------------------------------------------------------------------- #
def test_4_same_pert_split(adata: ad.AnnData, cfg: RobustnessConfig) -> TestResult:
    rng = np.random.default_rng(cfg.seed + 4)
    perts = _non_control_perts(adata, cfg)
    is_ctrl = (adata.obs[cfg.pert_col].astype(str) == cfg.control_pert).to_numpy()
    pert_vals = adata.obs[cfg.pert_col].astype(str).to_numpy()

    ab = np.array(["_"] * adata.n_obs, dtype=object)
    keep_perts = []
    for pert in perts:
        mask = pert_vals == pert
        sub = adata[mask]
        split = stratified_split(
            sub, group_col=cfg.pert_col, block_cols=[cfg.replicate_col, *cfg.block_cols], rng=rng
        )
        if (split == "A").sum() >= 2 and (split == "B").sum() >= 2:
            ab[np.where(mask)[0]] = split
            keep_perts.append(pert)
    if not keep_perts:
        return TestResult("test_4", "Same-Perturbation Split Reproducibility", "SKIP",
                          notes="No perturbation had >=2 cells in both halves.")
    ab[is_ctrl] = "both"
    sel = np.isin(pert_vals, [*keep_perts, cfg.control_pert])
    a_mask = sel & ((ab == "A") | (ab == "both"))
    b_mask = sel & ((ab == "B") | (ab == "both"))
    real = adata[a_mask].copy()
    pred = adata[b_mask].copy()
    res, _ = run_pipeline(real, pred, cfg, prefix="test4_split", profile="de")
    means = _agg_metric_means(res)
    corr = means.get("de_spearman_lfc_sig", means.get("de_spearman_sig", float("nan")))
    overlap = means.get("overlap_at_N", float("nan"))
    precision = means.get("precision_at_N", float("nan"))
    # This per-metric table IS the empirical ceiling: report model scores relative to it.
    ceiling = pl.DataFrame(
        [{"metric": k, "ceiling": v} for k, v in sorted(means.items())]
    )
    good = (np.isfinite(corr) and corr >= REPRO_CORR_GOOD) or (
        np.isfinite(overlap) and overlap >= REPRO_OVERLAP_GOOD
    )
    verdict = "PASS" if good else "WARN"
    return TestResult(
        "test_4", "Same-Perturbation Split Reproducibility", verdict,
        headline={"n_perts": len(keep_perts), "lfc_corr_ceiling": corr,
                  "overlap_at_N_ceiling": overlap, "precision_at_N_ceiling": precision},
        notes="A vs B halves of the same perturbation (each vs control) define the EMPIRICAL "
              "CEILING for every run-pipeline metric. Set-overlap/precision ceilings are usually "
              "well below 1.0 (sampling noise in the DE gene set), so a model's overlap_at_N should "
              "be read RELATIVE TO this ceiling (e.g. score / ceiling), not against 1.0. See the "
              "`ceiling` table for the per-metric maxima.",
        tables={"ceiling": ceiling},
    )


# --------------------------------------------------------------------------- #
# Test 5 — Same-Gene Independent sgRNA Reproducibility
# --------------------------------------------------------------------------- #
def test_5_same_gene_guides(adata: ad.AnnData, cfg: RobustnessConfig) -> TestResult:
    if not cfg.sgrna_col or not cfg.target_gene_col:
        return TestResult("test_5", "Same-Gene Independent sgRNA Reproducibility", "SKIP",
                          notes="Requires sgrna_col and target_gene_col in config.")
    guide_to_gene = _pert_to_gene(adata, cfg, cfg.sgrna_col)
    gene_to_guides: dict[str, list[str]] = {}
    for g, gene in guide_to_gene.items():
        if g == cfg.control_pert:
            continue
        gene_to_guides.setdefault(gene, []).append(g)
    multi = {gene: gs for gene, gs in gene_to_guides.items() if len(gs) >= 2}
    if cfg.max_conditions is not None:
        multi = dict(list(multi.items())[: cfg.max_conditions])
    if not multi:
        return TestResult("test_5", "Same-Gene Independent sgRNA Reproducibility", "SKIP",
                          notes="No gene has >=2 sgRNAs.")
    guides = sorted({g for gs in multi.values() for g in gs})
    # Control cells often do not carry control_pert in sgrna_col (they have their own
    # non-targeting guide ids). Build a unified guide column where controls collapse to
    # control_pert so it can serve as the DE reference level.
    is_ctrl = (adata.obs[cfg.pert_col].astype(str) == cfg.control_pert).to_numpy()
    guide_vals = adata.obs[cfg.sgrna_col].astype(str).to_numpy().copy()
    guide_vals[is_ctrl] = cfg.control_pert
    work = with_obs_column(adata, "_guide", guide_vals)
    sub = work[np.isin(guide_vals, [*guides, cfg.control_pert])].copy()
    de = run_pydeseq2_de(sub, cfg, groupby="_guide", reference=cfg.control_pert)

    rng = np.random.default_rng(cfg.seed + 5)
    same_rows, unrel_rows = [], []
    for gene, gs in multi.items():
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                m = _compare_two_targets(de, gs[i], gs[j], fdr_threshold=cfg.fdr_threshold)
                if m:
                    same_rows.append({"gene": gene, "g1": gs[i], "g2": gs[j], **m})
    # matched unrelated pairs: guides from different genes. Keep drawing until we
    # collect as many unrelated pairs as same-gene pairs (capped attempts).
    target_n = max(len(same_rows), 1)
    attempts = 0
    while len(unrel_rows) < target_n and attempts < target_n * 50:
        attempts += 1
        a, b = (str(x) for x in rng.choice(guides, size=2, replace=False))
        if guide_to_gene.get(a) == guide_to_gene.get(b):
            continue
        m = _compare_two_targets(de, a, b, fdr_threshold=cfg.fdr_threshold)
        if m:
            unrel_rows.append({"g1": a, "g2": b, **m})
    same = pl.DataFrame(same_rows) if same_rows else pl.DataFrame()
    unrel = pl.DataFrame(unrel_rows) if unrel_rows else pl.DataFrame()

    def _sep(metric: str) -> tuple[float, float, float]:
        s = float(same[metric].mean()) if same.height else float("nan")
        u = float(unrel[metric].mean()) if unrel.height else float("nan")
        return s, u, s - u

    # Whole-transcriptome LFC-Pearson saturates when perturbations share a global
    # response (e.g. essential-gene knockdowns), so it is a poor target discriminator.
    # Judge on rank- and set-based metrics, which carry the gene-level signal; report
    # Pearson only as a flagged diagnostic.
    p_same, p_unrel, p_sep = _sep("lfc_pearson")
    sp_same, sp_unrel, sp_sep = _sep("lfc_spearman")
    j_same, j_unrel, j_sep = _sep("sig_jaccard")
    seps = [s for s in (sp_sep, j_sep) if np.isfinite(s)]
    discriminating = max(seps) if seps else float("nan")
    verdict = "PASS" if (np.isfinite(discriminating) and discriminating > 0.05) else "WARN"
    pearson_saturated = (
        np.isfinite(p_same) and np.isfinite(p_unrel) and p_same > 0.8 and p_unrel > 0.8
    )
    note = ("Independent sgRNAs for the same gene should agree more than unrelated guides. "
            "Verdict uses rank/set-based separation (Spearman, sig-Jaccard).")
    if pearson_saturated:
        note += (f" NOTE: LFC-Pearson is saturated (same {p_same:.3f} vs unrelated {p_unrel:.3f}) "
                 "— a shared global response inflates it for all pairs; do not use it to "
                 "discriminate target identity here.")
    return TestResult(
        "test_5", "Same-Gene Independent sgRNA Reproducibility", verdict,
        headline={"n_genes": len(multi),
                  "spearman_sep": sp_sep, "sig_jaccard_sep": j_sep,
                  "same_gene_spearman": sp_same, "unrelated_spearman": sp_unrel,
                  "lfc_pearson_sep": p_sep, "pearson_saturated": pearson_saturated},
        notes=note,
        tables={"same_gene_pairs": same, "unrelated_pairs": unrel} if same.height else {},
    )


# --------------------------------------------------------------------------- #
# Test 6 — Target Gene Knockdown Recovery
# --------------------------------------------------------------------------- #
def test_6_target_knockdown(adata: ad.AnnData, cfg: RobustnessConfig) -> TestResult:
    if not cfg.target_gene_col:
        return TestResult("test_6", "Target Gene Knockdown Recovery", "SKIP",
                          notes="Requires target_gene_col in config.")
    pert_to_gene = _pert_to_gene(adata, cfg, cfg.pert_col)
    perts = _non_control_perts(adata, cfg)
    sub = adata[adata.obs[cfg.pert_col].astype(str).isin([*perts, cfg.control_pert])].copy()
    de = run_pydeseq2_de(sub, cfg, groupby=cfg.pert_col, reference=cfg.control_pert)
    genes_present = set(de["feature"].cast(str).unique().to_list())

    rows = []
    for pert in perts:
        target = pert_to_gene.get(pert)
        if target is None or target not in genes_present:
            continue
        d = de.filter(pl.col("target").cast(str) == pert)
        if d.height == 0:
            continue
        d = d.with_columns(
            pl.col("p_value").rank("ordinal").alias("rank_p"),
            pl.col("log2_fold_change").rank("ordinal").alias("rank_signed_lfc"),
            pl.col("log2_fold_change").abs().rank("ordinal", descending=True).alias("rank_abs_lfc"),
        )
        row = d.filter(pl.col("feature").cast(str) == target)
        if row.height == 0:
            continue
        rr = row.row(0, named=True)
        n = d.height
        rows.append({
            "perturbation": pert, "target_gene": target,
            "lfc": float(rr["log2_fold_change"]), "p_value": float(rr["p_value"]), "fdr": float(rr["fdr"]),
            "rank_p_pct": float(rr["rank_p"]) / n, "rank_abs_lfc_pct": float(rr["rank_abs_lfc"]) / n,
            "is_sig": bool(rr["fdr"] < cfg.fdr_threshold), "negative_direction": bool(rr["log2_fold_change"] < 0),
        })
    if not rows:
        return TestResult("test_6", "Target Gene Knockdown Recovery", "SKIP",
                          notes="No target genes found in the DE feature space.")
    table = pl.DataFrame(rows)
    frac_detect = float(table["is_sig"].mean())
    frac_neg = float(table["negative_direction"].mean())
    median_rank_pct = float(table["rank_abs_lfc_pct"].median())
    verdict = "PASS" if (frac_detect >= KNOCKDOWN_DETECT_GOOD and frac_neg >= KNOCKDOWN_DIR_GOOD) else "WARN"
    return TestResult(
        "test_6", "Target Gene Knockdown Recovery", verdict,
        headline={"n_targets": table.height, "frac_detected_sig": frac_detect,
                  "frac_negative_lfc": frac_neg, "median_abs_lfc_rank_pct": median_rank_pct},
        notes="Target gene should be DE (CRISPRi/KD => negative LFC) and rank near the top by |LFC|.",
        tables={"per_target": table},
    )


# --------------------------------------------------------------------------- #
# Test 7 — Curated Gene-to-Target Relationship Recovery
# --------------------------------------------------------------------------- #
def test_7_curated_targets(adata: ad.AnnData, cfg: RobustnessConfig) -> TestResult:
    if not cfg.curated_targets_csv:
        return TestResult("test_7", "Curated Gene-to-Target Recovery", "SKIP",
                          notes="Provide curated_targets_csv (cols: pert_gene,target_gene[,expected_direction,confidence]).")
    curated = pl.read_csv(cfg.curated_targets_csv)
    perts = _non_control_perts(adata, cfg)
    sub = adata[adata.obs[cfg.pert_col].astype(str).isin([*perts, cfg.control_pert])].copy()
    de = run_pydeseq2_de(sub, cfg, groupby=cfg.pert_col, reference=cfg.control_pert)
    genes_present = set(de["feature"].cast(str).unique().to_list())

    rows = []
    for rec in curated.iter_rows(named=True):
        pg, tg = str(rec.get("pert_gene")), str(rec.get("target_gene"))
        if pg not in set(perts) or tg not in genes_present:
            continue
        d = de.filter(pl.col("target").cast(str) == pg)
        if d.height == 0:
            continue
        d = d.with_columns(pl.col("p_value").rank("ordinal").alias("rk"))
        r = d.filter(pl.col("feature").cast(str) == tg)
        if r.height == 0:
            continue
        rr = r.row(0, named=True)
        exp = str(rec.get("expected_direction", "")).lower()
        dir_ok = None
        if exp in ("up", "+", "pos"):
            dir_ok = rr["log2_fold_change"] > 0
        elif exp in ("down", "-", "neg"):
            dir_ok = rr["log2_fold_change"] < 0
        rows.append({
            "pert_gene": pg, "target_gene": tg, "lfc": float(rr["log2_fold_change"]),
            "fdr": float(rr["fdr"]), "rank_p_pct": float(rr["rk"]) / d.height,
            "is_sig": bool(rr["fdr"] < cfg.fdr_threshold), "direction_ok": dir_ok,
            "confidence": rec.get("confidence"),
        })
    if not rows:
        return TestResult("test_7", "Curated Gene-to-Target Recovery", "SKIP",
                          notes="No curated relationships matched perturbations/features present.")
    table = pl.DataFrame(rows)
    frac_sig = float(table["is_sig"].mean())
    dir_vals = [v for v in table["direction_ok"].to_list() if v is not None]
    dir_acc = float(np.mean(dir_vals)) if dir_vals else float("nan")
    median_rank = float(table["rank_p_pct"].median())
    verdict = "PASS" if (frac_sig >= 0.4 and (not dir_vals or dir_acc >= 0.6)) else "WARN"
    return TestResult(
        "test_7", "Curated Gene-to-Target Recovery", verdict,
        headline={"n_relationships": table.height, "frac_sig": frac_sig,
                  "direction_accuracy": dir_acc, "median_rank_pct": median_rank},
        notes="Known downstream targets should be recovered (significant, top-ranked, correct sign).",
        tables={"per_relationship": table},
    )


# --------------------------------------------------------------------------- #
# Test 8 — Pathway / Regulon Recovery
# --------------------------------------------------------------------------- #
def test_8_pathway_recovery(adata: ad.AnnData, cfg: RobustnessConfig) -> TestResult:
    if not cfg.gene_sets_json:
        return TestResult("test_8", "Pathway / Regulon Recovery", "SKIP",
                          notes='Provide gene_sets_json: {"<pert_gene>": ["geneA","geneB", ...]}.')
    with open(cfg.gene_sets_json) as fh:
        gene_sets = json.load(fh)
    perts = _non_control_perts(adata, cfg)
    sub = adata[adata.obs[cfg.pert_col].astype(str).isin([*perts, cfg.control_pert])].copy()
    de = run_pydeseq2_de(sub, cfg, groupby=cfg.pert_col, reference=cfg.control_pert)

    rows = []
    for pert in perts:
        members = gene_sets.get(pert)
        if not members:
            continue
        d = de.filter(pl.col("target").cast(str) == pert)
        if d.height < 10:
            continue
        feat = d["feature"].cast(str).to_numpy()
        stat = d["log2_fold_change"].abs().to_numpy().astype(float)
        in_set = np.isin(feat, np.asarray(list(members), dtype=str))
        if in_set.sum() < 2 or in_set.sum() == in_set.size:
            continue
        # AUROC of |LFC| separating set members from the rest.
        auc = _auroc(stat, in_set)
        rows.append({"perturbation": pert, "set_size": int(in_set.sum()), "auroc_abs_lfc": auc})
    if not rows:
        return TestResult("test_8", "Pathway / Regulon Recovery", "SKIP",
                          notes="No perturbation had a usable gene set overlapping the feature space.")
    table = pl.DataFrame(rows)
    mean_auc = float(table["auroc_abs_lfc"].mean())
    verdict = "PASS" if mean_auc >= 0.6 else "WARN"
    return TestResult(
        "test_8", "Pathway / Regulon Recovery", verdict,
        headline={"n_perts": table.height, "mean_auroc": mean_auc},
        notes="Expected pathway/regulon genes should be enriched among top-|LFC| genes (AUROC>0.5).",
        tables={"per_pert": table},
    )


def _auroc(score: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U relationship; no sklearn dependency."""
    pos = score[labels]
    neg = score[~labels]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    u = stats.mannwhitneyu(pos, neg, alternative="greater").statistic
    return float(u / (pos.size * neg.size))


# --------------------------------------------------------------------------- #
# Tests 9 & 10 — Downsampling stress tests
# --------------------------------------------------------------------------- #
def _subsample_within(
    adata: ad.AnnData, mask: np.ndarray, n: int, by: str, rng: np.random.Generator
) -> np.ndarray:
    """Indices: keep all cells outside ``mask``; subsample ``n`` of the masked
    cells, stratified by ``by`` to preserve replicate representation."""
    keep = list(np.where(~mask)[0])
    idx = np.where(mask)[0]
    if idx.size <= n:
        return np.sort(np.concatenate([keep, idx])).astype(int)
    groups = adata.obs[by].astype(str).to_numpy()
    chosen: list[int] = []
    uniq = np.unique(groups[idx])
    per = max(1, n // len(uniq))
    for g in uniq:
        gidx = idx[groups[idx] == g]
        take = min(per, gidx.size)
        chosen.extend(rng.choice(gidx, size=take, replace=False).tolist())
    if len(chosen) > n:
        chosen = rng.choice(chosen, size=n, replace=False).tolist()
    return np.sort(np.concatenate([keep, chosen])).astype(int)


def _downsampling_test(
    adata: ad.AnnData, cfg: RobustnessConfig, *, target: str, name: str, title: str
) -> TestResult:
    """Shared engine for tests 9 (perturbed) and 10 (control)."""
    perts = _non_control_perts(adata, cfg)
    keep = adata[adata.obs[cfg.pert_col].astype(str).isin([*perts, cfg.control_pert])].copy()
    pert_vals = keep.obs[cfg.pert_col].astype(str).to_numpy()
    is_ctrl = pert_vals == cfg.control_pert

    try:
        de_full = run_pydeseq2_de(keep, cfg, groupby=cfg.pert_col, reference=cfg.control_pert)
    except Exception as exc:
        return TestResult(name, title, "SKIP", notes=f"Full-data DE failed: {exc}")
    full_summ = de_summary(de_full, cfg.fdr_threshold)
    full_nsig = dict(zip(full_summ["target"].to_list(), full_summ["n_sig"].to_list()))

    pool_mask = is_ctrl if target == "control" else ~is_ctrl
    max_pool = int(pool_mask.sum())
    base_grid = (
        cfg.control_downsample_grid
        if (target == "control" and cfg.control_downsample_grid)
        else cfg.downsample_grid
    )
    grid = [g for g in base_grid if g < max_pool]
    # Pseudobulk depth: control samples are split across replicates (one pseudobulk per
    # batch), so the per-sample cell count is roughly level / n_replicates.
    n_rep = int(np.unique(keep.obs[cfg.replicate_col].astype(str).to_numpy()[pool_mask]).size)
    rows = []
    for level in grid:
        for r in range(cfg.n_repeats):
            rng = np.random.default_rng(cfg.seed + 900 + level * 7 + r)
            sel = _subsample_within(keep, pool_mask, level, by=cfg.replicate_col, rng=rng)
            sa = keep[sel].copy()
            try:
                de = run_pydeseq2_de(sa, cfg, groupby=cfg.pert_col, reference=cfg.control_pert)
            except Exception as exc:
                logger.warning("%s level=%d rep=%d skipped (%s)", name, level, r, exc)
                continue
            cmp = compare_signatures(de_full, de, fdr_threshold=cfg.fdr_threshold)
            summ = de_summary(de, cfg.fdr_threshold)
            if cmp.height:
                rank_corr = _rank_corr_to_full(summ, full_nsig)
                rows.append({
                    "level": level, "repeat": r,
                    "mean_lfc_pearson_to_full": float(cmp["lfc_pearson"].mean()),
                    "mean_n_sig": float(summ["n_sig"].mean()),
                    "rank_corr_nsig_to_full": rank_corr,
                })
    if not rows:
        return TestResult(name, title, "SKIP", notes="No downsample level produced a valid design.")
    table = pl.DataFrame(rows)
    by_level = table.group_by("level").agg(
        pl.col("mean_lfc_pearson_to_full").mean().alias("lfc_corr_to_full_mean"),
        pl.col("mean_lfc_pearson_to_full").std().alias("lfc_corr_to_full_std"),
        pl.col("rank_corr_nsig_to_full").mean().alias("rank_corr_mean"),
    ).sort("level")
    by_level = by_level.with_columns(
        (pl.col("level") / max(n_rep, 1)).round(1).alias("cells_per_pseudobulk")
    )

    levels_sorted = by_level["level"].to_list()
    corr_by_level = by_level["lfc_corr_to_full_mean"].to_list()
    depth_by_level = by_level["cells_per_pseudobulk"].to_list()
    # Minimum cells for stable LFC estimates: first level whose corr-to-full >= threshold.
    min_cells_stable = next(
        (lv for lv, c in zip(levels_sorted, corr_by_level)
         if c is not None and c >= cfg.stable_lfc_corr),
        None,
    )
    max_corr = max((c for c in corr_by_level if c is not None), default=float("nan"))
    anti_correlated = any(c is not None and c < 0 for c in corr_by_level)
    shallow = any(d is not None and d < cfg.min_pseudobulk_cells for d in depth_by_level)

    if min_cells_stable is not None:
        verdict = "PASS"
    elif anti_correlated:
        verdict = "WARN"
    else:
        verdict = "WARN"

    note = (f"Metrics should stabilize (LFC-corr-to-full -> 1, rank-corr -> 1) as {target}-cell "
            f"count grows. Stable LFC threshold = {cfg.stable_lfc_corr}.")
    if min_cells_stable is not None:
        note += f" Minimum {target}-cell count for stable LFCs ≈ {min_cells_stable}."
    else:
        note += (f" LFC estimates never reach the stable threshold on this grid "
                 f"(max corr-to-full = {max_corr:.3f}); increase the grid upper bound.")
    if anti_correlated:
        note += (" ANOMALY: LFC-corr-to-full is NEGATIVE at some levels — tiny pseudobulks give "
                 "LFCs that anti-track the full-data estimates; do not trust LFC magnitudes there.")
    if shallow:
        note += (f" WARNING: some pseudobulk samples are built from < {cfg.min_pseudobulk_cells} "
                 "cells (see cells_per_pseudobulk) — increase the grid or coarsen replicate grouping.")
    return TestResult(
        name, title, verdict,
        headline={"levels": grid, "min_cells_for_stable_lfc": min_cells_stable,
                  "max_lfc_corr_to_full": max_corr,
                  "max_level_rank_corr": float(by_level["rank_corr_mean"][-1]),
                  "anti_correlated_lfc": anti_correlated},
        notes=note,
        tables={"by_level": by_level, "raw": table},
    )


def _rank_corr_to_full(summ: pl.DataFrame, full_nsig: dict[str, int]) -> float:
    common = [t for t in summ["target"].to_list() if t in full_nsig]
    if len(common) < 3:
        return float("nan")
    a = summ.filter(pl.col("target").is_in(common)).sort("target")["n_sig"].to_numpy().astype(float)
    b = np.array([full_nsig[t] for t in sorted(common)], dtype=float)
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def test_9_cell_downsampling(adata: ad.AnnData, cfg: RobustnessConfig) -> TestResult:
    return _downsampling_test(adata, cfg, target="perturbed", name="test_9",
                              title="Cell-Count Downsampling")


def test_10_control_downsampling(adata: ad.AnnData, cfg: RobustnessConfig) -> TestResult:
    return _downsampling_test(adata, cfg, target="control", name="test_10",
                              title="Control-Count Downsampling")


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
ALL_TESTS = {
    "test_1": test_1_within_split_null,
    "test_2": test_2_control_control_null,
    "test_3": test_3_label_permutation_null,
    "test_4": test_4_same_pert_split,
    "test_5": test_5_same_gene_guides,
    "test_6": test_6_target_knockdown,
    "test_7": test_7_curated_targets,
    "test_8": test_8_pathway_recovery,
    "test_9": test_9_cell_downsampling,
    "test_10": test_10_control_downsampling,
}


def select_tests(names: Sequence[str]) -> list[str]:
    if not names or "all" in names:
        return list(ALL_TESTS)
    bad = [n for n in names if n not in ALL_TESTS]
    if bad:
        raise ValueError(f"Unknown test(s): {bad}. Known: {list(ALL_TESTS)}")
    return list(names)
