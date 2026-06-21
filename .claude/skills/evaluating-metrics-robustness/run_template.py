#!/usr/bin/env python
"""Standalone, reproducible robustness experiment for cell-eval metrics (pdex backend).

Reads ``config.yaml`` + the dataset and runs the tests from ``TEST_PLAN.md`` using the *real*
cell-eval DE backend (``cell_eval._de_backends.build_de_frame``). Writes per-test tables, QQ /
calibration plots, a JSON summary, a context-first markdown report (dataset & experimental context,
how DE was run + the pseudoreplication caveat, plain-language per-test explanations, a glossary, all
verification p-values, power/sample-size analysis, every threshold, and the embedded test plan), an
optional MultiQC custom-content export, and renders ``report.pdf``.

DE unit of analysis (CRITICAL — see TEST_PLAN.md "Unit of analysis"):
  - ``pdex``     -> cell-level Wilcoxon rank-sum. Each cell is treated as an independent observation,
                   so the null-split tests are subject to PSEUDOREPLICATION (Squair et al. 2021,
                   Nat Commun 12:5692). The report foregrounds this and reads the nulls accordingly.
  - ``pydeseq2`` -> pseudobulk DESeq2 Wald (aggregates to replicate-level samples first).

pdex expects log-normalized input. ``cell-eval run`` normalizes inside ``MetricsEvaluator`` before
pdex; this runner drives ``build_de_frame`` directly, so it applies the same ``normalize_total`` +
``log1p`` once up front when ``normalize_if_raw`` is set and ``.X`` is raw integer counts
(per-cell ops, so normalizing once then subsetting == normalizing each subset). ``allow_discrete:
false`` then makes pdex treat the matrix as log1p data (``is_log1p=True``).

Reproducible: a single ``seed`` drives every split/permutation/injection. Re-running with the same
config and seed reproduces the report.

    python run.py --config config.yaml                 # full compute + report
    python run.py --config config.yaml --report-only   # re-render report from cached summary (cheap)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field

import anndata as ad
import numpy as np
import polars as pl
import scanpy as sc
import scipy.sparse as sp
import yaml
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cell_eval._de_backends import build_de_frame  # noqa: E402

log = logging.getLogger("robustness")
CHI2_MEDIAN_1DF = 0.4549  # chi2.ppf(0.5, df=1)
SQUAIR = "Squair et al. 2021, Nat Commun 12:5692 (https://www.nature.com/articles/s41467-021-25960-2)"


# Curated canonical (non-ribosomal) housekeeping genes — stably expressed; used to break down the
# Test-0 false-positive rate by gene class. Includes the Eisenberg–Levanon "stable" reference set.
HOUSEKEEPING = {
    "ACTB", "GAPDH", "B2M", "TBP", "HPRT1", "PGK1", "PPIA", "GUSB", "TFRC", "YWHAZ", "SDHA", "UBC",
    "HMBS", "EEF1A1", "EEF2", "PABPC1", "VCP", "PSMB2", "PSMB4", "TUBB", "TUBA1B", "TUBB4B", "CLTC",
    "HSP90AB1", "HSPA8", "CANX", "CALM1", "CALM2", "RAB7A", "CHMP2A", "EMC7", "REEP5", "SNRPD3",
    "VPS29", "C1orf43", "GPI", "LDHA", "ENO1", "PKM", "NACA", "CFL1", "PFN1",
}


def seed_everything(seed: int) -> None:
    """Seed every RNG source so the whole run is bit-for-bit reproducible (REQUIRED).

    All splits/permutations/injection sampling already use np.random.default_rng(seed + offset)
    (deterministic), but this also pins the global RNGs and PYTHONHASHSEED so any library that reads
    a global RNG, and any future stochastic step, stays deterministic. The DE backends (pdex
    Wilcoxon, pydeseq2 DESeq2) are themselves deterministic given fixed input."""
    import os
    import random as _random

    os.environ["PYTHONHASHSEED"] = str(seed)
    _random.seed(seed)
    np.random.seed(seed)
    try:
        sc.settings.seed = seed
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_cfg(path: str) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("num_threads", 8)
    cfg.setdefault("lfc_threshold", 0.1)
    cfg.setdefault("block_cols", [])
    cfg.setdefault("lambda_gc_warn", 1.05)
    cfg.setdefault("lambda_gc_fail", 1.10)
    cfg.setdefault("normalize_if_raw", False)
    cfg.setdefault("allow_discrete", False)
    cfg.setdefault("covariate_correction", "none")        # what (if anything) was regressed out of .X
    cfg.setdefault("celltype_cols", [])                   # obs cols to use for composition control
    cfg.setdefault("injection_deltas", [0.0, 0.25, 0.5, 1.0, 2.0])
    cfg.setdefault("injection_n_genes", 12)          # realistic: a perturbation moves ~a dozen genes
    cfg.setdefault("injection_n_repeats", 10)        # pool TPR/FPR over this many random gene-draws
    cfg.setdefault("injection_frac_genes", None)     # legacy large-fraction stress (used only if n_genes is null)
    cfg.setdefault("injection_max_cells_per_arm", 3000)
    cfg.setdefault("injection_n_hvg", 2000)          # # highly variable genes flagged for stratification
    cfg.setdefault("emit_multiqc", True)
    # `n_resamples` = # independent stratified-split repeats (Tests 0/1/2/4) or label permutations
    # (Test 3). Backward-compatible with the old name `n_permutations`.
    if "n_resamples" not in cfg and "n_permutations" in cfg:
        cfg["n_resamples"] = cfg["n_permutations"]
    cfg.setdefault("n_resamples", 10)
    if "test1_n_resamples" not in cfg and "test1_n_permutations" in cfg:
        cfg["test1_n_resamples"] = cfg["test1_n_permutations"]
    cfg.setdefault("test1_wellpowered_min_cells", 200)   # ≥200 total ≈ ≥100 cells per split-half
    # Test 3 (Label Permutation Null) p-value diagnostics:
    cfg.setdefault("test3_n_cellcount_strata", 3)        # # of cell-count strata (tertiles) for facets
    cfg.setdefault("test3_pooled_pval_subsample", 40000) # deterministic cap on pooled p-values for ECDF/QQ
    return cfg


# --------------------------------------------------------------------------- #
# rounding / presentation helpers
# --------------------------------------------------------------------------- #
def fmt(v, nd: int = 4) -> str:
    """Compact numeric formatting: <=nd decimals, scientific only for very small/large magnitudes."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if not np.isfinite(v):
            return str(v)
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
            return f"{v:.3g}"
        return f"{round(v, nd):g}"
    return str(v)


def kv(d: dict, nd: int = 4) -> str:
    return ", ".join(f"{k}={fmt(v, nd)}" for k, v in d.items())


# --------------------------------------------------------------------------- #
# dataset fingerprint / experimental context
# --------------------------------------------------------------------------- #
COVARIATE_HINTS = ("cell_type", "celltype", "cell.type", "cluster", "leiden", "louvain",
                   "phase", "cell_cycle", "cellcycle", "donor", "patient", "subject",
                   "sample", "tissue", "sex", "gender", "condition", "timepoint", "time",
                   "lane", "well", "plate", "replicate")


def fingerprint(adata: ad.AnnData, cfg: dict) -> dict:
    obs = adata.obs
    pc, ctrl, sg = cfg["pert_col"], cfg["control_pert"], cfg.get("sgrna_col")
    X = adata.X
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    data = data[np.isfinite(data)]
    samp = data if data.size <= 5_000_000 else data[np.random.default_rng(0).integers(0, data.size, 5_000_000)]
    raw_int = bool(np.allclose(samp, np.rint(samp))) if samp.size else False
    cols = {}
    for c in obs.columns:
        try:
            cols[c] = int(obs[c].nunique())
        except TypeError:
            cols[c] = None
    # perturbation / control accounting
    vc = obs[pc].astype(str).value_counts()
    n_ctrl = int(vc.get(ctrl, 0))
    perts = [g for g in vc.index if g != ctrl]
    per_pert = vc.drop(labels=[ctrl], errors="ignore")
    # guides per gene
    guides_per_gene = None
    if sg and sg in obs.columns:
        gpg = obs.groupby(pc, observed=True)[sg].nunique()
        guides_per_gene = gpg.drop(labels=[ctrl], errors="ignore")
    # which obs columns look like correctable / structural covariates that are present vs absent
    present_cov = [c for c in obs.columns if any(h in c.lower() for h in COVARIATE_HINTS)]
    has_celltype = any(any(k in c.lower() for k in ("cell_type", "celltype", "cluster", "leiden", "louvain"))
                       for c in obs.columns)
    has_cellcycle = any(any(k in c.lower() for k in ("phase", "cell_cycle", "cellcycle")) for c in obs.columns)
    return {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "layers": list(adata.layers.keys()),
        "obs_columns": cols,
        "X_raw_integer": raw_int,
        "X_min": float(samp.min()) if samp.size else None,
        "X_max": float(samp.max()) if samp.size else None,
        "X_mean": float(samp.mean()) if samp.size else None,
        "n_perturbations": len(perts),
        "control_label": ctrl,
        "n_control_cells": n_ctrl,
        "cells_per_pert_min": int(per_pert.min()) if len(per_pert) else None,
        "cells_per_pert_median": int(per_pert.median()) if len(per_pert) else None,
        "cells_per_pert_max": int(per_pert.max()) if len(per_pert) else None,
        "n_guides": int(obs[sg].nunique()) if (sg and sg in obs.columns) else None,
        "genes_with_ge2_guides": int((guides_per_gene >= 2).sum()) if guides_per_gene is not None else None,
        "present_covariate_like_cols": present_cov,
        "has_celltype_col": has_celltype,
        "has_cellcycle_col": has_cellcycle,
        "covariate_correction": cfg.get("covariate_correction", "none"),
    }


# --------------------------------------------------------------------------- #
# normalization (match cell-eval run when pdex receives raw counts)
# --------------------------------------------------------------------------- #
def _looks_raw_integer(adata: ad.AnnData) -> bool:
    X = adata.X
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        return False
    if data.size > 5_000_000:
        rng = np.random.default_rng(0)
        data = data[rng.integers(0, data.size, 5_000_000)]
    return bool(np.allclose(data, np.rint(data)))


def maybe_normalize(adata: ad.AnnData, cfg: dict) -> None:
    if not cfg.get("normalize_if_raw"):
        return
    if _looks_raw_integer(adata):
        log.info("normalize_if_raw: .X is raw integer counts -> normalize_total + log1p")
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
    else:
        log.info("normalize_if_raw set but .X does not look raw-integer -> skipping normalization")


# --------------------------------------------------------------------------- #
# DE wrapper (the real cell-eval code path)
# --------------------------------------------------------------------------- #
def run_de(adata: ad.AnnData, cfg: dict, groupby: str, reference: str) -> pl.DataFrame:
    return build_de_frame(
        mode="real",
        adata=adata,
        control_pert=reference,
        pert_col=groupby,
        num_threads=cfg["num_threads"],
        allow_discrete=cfg["allow_discrete"],
        de_method=cfg["de_method"],
        de_kwargs=None,
        counts_layer=cfg.get("counts_layer"),
        replicate_col=cfg.get("replicate_col"),
    )


# --------------------------------------------------------------------------- #
# shared metric helpers (TEST_PLAN.md "Shared Formulas")
# --------------------------------------------------------------------------- #
def _sig_mask(lfc, fdr, cfg):
    return np.isfinite(fdr) & (fdr <= cfg["fdr_threshold"]) & (np.abs(lfc) >= cfg["lfc_threshold"])


def lambda_gc(p: np.ndarray) -> float:
    p = p[np.isfinite(p)]
    p = np.clip(p, 1e-300, 1.0)
    if p.size == 0:
        return float("nan")
    chi2 = stats.chi2.isf(p, df=1)
    return float(np.median(chi2) / CHI2_MEDIAN_1DF)


def null_metrics(de: pl.DataFrame, cfg: dict) -> dict:
    lfc = de["log2_fold_change"].to_numpy().astype(float)
    p = de["p_value"].to_numpy().astype(float)
    fdr = de["fdr"].to_numpy().astype(float)
    fin = np.isfinite(lfc)
    G = int(fin.sum())
    sig = _sig_mask(lfc, fdr, cfg)
    pv = p[np.isfinite(p)]
    return {
        "G": G,
        "n_sig": int(np.nansum(sig)),
        "frac_sig": float(np.nansum(sig) / G) if G else float("nan"),
        "mean_lfc": float(np.nanmean(lfc[fin])) if G else float("nan"),
        "mean_abs_lfc": float(np.nanmean(np.abs(lfc[fin]))) if G else float("nan"),
        "ks_p_uniform": float(stats.kstest(pv, "uniform").pvalue) if pv.size >= 8 else float("nan"),
        "lambda_gc": lambda_gc(p),
    }


def qq_plot(pvals: np.ndarray, out_png: str, title: str) -> None:
    p = np.sort(pvals[np.isfinite(pvals)])
    G = p.size
    if G < 8:
        return
    p = np.clip(p, 1e-300, 1.0)
    obs = -np.log10(p)
    i = np.arange(1, G + 1)
    exp = -np.log10((i - 0.5) / G)
    lo = -np.log10(stats.beta.ppf(0.975, i, G - i + 1))
    hi = -np.log10(stats.beta.ppf(0.025, i, G - i + 1))
    if G > 3000:
        idx = np.unique(np.linspace(0, G - 1, 3000).astype(int))
        exp, obs, lo, hi = exp[idx], obs[idx], lo[idx], hi[idx]
    lam = lambda_gc(pvals)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.fill_between(exp, lo, hi, color="0.85", label="95% envelope")
    m = max(exp.max(), obs.max())
    ax.plot([0, m], [0, m], "r--", lw=1)
    ax.scatter(exp, obs, s=5, color="#1a3c6e")
    ax.set_xlabel("expected -log10(p)")
    ax.set_ylabel("observed -log10(p)")
    ax.set_title(f"{title}\nλ_GC = {lam:.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def compare_signatures(de_a: pl.DataFrame, de_b: pl.DataFrame, cfg: dict) -> dict:
    j = de_a.join(de_b, on="feature", how="inner", suffix="_b")
    if j.height < 5:
        return {}
    la = j["log2_fold_change"].to_numpy().astype(float)
    lb = j["log2_fold_change_b"].to_numpy().astype(float)
    fa = j["fdr"].to_numpy().astype(float)
    fb = j["fdr_b"].to_numpy().astype(float)
    ok = np.isfinite(la) & np.isfinite(lb)
    sa = _sig_mask(la, fa, cfg)
    sb = _sig_mask(lb, fb, cfg)
    union = int((sa | sb).sum())
    inter = int((sa & sb).sum())
    spear = float(stats.spearmanr(la[ok], lb[ok]).statistic) if ok.sum() >= 5 else float("nan")
    # additional: Spearman restricted to the UNION OF DE GENES (where the real signal lives) — the
    # all-genes ρ above is diluted by ~18k near-zero-effect genes; this isolates responsive genes.
    deg = (sa | sb) & ok
    spear_deg = float(stats.spearmanr(la[deg], lb[deg]).statistic) if int(deg.sum()) >= 5 else float("nan")
    direction = float(np.mean(np.sign(la[sa | sb]) == np.sign(lb[sa | sb]))) if union else float("nan")
    direction_all = float(np.mean(np.sign(la[ok]) == np.sign(lb[ok]))) if int(ok.sum()) else float("nan")
    auc = float("nan")
    if sb.sum() > 0 and (~sb).sum() > 0:
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(sb.astype(int), np.abs(la)))
        except Exception:
            auc = float("nan")
    return {
        "lfc_spearman": spear,
        "lfc_spearman_deg": spear_deg,
        "n_deg_union": int((sa | sb).sum()),
        "sig_jaccard": float(inter / union) if union else float("nan"),
        "direction_agreement": direction,
        "direction_agreement_all": direction_all,
        "auc_recovery": auc,
        "n_sig_a": int(sa.sum()),
        "n_sig_b": int(sb.sum()),
    }


def stratified_split(obs, strat_cols, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols = [c for c in strat_cols if c in obs.columns]
    out = np.empty(len(obs), dtype=object)
    if cols:
        groups = obs.groupby(cols, observed=True, sort=False).indices.values()
    else:
        groups = [np.arange(len(obs))]
    for idx in groups:
        idx = np.asarray(idx, dtype=int)
        rng.shuffle(idx)
        half = idx.size // 2
        out[idx[:half]] = "A"
        out[idx[half:]] = "B"
    return out.astype(str)


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclass
class TestResult:
    name: str
    title: str
    verdict: str  # PASS | WARN | FAIL | SKIP
    metrics: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    pvalues: dict = field(default_factory=dict)
    plot: str | None = None
    reason: str = ""  # one-line, LOCAL verdict reason


def verdict_from_null(m: dict, cfg: dict) -> tuple[str, list]:
    flags = []
    lam = m.get("lambda_gc", float("nan"))
    frac = m.get("frac_sig", float("nan"))
    ksp = m.get("ks_p_uniform", float("nan"))
    if np.isfinite(lam) and lam > cfg["lambda_gc_fail"]:
        flags.append(f"λ_GC={fmt(lam)} > {cfg['lambda_gc_fail']} (anti-conservative)")
        return "FAIL", flags
    if np.isfinite(frac) and frac > 5 * cfg["fdr_threshold"]:
        flags.append(f"frac_sig={fmt(frac)} >> fdr_threshold={cfg['fdr_threshold']}")
        return "FAIL", flags
    if (np.isfinite(lam) and lam > cfg["lambda_gc_warn"]) or (np.isfinite(frac) and frac > 2 * cfg["fdr_threshold"]):
        flags.append(f"λ_GC={fmt(lam)} / frac_sig={fmt(frac)} mildly elevated")
        return "WARN", flags
    defl = []
    if np.isfinite(lam) and lam < 0.9:
        defl.append(f"λ_GC={fmt(lam)} < 0.9 (deflated null — under-powered)")
    if np.isfinite(ksp) and ksp < 0.05:
        defl.append(f"ks_p_uniform={fmt(ksp)} < 0.05 (p-values not Uniform[0,1])")
    if defl:
        return "WARN", defl
    return "PASS", flags


# Pure verdict functions of cached metrics (reused by the live tests AND by --report-only, so a
# verdict-rule change can be re-applied to cached results without recomputing DE).
def verdict_test_0(m: dict, cfg: dict) -> tuple[str, str, list]:
    fpr_null = m.get("null_FPR", float("nan"))
    tpr_max = m.get("max_TPR", float("nan"))
    fpr_hi = m.get("FPR_at_max_delta", float("nan"))
    delta_hi = m.get("max_delta_log2fc", float("nan"))
    min_delta = m.get("min_resolvable_delta_log2fc", float("nan"))
    regime = m.get("injection_regime", "")
    flags, verdict = [], "PASS"
    if np.isfinite(fpr_null) and fpr_null > 2 * cfg["fdr_threshold"]:
        verdict = "FAIL"
        flags.append(f"null FPR={fmt(fpr_null)} > 2×fdr_threshold ({2*cfg['fdr_threshold']}) — anti-conservative")
    if np.isfinite(tpr_max) and tpr_max < 0.5:
        flags.append(f"max TPR={fmt(tpr_max)} < 0.5 even at δ={delta_hi} — resolves little real signal")
        if verdict != "FAIL":
            verdict = "WARN"
    coupling = np.isfinite(fpr_hi) and np.isfinite(fpr_null) and fpr_hi > max(2 * cfg["fdr_threshold"], fpr_null + 0.05)
    if coupling:
        flags.append(
            f"FPR rises to {fmt(fpr_hi)} at δ={delta_hi} (vs null {fmt(fpr_null)}) — strong injected effects "
            "induce false DE in UNTOUCHED genes via library-size renormalization (compositional coupling). "
            f"Injection regime: {regime}. This artifact scales with the FRACTION × magnitude of perturbed "
            "genes; it is a live risk only for broad/strong perturbations (a handful of perturbed genes "
            "barely shifts the library size). Appearing in the realistic small-N regime would be notable — "
            "prefer median-of-ratios / pseudobulk DESeq2 (or spike-in normalization) before interpreting "
            "non-target hits.")
        if verdict == "PASS":
            verdict = "WARN"
    if np.isfinite(fpr_null) and fpr_null <= cfg["fdr_threshold"] and verdict == "PASS":
        flags.append(f"null FPR={fmt(fpr_null)} ≤ α — conservative (no false positives)")
    if verdict == "FAIL":
        reason = f"FAIL: anti-conservative null FPR={fmt(fpr_null)}"
    elif verdict == "WARN" and coupling:
        reason = (f"WARN: null FPR={fmt(fpr_null)} clean but FPR→{fmt(fpr_hi)} at δ={delta_hi} "
                  "(compositional coupling — false DE in untouched genes under strong/broad effects)")
    elif verdict == "WARN":
        reason = f"WARN: under-powered (max TPR={fmt(tpr_max)})"
    else:
        reason = f"PASS: null FPR={fmt(fpr_null)} controlled; resolves δ≥{fmt(min_delta)} log2FC"
    return verdict, reason, flags


def verdict_test_5(m: dict, cfg: dict) -> tuple[str, str, list]:
    n_genes = m.get("n_genes", 0)
    n_pairs = m.get("n_same_pairs", 0)
    prim = m.get("primary_metric", "rho")  # default "rho" keeps old summaries (all-genes ρ) working
    prim_name = {"rho": "LFC ρ (all genes)", "rho_deg": "DEG LFC ρ", "jaccard": "DEG Jaccard",
                 "direction": "direction agreement"}.get(prim, prim)
    same = m.get(f"same_gene_mean_{prim}", m.get("same_gene_mean_rho", float("nan")))
    bg = m.get(f"background_mean_{prim}", m.get("background_mean_rho", float("nan")))
    sep = m.get("separation_z", float("nan"))
    auc = m.get(f"auc_{prim}", m.get("auc_rho"))            # common-language effect size = P(same > bg)
    mwu_p = m.get(f"mwu_p_{prim}", m.get("mwu_p_rho"))
    underpowered = (n_genes < 5) or (n_pairs < 5)
    flags = []
    if underpowered:
        # Never FAIL on an underpowered comparison — too few guide pairs to conclude anything.
        verdict = "WARN"
        trend = "trends higher than" if (np.isfinite(same) and np.isfinite(bg) and same > bg) else \
                ("≈" if (np.isfinite(same) and np.isfinite(bg)) else "vs")
        flags.append(
            f"UNINFORMATIVE / power-limited: only {n_genes} gene(s) with ≥2 guides ({n_pairs} same-gene "
            f"pair(s)). AUC={fmt(auc)} / MWU p={fmt(mwu_p)} carry essentially no statistical weight, so this "
            "is NOT evidence that guides are off-target or inefficacious. Same-gene guides also genuinely "
            "differ in knockdown efficacy, so only modest concordance is expected even with more pairs.")
        reason = (f"WARN (underpowered, cannot conclude): same-gene {prim_name}={fmt(same)} {trend} "
                  f"background {fmt(bg)} (AUC={fmt(auc)}) on only {n_pairs} pair(s)")
        return verdict, reason, flags
    # PRIMARY verdict: Mann-Whitney common-language effect size AUC = P(same-gene pair more concordant
    # than a random unrelated pair), GATED by the MWU significance. AUC is rank-based (robust to the wide,
    # skewed background that deflates the separation z) and not inflated by sample size (unlike a raw p).
    if auc is not None and np.isfinite(auc) and mwu_p is not None and np.isfinite(mwu_p):
        sig = mwu_p < cfg.get("fdr_threshold", 0.05)
        if auc >= 0.65 and sig:
            verdict = "PASS"
        elif auc >= 0.55 and sig:
            verdict = "WARN"
        else:
            verdict = "FAIL"
        strength = ("clear" if auc >= 0.65 else "modest" if auc >= 0.55 else "no")
        flags.append(f"AUC={fmt(auc)} = P(a random same-gene pair is more concordant than a random "
                     f"unrelated pair); 0.5 = no separation. Significance: MWU p={fmt(mwu_p)}. "
                     f"Tiers: PASS AUC≥0.65 & p<{cfg.get('fdr_threshold', 0.05)} · WARN AUC≥0.55 & sig · "
                     "FAIL otherwise. (Secondary, for reference: separation z="
                     f"{fmt(sep)} — z is variance-sensitive and understates a wide-background separation.)")
        reason = (f"{verdict}: {strength} same-gene separation — AUC={fmt(auc)} "
                  f"(P[same>unrelated]), MWU p={fmt(mwu_p)}; same-gene {prim_name}={fmt(same)} vs "
                  f"background {fmt(bg)}; {n_pairs} same-gene pairs (2° separation z={fmt(sep)})")
        return verdict, reason, flags
    # fallback (old summaries without AUC): the legacy separation-z rule
    if (not np.isfinite(sep) and np.isfinite(same) and np.isfinite(bg) and same > bg) or sep > 1.5:
        verdict = "PASS"
    elif np.isfinite(sep) and sep > 1.0:
        verdict = "WARN"
    else:
        verdict = "FAIL"
    reason = (f"{verdict}: same-gene {prim_name}={fmt(same)} vs background {fmt(bg)} "
              f"(separation z={fmt(sep)}, MWU p={fmt(mwu_p)}; primary={prim_name}; {n_pairs} pairs)")
    return verdict, reason, flags


# --------------------------------------------------------------------------- #
# TEST 0 — effect-size injection / calibration curve (NEW)
# --------------------------------------------------------------------------- #
def test_0_injection(adata_raw, cfg, outdir):
    """Inject known log2FC into a SMALL number of genes (~a dozen, the scale of a real perturbation) in
    a control-vs-control split, pooled over several random gene-draws; measure FPR-at-null and
    TPR-vs-effect-size. Distinguishes 'null=null' from 'pipeline dead' and reports the smallest effect
    the metric can resolve. Operates on RAW counts (before global normalization). Injecting a small
    NUMBER (not a large fraction) avoids the library-size shift that artificially creates compositional
    coupling, so it is faithful to how few genes a real perturbation actually moves."""
    pc, ctrl = cfg["pert_col"], cfg["control_pert"]
    mcg = cfg["min_cells_per_group"]
    cmask = (adata_raw.obs[pc].astype(str) == ctrl).to_numpy()
    ctrl_full = adata_raw[cmask]
    if ctrl_full.n_obs < 2 * mcg:
        return TestResult("test_0", "Effect-Size Injection / Calibration Curve", "SKIP",
                          flags=["too few control cells for an injection split"],
                          reason="SKIP: insufficient control cells")
    deltas = list(cfg.get("injection_deltas", [0.0, 0.25, 0.5, 1.0, 2.0]))
    n_genes_cfg = cfg.get("injection_n_genes", 12)    # REALISTIC: a real perturbation moves ~a dozen genes
    frac_cfg = cfg.get("injection_frac_genes", None)  # legacy large-fraction stress (used only if n_genes is null)
    n_hvg = int(cfg.get("injection_n_hvg", 2000))
    G = ctrl_full.n_vars
    feats = np.asarray(ctrl_full.var_names)

    # ===== Gene-class properties — computed over ALL control cells, so they are a DETERMINISTIC
    # ===== function of the h5ad (independent of the DE cell subsample / its seed). =====
    Xc = ctrl_full.X.tocsr() if sp.issparse(ctrl_full.X) else sp.csr_matrix(np.asarray(ctrl_full.X))
    n_c = Xc.shape[0]
    expressed = np.where(np.asarray(Xc.sum(0)).ravel() > 0)[0]
    ctrlln = ctrl_full.copy()
    sc.pp.normalize_total(ctrlln)
    sc.pp.log1p(ctrlln)
    Xn = ctrlln.X.tocsr() if sp.issparse(ctrlln.X) else sp.csr_matrix(np.asarray(ctrlln.X))
    del ctrlln
    mean_expr = np.asarray(Xn.mean(0)).ravel()                 # log-normalized mean per gene (all controls)
    try:
        hvtmp = ad.AnnData(X=Xn.copy(), var=ctrl_full.var.copy())
        sc.pp.highly_variable_genes(hvtmp, n_top_genes=min(n_hvg, max(1, expressed.size // 2)), flavor="seurat")
        hvg_bool = hvtmp.var["highly_variable"].to_numpy().astype(bool)
        del hvtmp
    except Exception as e:  # noqa: BLE001
        log.warning("test0 HVG computation failed (%s); treating all genes as non-HVG", e)
        hvg_bool = np.zeros(G, dtype=bool)

    # --- EXPRESSION tertiles (static): low = sparse/hard ... high = abundant/easy. Anchors are drawn
    #     evenly across tiers so the TPR-vs-δ curve is interpretable PER tier (the smallest resolvable δ
    #     for sparse vs typical vs abundant genes), instead of an artifact of the mix. ---
    ee = mean_expr[expressed]
    t33, t67 = float(np.quantile(ee, 1.0 / 3)), float(np.quantile(ee, 2.0 / 3))
    EXPR_TIERS = ("low", "mid", "high")
    expr_tier_all = np.full(G, "none", dtype=object)
    expr_tier_all[expressed] = np.where(mean_expr[expressed] <= t33, "low",
                                        np.where(mean_expr[expressed] >= t67, "high", "mid"))

    # --- REALISM: inject a SMALL fixed NUMBER of genes (~a dozen, the scale of a real perturbation),
    #     repeated over several independent random draws and POOLED, so the rates are stable while each
    #     DE stays realistic. Compositional coupling (false DE in untouched genes via library-size
    #     renormalization) only appears when a large FRACTION of genes is perturbed and the per-cell
    #     total shifts; a dozen genes barely move it, so this is the faithful test. The legacy fraction
    #     mode (set injection_n_genes: null + injection_frac_genes) is kept as a large-fraction stress. ---
    if n_genes_cfg is not None:
        n_inj = max(3, int(n_genes_cfg))
        n_repeats = max(1, int(cfg.get("injection_n_repeats", 10)))
        regime = f"{n_inj} genes × {n_repeats} draws (realistic small-perturbation regime)"
    else:
        n_inj = max(1, int(float(frac_cfg or 0.10) * expressed.size))
        n_repeats = 1
        regime = f"{n_inj} genes (= {float(frac_cfg or 0.10):.0%} of expressed; large-fraction stress)"

    # --- per-gene moments reused for the per-draw anchor correlation ---
    sumsq_g = np.asarray(Xn.multiply(Xn).sum(0)).ravel()
    var_g = np.clip(sumsq_g / n_c - mean_expr ** 2, 0.0, None)

    # --- static FPR gene-class masks (draw-independent); AnchorCorr is added per-draw inside the loop ---
    expressed_mask = np.zeros(G, dtype=bool)
    expressed_mask[expressed] = True
    hk_mask = np.array([g in HOUSEKEEPING for g in feats])
    marker_set = set(cfg.get("injection_marker_genes", []) or [])
    marker_mask = np.array([g in marker_set for g in feats])
    heg_hi, leg_lo = float(np.quantile(ee, 0.80)), float(np.quantile(ee, 0.20))
    rng_grp = np.random.default_rng(cfg["seed"] + 9)
    random_mask = np.zeros(G, dtype=bool)
    random_mask[rng_grp.choice(G, size=max(1, int(0.10 * G)), replace=False)] = True
    STATIC_GROUPS = {  # diagnostic lenses (may overlap); FPR is measured over UNTOUCHED genes in each
        "HighlyExpr": expressed_mask & (mean_expr >= heg_hi),    # abundant -> high signal, easy
        "LowlyExpr": expressed_mask & (mean_expr <= leg_lo),     # sparse -> hard, zero-dominated
        "HouseKeeping": hk_mask,                                 # constitutive -> predictable, watch memorization
        "Marker": marker_mask,                                   # cell-type identity (N/A without annotation)
        "HighlyVarG": hvg_bool,                                  # high-variance complement to anchors
        "Random": random_mask,                                   # unbiased baseline across detection spectrum
    }
    GROUP_NAMES = ["AnchorCorr"] + list(STATIC_GROUPS.keys())    # AnchorCorr depends on the per-draw anchors
    pos = {f: i for i, f in enumerate(feats.tolist())}

    # ===== DE cell subsample (fixed once; the question is about gene injection, not split luck) =====
    cap = int(cfg.get("injection_max_cells_per_arm", 3000))
    rng = np.random.default_rng(cfg["seed"] + 7)
    order = np.arange(ctrl_full.n_obs)
    rng.shuffle(order)
    sub = ctrl_full[order[: 2 * cap]].copy()
    arm = stratified_split(sub.obs, cfg["block_cols"], cfg["seed"] + 7)
    bmask = arm == "B"
    base = sub.X
    base = base.toarray().astype(np.float64) if sp.issparse(base) else np.asarray(base, float)
    normalize = bool(cfg.get("normalize_if_raw"))  # pdex path normalizes; pydeseq2 consumes raw counts

    # ===== pool TPR/FPR over n_repeats independent small gene-draws (accumulate COUNTS, not rates) =====
    acc = {float(d): {"tpr_n": 0, "tpr_d": 0, "fpr_n": 0, "fpr_d": 0,
                      "tier_n": {t: 0 for t in EXPR_TIERS}, "tier_d": {t: 0 for t in EXPR_TIERS},
                      "grp_n": {g: 0 for g in GROUP_NAMES}, "grp_d": {g: 0 for g in GROUP_NAMES},
                      "lfc_inj": [], "pvals": []} for d in deltas}
    n_inj_tier_tot = {t: 0 for t in EXPR_TIERS}
    draws_done = 0
    for rep in range(n_repeats):
        rng_sel = np.random.default_rng(cfg["seed"] + 8 + rep)
        per = max(1, n_inj // 3)
        inj_parts = []
        for t in EXPR_TIERS:
            pool = np.where(expr_tier_all == t)[0]
            if pool.size:
                inj_parts.append(rng_sel.choice(pool, size=min(per, pool.size), replace=False))
        inj_idx = np.sort(np.concatenate(inj_parts).astype(int)) if inj_parts else np.array([], int)
        if inj_idx.size == 0:
            continue
        inj_mask_all = np.zeros(G, dtype=bool)
        inj_mask_all[inj_idx] = True
        for t in EXPR_TIERS:
            n_inj_tier_tot[t] += int((expr_tier_all[inj_idx] == t).sum())
        # |Pearson r| of every gene with THIS draw's anchor signature, over ALL controls (sparse, exact)
        sgn = np.asarray(Xn[:, inj_idx].mean(1)).ravel()
        e_gs = np.asarray(Xn.T.dot(sgn)).ravel() / n_c
        cov = e_gs - mean_expr * sgn.mean()
        anchor_corr = np.abs(cov / (np.sqrt(var_g) * sgn.std() + 1e-12))
        groups = dict(STATIC_GROUPS)
        groups["AnchorCorr"] = anchor_corr >= float(np.quantile(anchor_corr, 0.90))
        for d in deltas:
            Xd = base.copy()
            if d > 0:
                Xd[np.ix_(np.where(bmask)[0], inj_idx)] *= 2.0 ** d
            if not normalize:
                Xd = np.rint(Xd)  # keep integer counts for pseudobulk DESeq2
            sad = ad.AnnData(X=Xd.astype(np.float32), obs=sub.obs.copy(), var=sub.var.copy())
            sad.obs["_arm"] = arm.astype(str)
            if normalize:
                sc.pp.normalize_total(sad, inplace=True)
                sc.pp.log1p(sad)
            try:
                de = run_de(sad, cfg, groupby="_arm", reference="A")  # target = B (anchors injected up)
            except Exception as e:  # noqa: BLE001
                log.warning("test0 rep=%d delta=%s: %s", rep, d, e)
                continue
            feat_arr = de["feature"].cast(str).to_numpy()
            fpos = np.array([pos[f] for f in feat_arr])           # map DE rows -> gene-axis positions
            lfc = de["log2_fold_change"].to_numpy().astype(float)
            fdr = de["fdr"].to_numpy().astype(float)
            pvl = de["p_value"].to_numpy().astype(float)
            sig = _sig_mask(lfc, fdr, cfg)
            is_inj = inj_mask_all[fpos]
            A = acc[float(d)]
            A["tpr_n"] += int(sig[is_inj].sum()); A["tpr_d"] += int(is_inj.sum())
            A["fpr_n"] += int(sig[~is_inj].sum()); A["fpr_d"] += int((~is_inj).sum())
            A["lfc_inj"].extend(lfc[is_inj].tolist())
            A["pvals"].extend(pvl.tolist())
            tier_at = expr_tier_all[fpos]
            for t in EXPR_TIERS:
                mt = is_inj & (tier_at == t)
                A["tier_n"][t] += int(sig[mt].sum()); A["tier_d"][t] += int(mt.sum())
            for g in GROUP_NAMES:
                unt = (~is_inj) & groups[g][fpos]
                A["grp_n"][g] += int(sig[unt].sum()); A["grp_d"][g] += int(unt.sum())
        draws_done += 1
    del Xn
    log.info("test0 injected anchors: %s; pooled over %d draw(s) [low=%d mid=%d high=%d genes total]",
             regime, draws_done, n_inj_tier_tot["low"], n_inj_tier_tot["mid"], n_inj_tier_tot["high"])

    rows = []
    for d in deltas:
        A = acc[float(d)]
        if A["tpr_d"] == 0:
            continue
        row = {"delta_log2fc": float(d), "n_injected": int(A["tpr_d"]),
               "TPR": A["tpr_n"] / A["tpr_d"] if A["tpr_d"] else float("nan"),
               "FPR": A["fpr_n"] / A["fpr_d"] if A["fpr_d"] else float("nan"),
               "median_observed_lfc_injected": float(np.nanmedian(A["lfc_inj"])) if A["lfc_inj"] else float("nan"),
               "lambda_gc": lambda_gc(np.asarray(A["pvals"])) if A["pvals"] else float("nan")}
        # FPR among UNTOUCHED genes, broken down by the diagnostic gene classes (lenses)
        for g in GROUP_NAMES:
            row[f"FPR_{g}"] = A["grp_n"][g] / A["grp_d"][g] if A["grp_d"][g] else float("nan")
            row[f"n_untouched_{g}"] = int(A["grp_d"][g])
        # TPR among INJECTED/anchor genes, broken down by EXPRESSION TIER (detection power)
        for t in EXPR_TIERS:
            row[f"TPR_{t}expr"] = A["tier_n"][t] / A["tier_d"][t] if A["tier_d"][t] else float("nan")
            row[f"n_injected_{t}expr"] = int(A["tier_d"][t])
        rows.append(row)
    if not rows:
        return TestResult("test_0", "Effect-Size Injection / Calibration Curve", "SKIP",
                          flags=["no injection DE succeeded"], reason="SKIP: no DE result")
    df = pl.DataFrame(rows)
    df.write_csv(os.path.join(outdir, "tables", "test_0__calibration_curve.csv"))
    # plot overall TPR/FPR, plus FPR-by-class and TPR-by-class panels
    png = os.path.join(outdir, "plots", "test_0_injection.png")
    fig, (axl, axm, axr) = plt.subplots(1, 3, figsize=(13.0, 4.0))
    dd = df["delta_log2fc"].to_numpy()
    axl.plot(dd, df["TPR"].to_numpy(), "-o", color="#1a3c6e", label="TPR (injected)")
    axl.plot(dd, df["FPR"].to_numpy(), "-s", color="#b22222", label="FPR (untouched)")
    axl.axhline(cfg["fdr_threshold"], ls="--", color="0.5", lw=1, label=f"α={cfg['fdr_threshold']}")
    axl.set_xlabel("injected δ (log2 fold-change)"); axl.set_ylabel("rate"); axl.set_ylim(-0.02, 1.02)
    axl.set_title("Overall TPR / FPR", fontsize=9); axl.legend(fontsize=7)
    # middle: FPR by gene class (untouched genes)
    grp_colors = {"AnchorCorr": "#b22222", "HighlyExpr": "#1a9850", "LowlyExpr": "#4575b4",
                  "HouseKeeping": "#762a83", "Marker": "#e08214", "HighlyVarG": "#d6604d",
                  "Random": "#888888"}
    for c, col in grp_colors.items():
        cc = f"FPR_{c}"
        if cc in df.columns and not df[cc].is_null().all():
            axm.plot(dd, df[cc].to_numpy(), "-o", ms=4, color=col, label=c)
    axm.axhline(cfg["fdr_threshold"], ls="--", color="0.5", lw=1, label=f"α={cfg['fdr_threshold']}")
    axm.set_xlabel("injected δ (log2 fold-change)"); axm.set_ylabel("FPR (untouched)"); axm.set_ylim(-0.02, 1.02)
    axm.set_title("FPR by gene class (untouched; anchors=injected)", fontsize=8.5); axm.legend(fontsize=6.5)
    # right: TPR by EXPRESSION TIER of the injected/anchor gene (detection power)
    tier_colors = {"high": "#1a9850", "mid": "#4575b4", "low": "#b22222"}
    for t, col in tier_colors.items():
        cc = f"TPR_{t}expr"
        if cc in df.columns and not df[cc].is_null().all():
            axr.plot(dd, df[cc].to_numpy(), "-o", ms=4, color=col, label=f"{t} expr")
    axr.axhline(0.5, ls=":", color="0.5", lw=1, label="TPR=0.5")
    axr.set_xlabel("injected δ (log2 fold-change)"); axr.set_ylabel("TPR (injected)"); axr.set_ylim(-0.02, 1.02)
    axr.set_title("TPR by injected-gene expression tier", fontsize=8.5); axr.legend(fontsize=6.5)
    fig.suptitle("Test 0 — injection calibration curve", fontsize=11)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)
    fpr0 = df.filter(pl.col("delta_log2fc") == 0.0)["FPR"]
    fpr_null = float(fpr0[0]) if fpr0.len() else float("nan")
    pos = df.filter(pl.col("delta_log2fc") > 0)
    tpr_max = float(pos["TPR"].max()) if pos.height else float("nan")
    delta_hi = float(pos["delta_log2fc"].max()) if pos.height else float("nan")
    fpr_hi = float(pos.filter(pl.col("delta_log2fc") == delta_hi)["FPR"][0]) if pos.height else float("nan")
    res = pos.filter(pl.col("TPR") >= 0.5)
    min_delta = float(res["delta_log2fc"].min()) if res.height else float("nan")
    hi_row = pos.filter(pl.col("delta_log2fc") == delta_hi) if pos.height else df.head(0)
    metrics = {"null_FPR": fpr_null, "max_TPR": tpr_max, "min_resolvable_delta_log2fc": min_delta,
               "FPR_at_max_delta": fpr_hi, "max_delta_log2fc": delta_hi,
               "n_injected_genes_per_draw": int(n_inj), "n_draws": int(draws_done),
               "n_injected_total": int(sum(n_inj_tier_tot.values())),
               "n_injected_low_expr": int(n_inj_tier_tot["low"]), "n_injected_mid_expr": int(n_inj_tier_tot["mid"]),
               "n_injected_high_expr": int(n_inj_tier_tot["high"]), "injection_stratified_by": "expression tertile",
               "injection_regime": regime,
               "n_expressed_genes": int(expressed.size), "n_total_genes": int(G),
               "control_cells_used": int(sub.n_obs), "cells_per_arm": int(min(bmask.sum(), (~bmask).sum()))}
    # per-gene-class FPR (untouched) + per-expression-tier TPR (injected) at the largest injected δ
    for c in GROUP_NAMES:
        col = f"FPR_{c}"
        if hi_row.height and col in hi_row.columns:
            metrics[f"FPR_{c}_at_max_delta"] = float(hi_row[col][0])
    for t in EXPR_TIERS:
        col = f"TPR_{t}expr"
        if hi_row.height and col in hi_row.columns:
            metrics[f"TPR_{t}expr_at_max_delta"] = float(hi_row[col][0])
    # min resolvable δ per expression tier (smallest δ reaching TPR ≥ 0.5 in that tier)
    for t in EXPR_TIERS:
        col = f"TPR_{t}expr"
        if col in pos.columns:
            r = pos.filter(pl.col(col) >= 0.5)
            metrics[f"min_resolvable_delta_{t}expr"] = float(r["delta_log2fc"].min()) if r.height else float("nan")
    verdict, reason, flags = verdict_test_0(metrics, cfg)
    return TestResult("test_0", "Effect-Size Injection / Calibration Curve", verdict, metrics=metrics,
                      flags=flags, pvalues={"null_FPR": fpr_null}, plot="plots/test_0_injection.png",
                      reason=reason)


# --------------------------------------------------------------------------- #
# Composition control diagnostic (NEW)
# --------------------------------------------------------------------------- #
def composition_diagnostic(adata, cfg, outdir):
    """Check whether perturbations shift cell-type / cluster / cell-cycle proportions vs control —
    such shifts make cross-population DE read composition change as expression change. Requires a
    categorical cell-state column; gracefully SKIPs (with guidance) if none exists."""
    obs = adata.obs
    explicit = [c for c in cfg.get("celltype_cols", []) if c in obs.columns]
    auto = [c for c in obs.columns
            if any(k in c.lower() for k in ("cell_type", "celltype", "cluster", "leiden", "louvain",
                                            "phase", "cell_cycle", "cellcycle"))]
    cols = explicit or auto
    if not cols:
        flag = ("no cell-type / cluster / cell-cycle column in obs — composition shift cannot be "
                "assessed. Only `batch` is present. A perturbation that changes proliferation or "
                "differentiation will alter cell-state proportions, and cross-population DE reads "
                "that as expression change. Recommend annotating cell type (and cell-cycle phase) "
                "and re-running, or interpreting strong hits with this confounder in mind.")
        return TestResult("composition", "Composition Control", "SKIP", flags=[flag],
                          reason="SKIP: no cell-state column to assess composition shift")
    col = cols[0]
    pc, ctrl = cfg["pert_col"], cfg["control_pert"]
    cats = sorted(obs[col].astype(str).unique())
    ref = obs.loc[obs[pc].astype(str) == ctrl, col].astype(str).value_counts(normalize=True)
    ref = ref.reindex(cats, fill_value=0.0).to_numpy()
    rows = []
    for pert in perts_in_use(adata, cfg):
        q = obs.loc[obs[pc].astype(str) == pert, col].astype(str).value_counts(normalize=True)
        q = q.reindex(cats, fill_value=0.0).to_numpy()
        tvd = float(0.5 * np.abs(q - ref).sum())  # total variation distance from control
        rows.append({"condition": pert, "tv_distance_vs_control": tvd})
    df = pl.DataFrame(rows)
    df.write_csv(os.path.join(outdir, "tables", "composition__tvd.csv"))
    med = float(df["tv_distance_vs_control"].median())
    mx = float(df["tv_distance_vs_control"].max())
    n_shift = int((df["tv_distance_vs_control"] > 0.10).sum())
    verdict = "WARN" if n_shift > 0 else "PASS"
    flags = []
    if n_shift:
        flags.append(f"{n_shift} perturbation(s) shift `{col}` proportions by TVD>0.10 vs control — "
                     "their DE may partly reflect composition, not per-cell expression change")
    reason = (f"{verdict}: {n_shift} perturbation(s) with TVD>0.10 on `{col}` (median={fmt(med)})")
    return TestResult("composition", "Composition Control", verdict,
                      metrics={"cell_state_col": col, "median_tvd": med, "max_tvd": mx,
                               "n_perts_tvd_gt_0.10": n_shift},
                      flags=flags, reason=reason)


# --------------------------------------------------------------------------- #
# the six tests (logic unchanged; verdict reasons made local)
# --------------------------------------------------------------------------- #
def perts_in_use(adata, cfg):
    tg = adata.obs[cfg["pert_col"]].astype(str)
    perts = [g for g in sorted(tg.unique()) if g != cfg["control_pert"]]
    if cfg.get("max_conditions"):
        perts = perts[: cfg["max_conditions"]]
    return perts


def _de_two(adata, pos1, pos2, cfg, lab1="G1", lab2="G2"):
    """DE of cells at global positions pos1 (label lab1) vs pos2 (label lab2), reference=lab2."""
    sel = np.concatenate([np.asarray(pos1, int), np.asarray(pos2, int)])
    s = adata[sel].copy()
    s.obs["_g"] = np.array([lab1] * len(pos1) + [lab2] * len(pos2))
    return run_de(s, cfg, groupby="_g", reference=lab2)


# Reproducibility verdict tiers (Tests 1 & 4) — strong/moderate/low thresholds per metric.
# (LFC ρ drives the verdict; Jaccard/direction are tiered too, from TEST_PLAN §Test 4 "Expected Behaviour".)
REPRO_PASS, REPRO_WARN = 0.6, 0.3
# metric key -> (display name, strong threshold, moderate threshold). value > strong = strong (PASS),
# >= moderate = moderate (WARN), else low (FAIL). "strong→PASS / moderate→WARN / low→FAIL".
REPRO_METRIC_TIERS = {
    "lfc_spearman":        ("split-half LFC Spearman ρ", 0.6, 0.3),
    "sig_jaccard":         ("DEG-set Jaccard",           0.3, 0.1),
    "direction_agreement": ("direction agreement",       0.7, 0.6),
}
TIER_TO_VERDICT = {"strong": "PASS", "moderate": "WARN", "low": "FAIL", "n/a": "SKIP"}
REPRO_LEGEND = (f"LFC ρ (verdict driver): >{REPRO_PASS} strong/PASS, {REPRO_WARN}–{REPRO_PASS} "
                f"moderate/WARN, <{REPRO_WARN} low/FAIL. DEG Jaccard: >0.3 strong, 0.1–0.3 moderate, "
                "<0.1 low. Direction agreement: >0.7 strong, 0.6–0.7 moderate, ≈0.5 low. Low = the two "
                "half-signatures disagree, so downstream metrics cannot be trusted above this ceiling.")


def repro_tier(value: float, strong: float, moderate: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    if value > strong:
        return "strong"
    if value >= moderate:
        return "moderate"
    return "low"


def metric_tier(metric_key: str, value: float) -> str:
    """Tier label (strong/moderate/low) for a reproducibility metric using REPRO_METRIC_TIERS."""
    if metric_key not in REPRO_METRIC_TIERS:
        return "n/a"
    _, strong, moderate = REPRO_METRIC_TIERS[metric_key]
    return repro_tier(value, strong, moderate)


def verdict_reproducibility(rho: float) -> str:
    return TIER_TO_VERDICT[repro_tier(rho, REPRO_PASS, REPRO_WARN)]


def verdict_test_4(m: dict, cfg: dict) -> tuple[str, str, list]:
    """Test 4 verdict — guide-level reproducibility, identical tiers to Test 1 (delegates to the shared
    metrics-based verdict so the live run and --report-only stay consistent)."""
    return _repro_verdict_from_metrics(m, cfg, "guide")


# Reproducibility metrics shown vs cell count, one panel each. (rdf column, panel title, tier key into
# REPRO_METRIC_TIERS, y-limits). NOTE: Jaccard's value is independent of the gene universe (it is the
# overlap of the two SIGNIFICANT sets), so "all-genes Jaccard" ≡ "DEG Jaccard" — shown once.
REPRO_PANELS = [
    ("lfc_spearman",            "Spearman LFC — ALL genes",        "lfc_spearman",        (-0.25, 1.0)),
    ("lfc_spearman_deg",        "Spearman LFC — DE genes",         "lfc_spearman",        (-0.25, 1.0)),
    ("sig_jaccard",             "DEG-set Jaccard (≡ all-genes)",   "sig_jaccard",         (0.0, 1.0)),
    ("direction_agreement_all", "Direction agreement — ALL genes", "direction_agreement", (0.0, 1.02)),
    ("direction_agreement",     "Direction agreement — DE genes",  "direction_agreement", (0.0, 1.02)),
]


def repro_vs_ncells(rdf, cfg):
    """Per-condition reproducibility (median over reps) vs the condition's CELL COUNT — ONE POINT PER
    PERTURBATION (no binning), for EVERY reproducibility statistic in REPRO_PANELS. Separates 'the method
    is low-reproducibility' from 'this perturbation just didn't have enough cells': values that climb with
    cell count and plateau => power-limited; values that stay low even at high cell counts => a genuine
    method limitation. Returns (per_pert_df[condition, n_cells, <metric cols>], primary_trend_metrics,
    trends_df[metric, spearman_ncells, well/under medians])."""
    if rdf is None or "n_pert_cells" not in rdf.columns:
        return None, {}, None
    metric_cols = [c for c, *_ in REPRO_PANELS if c in rdf.columns]
    aggs = [pl.col(c).median().alias(c) for c in metric_cols] + [pl.col("n_pert_cells").first().alias("n_cells")]
    for sc in ("n_sig_a", "n_sig_b", "n_deg_union"):  # for the DEG-count point encoding
        if sc in rdf.columns:
            aggs.append(pl.col(sc).median().alias(sc))
    pc = rdf.group_by("condition").agg(aggs).drop_nulls(["n_cells"]).sort("n_cells")
    # n_deg = # DE genes in the UNION of split A & B (the set the DEG metrics use). Prefer the exact union
    # count (n_deg_union); fall back to the mean of the two arms' significant counts if union isn't present.
    if "n_deg_union" in pc.columns:
        pc = pc.with_columns(pl.col("n_deg_union").alias("n_deg"))
    elif "n_sig_a" in pc.columns and "n_sig_b" in pc.columns:
        pc = pc.with_columns(((pl.col("n_sig_a") + pl.col("n_sig_b")) / 2.0).alias("n_deg"))
    nc = pc["n_cells"].to_numpy().astype(float)
    wp_min = int(cfg.get("test1_wellpowered_min_cells", 200))   # >=200 total ~ >=100 cells per half
    tm = {"wellpowered_min_cells": wp_min}
    trows = []
    for c in metric_cols:
        v = pc[c].to_numpy().astype(float)
        ok = np.isfinite(v) & np.isfinite(nc)
        if ok.sum() >= 3:
            vo, nco = v[ok], nc[ok]
            tr = float(stats.spearmanr(nco, vo).correlation) if vo.size > 2 else float("nan")
            wp, up = vo[nco >= wp_min], vo[nco < wp_min]
            wpm = float(np.median(wp)) if wp.size else float("nan")
            upm = float(np.median(up)) if up.size else float("nan")
            nwp, nup = int(wp.size), int(up.size)
        else:
            tr = wpm = upm = float("nan"); nwp = nup = 0
        trows.append({"metric": c, "spearman_ncells": tr, "wellpowered_median": wpm,
                      "underpowered_median": upm, "n_wellpowered": nwp, "n_underpowered": nup})
        if c == "lfc_spearman":  # primary -> canonical keys used by the verdict flag + report summary
            tm.update({"rho_vs_ncells_spearman": tr, "repro_rho_wellpowered": wpm,
                       "repro_rho_underpowered": upm, "n_wellpowered": nwp, "n_underpowered": nup})
    return pc, tm, (pl.DataFrame(trows) if trows else None)


def plot_repro_vs_ncells(modes, path, cfg, title_prefix="Test 1", unit_label="perturbation"):
    """modes: {mode: {"_pc": per_pert_df}}. A GRID of scatter panels — one per reproducibility statistic
    in REPRO_PANELS — each with one point PER UNIT (perturbation for Test 1, guide for Test 4;
    x = cell count, log). The 'is it power or the method?' figure, now for every statistic."""
    if not modes:
        return
    panels = [p for p in REPRO_PANELS
              if any((d.get("_pc") is not None and p[0] in d["_pc"].columns) for d in modes.values())]
    if not panels:
        return
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.5 * nrow), squeeze=False)
    colors = {"no_match": "#1a3c6e"}
    wp_min = int(cfg.get("test1_wellpowered_min_cells", 200))
    allnc = [d["_pc"]["n_cells"].to_numpy().astype(float) for d in modes.values()
             if d.get("_pc") is not None and d["_pc"].height]
    allnc = np.concatenate(allnc) if allnc else np.array([])
    cand = [100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000]
    ticks = [t for t in cand if allnc.size and allnc.min() * 0.9 <= t <= allnc.max() * 1.1]
    # marker AREA encodes the # of DE genes in the UNION of split A and B (the gene set the DEG-restricted
    # metrics are computed over), sqrt-scaled so a wide range stays readable.
    def sz(nd):
        return np.clip(6.0 + 5.5 * np.sqrt(np.clip(np.asarray(nd, float), 0, None)), 6, 240)
    have_ndeg = any(d.get("_pc") is not None and "n_deg" in d["_pc"].columns for d in modes.values())
    for i, (col_name, title, tk, yl) in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        for mode, d in modes.items():
            pcd = d.get("_pc")
            if pcd is None or col_name not in pcd.columns:
                continue
            nc = pcd["n_cells"].to_numpy().astype(float)
            v = pcd[col_name].to_numpy().astype(float)
            col = colors.get(mode, "#444444")
            s = sz(pcd["n_deg"].to_numpy()) if "n_deg" in pcd.columns else 20
            ax.scatter(nc, v, s=s, alpha=0.55, color=col, edgecolor="white", linewidth=0.3,
                       label=f"median={fmt(float(np.nanmedian(v)))}")
        if tk in REPRO_METRIC_TIERS:
            for thr in REPRO_METRIC_TIERS[tk][1:3]:
                ax.axhline(thr, ls="--", color="0.6", lw=0.8)
        ax.axvline(wp_min, ls=":", color="0.5", lw=0.8)
        ax.set_xscale("log"); ax.set_ylim(*yl)
        if len(ticks) >= 2:
            ax.set_xticks(ticks); ax.set_xticklabels([f"{t:,}" for t in ticks], rotation=45, fontsize=6)
            ax.minorticks_off()
        ax.set_title(title, fontsize=8); ax.tick_params(labelsize=6.5)
        ax.legend(fontsize=6, loc="lower right")
        if i % ncol == 0:
            ax.set_ylabel(f"per-{unit_label} value", fontsize=7)
    for j in range(len(panels), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    # DE-gene size legend — figure-level (bottom), always shown regardless of grid layout
    if have_ndeg:
        allnd = np.concatenate([d["_pc"]["n_deg"].to_numpy().astype(float) for d in modes.values()
                                if d.get("_pc") is not None and "n_deg" in d["_pc"].columns])
        allnd = allnd[np.isfinite(allnd)]
        refs = [r for r in (10, 50, 100, 500, 1000, 2000, 5000)
                if allnd.size and allnd.min() <= r <= allnd.max()] or ([int(np.median(allnd))] if allnd.size else [])
        h0 = axes[0][0]
        handles, labels = [], []
        for r in refs:
            handles.append(h0.scatter([], [], s=float(sz(r)), color="#1a3c6e", alpha=0.55, edgecolor="white",
                                      linewidth=0.3))
            labels.append(f"{r:,}")
        fig.legend(handles, labels, title="marker size = # DE genes (union of split A & B)",
                   loc="lower center", ncol=max(1, len(labels)), fontsize=7, title_fontsize=7, frameon=True,
                   columnspacing=1.4, handletextpad=0.4)
    fig.suptitle(f"{title_prefix} — reproducibility vs cell count, one point per {unit_label} "
                 "(x = cell count, log)\npoint size = # DE genes (union of split A & B) · dotted = "
                 "well-powered cutoff · dashed = tier cutoffs", fontsize=9)
    fig.tight_layout(rect=[0, 0.07, 1, 0.93]); fig.savefig(path, dpi=120); plt.close(fig)


def _repro_scenario(adata, cfg, outdir, cond_positions, pos_C_all, ctrl_obs_all, tprefix):
    """Shared Test-1 / Test-4 within-UNIT reproducibility scenario (mode = 'no_match').

    ``cond_positions`` is an ordered dict ``{unit_label: np.ndarray of global cell positions}``. A *unit*
    is a perturbation/gene for Test 1 and a guide/sgRNA for Test 4 — the design is otherwise identical.
    For each unit with ≥ 2×min_cells_per_group cells, split its cells into halves A/B (batch-stratified)
    and run two independent unit-vs-control DEs (DE_A = A vs ctrl_half_A, DE_B = B vs ctrl_half_B; the
    controls are also split in half, NO 1:1 cell matching) plus the difference-is-null DE_AB = A vs B
    (same unit ⇒ should be ≈null). Repeated for ``test1_n_resamples`` draws per unit. Writes
    ``{tprefix}__reproducibility_no_match.csv``, ``__difference_null_no_match.csv``,
    ``__repro_vs_ncells_no_match.csv`` and ``__repro_vs_ncells_trends_no_match.csv``; returns the
    aggregate dict (with private ``_rhos`` / ``_pc`` / ``_pooled_p`` for plotting) or ``None``."""
    mcg = cfg["min_cells_per_group"]
    nperm = int(cfg.get("test1_n_resamples", min(3, cfg["n_resamples"])))
    repro_rows, null_rows, pooled, skipped = [], [], [], 0
    for ci, (cond, pos_P) in enumerate(cond_positions.items()):
        pos_P = np.asarray(pos_P, int)
        if pos_P.size < 2 * mcg:
            skipped += 1
            continue
        cond_obs = adata.obs.iloc[pos_P]
        for r in range(nperm):
            seed = cfg["seed"] + 1000 + ci * 17 + r
            arm = stratified_split(cond_obs, cfg["block_cols"], seed)
            ai, bi = np.where(arm == "A")[0], np.where(arm == "B")[0]
            carm = stratified_split(ctrl_obs_all, cfg["block_cols"], seed + 7)
            cai, cbi = np.where(carm == "A")[0], np.where(carm == "B")[0]
            if min(ai.size, bi.size, cai.size, cbi.size) < mcg:
                continue
            A_pert, B_pert = pos_P[ai], pos_P[bi]
            A_ctrl, B_ctrl = pos_C_all[cai], pos_C_all[cbi]
            try:
                de_A = _de_two(adata, A_pert, A_ctrl, cfg, "pert", "ctrl")
                de_B = _de_two(adata, B_pert, B_ctrl, cfg, "pert", "ctrl")
                de_AB = _de_two(adata, A_pert, B_pert, cfg, "A", "B")  # difference-is-null
            except Exception as e:  # noqa: BLE001
                log.warning("%s %s rep%d: %s", tprefix, cond, r, e)
                continue
            rep = compare_signatures(de_A, de_B, cfg)
            if rep:
                rep["condition"] = cond
                rep["n_pert_cells"] = int(pos_P.size)       # total cells this unit has
                rep["cells_per_arm"] = int(min(ai.size, bi.size))
                repro_rows.append(rep)
            nm = null_metrics(de_AB, cfg)
            nm["condition"] = cond
            null_rows.append(nm)
            pooled.append(de_AB["p_value"].to_numpy().astype(float))
    if not repro_rows:
        return None
    rdf = pl.DataFrame(repro_rows)
    rdf.write_csv(os.path.join(outdir, "tables", f"{tprefix}__reproducibility_no_match.csv"))
    ndf = pl.DataFrame(null_rows) if null_rows else None
    if ndf is not None:
        ndf.write_csv(os.path.join(outdir, "tables", f"{tprefix}__difference_null_no_match.csv"))
    rhos = rdf["lfc_spearman"].to_numpy().astype(float)
    rhos = rhos[np.isfinite(rhos)]
    # reproducibility vs cell count, one point PER UNIT, for EVERY statistic
    pp_df, trend, trends_df = repro_vs_ncells(rdf, cfg)
    if pp_df is not None:
        pp_df.write_csv(os.path.join(outdir, "tables", f"{tprefix}__repro_vs_ncells_no_match.csv"))
    if trends_df is not None:
        trends_df.write_csv(os.path.join(outdir, "tables", f"{tprefix}__repro_vs_ncells_trends_no_match.csv"))
    rhos_deg = rdf["lfc_spearman_deg"].to_numpy().astype(float)
    rhos_deg = rhos_deg[np.isfinite(rhos_deg)]
    ag = {
        "repro_lfc_spearman": float(np.median(rhos)) if rhos.size else float("nan"),
        "repro_lfc_spearman_deg": float(np.median(rhos_deg)) if rhos_deg.size else float("nan"),
        "repro_jaccard": float(rdf["sig_jaccard"].median()),
        "repro_direction": float(rdf["direction_agreement"].median()),
        "n_strong_rho_gt0.6": int((rhos > REPRO_PASS).sum()),
        "n_moderate_rho_0.3_0.6": int(((rhos >= REPRO_WARN) & (rhos <= REPRO_PASS)).sum()),
        "n_low_rho_lt0.3": int((rhos < REPRO_WARN).sum()),
        # secondary: difference-is-null (A vs B — same unit, should be ≈null)
        "diffnull_lambda_gc": float(ndf["lambda_gc"].mean()) if ndf is not None else float("nan"),
        "diffnull_frac_sig": float(ndf["frac_sig"].mean()) if ndf is not None else float("nan"),
        "diffnull_ks_p_uniform": float(ndf["ks_p_uniform"].mean()) if ndf is not None else float("nan"),
        "n_conditions": int(rdf["condition"].n_unique()), "skipped": skipped,
        "_rhos": rhos, "_pc": pp_df,
        "_pooled_p": np.concatenate(pooled) if pooled else np.array([]),
    }
    ag.update(trend)
    return ag


def _plot_repro_main(ag, png, title_prefix, unit_label):
    """Inline reproducibility figure shared by Test 1 & Test 4: (left) per-unit split-half ρ histogram;
    (right) difference-is-null QQ (A vs B, the same unit ⇒ ≈null) with the exact Beta(i,G−i+1) 95%
    envelope. ``unit_label`` is 'perturbation' (Test 1) or 'guide' (Test 4)."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    axl.hist(ag["_rhos"], bins=20, range=(-1, 1), alpha=0.65, color="#1a3c6e",
             label=f"median ρ={fmt(ag['repro_lfc_spearman'])} (n={ag['n_conditions']})")
    for thr in (REPRO_WARN, REPRO_PASS):
        axl.axvline(thr, ls="--", color="0.5", lw=1)
    axl.set_xlabel(f"split-half Spearman LFC ρ (per {unit_label})")
    axl.set_ylabel(f"{unit_label}s")
    axl.set_title("Reproducibility: DE_A vs DE_B (dashed=tier cutoffs 0.3/0.6)", fontsize=8.5)
    axl.legend(fontsize=7)
    p = np.sort(ag["_pooled_p"][np.isfinite(ag["_pooled_p"])])
    if p.size >= 8:
        p = np.clip(p, 1e-300, 1.0)
        G = p.size
        i = np.arange(1, G + 1)
        exp = -np.log10((i - 0.5) / G)
        obs = -np.log10(p)
        lo = -np.log10(stats.beta.ppf(0.975, i, G - i + 1))
        hi = -np.log10(stats.beta.ppf(0.025, i, G - i + 1))
        if G > 3000:
            k = np.unique(np.linspace(0, G - 1, 3000).astype(int))
            exp, obs, lo, hi = exp[k], obs[k], lo[k], hi[k]
        axr.fill_between(exp, lo, hi, color="0.85")
        m = max(exp.max(), obs.max())
        axr.plot([0, m], [0, m], "r--", lw=1)
        axr.scatter(exp, obs, s=5, color="#1a3c6e")
        axr.set_title(f"Difference-is-null QQ: A vs B ({unit_label})\nλ_GC={fmt(ag['diffnull_lambda_gc'])}",
                      fontsize=8.5)
    else:
        axr.set_title("Difference-is-null QQ (insufficient genes)", fontsize=8.5)
    axr.set_xlabel("expected -log10(p)")
    axr.set_ylabel("observed -log10(p)")
    fig.suptitle(f"{title_prefix} — within-{unit_label} reproducibility + difference-is-null", fontsize=11)
    fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)


def _repro_verdict_from_metrics(m, cfg, unit_label):
    """PURE function of cached/flattened metrics → (verdict, reason, flags) for Test 1 / Test 4. Reads
    the ``no_match__*`` flattened keys (live + cached), falling back to ``median_*`` for old summaries.
    Shared by the live path (_finish_repro) and --report-only verdict re-derivation (verdict_test_4)."""
    def g(k):
        v = m.get(f"no_match__{k}")
        return v if v is not None else m.get(k)
    rho = g("repro_lfc_spearman")
    rho = m.get("median_lfc_spearman", float("nan")) if rho is None else rho
    jac = g("repro_jaccard")
    jac = m.get("median_jaccard", float("nan")) if jac is None else jac
    dirn = g("repro_direction")
    dirn = m.get("median_direction", float("nan")) if dirn is None else dirn
    verdict = verdict_reproducibility(rho)
    tier = {"PASS": "strong", "WARN": "moderate", "FAIL": "low", "SKIP": "n/a"}[verdict]
    flags = []
    wp, up = g("repro_rho_wellpowered"), g("repro_rho_underpowered")
    wpn, upn = g("n_wellpowered"), g("n_underpowered")
    flags.append(
        f"reproducibility vs cell count: well-powered (≥{g('wellpowered_min_cells')} cells, "
        f"n={wpn}) median ρ={fmt(wp)} vs under-powered (n={upn}) median ρ={fmt(up)}; "
        f"Spearman(cell count, ρ)={fmt(g('rho_vs_ncells_spearman'))}. "
        + ("Well-powered ρ ≈ overall ⇒ low reproducibility is a GENUINE method limitation, not undersampling."
           if (isinstance(wp, float) and np.isfinite(wp) and verdict_reproducibility(wp) == verdict)
           else f"Well-powered {unit_label}s reproduce better ⇒ part of the low score is an undersampling artifact."))
    flags.append(f"per-{unit_label} reproducibility tiers: {g('n_strong_rho_gt0.6')} strong / "
                 f"{g('n_moderate_rho_0.3_0.6')} moderate / {g('n_low_rho_lt0.3')} low "
                 f"(of {g('n_conditions')})")
    flags.append("thresholds — " + REPRO_LEGEND)
    flags.append(f"secondary difference-is-null (A vs B, same {unit_label} ⇒ should be ≈null): "
                 f"λ_GC={fmt(g('diffnull_lambda_gc'))}, frac_sig={fmt(g('diffnull_frac_sig'))}, "
                 f"ks_p_uniform={fmt(g('diffnull_ks_p_uniform'))}")
    reason = (f"{verdict}: median split-half LFC ρ={fmt(rho)} ({tier} reproducibility; "
              f"PASS>{REPRO_PASS}/WARN≥{REPRO_WARN}/FAIL<{REPRO_WARN}); Jaccard={fmt(jac)}, "
              f"direction={fmt(dirn)}; (2°) difference-is-null λ_GC={fmt(g('diffnull_lambda_gc'))}, "
              f"frac_sig={fmt(g('diffnull_frac_sig'))}")
    return verdict, reason, flags


def _finish_repro(out, name, title, cfg, plot_png, unit_label):
    """Build the TestResult (flattened {mode}__ metrics + shared verdict/flags/reason) for Test 1/Test 4
    from the scenario aggregate(s) in ``out`` (here only the 'no_match' mode)."""
    a = out["no_match"]
    metrics = {"scenario_primary": "no_match"}
    for mode, ag in out.items():
        for k, v in ag.items():
            if not k.startswith("_"):
                metrics[f"{mode}__{k}"] = v
    # also expose the canonical median_* keys so verdict_test_4 / --report-only can re-derive cheaply
    metrics["median_lfc_spearman"] = a["repro_lfc_spearman"]
    metrics["median_jaccard"] = a["repro_jaccard"]
    metrics["median_direction"] = a["repro_direction"]
    metrics["n_guides" if name == "test_4" else "n_conditions"] = a["n_conditions"]
    verdict, reason, flags = _repro_verdict_from_metrics(metrics, cfg, unit_label)
    return TestResult(name, title, verdict, metrics=metrics, flags=flags, pvalues={},
                      plot=plot_png, reason=reason)


def test_1(adata, cfg, outdir):
    """Test 1 — within-condition REPRODUCIBILITY (NOT a uniformity null; that is Test 2).

    For each perturbation, split its cells into halves A and B (batch-stratified) and run two
    independent perturbation-vs-control DEs (DE_A, DE_B) — each carries the REAL perturbation signal
    (not uniform). The test asks whether the two half-signatures AGREE: DE_A ≈ DE_B. Controls are split
    into two halves: DE_A = A vs ctrl_half_A, DE_B = B vs ctrl_half_B (cell-eval DE, NO 1:1 cell
    matching) — each DE uses half of the total control cells.

    Reproducibility = median over perturbations of Spearman(LFC_A, LFC_B) (+ DEG Jaccard, direction).
    Verdict tiers (on median ρ): PASS > 0.6, WARN 0.3–0.6, FAIL < 0.3. Also reported per perturbation
    against cell count (separates a low-reproducibility method from undersampled conditions)."""
    pc, mcg = cfg["pert_col"], cfg["min_cells_per_group"]
    ctrl_lab = cfg["control_pert"]
    perts = perts_in_use(adata, cfg)
    labels = adata.obs[pc].astype(str).to_numpy()
    pos_C_all = np.where(labels == ctrl_lab)[0]
    ctrl_obs_all = adata.obs.iloc[pos_C_all]
    cond_positions = {pert: np.where(labels == pert)[0] for pert in perts}
    ag = _repro_scenario(adata, cfg, outdir, cond_positions, pos_C_all, ctrl_obs_all, "test_1")
    if ag is None:
        return TestResult("test_1", "Within-Condition Reproducibility", "SKIP",
                          flags=["no condition had enough cells for an A/B split"],
                          reason="SKIP: insufficient cells")
    out = {"no_match": ag}
    _plot_repro_main(ag, os.path.join(outdir, "plots", "test_1_reproducibility.png"), "Test 1", "perturbation")
    plot_repro_vs_ncells(out, os.path.join(outdir, "plots", "test_1_undersampling.png"), cfg,
                         title_prefix="Test 1", unit_label="perturbation")
    return _finish_repro(out, "test_1", "Within-Condition Reproducibility", cfg,
                         "plots/test_1_reproducibility.png", "perturbation")


def test_2(adata, cfg, outdir):
    """Control-control split null. Both arms are control cells (same population ⇒ a true global null),
    split into A/B by a **stratified** random split balanced within `block_cols` (batch). A calibrated
    pipeline should produce Uniform[0,1] p-values (λ_GC≈1, frac_sig≈α). (No 1:1 control–control
    'matching' mode: with a single population a stratified split already balances batch/depth, so
    matching controls-to-controls is uninformative — that batch-controlled design belongs to Test 1,
    where two *different* populations are compared.)"""
    pc, mcg = cfg["pert_col"], cfg["min_cells_per_group"]
    ctrl = adata[adata.obs[pc].astype(str) == cfg["control_pert"]]
    if ctrl.n_obs < 2 * mcg:
        return TestResult("test_2", "Control-Control Split Null", "SKIP", flags=["too few control cells"],
                          reason="SKIP: too few control cells")
    rows, pooled, design = [], [], {}
    for r in range(cfg["n_resamples"]):
        s = ctrl.copy()
        s.obs["_arm"] = stratified_split(ctrl.obs, cfg["block_cols"], cfg["seed"] + 100 + r)
        if not design:  # capture the sample design once (deterministic across splits)
            na = int((s.obs["_arm"].astype(str) == "A").sum())
            nb = int((s.obs["_arm"].astype(str) == "B").sum())
            design = {"design_n_control_cells": int(ctrl.n_obs), "design_cells_per_arm": int(min(na, nb))}
            if cfg["de_method"] == "pydeseq2" and cfg.get("replicate_col") in s.obs.columns:
                rc = cfg["replicate_col"]
                design["design_n_pseudobulk_samples"] = int(
                    s.obs.groupby([rc, "_arm"], observed=True).ngroups)
                design["design_n_replicate_units"] = int(s.obs[rc].nunique())
        try:
            de = run_de(s, cfg, groupby="_arm", reference="B")
        except Exception as e:  # noqa: BLE001
            log.warning("test2 rep%d: %s", r, e)
            continue
        m = null_metrics(de, cfg)
        m["rep"] = r
        rows.append(m)
        pooled.append(de["p_value"].to_numpy().astype(float))
    if not rows:
        return TestResult("test_2", "Control-Control Split Null", "SKIP", flags=["no valid control split"],
                          reason="SKIP: no valid control split")
    df = pl.DataFrame(rows)
    df.write_csv(os.path.join(outdir, "tables", "test_2__per_split.csv"))
    agg = {k: float(df[k].mean()) for k in ("frac_sig", "mean_lfc", "mean_abs_lfc", "lambda_gc", "ks_p_uniform")}
    agg["lambda_gc_sd"] = float(df["lambda_gc"].std()) if df.height > 1 else 0.0
    agg["design_n_splits"] = int(df.height)
    agg.update(design)
    qq_plot(np.concatenate(pooled), os.path.join(outdir, "plots", "test_2_qq.png"),
            "Test 2 — control-control null (stratified split)")
    verdict, flags = verdict_from_null(agg, cfg)
    if np.isfinite(agg["lambda_gc_sd"]) and agg["lambda_gc_sd"] > 0.05:
        flags.append(f"split-to-split SD(λ_GC)={fmt(agg['lambda_gc_sd'])} > 0.05 (residual confounding?)")
        if verdict == "PASS":
            verdict = "WARN"
    reason = _null_reason(verdict, agg)
    return TestResult("test_2", "Control-Control Split Null", verdict, metrics=agg, flags=flags,
                      pvalues={"ks_p_uniform_mean": agg["ks_p_uniform"], "lambda_gc": agg["lambda_gc"]},
                      plot="plots/test_2_qq.png", reason=reason)


def _null_reason(verdict, agg):
    lam, frac, ksp = agg.get("lambda_gc"), agg.get("frac_sig"), agg.get("ks_p_uniform")
    if verdict == "PASS":
        return f"PASS: well-calibrated (λ_GC={fmt(lam)}, frac_sig={fmt(frac)}, p-values ~Uniform)"
    if verdict == "FAIL":
        return f"FAIL: anti-conservative (λ_GC={fmt(lam)}, frac_sig={fmt(frac)})"
    # WARN
    if np.isfinite(lam) and lam < 0.9:
        return (f"WARN: deflated/under-powered null (λ_GC={fmt(lam)}<0.9, ks_p_uniform={fmt(ksp)}); "
                "no false positives but p-values are not calibrated")
    if np.isfinite(ksp) and ksp < 0.05:
        return f"WARN: p-values not Uniform[0,1] (ks_p_uniform={fmt(ksp)}); frac_sig={fmt(frac)}"
    return f"WARN: mild inflation (λ_GC={fmt(lam)}, frac_sig={fmt(frac)})"


def _signal_metric(de: pl.DataFrame, cfg: dict) -> float:
    lfc = de["log2_fold_change"].to_numpy().astype(float)
    fdr = de["fdr"].to_numpy().astype(float)
    sig = _sig_mask(lfc, fdr, cfg)
    tgt = de["target"].to_numpy().astype(str)
    fr = []
    for t in np.unique(tgt):
        msk = tgt == t
        fr.append(sig[msk].sum() / max(msk.sum(), 1))
    return float(np.mean(fr)) if fr else float("nan")


def _t3_per_pert_pvalues(de: pl.DataFrame, ncells: dict[str, int], cfg: dict, kind: str) -> list[dict]:
    """Per-perturbation p-value summaries from a DE frame (real or shuffled). Returns one row per
    `target` (perturbation) with cell count and uniformity/signal summaries. `kind` ∈ {real, shuffled}."""
    tgt = de["target"].to_numpy().astype(str)
    pv = de["p_value"].to_numpy().astype(float)
    lfc = de["log2_fold_change"].to_numpy().astype(float)
    fdr = de["fdr"].to_numpy().astype(float)
    rows = []
    for t in np.unique(tgt):
        if t == cfg["control_pert"]:
            continue
        m = tgt == t
        p = pv[m]
        p = p[np.isfinite(p)]
        if p.size == 0:
            continue
        sig = _sig_mask(lfc[m], fdr[m], cfg)
        rows.append({
            "perturbation": t,
            "kind": kind,
            "n_cells": int(ncells.get(t, 0)),
            "n_genes": int(p.size),
            "frac_sig": float(np.nansum(sig) / max(int(m.sum()), 1)),
            "frac_p_lt_05": float(np.mean(p < 0.05)),
            "lambda_gc": lambda_gc(p),
            "ks_p_uniform": float(stats.kstest(p, "uniform").pvalue) if p.size >= 8 else float("nan"),
        })
    return rows


def _t3_subsample_pooled(pvals: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Deterministically subsample a pooled p-value vector to <= cap entries (for ECDF/QQ memory)."""
    p = pvals[np.isfinite(pvals)]
    if p.size <= cap:
        return p
    idx = np.random.default_rng(seed).choice(p.size, size=cap, replace=False)
    return p[np.sort(idx)]


def _ecdf(p: np.ndarray):
    s = np.sort(p)
    return s, np.arange(1, s.size + 1) / s.size


def plot_test3_pvalue_diagnostics(real_pooled, shuf_pooled, per_pert_df, strata_pooled, path, cfg):
    """Diagnostic for Test 3: compare p-value distributions of UNSHUFFLED (real) DE vs SHUFFLED
    (label-permuted) DE, pooled over all perturbations × tested genes, then stratified by each
    perturbation's cell count.

    Panels:
      (A) overlaid p-value ECDF — real should bow ABOVE the diagonal (small-p excess = signal);
          shuffled should track the diagonal (Uniform = calibrated null).
      (B) QQ vs Uniform (-log10) — real points rise above the y=x line (signal); shuffled hug it.
      (C) per-perturbation frac(p<0.05) real vs shuffled scattered against cell count (log-x).
      (D) ECDF faceted by cell-count stratum (real vs shuffled per stratum) — does the shuffled null
          stay Uniform, and the real small-p excess hold, for small- vs large-cell-count perturbations?
    """
    real_pooled = np.asarray(real_pooled, float)
    shuf_pooled = np.asarray(shuf_pooled, float)
    real_pooled = np.clip(real_pooled[np.isfinite(real_pooled)], 1e-300, 1.0)
    shuf_pooled = np.clip(shuf_pooled[np.isfinite(shuf_pooled)], 1e-300, 1.0)
    if real_pooled.size < 8 or shuf_pooled.size < 8:
        return False
    C_REAL, C_SHUF = "#1a3c6e", "#c0392b"
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.4))
    axA, axB, axC, axD = axes.ravel()

    # (A) overlaid ECDF
    for p, lab, col in ((real_pooled, "real (unshuffled)", C_REAL), (shuf_pooled, "shuffled", C_SHUF)):
        xs, ys = _ecdf(p)
        if xs.size > 4000:
            k = np.unique(np.linspace(0, xs.size - 1, 4000).astype(int)); xs, ys = xs[k], ys[k]
        axA.plot(xs, ys, color=col, lw=1.6,
                 label=f"{lab} (frac p<0.05={fmt(float(np.mean(p < 0.05)))}, λ_GC={fmt(lambda_gc(p))})")
    axA.plot([0, 1], [0, 1], "k--", lw=1, label="Uniform")
    axA.set_xlabel("p-value"); axA.set_ylabel("ECDF")
    axA.set_title("(A) Pooled p-value ECDF — real vs shuffled\n(real bows above diagonal = signal; "
                  "shuffled on diagonal = calibrated null)", fontsize=8.5)
    axA.legend(fontsize=7, loc="lower right")

    # (B) QQ vs Uniform (-log10)
    for p, lab, col in ((real_pooled, "real", C_REAL), (shuf_pooled, "shuffled", C_SHUF)):
        sp = np.sort(p); G = sp.size
        i = np.arange(1, G + 1)
        exp = -np.log10((i - 0.5) / G); obs = -np.log10(sp)
        if G > 3000:
            k = np.unique(np.linspace(0, G - 1, 3000).astype(int)); exp, obs = exp[k], obs[k]
        axB.scatter(exp, obs, s=5, color=col, label=lab)
    m = float(max(-np.log10(0.5 / real_pooled.size), -np.log10(0.5 / shuf_pooled.size)))
    axB.plot([0, m], [0, m], "k--", lw=1)
    axB.set_xlabel("expected -log10(p) [Uniform]"); axB.set_ylabel("observed -log10(p)")
    axB.set_title("(B) QQ vs Uniform — pooled\n(real above line = small-p excess; shuffled on line)",
                  fontsize=8.5)
    axB.legend(fontsize=7, loc="upper left")

    # (C) per-perturbation frac(p<0.05) vs cell count
    if per_pert_df is not None and per_pert_df.height:
        for kind, col in (("real", C_REAL), ("shuffled", C_SHUF)):
            d = per_pert_df.filter(pl.col("kind") == kind)
            if d.height:
                axC.scatter(d["n_cells"].to_numpy(), d["frac_p_lt_05"].to_numpy(),
                            s=26, alpha=0.7, color=col, edgecolor="white", linewidth=0.4,
                            label=f"{kind} (median={fmt(float(d['frac_p_lt_05'].median()))})")
        axC.axhline(0.05, ls=":", color="0.5", lw=1)
        axC.text(axC.get_xlim()[1], 0.05, " 0.05 (Uniform)", va="center", fontsize=7, color="0.4")
        axC.set_xscale("log")
        axC.set_xlabel("perturbation cell count (log scale)")
        axC.set_ylabel("fraction of genes with p < 0.05")
        axC.set_title("(C) Per-perturbation small-p fraction vs cell count\n"
                      "(real >> shuffled = signal; shuffled flat near 0.05 = calibrated)", fontsize=8.5)
        axC.legend(fontsize=7, loc="upper left")

    # (D) ECDF faceted by cell-count stratum
    if strata_pooled:
        strata = sorted(strata_pooled.keys())
        cmap_real = plt.cm.Blues(np.linspace(0.5, 0.95, len(strata)))
        cmap_shuf = plt.cm.Reds(np.linspace(0.5, 0.95, len(strata)))
        for si, sname in enumerate(strata):
            d = strata_pooled[sname]
            for kind, cm in (("real", cmap_real), ("shuffled", cmap_shuf)):
                p = np.clip(np.asarray(d.get(kind, []), float), 1e-300, 1.0)
                p = p[np.isfinite(p)]
                if p.size < 8:
                    continue
                xs, ys = _ecdf(p)
                if xs.size > 3000:
                    k = np.unique(np.linspace(0, xs.size - 1, 3000).astype(int)); xs, ys = xs[k], ys[k]
                axD.plot(xs, ys, color=cm[si], lw=1.4, ls="-" if kind == "real" else "--",
                         label=f"{sname} {kind}")
        axD.plot([0, 1], [0, 1], "k:", lw=1)
        axD.set_xlabel("p-value"); axD.set_ylabel("ECDF")
        axD.set_title("(D) p-value ECDF by cell-count stratum\n(solid=real, dashed=shuffled; does the "
                      "null stay Uniform for small-cell-count perts?)", fontsize=8.5)
        axD.legend(fontsize=6.5, loc="lower right", ncol=2)
    fig.suptitle("Test 3 — label-permutation p-value diagnostics (real vs shuffled), "
                 "stratified by perturbation cell count", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=115); plt.close(fig)
    return True


def test_3(adata, cfg, outdir, de_true=None):
    pc = cfg["pert_col"]
    perts = perts_in_use(adata, cfg)
    keep = adata[adata.obs[pc].astype(str).isin(perts + [cfg["control_pert"]])].copy()
    if de_true is None:
        de_true = run_de(keep, cfg, groupby=pc, reference=cfg["control_pert"])
    true_metric = _signal_metric(de_true, cfg)
    # per-perturbation cell counts (from the cells actually used here) for cell-count stratification
    ncells = keep.obs[pc].astype(str).value_counts().to_dict()
    rng = np.random.default_rng(cfg["seed"] + 3)
    blocks = [c for c in cfg["block_cols"] if c in keep.obs.columns]
    perm_metrics = []
    # p-value diagnostics: per-pert summaries (real once + every shuffle) and pooled p-values
    pp_rows = _t3_per_pert_pvalues(de_true, ncells, cfg, "real")
    real_pooled = de_true["p_value"].to_numpy().astype(float)
    _t3_shuf_targets = []  # list of (target_array, pvalue_array) per successful permutation
    labels0 = keep.obs[pc].astype(str).to_numpy()
    is_ctrl = labels0 == cfg["control_pert"]
    for r in range(cfg["n_resamples"]):
        lab = labels0.copy()
        if blocks:
            grp = keep.obs.groupby(blocks, observed=True, sort=False).indices.values()
        else:
            grp = [np.arange(len(lab))]
        for idx in grp:
            idx = np.asarray(idx, dtype=int)
            nc = idx[~is_ctrl[idx]]
            if nc.size > 1:
                perm = nc.copy()
                rng.shuffle(perm)
                lab[nc] = lab[perm]
        s = keep.copy()
        s.obs["_perm"] = lab
        try:
            de_p = run_de(s, cfg, groupby="_perm", reference=cfg["control_pert"])
            perm_metrics.append(_signal_metric(de_p, cfg))
            # the shuffled "perturbation" label is arbitrary; tie each shuffled DE row's cell count
            # to the count of the (shuffled) label group so strata pool comparable group sizes
            pp_rows += _t3_per_pert_pvalues(de_p, ncells, cfg, "shuffled")
            _t3_shuf_targets.append((de_p["target"].to_numpy().astype(str),
                                     de_p["p_value"].to_numpy().astype(float)))
        except Exception as e:  # noqa: BLE001
            log.warning("test3 perm%d: %s", r, e)
    if len(perm_metrics) < 2:
        return TestResult("test_3", "Label Permutation Null", "SKIP", flags=["not enough permutations"],
                          reason="SKIP: <2 permutations succeeded"), de_true
    pm = np.array(perm_metrics, dtype=float)
    sep = (true_metric - pm.mean()) / pm.std(ddof=1) if pm.std(ddof=1) > 0 else float("inf")
    perm_p = float((pm >= true_metric).mean())
    pl.DataFrame({"perm_signal": pm}).write_csv(os.path.join(outdir, "tables", "test_3__permutations.csv"))

    # ---- cell-count-stratified p-value diagnostics (real vs shuffled) ----
    diag_metrics, diag_flags = {}, []
    plot_rel = None
    pp_df = pl.DataFrame(pp_rows) if pp_rows else None
    if pp_df is not None and pp_df.height:
        # assign each perturbation to a cell-count stratum (tertiles by default) from the REAL rows'
        # distinct perturbation cell counts, so the same edges apply to real and shuffled rows
        nstr = max(2, int(cfg.get("test3_n_cellcount_strata", 3)))
        real_nc = (pp_df.filter(pl.col("kind") == "real")
                        .select(["perturbation", "n_cells"]).unique())
        ncv = real_nc["n_cells"].to_numpy().astype(float)
        if ncv.size >= nstr:
            qs = np.quantile(ncv, np.linspace(0, 1, nstr + 1))
            def _stratum(n):
                k = int(np.clip(np.searchsorted(qs, n, side="right") - 1, 0, nstr - 1))
                lo, hi = qs[k], qs[k + 1]
                return f"S{k + 1} [{int(round(lo)):,}-{int(round(hi)):,}]"
            pp_df = pp_df.with_columns(
                pl.col("n_cells").map_elements(_stratum, return_dtype=pl.Utf8).alias("cellcount_stratum"))
        else:
            pp_df = pp_df.with_columns(pl.lit("all").alias("cellcount_stratum"))
        pp_df.write_csv(os.path.join(outdir, "tables", "test_3__pvalues_by_cellcount.csv"))

        # pooled p-values for ECDF/QQ, deterministically subsampled for memory
        cap = int(cfg.get("test3_pooled_pval_subsample", 40000))
        real_p = _t3_subsample_pooled(real_pooled, cap, cfg["seed"] + 31)
        shuf_all = (np.concatenate([c[1] for c in _t3_shuf_targets]) if _t3_shuf_targets else np.array([]))
        shuf_p = _t3_subsample_pooled(shuf_all, cap, cfg["seed"] + 32)
        pl.DataFrame({"kind": ["real"] * real_p.size + ["shuffled"] * shuf_p.size,
                      "p_value": np.concatenate([real_p, shuf_p]) if (real_p.size or shuf_p.size)
                      else np.array([])}).write_csv(
            os.path.join(outdir, "tables", "test_3__pooled_pvalues.csv"))

        # per-stratum pooled p-values (subsampled per stratum/kind) for the faceted ECDF. A perturbation
        # maps to a cell-count stratum via the REAL per-pert cell counts; the same map is applied to the
        # shuffled DE rows by (shuffled) label group, so each stratum pools comparable group sizes.
        strata_pooled = {}
        if "cellcount_stratum" in pp_df.columns:
            pert2str = dict(zip(pp_df.filter(pl.col("kind") == "real")["perturbation"].to_list(),
                                pp_df.filter(pl.col("kind") == "real")["cellcount_stratum"].to_list()))
            snames = sorted(set(pert2str.values()))
            per_cap = max(2000, cap // max(len(snames), 1))
            real_tgt = de_true["target"].to_numpy().astype(str)
            real_pv = de_true["p_value"].to_numpy().astype(float)
            real_str = np.array([pert2str.get(t, None) for t in real_tgt], dtype=object)
            # pool shuffled p-values by stratum across all permutation chunks (label group sizes are the
            # same control-vs-pert group counts, so the same n_cells/stratum mapping applies)
            shuf_tgt = (np.concatenate([c[0] for c in _t3_shuf_targets]) if _t3_shuf_targets else np.array([], str))
            shuf_pv = (np.concatenate([c[1] for c in _t3_shuf_targets]) if _t3_shuf_targets else np.array([]))
            shuf_str = np.array([pert2str.get(t, None) for t in shuf_tgt], dtype=object)
            for si, sname in enumerate(snames):
                strata_pooled[sname] = {
                    "real": _t3_subsample_pooled(real_pv[real_str == sname], per_cap, cfg["seed"] + 33 + si),
                    "shuffled": _t3_subsample_pooled(shuf_pv[shuf_str == sname], per_cap,
                                                     cfg["seed"] + 50 + si),
                }
        else:
            strata_pooled = {}

        plot_path = os.path.join(outdir, "plots", "test_3_pvalue_diagnostics.png")
        if plot_test3_pvalue_diagnostics(real_p, shuf_p, pp_df, strata_pooled, plot_path, cfg):
            plot_rel = "plots/test_3_pvalue_diagnostics.png"

        rr = pp_df.filter(pl.col("kind") == "real")
        ss = pp_df.filter(pl.col("kind") == "shuffled")
        diag_metrics = {
            "real_frac_p_lt_05": float(rr["frac_p_lt_05"].median()) if rr.height else float("nan"),
            "shuffled_frac_p_lt_05": float(ss["frac_p_lt_05"].median()) if ss.height else float("nan"),
            "real_frac_sig": float(rr["frac_sig"].median()) if rr.height else float("nan"),
            "shuffled_frac_sig": float(ss["frac_sig"].median()) if ss.height else float("nan"),
            "real_lambda_gc_pooled": lambda_gc(real_p),
            "shuffled_lambda_gc_pooled": lambda_gc(shuf_p),
            "shuffled_ks_p_uniform_median": float(ss["ks_p_uniform"].median()) if ss.height else float("nan"),
        }
        diag_flags.append(
            f"p-value diagnostic (pooled over perturbations × genes): real frac(p<0.05)="
            f"{fmt(diag_metrics['real_frac_p_lt_05'])} / λ_GC={fmt(diag_metrics['real_lambda_gc_pooled'])} "
            f"vs shuffled frac(p<0.05)={fmt(diag_metrics['shuffled_frac_p_lt_05'])} / "
            f"λ_GC={fmt(diag_metrics['shuffled_lambda_gc_pooled'])} "
            "(real >> shuffled small-p excess = signal; shuffled ≈ Uniform/0.05 = calibrated null). "
            "Stratified by cell count in plots/test_3_pvalue_diagnostics.png + "
            "tables/test_3__pvalues_by_cellcount.csv.")

    if not np.isfinite(sep):
        verdict = "PASS" if true_metric > pm.max() else "WARN"
    elif sep > 2:
        verdict = "PASS"
    elif sep > 1:
        verdict = "WARN"
    else:
        verdict = "FAIL"
    reason = (f"{verdict}: true signal {fmt(true_metric)} vs shuffled {fmt(float(pm.mean()))} "
              f"(separation z={fmt(float(sep))}, perm_p={fmt(perm_p)})")
    metrics = {"true_signal": true_metric, "perm_mean": float(pm.mean()),
               "perm_sd": float(pm.std(ddof=1)), "separation_z": float(sep)}
    metrics.update(diag_metrics)
    return TestResult("test_3", "Label Permutation Null", verdict,
                      metrics=metrics, flags=diag_flags,
                      pvalues={"perm_p": perm_p, "separation_z": float(sep)},
                      plot=plot_rel, reason=reason), de_true


def test_4(adata, cfg, outdir):
    """Test 4 — within-GUIDE reproducibility: Test 1 run at the sgRNA (guide) level.

    Identical design to Test 1 but the *unit* is an individual guide rather than a perturbation/gene.
    For each guide with ≥ 2×min_cells_per_group cells, split the guide's cells into halves A/B
    (batch-stratified), split the controls in half, and run DE_A = A vs ctrl_half_A, DE_B = B vs
    ctrl_half_B (NO 1:1 cell matching) plus the difference-is-null DE_AB = A vs B, repeated for
    ``test1_n_resamples`` draws. Reproducibility = median over guides of Spearman(LFC_A, LFC_B) (+ DEG
    Jaccard, direction); same tiers as Test 1 (PASS > 0.6 / WARN 0.3–0.6 / FAIL < 0.3). Also reported
    per guide against cell count, and as a difference-is-null QQ — the same two plots as Test 1, at the
    guide level. This is the empirical reproducibility ceiling for any per-guide downstream metric."""
    pc, sg, mcg = cfg["pert_col"], cfg.get("sgrna_col"), cfg["min_cells_per_group"]
    if not sg:
        return TestResult("test_4", "Same-sgRNA Split Reproducibility (guide-level Test 1)", "SKIP",
                          flags=["no sgrna_col"], reason="SKIP: no sgrna_col")
    perts = perts_in_use(adata, cfg)
    ctrl_lab = cfg["control_pert"]
    tg = adata.obs[pc].astype(str).to_numpy()
    sgv = adata.obs[sg].astype(str).to_numpy()
    in_use = np.isin(tg, list(perts))                    # guides of perturbations under test (excl. control)
    pos_C_all = np.where(tg == ctrl_lab)[0]
    ctrl_obs_all = adata.obs.iloc[pos_C_all]
    guides = sorted(set(sgv[in_use]))
    # one unit per guide; only keep guides with enough cells for an A/B split (others are SKIPped inside)
    cond_positions = {g: np.where((sgv == g) & in_use)[0] for g in guides}
    cond_positions = {g: p for g, p in cond_positions.items() if p.size >= 2 * mcg}
    if not cond_positions:
        return TestResult("test_4", "Same-sgRNA Split Reproducibility (guide-level Test 1)", "SKIP",
                          flags=[f"no guide had ≥2×min_cells_per_group ({2 * mcg}) cells"],
                          reason="SKIP: no guide had enough cells for an A/B split")
    ag = _repro_scenario(adata, cfg, outdir, cond_positions, pos_C_all, ctrl_obs_all, "test_4")
    if ag is None:
        return TestResult("test_4", "Same-sgRNA Split Reproducibility (guide-level Test 1)", "SKIP",
                          flags=["no guide produced a usable A/B split"],
                          reason="SKIP: no guide produced a usable A/B split")
    out = {"no_match": ag}
    _plot_repro_main(ag, os.path.join(outdir, "plots", "test_4_reproducibility.png"), "Test 4", "guide")
    plot_repro_vs_ncells(out, os.path.join(outdir, "plots", "test_4_undersampling.png"), cfg,
                         title_prefix="Test 4", unit_label="guide")
    return _finish_repro(out, "test_4", "Same-sgRNA Split Reproducibility (guide-level Test 1)", cfg,
                         "plots/test_4_reproducibility.png", "guide")


# Test 5 reproducibility metrics carried for the same-gene-vs-background separation (display name,
# legacy/short label used in metric keys, value range for the distribution plot).
TEST5_METRICS = [
    ("lfc_spearman",        "Spearman LFC ρ — all genes", "rho",       (-1.0, 1.0)),
    ("lfc_spearman_deg",    "Spearman LFC ρ — DE genes",  "rho_deg",   (-1.0, 1.0)),
    ("sig_jaccard",         "DEG-set Jaccard",            "jaccard",   (0.0, 1.0)),
    ("direction_agreement", "Direction agreement",        "direction", (0.0, 1.0)),
]


def _separation(same, bg):
    """(mean_same, mean_bg, sd_bg, z, emp_p, mwu_p, auc) for one metric. ``z`` = (mean_same − mean_bg) /
    SD(background); ``emp_p`` = fraction of background pairs ≥ the same-gene mean (one-sided); ``mwu_p``
    = Mann-Whitney U p that same-gene concordance > background; ``auc`` = the Mann-Whitney common-language
    effect size U/(n_same·n_bg) = P(a random same-gene pair is more concordant than a random background
    pair), 0.5 = no separation. AUC is the rank-based effect size that pairs with mwu_p (robust to the
    wide/skewed background that deflates z, and not inflated by sample size like a raw p)."""
    same = np.asarray([v for v in np.asarray(same, float) if np.isfinite(v)], float)
    bg = np.asarray([v for v in np.asarray(bg, float) if np.isfinite(v)], float)
    if same.size == 0 or bg.size == 0:
        return (float("nan"),) * 7
    ms, mb = float(same.mean()), float(bg.mean())
    sd = float(bg.std(ddof=1)) if bg.size > 1 else 0.0
    z = (ms - mb) / sd if sd > 0 else float("inf")
    emp_p = float((bg >= ms).mean())
    try:
        U = stats.mannwhitneyu(same, bg, alternative="greater")
        mwu_p = float(U.pvalue)
        auc = float(U.statistic / (same.size * bg.size))
    except Exception:  # noqa: BLE001
        mwu_p, auc = float("nan"), float("nan")
    return ms, mb, sd, z, emp_p, mwu_p, auc


def _plot_test5_separation(same_df, bg_df, png):
    """Overlaid same-gene vs unrelated (background) guide-pair distributions for each reproducibility
    metric, with mean lines + separation z — the Test-5 analog of the Test-1/4 reproducibility figure."""
    panels = [(c, t, xr) for c, t, _, xr in TEST5_METRICS]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.7), squeeze=False)
    for ax, (col, title, xr) in zip(axes[0], panels):
        s = same_df[col].to_numpy().astype(float); s = s[np.isfinite(s)]
        b = bg_df[col].to_numpy().astype(float); b = b[np.isfinite(b)]
        bins = np.linspace(xr[0], xr[1], 26)
        if b.size:
            ax.hist(b, bins=bins, density=True, alpha=0.5, color="#999999", label=f"background (n={b.size})")
        if s.size:
            ax.hist(s, bins=bins, density=True, alpha=0.6, color="#1a3c6e", label=f"same-gene (n={s.size})")
        mb = float(b.mean()) if b.size else float("nan")
        ms = float(s.mean()) if s.size else float("nan")
        sd = float(b.std(ddof=1)) if b.size > 1 else 0.0
        z = (ms - mb) / sd if sd > 0 else float("nan")
        if b.size:
            ax.axvline(mb, ls="--", color="#666666", lw=1)
        if s.size:
            ax.axvline(ms, ls="--", color="#1a3c6e", lw=1.3)
        ax.set_title(f"{title}\nsame={fmt(ms)} · bg={fmt(mb)} · sep z={fmt(z)}", fontsize=8.3)
        ax.set_xlabel("pair concordance")
        ax.set_ylabel("density (guide pairs)")
        ax.set_xlim(*xr); ax.legend(fontsize=6.5)
    fig.suptitle("Test 5 — same-gene vs unrelated guide-pair concordance "
                 "(separation = same-gene signal above the background distribution)", fontsize=10)
    fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)


def _plot_test5_vs_ncells(same_df, bg_df, prim_col, png):
    """Same-gene pair concordance (PRIMARY metric) vs the pair's cells-per-guide (min of the two guides),
    with the background mean ± SD band — the Test-5 analog of the Test-4 undersampling plot: does the
    (modest) same-gene separation strengthen with more cells, i.e. is it real-but-undersampled?"""
    if "min_cells" not in same_df.columns:
        return
    mc = same_df["min_cells"].to_numpy().astype(float)
    v = same_df[prim_col].to_numpy().astype(float)
    ok = np.isfinite(mc) & np.isfinite(v)
    if ok.sum() < 3:
        return
    mc, v = mc[ok], v[ok]
    b = bg_df[prim_col].to_numpy().astype(float); b = b[np.isfinite(b)]
    rho = float(stats.spearmanr(mc, v).correlation)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if b.size:
        bm, bs = float(b.mean()), float(b.std(ddof=1)) if b.size > 1 else 0.0
        ax.axhspan(bm - bs, bm + bs, color="#cccccc", alpha=0.6, label=f"background mean±SD ({fmt(bm)})")
        ax.axhline(bm, ls="--", color="#777777", lw=1)
    ax.axhline(float(v.mean()), ls="--", color="#1a3c6e", lw=1.2, label=f"same-gene mean ({fmt(float(v.mean()))})")
    ax.scatter(mc, v, s=26, alpha=0.6, color="#1a3c6e", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("cells per guide in the pair (min of the two guides, log)")
    ax.set_ylabel("same-gene pair concordance (primary: DEG LFC ρ)")
    ax.set_title(f"Test 5 — same-gene concordance vs cells-per-guide\nSpearman(cells, concordance)="
                 f"{fmt(rho)} (n={mc.size} same-gene pairs)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)


def test_5(adata, cfg, outdir):
    """Test 5 — Same-Gene Independent sgRNA Reproducibility (improved, in the spirit of Test 4).

    For each gene with ≥2 guides, DE each guide vs control, then compute the **full Test-4
    reproducibility metrics** (Spearman LFC ρ over all genes & over DE genes, DEG-set Jaccard, direction
    agreement) for every **same-gene guide pair** and for a matched **background** of unrelated
    (different-gene) guide pairs. For each metric report the **separation** (same-gene mean − background
    mean)/SD(background) plus an **empirical p** (fraction of background pairs ≥ the same-gene mean) and
    a Mann-Whitney U p. Emits per-pair tables + a same-gene-vs-background distribution plot. Verdict
    (on LFC ρ): PASS sep>1.5 / WARN 1–1.5 / FAIL else — but WARN-only (uninformative) when <5 genes with
    ≥2 guides / <5 same-gene pairs (never FAIL on an underpowered comparison)."""
    pc, sg, mcg = cfg["pert_col"], cfg.get("sgrna_col"), cfg["min_cells_per_group"]
    if not sg:
        return TestResult("test_5", "Same-Gene Independent sgRNA Reproducibility", "SKIP", flags=["no sgrna_col"],
                          reason="SKIP: no sgrna_col")
    perts = perts_in_use(adata, cfg)
    tg = adata.obs[pc].astype(str)
    sgv = adata.obs[sg].astype(str)
    gene2guides = {}
    for gene in perts:
        gs = sorted(sgv[tg == gene].unique())
        if len(gs) >= 2:
            gene2guides[gene] = gs
    if not gene2guides:
        return TestResult("test_5", "Same-Gene Independent sgRNA Reproducibility", "SKIP",
                          flags=["no gene with >=2 guides among selected conditions"],
                          reason="SKIP: no gene has ≥2 guides")
    allg = sorted({g for gs in gene2guides.values() for g in gs})
    cmask = (tg == cfg["control_pert"]).to_numpy()
    de_by_guide, n_cells = {}, {}
    for guide in allg:
        gmask = (sgv == guide).to_numpy()
        if gmask.sum() < mcg:
            continue
        keepmask = cmask | gmask
        s = adata[keepmask].copy()
        lab = np.where((s.obs[pc].astype(str) == cfg["control_pert"]).to_numpy(), cfg["control_pert"], guide)
        s.obs["_g"] = lab
        try:
            de_by_guide[guide] = run_de(s, cfg, groupby="_g", reference=cfg["control_pert"])
            n_cells[guide] = int(gmask.sum())
        except Exception as e:  # noqa: BLE001
            log.warning("test5 %s: %s", guide, e)
    guide_gene = {g: gene for gene, gs in gene2guides.items() for g in gs}
    metric_cols = [c for c, *_ in TEST5_METRICS]
    same_rows = []
    for gene, gs in gene2guides.items():
        gs = [g for g in gs if g in de_by_guide]
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                c = compare_signatures(de_by_guide[gs[i]], de_by_guide[gs[j]], cfg)
                if c:
                    ni, nj = n_cells.get(gs[i]), n_cells.get(gs[j])
                    same_rows.append({"gene": gene, "guide_i": gs[i], "guide_j": gs[j],
                                      "n_cells_i": ni, "n_cells_j": nj,
                                      "min_cells": (min(ni, nj) if (ni is not None and nj is not None) else None),
                                      **{k: c.get(k) for k in metric_cols}})
    # background: random UNRELATED (different-gene) guide pairs, deterministic, deduplicated
    glist = list(de_by_guide)
    rng = np.random.default_rng(cfg["seed"] + 5)
    n_target_bg = int(min(2000, max(200, 5 * len(same_rows))))
    seen, bg_rows, tries = set(), [], 0
    while len(bg_rows) < n_target_bg and tries < 60 * n_target_bg + 200 and len(glist) >= 2:
        tries += 1
        a, b = sorted(map(str, rng.choice(glist, size=2, replace=False)))
        if guide_gene.get(a) == guide_gene.get(b) or (a, b) in seen:
            continue
        seen.add((a, b))
        c = compare_signatures(de_by_guide[a], de_by_guide[b], cfg)
        if c:
            na, nb = n_cells.get(a), n_cells.get(b)
            bg_rows.append({"guide_i": a, "guide_j": b, "n_cells_i": na, "n_cells_j": nb,
                            "min_cells": (min(na, nb) if (na is not None and nb is not None) else None),
                            **{k: c.get(k) for k in metric_cols}})
    if not same_rows or not bg_rows:
        return TestResult("test_5", "Same-Gene Independent sgRNA Reproducibility", "SKIP",
                          flags=["insufficient same-gene or background pairs"],
                          reason="SKIP: insufficient same-gene or background pairs")
    same_df = pl.DataFrame(same_rows)
    bg_df = pl.DataFrame(bg_rows)
    same_df.write_csv(os.path.join(outdir, "tables", "test_5__same_gene_pairs.csv"))
    bg_df.write_csv(os.path.join(outdir, "tables", "test_5__background_pairs.csv"))
    metrics = {"n_genes": len(gene2guides), "n_guides_tested": len(de_by_guide),
               "n_same_pairs": same_df.height, "n_background_pairs": bg_df.height,
               # PRIMARY metric = DEG-restricted LFC ρ (Test 5 is a SEPARATION test: the all-genes ρ is
               # ~0 for both same & background — diluted by ~thousands of noise genes — and cannot
               # discriminate biology; the shared signal lives in the responsive (DE) genes).
               "primary_metric": "rho_deg"}
    for col, _disp, lab, _xr in TEST5_METRICS:
        ms, mb, _sd, z, emp_p, mwu_p, auc = _separation(same_df[col].to_numpy(), bg_df[col].to_numpy())
        metrics[f"same_gene_mean_{lab}"] = ms
        metrics[f"background_mean_{lab}"] = mb
        metrics[f"separation_z_{lab}"] = z
        metrics[f"emp_p_{lab}"] = emp_p
        metrics[f"mwu_p_{lab}"] = mwu_p
        metrics[f"auc_{lab}"] = auc
    prim = metrics["primary_metric"]
    prim_col = next(c for c, _d, lab, _x in TEST5_METRICS if lab == prim)
    # legacy/verdict-driver key points at the PRIMARY metric's separation
    metrics["separation_z"] = metrics[f"separation_z_{prim}"]
    # cell-count dependence: does same-gene concordance climb with cells-per-guide? (Test-4 undersampling
    # analog) — Spearman of the pair's smaller guide cell count vs the primary concordance metric.
    sg_mc = same_df["min_cells"].to_numpy().astype(float)
    sg_v = same_df[prim_col].to_numpy().astype(float)
    ok = np.isfinite(sg_mc) & np.isfinite(sg_v)
    metrics["samegene_primary_vs_ncells_spearman"] = (
        float(stats.spearmanr(sg_mc[ok], sg_v[ok]).correlation) if ok.sum() >= 3 else float("nan"))
    _plot_test5_separation(same_df, bg_df, os.path.join(outdir, "plots", "test_5_separation.png"))
    _plot_test5_vs_ncells(same_df, bg_df, prim_col, os.path.join(outdir, "plots", "test_5_vs_ncells.png"))
    verdict, reason, flags = verdict_test_5(metrics, cfg)
    pvalues = {f"separation_z ({prim})": metrics["separation_z"],
               f"emp_p ({prim})": metrics[f"emp_p_{prim}"], f"mwu_p ({prim})": metrics[f"mwu_p_{prim}"]}
    return TestResult("test_5", "Same-Gene Independent sgRNA Reproducibility", verdict,
                      metrics=metrics, flags=flags, pvalues=pvalues, plot="plots/test_5_separation.png",
                      reason=reason)


def test_6(adata, cfg, outdir, de_true=None):
    pc = cfg["pert_col"]
    perts = perts_in_use(adata, cfg)
    if de_true is None:
        keep = adata[adata.obs[pc].astype(str).isin(perts + [cfg["control_pert"]])].copy()
        de_true = run_de(keep, cfg, groupby=pc, reference=cfg["control_pert"])
    genes_present = set(de_true["feature"].cast(str).unique().to_list())
    rows = []
    for pert in perts:
        target = pert
        if target not in genes_present:
            continue
        d = de_true.filter(pl.col("target").cast(str) == pert)
        if d.height == 0:
            continue
        n = d.height
        d = d.with_columns(
            pl.col("p_value").rank("ordinal").alias("rk_p"),
            pl.col("log2_fold_change").abs().rank("ordinal", descending=True).alias("rk_lfc"),
        )
        row = d.filter(pl.col("feature").cast(str) == target)
        if row.height == 0:
            continue
        rr = row.row(0, named=True)
        rows.append({
            "condition": pert, "target_gene": target,
            "lfc_target": float(rr["log2_fold_change"]), "padj_target": float(rr["fdr"]),
            "pval_rank": float(rr["rk_p"]) / n, "lfc_rank": float(rr["rk_lfc"]) / n,
            "detected": bool(rr["fdr"] <= cfg["fdr_threshold"]),
            "correct_direction": bool(rr["log2_fold_change"] < 0),
        })
    if not rows:
        return TestResult("test_6", "Target Gene Knockdown Recovery", "SKIP", flags=["no target genes found"],
                          reason="SKIP: no target genes found in var_names"), de_true
    df = pl.DataFrame(rows)
    df.write_csv(os.path.join(outdir, "tables", "test_6__per_target.csv"))
    rec = float(df["detected"].mean())
    dirr = float(df["correct_direction"].mean())
    if rec > 0.5 and dirr > 0.8:
        verdict = "PASS"
    elif rec < 0.2 or dirr < 0.6:
        verdict = "FAIL"
    else:
        verdict = "WARN"
    reason = (f"{verdict}: recovery_rate={fmt(rec)}, direction_rate={fmt(dirr)} over {df.height} target "
              "genes (one DE per perturbation/gene, not per guide; also reflects assay/guide quality, "
              "not the metric alone)")
    return TestResult("test_6", "Target Gene Knockdown Recovery", verdict,
                      metrics={"n_targets": df.height, "recovery_rate": rec, "direction_rate": dirr,
                               "median_pval_rank": float(df["pval_rank"].median()),
                               "median_lfc_rank": float(df["lfc_rank"].median())},
                      flags=[], pvalues={"median_padj_target": float(df["padj_target"].median())},
                      reason=reason), de_true


# --------------------------------------------------------------------------- #
# plain-language docs + glossary
# --------------------------------------------------------------------------- #
TEST_DOC = {
    "test_0": "Spike a known log2 fold-change into a SMALL number of genes (~a dozen, the scale of a "
              "real perturbation), pooled over several random gene-draws, in a control-vs-control split; "
              "then measure how many injected genes we recover (TPR) and how many untouched genes are "
              "wrongly called (FPR) across effect sizes. Injecting a small NUMBER (not a large fraction) "
              "is faithful to real biology and avoids the library-size shift that artificially induces "
              "false DE in untouched genes (compositional coupling). This separates a genuinely "
              "calibrated null from a pipeline that simply finds nothing, and tells you the smallest "
              "effect the metric can resolve.",
    "test_1": "REPRODUCIBILITY. Split each perturbation's cells into halves A and B and run each vs "
              "control (DE_A, DE_B) — both carry the real perturbation signal. Controls are split into "
              "two halves (DE_A = A vs ctrl_half_A, DE_B = B vs ctrl_half_B; NO 1:1 cell matching). The "
              "question is whether the two independent half-signatures AGREE (DE_A ≈ DE_B): median "
              "split-half Spearman LFC ρ (PASS>0.6 strong / 0.3–0.6 moderate / <0.3 low). To separate a "
              "genuinely low-reproducibility method from merely undersampled perturbations, ρ is also "
              "plotted PER PERTURBATION against that perturbation's cell count (climbs→plateau = "
              "power-limited; flat/low at high cell counts = genuine method limitation). (Secondary "
              "sanity check: the direct A_pert-vs-B_pert contrast, same perturbation, should be ≈null.)",
    "test_2": "CALIBRATION / NULL. Split only control cells into A and B (a stratified random split "
              "balanced within batch; same population ⇒ true null) and run DE between them: a calibrated "
              "pipeline should produce Uniform[0,1] p-values (λ_GC≈1, frac_sig≈α). Inflation (λ_GC>1) = "
              "false positives; deflation (λ_GC<1) = under-powered (cell-level pseudoreplication).",
    "test_3": "Shuffle the perturbation labels (within batch) and recompute. Real biological signal "
              "should sit far outside the shuffled distribution (high separation z); if not, the "
              "metric cannot tell signal from noise. A diagnostic compares the per-gene p-value "
              "distributions of the real vs shuffled DE, pooled over all perturbations × genes and "
              "stratified by each perturbation's cell count: the real run should show a small-p "
              "excess (signal) while the shuffled run should track Uniform[0,1] (a calibrated null), "
              "consistently across small- and large-cell-count perturbations.",
    "test_4": "Split each guide's cells in two and compare each half vs control. Agreement between "
              "the halves is the reproducibility ceiling — no model can score higher on a perturbation "
              "than the data agrees with itself.",
    "test_5": "Do two different guides for the same gene agree more than two unrelated guides? Note "
              "that guides for the same gene often differ in knockdown efficacy, so only modest "
              "concordance is expected; a low score with few guide pairs is power-limited, not proof "
              "of off-target activity.",
    "test_6": "Is the targeted gene itself knocked down (negative LFC, significant)? Computed per target "
              "gene (one DE per perturbation, pooling that gene's guides — not per individual guide). "
              "High recovery is reassuring, but it partly reflects assay and guide quality, not the "
              "metric alone.",
    "composition": "Do perturbations shift cell-type / cluster / cell-cycle proportions vs control? "
                   "Such shifts make cross-population DE read composition change as expression change. "
                   "Requires a cell-state annotation in obs.",
}

# Per-test tier thresholds, rendered inline in each test's result section.
TIER_DOC = {
    "test_0": "**Tiers** — null FPR (δ=0): ≤ α PASS · > 2α FAIL (anti-conservative). max TPR across δ: "
              "> 0.5 = resolves effects · < 0.5 WARN (under-powered). FPR-by-class at high δ: ≈ null "
              "FPR = clean · ≫ null FPR = compositional coupling (worst in HighlyExpr/AnchorCorr, "
              "spares LowlyExpr).",
    "test_1": "**Tiers** (per metric, strong/moderate/low) — LFC Spearman ρ (verdict driver): > 0.6 "
              "PASS · 0.3–0.6 WARN · < 0.3 FAIL. DEG Jaccard: > 0.3 · 0.1–0.3 · < 0.1. Direction "
              "agreement: > 0.7 · 0.6–0.7 · ≈ 0.5. Low ρ = the two half-signatures disagree → ceiling "
              "for downstream metrics. (Secondary difference-is-null should be ≈null: λ_GC≈1, p uniform.)",
    "test_2": "**Tiers** — λ_GC: ≈ 1 calibrated · 1.05–1.10 or < 0.9 WARN · > 1.10 FAIL (anti-"
              "conservative). frac_sig: ≈ α good · ≫ α FAIL. ks_p_uniform: > 0.05 good · < 0.05 WARN "
              "(p-values not Uniform[0,1]). Arms are a stratified control-control split (true null).",
    "test_3": "**Tiers** — separation z (true signal vs permuted null): > 2 PASS · 1–2 WARN · ≤ 1 FAIL "
              "(metric cannot tell signal from noise). Diagnostic: real p-values should show a small-p "
              "excess (ECDF above diagonal / QQ above the line) while shuffled p-values track "
              "Uniform[0,1] (frac(p<0.05)≈0.05, λ_GC≈1) — across all cell-count strata.",
    "test_4": "Test 1's within-condition reproducibility design run at the **guide (sgRNA) level** — the "
              "unit is one guide, split A/B vs split control (DE_A, DE_B), plus the difference-is-null "
              "QQ and the ρ-vs-cell-count diagnostic, same as Test 1. **Tiers** (same as Test 1) — LFC "
              "Spearman ρ: > 0.6 strong/PASS · 0.3–0.6 moderate/WARN · < 0.3 low/FAIL. DEG Jaccard: > 0.3 "
              "· 0.1–0.3 · < 0.1. Direction: > 0.7 · 0.6–0.7 · ≈ 0.5. This is the empirical reproducibility "
              "ceiling for any per-guide downstream metric.",
    "test_5": "A separation test (like Test 3) using Test 4's reproducibility metrics: same-gene guide "
              "pairs vs an unrelated-pair background, on **DEG-restricted LFC ρ** (all-genes ρ is ≈0 for "
              "both — diluted by non-responsive genes — so it can't discriminate; the signal is in the DE "
              "genes). The verdict is driven by the **AUC effect size** = P(a random same-gene pair is "
              "more concordant than a random unrelated pair) **gated by Mann-Whitney significance**: PASS "
              "AUC≥0.65 & p<α · WARN AUC≥0.55 & p<α (real but modest) · FAIL otherwise. (Separation z is "
              "reported but not the driver — it is variance-sensitive and understates a wide-background "
              "separation.) Plus a cells-per-guide trend (is weak concordance biology or undersampling?). "
              "With < ~5 genes with ≥2 guides / < ~5 pairs ⇒ WARN (power-limited, never FAIL).",
    "test_6": "**Tiers** — recovery_rate (targets detected as DE): > 0.5 PASS · 0.2–0.5 WARN · < 0.2 "
              "FAIL. direction_rate (knocked down in expected direction): > 0.8 PASS · 0.6–0.8 WARN · "
              "< 0.6 FAIL. (Also reflects assay/guide quality, not the metric alone.)",
    "composition": "**Tiers** — per-perturbation total-variation distance (TVD) of cell-state "
                   "proportions vs control: > 0.10 ⇒ flagged (its DE may be partly compositional). "
                   "SKIP when no cell-state column exists.",
}

GLOSSARY = [
    ("unit of analysis", "Whether each *cell* (pdex/Wilcoxon) or each *replicate/sample* (pseudobulk "
     "DESeq2) is one observation. Cell-level tests treat correlated cells from one sample as "
     f"independent (pseudoreplication), distorting null p-values — see {SQUAIR}."),
    ("arm", "One side of the two-group split that DE compares (clinical-trial term: reference arm vs "
     "treatment arm). In the null/injection tests both arms are drawn from the same cells; arm B is "
     "the pretend-perturbation (and receives the injected effect in Test 0), arm A is the reference, "
     "and DE is run B vs A."),
    ("stratified split (block_cols)", "Cells are divided into arms separately within each level of "
     "block_cols (e.g. each batch) and then pooled, so both arms have the same batch composition. "
     "This prevents technical structure (batch effects) from masquerading as signal; it is why the "
     "per-arm counts are slightly uneven (odd-sized batches can't halve exactly)."),
    ("λ_GC (genomic inflation)", "Median observed χ² statistic / its null median. ≈1 well-calibrated; "
     ">1 inflated (false positives); <1 deflated (conservative, under-powered)."),
    ("ks_p_uniform", "KS-test p-value that the per-gene p-values are Uniform[0,1] as a true null "
     "requires. Small (<0.05) ⇒ the p-value distribution is mis-shaped."),
    ("frac_sig", "Fraction of genes called significant (FDR≤threshold AND |LFC|≥threshold). Under a "
     "true null this is the empirical false-positive rate; it should be ≈α."),
    ("mean_abs_lfc", "Mean |log2 fold-change| across genes — magnitude inflation under the null."),
    ("TPR / FPR (Test 0)", "Of injected genes, the fraction recovered (true-positive rate); of "
     "untouched genes, the fraction wrongly called (false-positive rate)."),
    ("max_TPR / min resolvable δ (Test 0)", "max_TPR = the highest TPR across the injected δ tiers — "
     "the sensitivity ceiling (best-case recovery of true effects). min resolvable δ = the smallest "
     "injected log2FC that reaches TPR ≥ 0.5, i.e. the smallest real effect the metric reliably "
     "detects. A max_TPR well below 1 means even strong real effects are partly missed (false negatives)."),
    ("separation z (Tests 3,5)", "(observed − null mean) / null SD. >2 means the true signal is clearly "
     "outside the permuted/background null. Variance-sensitive: a wide/skewed null deflates it (which is "
     "why Test 5's verdict uses the AUC effect size instead)."),
    ("AUC / probability of superiority (Test 5)", "The Mann-Whitney common-language effect size = "
     "U/(n_same·n_bg) = P(a random same-gene guide pair is more concordant than a random unrelated pair). "
     "0.5 = no separation, 1.0 = perfect. Rank-based (robust to the wide, skewed background that deflates "
     "the separation z) and not inflated by sample size (unlike a raw p-value); it is the effect-size "
     "companion to the Mann-Whitney p, so Test 5 reports both — AUC for magnitude, p for significance."),
    ("LFC Spearman ρ / Jaccard (Tests 1, 4)", "Rank correlation of fold-changes / overlap of "
     "significant gene sets between two split halves — the empirical reproducibility ceiling."),
]


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def global_verdict(results):
    gates = {r.name: r.verdict for r in results if r.name in ("test_0", "test_1", "test_2", "test_3")}
    if any(v == "FAIL" for v in gates.values()):
        return "FAIL"
    if any(v == "WARN" for v in gates.values()):
        return "WARN"
    return "PASS"


def global_reason(results):
    """One-line summary enumerating EVERY distinct gate failure mode (so the headline matches the body)."""
    by = {r.name: r for r in results}
    modes = []
    t0 = by.get("test_0")
    if t0 and t0.verdict in ("WARN", "FAIL"):
        if any("compositional coupling" in f for f in t0.flags):
            modes.append("effect-driven FPR inflation (compositional coupling under strong/broad perturbation)")
        if t0.metrics.get("max_TPR", 1) < 0.5:
            modes.append("under-powered effect detection")
        if t0.metrics.get("null_FPR", 0) > 0.1:
            modes.append("anti-conservative null")
    for n in ("test_1", "test_2"):
        r = by.get(n)
        if r and r.verdict in ("WARN", "FAIL"):
            lam = r.metrics.get("lambda_gc", float("nan"))
            if np.isfinite(lam) and lam < 0.9:
                modes.append(f"deflated/under-powered cell-level null ({n}: λ_GC={fmt(lam)})")
            elif np.isfinite(lam) and lam > 1.05:
                modes.append(f"inflated null ({n}: λ_GC={fmt(lam)})")
            else:
                modes.append(f"non-uniform null p-values ({n})")
    t3 = by.get("test_3")
    if t3 and t3.verdict in ("WARN", "FAIL"):
        modes.append(f"weak signal/permutation separation ({fmt(t3.metrics.get('separation_z'))})")
    # de-duplicate while preserving order
    seen, uniq = set(), []
    for m in modes:
        if m not in seen:
            seen.add(m); uniq.append(m)
    return "; ".join(uniq) if uniq else "all validity gates clean"


VERDICT_EMOJI = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}


def _unit_of_analysis(cfg):
    if cfg["de_method"] == "pdex":
        return ("individual cell (Wilcoxon rank-sum)",
                "Each cell is one observation, so cells from the same sample/batch are treated as "
                f"independent — **pseudoreplication** ({SQUAIR}). Expect the null-split tests to be "
                "deflated/under-powered and p-values non-uniform even when there are no false "
                "positives. For calibrated significance on raw counts, prefer the pseudobulk "
                "`pydeseq2` backend; here pdex output is best read as a *ranking*, not calibrated FDR.")
    if cfg["de_method"] == "pydeseq2":
        return ("pseudobulk replicate / sample (DESeq2 Wald)",
                "Counts are aggregated to replicate-level pseudobulk samples before testing, which "
                "respects the true unit of replication and avoids pseudoreplication.")
    return (cfg["de_method"], "")


def write_report(results, cfg, power, fp, outdir):
    gv = global_verdict(results)
    scale = "full (all conditions)" if not cfg.get("max_conditions") else f"max_conditions={cfg['max_conditions']}"
    unit, unit_caveat = _unit_of_analysis(cfg)
    by = {r.name: r for r in results}
    L = [f"# Robustness report — {os.path.basename(os.path.abspath(outdir))}", ""]

    # ---- global verdict (top, with one-line reason) ----
    L += [f"## Global verdict: {VERDICT_EMOJI.get(gv,'')} **{gv}**", "",
          "Validity gates (Tests 0–3) must hold before the sensitivity diagnostics (Tests 4–6) are "
          "interpretable.", ""]
    gate_lines = [f"- {VERDICT_EMOJI.get(by[n].verdict,'')} **{by[n].name}** — {by[n].reason}"
                  for n in ("test_0", "test_1", "test_2", "test_3") if n in by]
    L += gate_lines + [""]
    greason = global_reason(results)
    if gv == "FAIL":
        L += [f"> ❌ A validity gate FAILED — **do not interpret the biology** until the pipeline is fixed. "
              f"Failure mode(s): {greason}.", ""]
    elif gv == "WARN":
        L += [f"> ⚠️ Validity gates WARNed — results are usable but **read the caveats first**. Distinct "
              f"issues found: {greason}. With a cell-level test, treat pdex output as a **gene ranking, "
              "not calibrated FDR**.", ""]

    # ---- dataset & experimental context (FIRST, per reviewer) ----
    L += ["## 1. Dataset & experimental context", ""]
    L += [f"- **file**: `{cfg['adata_path']}`",
          f"- **size**: {fp['n_cells']:,} cells × {fp['n_genes']:,} genes; layers={fp['layers'] or 'none'}",
          f"- **perturbations**: {fp['n_perturbations']} + control `{fp['control_label']}` "
          f"({fp['n_control_cells']:,} control cells)",
          f"- **cells per perturbation**: min {fp['cells_per_pert_min']}, median {fp['cells_per_pert_median']}, "
          f"max {fp['cells_per_pert_max']}",
          f"- **guides**: {fp['n_guides']} total; **{fp['genes_with_ge2_guides']} gene(s) have ≥2 guides** "
          "(limits Test 5)",
          f"- **obs columns**: " + ", ".join(f"`{c}`(n={n})" for c, n in fp["obs_columns"].items()),
          f"- **.X**: raw-integer={fp['X_raw_integer']}, min/max/mean="
          f"{fmt(fp['X_min'])}/{fmt(fp['X_max'])}/{fmt(fp['X_mean'])}", ""]
    # explicit "what context is missing" so the reviewer doesn't have to guess
    missing = []
    if not fp["has_celltype_col"]:
        missing.append("cell type / cluster")
    if not fp["has_cellcycle_col"]:
        missing.append("cell-cycle phase")
    for want in ("donor", "patient", "sex", "tissue", "timepoint"):
        if not any(want in c.lower() for c in fp["obs_columns"]):
            missing.append(want)
    L += [f"- **structural covariates present**: {fp['present_covariate_like_cols'] or 'only the columns above'}",
          f"- **NOT available in this dataset** (assumed absent, interpret accordingly): "
          f"{', '.join(missing) if missing else 'none'}", ""]

    # ---- how DE was run (method + unit of analysis + correction state) ----
    L += ["## 2. How DE was computed", "",
          f"- **DE backend**: `{cfg['de_method']}`",
          f"- **unit of analysis**: **{unit}**",
          f"- **covariate correction of .X**: **{cfg.get('covariate_correction','none')}** "
          "(this run evaluates the metric on this state of the data)",
          f"- **input scaling**: normalize_if_raw={cfg.get('normalize_if_raw')}, allow_discrete="
          f"{cfg['allow_discrete']} (pdex is_log1p={not cfg['allow_discrete']})",
          f"- **stratification / blocking**: {cfg.get('block_cols')}",
          f"- **scale**: {scale}, n_resamples={cfg['n_resamples']}, seed={cfg['seed']}", ""]
    if unit_caveat:
        L += [f"> **Unit-of-analysis caveat.** {unit_caveat}", ""]
    L += ["> **Correction note.** Many groups regress out batch / cell-cycle before DE. An ideal "
          "perturbation metric is insensitive to this, but real ones are not — evaluate on **both "
          "corrected and uncorrected** data. This run is "
          f"**{cfg.get('covariate_correction','none')}**; set `covariate_correction` and re-run to "
          "compare.", ""]

    # ---- at-a-glance summary ----
    L += ["## 3. At-a-glance", "",
          "**What each test is for** (validity gates 0–3 must hold before the sensitivity diagnostics "
          "4–6 are interpretable):", "",
          "| test | role | the question it answers |", "|---|---|---|",
          "| test_0 | gate — calibration + power | Inject a known effect: what false-positive rate at "
          "the null, and what effect size can the metric resolve (TPR vs δ)? |",
          "| test_1 | **reproducibility** | Split a perturbation in half vs control — do the two "
          "half-signatures agree (DE_A≈DE_B)? And does that agreement depend on cell count? |",
          "| test_2 | gate — null calibration | Split control cells into A/B (stratified, true null) — "
          "are the p-values Uniform[0,1] (λ_GC≈1, frac_sig≈α)? |",
          "| test_3 | gate — signal vs noise | Shuffle perturbation labels — is real signal far outside "
          "the permuted null (separation z)? Diagnostic: do real p-values show a small-p excess while "
          "shuffled p-values stay Uniform, across cell-count strata? |",
          "| test_4 | sensitivity — ceiling | **Test 1 at the guide level**: split each guide's cells in "
          "half vs control — do the two half-signatures agree, and does that depend on cell count? The "
          "empirical reproducibility ceiling for any per-guide downstream metric. |",
          "| test_5 | sensitivity | Do independent guides for the same gene agree more than unrelated "
          "guides? Verdict = AUC effect size (P[same>unrelated]) gated by Mann-Whitney significance; "
          "power-limited with few guides. |",
          "| test_6 | sensitivity | Is the targeted gene itself knocked down in the right direction? "
          "(also reflects assay/guide quality) |",
          "| composition | confounder | Do perturbations shift cell-state proportions vs control? "
          "(needs a cell-type column) |", "",
          "| test | verdict | what it tells you (local reason) |", "|---|---|---|"]
    for r in results:
        L.append(f"| **{r.name}** — {r.title} | {VERDICT_EMOJI.get(r.verdict,'')} {r.verdict} | {r.reason} |")
    L.append("")
    if cfg.get("emit_multiqc"):
        L += ["_This verdict table is also exported as a **MultiQC** custom-content file "
              "(`multiqc/cell_eval_robustness_mqc.txt`) — run `multiqc .` in this folder to render an "
              "interactive HTML report that integrates with downstream single-cell pipelines._", ""]

    # ---- per-test detail ----
    L += ["## 4. Per-test detail", ""]
    for r in results:
        L.append(f"### {r.name} — {r.title}  {VERDICT_EMOJI.get(r.verdict,'')} {r.verdict}")
        L.append("")
        if r.name in TEST_DOC:
            L.append(f"*What this tests.* {TEST_DOC[r.name]}")
            L.append("")
        if r.name in TIER_DOC:
            L.append(TIER_DOC[r.name])
            L.append("")
        if r.name == "test_0" and r.verdict != "SKIP" and r.metrics:
            m = r.metrics
            n_per = m.get("n_injected_genes_per_draw")
            n_draws = m.get("n_draws")
            n_inj_tot = m.get("n_injected_total")
            regime = m.get("injection_regime", "")
            n_expr = m.get("n_expressed_genes")
            n_tot = m.get("n_total_genes") or fp.get("n_genes")
            cpa = m.get("cells_per_arm")
            cused = m.get("control_cells_used") or (2 * cpa if cpa else None)
            deltas = cfg.get("injection_deltas")
            nctrl = fp.get("n_control_cells")
            cap = cfg.get("injection_max_cells_per_arm")
            n_lo, n_mid, n_hi = m.get("n_injected_low_expr"), m.get("n_injected_mid_expr"), m.get("n_injected_high_expr")
            L.append(
                "*Injection design (ground truth).* An **arm** is one side of the two-group split that "
                "differential expression compares (term borrowed from clinical trials: a reference arm "
                "vs a treatment arm). Here **both arms are control cells** (so they are truly identical) "
                "— **arm A** is the reference and **arm B** is the pretend-perturbation that receives the "
                "injected effect; DE is run as *B vs A*.")
            L.append("")
            L.append(
                f"From the **{nctrl:,} `{cfg.get('control_pert')}` control cells**, "
                f"{('a random ~' + format(cused, ',') + ' were subsampled and ') if cused else ''}"
                f"split into the two arms of **~{cpa:,} cells each**, stratified within "
                f"`{cfg.get('block_cols')}` (the per-arm count is capped at "
                f"`injection_max_cells_per_arm`={cap} for speed/memory — DE is re-run for every δ tier — "
                "and is slightly under the cap because the split halves each batch with integer "
                "rounding; raise the cap to use more of the available control cells). "
                f"On **each draw, only {n_per} genes** ({n_per:,} of the {n_expr:,} expressed; {n_tot:,} "
                "total) were chosen as the *injected* (**anchor**) set — a deliberately **small number, "
                "the scale of a real perturbation**. Injecting a handful of genes (rather than a large "
                "*fraction*) barely shifts the per-cell library size, so it does **not** artificially "
                "induce false DE in untouched genes via renormalization (compositional coupling) — the "
                "faithful test. To get stable rates from so few genes per draw, the injection is **repeated "
                f"over {n_draws} independent random draws and pooled** ({n_inj_tot:,} injected-gene "
                f"observations total: **{n_lo} low · {n_mid} mid · {n_hi} high** expression). Anchors are "
                "**stratified across the expression (detection) spectrum** (equal shares from the low / mid "
                "/ high mean-expression tertiles) because detectability is dominated by expression level, "
                "so this makes the TPR-vs-δ curve readable *per tier* (the smallest resolvable δ for sparse "
                "vs typical vs abundant genes). In arm B their "
                f"{'raw counts were multiplied' if not cfg.get('normalize_if_raw') else 'expression was scaled'} "
                f"by 2^δ for each effect size **δ ∈ {deltas}** (log2 fold-change; δ=0 = untouched null). "
                "The remaining (untouched) genes on each draw are used to measure the false-positive rate. "
                "TPR is recovery of the injected/anchor set (reported per expression tier); FPR is false "
                f"calls among the untouched set (reported per gene class). Regime: *{regime}*.")
            L.append("")
            L.append(
                "*Gene-class FPR breakdown.* The untouched-gene FPR is broken down into gene classes — "
                "**AnchorCorr** (genes most correlated with the injected anchors — the easiest/upper-bound "
                "case and the one most exposed to compositional coupling), **HighlyExpr** (abundant, high "
                "signal), **LowlyExpr** (sparse, zero-dominated, hard), **HouseKeeping** (constitutive — "
                "should be predictable; watch for memorization), **Marker** (cell-type identity genes — "
                "the biologically interesting case; N/A without a cell-type annotation), **HighlyVarG** "
                "(high-variance complement to the anchors), and **Random** (unbiased baseline). All class "
                "memberships are computed **deterministically over all control cells** (a fixed function "
                "of the h5ad), independent of the DE cell subsample.")
            L.append("")
        L.append(f"**Verdict reason:** {r.reason}")
        L.append("")
        # Test 1 (perturbation-level) & Test 4 (guide-level) share this reproducibility renderer
        if r.name in ("test_1", "test_4") and any(k.startswith("no_match__") for k in r.metrics):
            m = r.metrics
            present = ["no_match"]
            tprefix = r.name
            unit = "perturbation" if r.name == "test_1" else "guide"
            U = unit.capitalize()
            ab = "A_pert vs B_pert" if r.name == "test_1" else "A_guide vs B_guide"
            if r.name == "test_4":
                L.append("**What this measures.** This is **Test 1 run at the sgRNA (guide) level** — the "
                         "*unit* is an individual guide, not a perturbation/gene. Split each guide's cells "
                         "into two halves A and B and run each half against control (`DE_A`, `DE_B`), with "
                         "the control cells also split into two halves (`DE_A = A vs ctrl_half_A`, `DE_B = B "
                         "vs ctrl_half_B`; **no 1:1 cell matching**). The **primary** question is "
                         "*reproducibility* — do the two independent half-signatures of the **same guide** "
                         "agree (`DE_A ≈ DE_B`)? A **secondary** difference-is-null stat (direct `A_guide vs "
                         "B_guide`, the same guide ⇒ should be ≈null) is reported as a sanity check. With "
                         "many guides per gene, this is the empirical reproducibility ceiling at the "
                         "resolution downstream per-guide metrics actually operate.")
            else:
                L.append("**What this measures.** Split each perturbation's cells into two halves A and B "
                         "and run each half against control (`DE_A`, `DE_B`), with the control cells also "
                         "split into two halves (`DE_A = A vs ctrl_half_A`, `DE_B = B vs ctrl_half_B`; **no "
                         "1:1 cell matching**). The **primary** question is *reproducibility* — do the two "
                         "independent half-signatures agree (`DE_A ≈ DE_B`)? A **secondary** difference-is-null "
                         "stat (direct `A_pert vs B_pert`, the same perturbation ⇒ should be ≈null) is "
                         "reported as a sanity check.")
            L.append("")
            # --- design: what's compared, how DE is defined, thresholds, and per-unit cell counts ---
            de_desc = ("cell-level **Wilcoxon rank-sum** (each cell is a unit; log2FC = "
                       "log2(mean_perturbed / mean_control), BH-FDR)" if cfg.get("de_method") == "pdex"
                       else "**pseudobulk DESeq2 Wald** (cells summed per replicate/`%s`, negative-binomial GLM)"
                       % cfg.get("replicate_col", "batch") if cfg.get("de_method") == "pydeseq2"
                       else f"`{cfg.get('de_method')}`")
            nperm_t1 = cfg.get("test1_n_resamples", min(3, cfg.get("n_resamples", 10)))
            ncond = m.get("no_match__n_conditions")
            nctrl = fp.get("n_control_cells")
            ppcsv = os.path.join(outdir, "tables", f"{tprefix}__repro_vs_ncells_no_match.csv")
            cell_txt = ""
            if os.path.exists(ppcsv):
                ncv = pl.read_csv(ppcsv)["n_cells"].to_numpy()
                if ncv.size:
                    cell_txt = (f" Across the **{ncv.size} tested {unit}s**, total cells per "
                                f"{unit} range **{int(ncv.min()):,}–{int(ncv.max()):,}** "
                                f"(median {int(np.median(ncv)):,}); each A/B half uses ~half of those "
                                f"(the per-{unit} counts are in the table below / "
                                f"`tables/{tprefix}__repro_vs_ncells_no_match.csv`).")
            L.append(
                "*Design (ground truth).* An **arm** is one side of the split DE compares. `DE_A` and "
                f"`DE_B` are computed with the **same DE method as the main analysis** — {de_desc} — and a "
                f"gene is called a DEG when **FDR < {cfg.get('fdr_threshold')}** and **|log2FC| > "
                f"{cfg.get('lfc_threshold')}**. Reproducibility compares the two signatures gene-by-gene: "
                "**Spearman ρ of their log2FC vectors** (primary), **DEG-set Jaccard**, and **direction "
                f"agreement**. The split is repeated for **{nperm_t1} draws** (`test1_n_resamples`) per "
                f"{unit} (each {unit}'s reported ρ is the median over its {nperm_t1} draws). "
                f"Controls ({nctrl:,} cells) are split into two halves (~{nctrl // 2:,} each).{cell_txt}")
            L.append("")
            L.append(f"*Thresholds (reproducibility tiers).* split-half LFC Spearman ρ: **PASS > "
                     f"{REPRO_PASS}** (strong) / **WARN {REPRO_WARN}–{REPRO_PASS}** (moderate) / **FAIL < "
                     f"{REPRO_WARN}** (low) — this drives the verdict. DEG Jaccard: strong ≥ "
                     f"{REPRO_METRIC_TIERS['sig_jaccard'][1]} / moderate ≥ {REPRO_METRIC_TIERS['sig_jaccard'][2]}. "
                     f"Direction agreement: strong ≥ {REPRO_METRIC_TIERS['direction_agreement'][1]} / moderate "
                     f"≥ {REPRO_METRIC_TIERS['direction_agreement'][2]}.")
            L.append("")
            L.append("| statistic | value |")
            L.append("|---|---|")
            rowdefs = [
                ("**reproducibility** — median Spearman LFC ρ, ALL genes (PRIMARY)", "repro_lfc_spearman"),
                ("reproducibility — median Spearman LFC ρ, DE genes only (2°)", "repro_lfc_spearman_deg"),
                ("reproducibility — median DEG Jaccard", "repro_jaccard"),
                ("reproducibility — median direction agreement", "repro_direction"),
                (f"# {unit}s strong (ρ>0.6)", "n_strong_rho_gt0.6"),
                ("# moderate (0.3–0.6)", "n_moderate_rho_0.3_0.6"),
                ("# low (<0.3)", "n_low_rho_lt0.3"),
                ("(2°) difference-is-null λ_GC", "diffnull_lambda_gc"),
                ("(2°) difference-is-null frac_sig", "diffnull_frac_sig"),
                ("(2°) difference-is-null ks_p_uniform", "diffnull_ks_p_uniform"),
                (f"# {unit}s tested", "n_conditions"),
            ]
            tier_key = {"repro_lfc_spearman": "lfc_spearman", "repro_lfc_spearman_deg": "lfc_spearman",
                        "repro_jaccard": "sig_jaccard", "repro_direction": "direction_agreement"}
            for lbl, key in rowdefs:
                cells = []
                for mode in present:
                    v = m.get(f"{mode}__{key}")
                    if key in tier_key and isinstance(v, (int, float)):
                        cells.append(f"{fmt(v)} ({metric_tier(tier_key[key], v)})")
                    else:
                        cells.append(fmt(v))
                L.append(f"| {lbl} | " + " | ".join(cells) + " |")
            L.append("")
            L.append(f"_Tiers (strong/moderate/low) per metric: {REPRO_LEGEND} The split-half LFC ρ "
                     f"(all genes) drives the verdict. difference-is-null: A and B are the same {unit}, "
                     "so a calibrated pipeline calls ≈α (λ_GC≈1, p uniform)._")
            L.append("")
            # which gene set each reproducibility metric uses
            L.append("*Which genes each metric uses:*")
            L.append("")
            L.append("| metric | gene set used |")
            L.append("|---|---|")
            L.append("| `lfc_spearman` (primary, verdict driver) | **ALL genes** with a finite log2FC in both "
                     "halves — the full transcriptome (~all measured genes), no significance filter |")
            L.append(f"| `lfc_spearman_deg` (additional) | **DE genes only** — Spearman over the *union* of "
                     f"the two significant sets (FDR below {cfg.get('fdr_threshold')} and abs(log2FC) above "
                     f"{cfg.get('lfc_threshold')}); isolates the responsive genes, undiluted by near-zero-"
                     "effect genes |")
            L.append("| `sig_jaccard` | **DE genes only** — intersection ÷ union of the two significant sets |")
            L.append("| `direction_agreement` | **DE genes only** — sign concordance over the union of "
                     "significant genes (significant in A or B) |")
            L.append("")
            # Reproducibility vs cell count — is it the method, or undersampling?
            if any(f"{mode}__rho_vs_ncells_spearman" in m for mode in present):
                wp_min = next((m.get(f"{mode}__wellpowered_min_cells") for mode in present
                               if m.get(f"{mode}__wellpowered_min_cells")), 200)
                L.append("**Reproducibility vs cell count — is it the method, or undersampling?** Splitting a "
                         f"{unit} into halves starves small {unit}s of cells, so each {unit} is "
                         "plotted as **one point** (its total cell count vs its median split-half ρ). Points "
                         "that **climb with cell count and plateau** ⇒ low scores are power-limited (not enough "
                         "cells); points that **stay low even at high cell counts** ⇒ a genuine method "
                         f"limitation. Well-powered = ≥{wp_min} total cells (≈≥100 per half).")
                L.append("")
                for mode in present:
                    if f"{mode}__rho_vs_ncells_spearman" not in m:
                        continue
                    wp, up = m.get(f"{mode}__repro_rho_wellpowered"), m.get(f"{mode}__repro_rho_underpowered")
                    wpn, upn = m.get(f"{mode}__n_wellpowered"), m.get(f"{mode}__n_underpowered")
                    tr = m.get(f"{mode}__rho_vs_ncells_spearman")
                    overall_mode = m.get(f"{mode}__repro_lfc_spearman")
                    lifts = (isinstance(wp, (int, float)) and np.isfinite(wp)
                             and isinstance(overall_mode, (int, float))
                             and verdict_reproducibility(wp) != verdict_reproducibility(overall_mode))
                    note = (f"→ restricting to well-powered {unit}s **lifts** ρ across a tier boundary ⇒ "
                            "part of the low score is an undersampling artifact." if lifts else
                            "→ well-powered ρ ≈ this scenario's overall ρ ⇒ the score reflects a **genuine "
                            "method property**, not undersampling.")
                    L.append(f"- **{mode}**: well-powered (n={wpn}) median ρ={fmt(wp)} · under-powered "
                             f"(n={upn}) median ρ={fmt(up)} · Spearman(cell count, ρ)={fmt(tr)} {note}")
                L.append("")
                L.append("The figure shows this for **every** reproducibility statistic (one panel each), "
                         f"one point per {unit}:")
                L.append("")
                upng = f"plots/{tprefix}_undersampling.png"
                if os.path.exists(os.path.join(outdir, upng)):
                    L.append(f"![reproducibility vs cell count]({upng})")
                    L.append("")
                # per-statistic cell-count trend table (Spearman of cell count vs the metric + well/under medians)
                tcsv = os.path.join(outdir, "tables", f"{tprefix}__repro_vs_ncells_trends_no_match.csv")
                if os.path.exists(tcsv):
                    tdf = pl.read_csv(tcsv)
                    L.append(f"Per-statistic cell-count dependence (one point per {unit}):")
                    L.append("")
                    L.append("| statistic | Spearman(cells, value) | well-powered median | under-powered median |")
                    L.append("|---|---|---|---|")
                    nice = {"lfc_spearman": "Spearman LFC — all genes", "lfc_spearman_deg": "Spearman LFC — DE genes",
                            "sig_jaccard": "DEG-set Jaccard (≡ all-genes)",
                            "direction_agreement_all": "Direction agreement — all genes",
                            "direction_agreement": "Direction agreement — DE genes"}
                    for row in tdf.iter_rows(named=True):
                        L.append(f"| {nice.get(row['metric'], row['metric'])} | {fmt(row['spearman_ncells'])} | "
                                 f"{fmt(row['wellpowered_median'])} (n={row['n_wellpowered']}) | "
                                 f"{fmt(row['underpowered_median'])} (n={row['n_underpowered']}) |")
                    L.append("")
                    L.append("_A positive Spearman (cells↑ ⇒ value↑) with a higher well-powered median = "
                             "power-limited for that statistic; ≈0 / negative with well-powered ≈ overall = a "
                             "genuine method property. Jaccard is universe-independent, so its all-genes and "
                             "DE-genes values coincide._")
                    L.append("")
                L.append(f"_Per-{unit} points: `tables/{tprefix}__repro_vs_ncells_no_match.csv`; "
                         f"per-statistic trends: `tables/{tprefix}__repro_vs_ncells_trends_no_match.csv`._")
                L.append("")
        elif r.name == "test_5" and r.verdict != "SKIP" and r.metrics:
            m = r.metrics
            L.append("**What this measures.** Do **two different guides targeting the same gene** produce "
                     "more similar perturbation signatures than two **unrelated** guides (different "
                     "genes)? Each guide is DE'd vs control; then the **same Test-4 reproducibility "
                     "metrics** (Spearman LFC ρ over all genes & over DE genes, DEG-set Jaccard, direction "
                     "agreement) are computed for every **same-gene guide pair** and for a matched "
                     "**background** of unrelated pairs. The headline statistic is the **AUC** = "
                     "P(a random same-gene pair is more concordant than a random unrelated pair) — the "
                     "Mann-Whitney common-language effect size (0.5 = no separation, 1.0 = perfect). A "
                     "biology-rewarding metric puts same-gene pairs above background. Expect only **modest** "
                     "concordance — guides for one gene genuinely differ in knockdown efficacy.")
            L.append("")
            prim = m.get("primary_metric", "rho_deg")
            prim_disp = next((d for _c, d, lab, _x in TEST5_METRICS if lab == prim), prim)
            fdr = cfg.get("fdr_threshold", 0.05)
            L.append(f"*Design & verdict.* **{m.get('n_genes')} genes with ≥2 guides** "
                     f"({m.get('n_guides_tested')} guides DE'd vs control) → **{m.get('n_same_pairs')} "
                     f"same-gene pairs** vs **{m.get('n_background_pairs')} unrelated background pairs**. The "
                     f"**primary metric is {prim_disp}** (all-genes ρ is diluted to ≈0 by thousands of "
                     "non-responsive genes for *both* same and background pairs, so it cannot discriminate — "
                     "the signal lives in the DE genes). The verdict is driven by the **AUC effect size "
                     f"gated by Mann-Whitney significance**: **PASS** AUC ≥ 0.65 & p < {fdr} (clear "
                     f"discrimination) · **WARN** AUC ≥ 0.55 & p < {fdr} (real but modest) · **FAIL** "
                     "otherwise. With < 5 genes-with-≥2-guides or < 5 same-gene pairs ⇒ **WARN-only "
                     "(uninformative — never FAIL)**. (Separation z = (same−bg)/SD(bg) is reported as a "
                     "secondary number but is *not* the driver: it is variance-sensitive and understates a "
                     "separation when the background distribution is wide, as it is for essential-gene screens.)")
            L.append("")
            L.append("| metric | AUC P(same>bg) | MWU p | same-gene mean | background mean | separation z | empirical p |")
            L.append("|---|---|---|---|---|---|---|")
            for _col, disp, lab, _xr in TEST5_METRICS:
                tag = " **(primary)**" if lab == prim else ""
                L.append(f"| {disp}{tag} | {fmt(m.get('auc_' + lab))} | {fmt(m.get('mwu_p_' + lab))} | "
                         f"{fmt(m.get('same_gene_mean_' + lab))} | {fmt(m.get('background_mean_' + lab))} | "
                         f"{fmt(m.get('separation_z_' + lab))} | {fmt(m.get('emp_p_' + lab))} |")
            L.append("")
            L.append("_**AUC** (verdict driver) = P(same-gene pair more concordant than a random unrelated "
                     "pair), the Mann-Whitney effect size — rank-based, so robust to the wide/skewed "
                     "background, and not inflated by sample size. **MWU p** = significance that same-gene > "
                     "background. Separation z & empirical p are secondary. Per-pair values: "
                     "`tables/test_5__same_gene_pairs.csv`, `tables/test_5__background_pairs.csv`._")
            L.append("")
            # cell-count dependence (Test-4 undersampling analog): is weak concordance a biology problem
            # or just undersampled guides?
            vsn = m.get("samegene_primary_vs_ncells_spearman")
            if vsn is not None and isinstance(vsn, (int, float)) and np.isfinite(vsn):
                interp = ("a clear **positive** trend ⇒ the (modest) same-gene concordance is real but "
                          "**cell-limited** (more cells per guide ⇒ stronger agreement), mirroring the "
                          "Test-4 undersampling finding." if vsn >= 0.15 else
                          ("**≈ 0** ⇒ low same-gene concordance persists even for well-powered guides — a "
                           "**genuine ceiling**, not just undersampling." if vsn > -0.15 else
                           "**negative** ⇒ no evidence that more cells per guide improve same-gene "
                           "concordance here (the low concordance is not explained by undersampling)."))
                L.append(f"**Is weak concordance biology or undersampling?** Spearman(cells-per-guide, "
                         f"same-gene {prim_disp}) = **{fmt(vsn)}** — {interp}")
                L.append("")
                vpng = "plots/test_5_vs_ncells.png"
                if os.path.exists(os.path.join(outdir, vpng)):
                    L.append(f"![same-gene concordance vs cells per guide]({vpng})")
                    L.append("")
        elif r.metrics:
            L.append("| metric | value |")
            L.append("|---|---|")
            for k, v in r.metrics.items():
                L.append(f"| `{k}` | {fmt(v)} |")
            L.append("")
        # Test 2: state the sample design (cells/arm, # pseudobulk samples, # repeats) in plain prose
        if r.name == "test_2" and r.verdict != "SKIP" and r.metrics:
            mm = r.metrics
            nctrl = mm.get("design_n_control_cells") or fp.get("n_control_cells")
            cpa = mm.get("design_cells_per_arm")
            nsplit = mm.get("design_n_splits") or cfg.get("n_resamples")
            pb = mm.get("design_n_pseudobulk_samples")
            nbatch = mm.get("design_n_replicate_units")
            agg_txt = ""
            if cfg.get("de_method") == "pydeseq2" and pb:
                agg_txt = (f" For DESeq2 these cells are then **aggregated into ~{pb} pseudobulk samples** "
                           f"({nbatch} {cfg.get('replicate_col', 'batch')}es × 2 arms), and the Wald test is "
                           "run on those replicate-level pseudobulk profiles.")
            L.append(
                "*Sample design (ground truth).* Both arms are **control cells** (same population ⇒ a true "
                f"null). All **{nctrl:,} control cells** are used every repeat (no cap/subsampling): each "
                f"repeat is a fresh batch-stratified random split into two arms of **~{cpa:,} cells each** "
                f"(balanced within `{cfg.get('block_cols')}`).{agg_txt} The split is repeated for "
                f"**{nsplit} draws** (`n_resamples`); per-draw λ_GC / frac_sig / ks_p are averaged "
                "(draw-to-draw SD of λ_GC reported), and all draws' p-values are pooled for the QQ plot.")
            L.append("")
        # Test 3: cell-count-stratified p-value diagnostics (real vs shuffled) — how to read them
        if r.name == "test_3" and r.verdict != "SKIP":
            m3 = r.metrics
            nperm_t3 = cfg.get("n_resamples")
            blk = [c for c in cfg.get("block_cols", []) if c]
            blk_txt = (f"within `{blk}`" if blk else "globally (no block columns)")
            L.append(
                "*Design (ground truth).* The perturbation labels are randomly **shuffled among the "
                f"non-control cells {blk_txt}** (the control cells keep their label) and the **same DE is "
                f"re-run** — repeated for **{nperm_t3} permutations** (`n_resamples`). This destroys the "
                "real perturbation→expression coupling while preserving batch structure, library-size, and "
                "the number of cells per group, so the shuffled run is a matched **negative control**.")
            L.append("")
            L.append(
                "**How to read the p-value diagnostic** (`plots/test_3_pvalue_diagnostics.png`). Each "
                "perturbation contributes the per-gene p-values of its perturbation-vs-control DE, pooled "
                "over **all perturbations × all tested genes**, for the **real (unshuffled)** run and the "
                "**shuffled** run:\n"
                "- **(A) ECDF** — the real curve should bow **above** the y=x diagonal (an **excess of "
                "small p-values** = genuine signal); the shuffled curve should **track the diagonal** "
                "(p ~ Uniform[0,1] = a calibrated null).\n"
                "- **(B) QQ vs Uniform** (−log10) — real points rise **above** the y=x line (small-p "
                "excess); shuffled points **hug** the line if the null is calibrated, or rise above it if "
                "the pipeline is anti-conservative even with no real signal.\n"
                "- **(C) per-perturbation** fraction of genes with p<0.05 vs **cell count** (log-x): real "
                "should sit **well above** shuffled, and shuffled should hover near the 0.05 Uniform "
                "reference regardless of cell count.\n"
                "- **(D) ECDF stratified by cell-count tertile** — checks whether the shuffled null stays "
                "Uniform and the real small-p excess holds for **small- vs large-cell-count** "
                "perturbations (a cell-count-dependent shuffled curve would flag size-driven miscalibration).")
            L.append("")
            L.append(
                f"**What the numbers say here.** Pooled over perturbations × genes, the **real** run has "
                f"median frac(p<0.05)={fmt(m3.get('real_frac_p_lt_05'))} (frac_sig={fmt(m3.get('real_frac_sig'))}, "
                f"pooled λ_GC={fmt(m3.get('real_lambda_gc_pooled'))}) versus the **shuffled** run's "
                f"frac(p<0.05)={fmt(m3.get('shuffled_frac_p_lt_05'))} "
                f"(frac_sig={fmt(m3.get('shuffled_frac_sig'))}, pooled λ_GC="
                f"{fmt(m3.get('shuffled_lambda_gc_pooled'))}, median per-pert ks_p_uniform="
                f"{fmt(m3.get('shuffled_ks_p_uniform_median'))}). A large real-vs-shuffled gap with a "
                "near-Uniform shuffled null is the desired outcome; if the shuffled null itself shows a "
                "small-p excess (frac(p<0.05) ≫ 0.05 or λ_GC ≫ 1), the metric is anti-conservative and "
                "the separation z above is overstated. Full tidy numbers: "
                "`tables/test_3__pvalues_by_cellcount.csv` (per-perturbation, per-kind, with stratum) and "
                "`tables/test_3__pooled_pvalues.csv` (subsampled pooled p-values).")
            L.append("")
        # Test 0: show the full δ-by-δ calibration curve inline (the λ_GC blow-up is the headline number)
        if r.name == "test_0":
            ccsv = os.path.join(outdir, "tables", "test_0__calibration_curve.csv")
            if os.path.exists(ccsv):
                cdf = pl.read_csv(ccsv)
                L.append("Calibration curve (injected δ in log2FC; δ=0 is the null FPR baseline):")
                L.append("")
                L.append("| δ (log2FC) | TPR (injected) | FPR (untouched) | median obs LFC | λ_GC |")
                L.append("|---|---|---|---|---|")
                for row in cdf.iter_rows(named=True):
                    L.append(f"| {fmt(row['delta_log2fc'])} | {fmt(row['TPR'])} | {fmt(row['FPR'])} | "
                             f"{fmt(row['median_observed_lfc_injected'])} | {fmt(row['lambda_gc'])} |")
                L.append("")
                _fn, _fh = r.metrics.get("null_FPR"), r.metrics.get("FPR_at_max_delta")
                _coupled = (isinstance(_fn, (int, float)) and isinstance(_fh, (int, float))
                            and _fh > max(2 * cfg["fdr_threshold"], _fn + 0.05))
                if _coupled:
                    L.append("_FPR (and often λ_GC) climbs steeply as δ grows — the compositional-coupling "
                             "artifact (false DE in untouched genes from library-size renormalization under "
                             "strong, widespread injected effects). Note this realistic regime injects only "
                             "~a dozen genes, so coupling here is notable._")
                else:
                    L.append("_FPR stays flat near the null as δ grows — **no compositional coupling**: "
                             "injecting only ~a dozen genes barely shifts the per-cell library size, so "
                             "untouched genes are not renormalized into false positives. The TPR ramp with δ "
                             "is the resolving-power curve._")
                L.append("")
                # per-gene-class FPR (untouched genes) by δ
                grp_cols = [(c, f"FPR_{c}") for c in ("AnchorCorr", "HighlyExpr", "LowlyExpr",
                            "HouseKeeping", "Marker", "HighlyVarG", "Random")
                            if f"FPR_{c}" in cdf.columns]
                if grp_cols:
                    L.append("FPR by gene class (false calls among **untouched** genes; anchors = injected):")
                    L.append("")
                    L.append("| δ (log2FC) | " + " | ".join(c for c, _ in grp_cols) + " |")
                    L.append("|---|" + "|".join("---" for _ in grp_cols) + "|")
                    for rr in cdf.iter_rows(named=True):
                        L.append(f"| {fmt(rr['delta_log2fc'])} | "
                                 + " | ".join(fmt(rr[col]) for _, col in grp_cols) + " |")
                    L.append("")
                    L.append("_Compare classes: if compositional coupling were present, "
                             "AnchorCorr/HighlyExpr/HighlyVarG would inflate first; LowlyExpr stays low "
                             "(zero-dominated); Random is the baseline; HouseKeeping flags memorization; "
                             "Marker is N/A without cell-type labels._")
                    L.append("")
                tpr_cols = [(f"{t} expr", f"TPR_{t}expr") for t in ("high", "mid", "low")
                            if f"TPR_{t}expr" in cdf.columns]
                if tpr_cols:
                    L.append("TPR by **injected-gene expression tier** (recovery of the anchors; detection "
                             "power is dominated by expression level):")
                    L.append("")
                    L.append("| δ (log2FC) | " + " | ".join(c for c, _ in tpr_cols) + " |")
                    L.append("|---|" + "|".join("---" for _ in tpr_cols) + "|")
                    for rr in cdf.iter_rows(named=True):
                        L.append(f"| {fmt(rr['delta_log2fc'])} | "
                                 + " | ".join(fmt(rr[col]) for _, col in tpr_cols) + " |")
                    L.append("")
                    mtr = {t: m.get(f"min_resolvable_delta_{t}expr") for t in ("high", "mid", "low")}
                    L.append(f"_min resolvable δ (TPR≥0.5): high-expr={fmt(mtr['high'])}, "
                             f"mid={fmt(mtr['mid'])}, low={fmt(mtr['low'])}. Abundant genes are easiest "
                             "to recover; sparse (low-expr) genes are the conservative floor — the real "
                             "limit on what the metric can resolve._")
                    L.append("")
        if r.pvalues:
            L.append("**Verification p-values:** " + ", ".join(f"`{k}`={fmt(v)}" for k, v in r.pvalues.items()))
            L.append("")
        if r.flags:
            for f in r.flags:
                L.append(f"- ⚠️ {f}")
            L.append("")
        if r.plot:
            L.append(f"![{r.name}]({r.plot})")
            L.append("")
        L.append(f"_Full per-gene/per-split numbers: `tables/{r.name}__*.csv`_")
        L.append("")


    # ---- thresholds ----
    L += ["## 5. Verification parameters & thresholds", "",
          "| parameter | value | source |", "|---|---|---|",
          f"| de_method / unit | {cfg['de_method']} / {unit} | config |",
          f"| covariate_correction | {cfg.get('covariate_correction','none')} | config |",
          f"| fdr_threshold | {cfg['fdr_threshold']} | config |",
          f"| lfc_threshold | {cfg['lfc_threshold']} | config |",
          f"| lambda_gc_warn / fail | {cfg['lambda_gc_warn']} / {cfg['lambda_gc_fail']} | TEST_PLAN.md |",
          "| λ_GC deflation WARN / ks_p_uniform WARN | < 0.90 / < 0.05 | skill notes |",
          f"| injection δ tiers (log2FC) | {cfg['injection_deltas']} | config |",
          (f"| injection genes/draw × draws | {cfg.get('injection_n_genes')} × {cfg.get('injection_n_repeats')} | config |"
           if cfg.get('injection_n_genes') is not None
           else f"| injection fraction of genes | {cfg.get('injection_frac_genes')} | config (legacy frac mode) |"),
          f"| n_resamples / min_cells_per_group / seed | {cfg['n_resamples']} / "
          f"{cfg['min_cells_per_group']} / {cfg['seed']} | config |",
          f"| max_conditions / block_cols | {cfg.get('max_conditions')} / {cfg.get('block_cols')} | config |",
          "| test-0 null FPR FAIL | > 2×α | skill notes |",
          "| test-3 separation PASS / WARN | > 2 / 1–2 | TEST_PLAN.md |",
          "| test-4 LOW-REPRODUCIBILITY flag | median LFC ρ < 0.2 | TEST_PLAN.md |",
          "| test-5 separation PASS / WARN | > 1.5 / 1.0–1.5 | TEST_PLAN.md |",
          "| test-6 recovery / direction PASS | > 0.5 / > 0.8 | TEST_PLAN.md |", ""]

    # ---- glossary ----
    L += ["## 6. Glossary", ""]
    for term, definition in GLOSSARY:
        L.append(f"- **{term}** — {definition}")
    L.append("")

    # ---- limitations / next steps ----
    L += ["## 7. Known limitations & recommended next steps", "",
          "- **Single dataset / modality.** Metric behaviour is partly experiment-dependent; validate "
          "on ≥2 datasets/modalities (e.g. a CRISPRi Perturb-seq screen and a chemical/Tahoe-style "
          "screen) before trusting thresholds.",
          "- **Cell-level unit of analysis.** With pdex, null p-values are not calibrated (see §2). "
          "Re-run with `de_method: pydeseq2` for a pseudobulk comparison.",
          "- **Composition not assessed** when no cell-state column is present (see the Composition "
          "test). Proliferation/differentiation shifts can masquerade as expression change.",
          "- **Corrected vs uncorrected.** Evaluate the metric on both; this run is "
          f"`{cfg.get('covariate_correction','none')}`.",
          "- Gene-set / pathway enrichment metrics are intentionally **not** included: enrichment is "
          "unsolved (databases disagree; many effects need multi-gene perturbations) and would be a "
          "noisy gate.", ""]

    # ---- appendix: full test plan (LAST) ----
    L += ["## Appendix — embedded TEST_PLAN.md", ""]
    plan = os.path.join(outdir, "TEST_PLAN.md")
    if os.path.exists(plan):
        with open(plan) as fh:
            for ln in fh.read().splitlines():
                L.append(("#" + ln) if ln.startswith("#") else ln)
    md = "\n".join(L)
    with open(os.path.join(outdir, "robustness_report.md"), "w") as fh:
        fh.write(md)
    return md


def write_summary(results, cfg, power, fp, outdir):
    summary = {"config": cfg, "global_verdict": global_verdict(results), "power": power,
               "fingerprint": fp,
               "tests": [asdict(r) for r in results]}
    with open(os.path.join(outdir, "robustness_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)


def write_multiqc(results, cfg, outdir):
    """Emit a MultiQC custom-content table (experimental — github.com/MultiQC/MultiQC)."""
    if not cfg.get("emit_multiqc"):
        return
    d = os.path.join(outdir, "multiqc")
    os.makedirs(d, exist_ok=True)
    header = [
        "# id: 'cell_eval_robustness'",
        "# section_name: 'cell-eval metric robustness'",
        f"# description: 'DE backend {cfg['de_method']}; global verdict "
        f"{global_verdict(results)}. Validity gates 0-3 must pass before sensitivity tests 4-6.'",
        "# plot_type: 'table'",
        "# pconfig:",
        "#     namespace: 'cell-eval robustness'",
        "test\tverdict\treason",
    ]
    lines = header + [f"{r.name} — {r.title}\t{r.verdict}\t{r.reason}" for r in results]
    with open(os.path.join(d, "cell_eval_robustness_mqc.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def render_pdf(md_path, pdf_path):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from reportlab.lib.utils import ImageReader
        import re

        base, bold = "Helvetica", "Helvetica-Bold"
        try:
            import matplotlib as mpl
            fd = os.path.join(os.path.dirname(mpl.__file__), "mpl-data", "fonts", "ttf")
            pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(fd, "DejaVuSans.ttf")))
            pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(fd, "DejaVuSans-Bold.ttf")))
            base, bold = "DejaVu", "DejaVu-Bold"
        except Exception:
            pass
        ss = getSampleStyleSheet()
        body = ParagraphStyle("b", parent=ss["BodyText"], fontName=base, fontSize=8.5, leading=11)
        h1 = ParagraphStyle("h1", parent=ss["Title"], fontName=bold, fontSize=15)
        h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName=bold, fontSize=11, textColor=colors.HexColor("#1a3c6e"))
        h3 = ParagraphStyle("h3", parent=ss["Heading3"], fontName=bold, fontSize=9.5, textColor=colors.HexColor("#333333"))
        cell = ParagraphStyle("cell", fontName=base, fontSize=7.3, leading=8.8)
        cell_hdr = ParagraphStyle("cellh", fontName=bold, fontSize=7.5, leading=9, textColor=colors.white)
        USABLE = 7.1 * inch

        def inline(s: str) -> str:
            s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
            s = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", s)
            return s

        def split_row(line: str):
            return [c.strip() for c in line.strip().strip("|").split("|")]

        def is_sep(line: str) -> bool:
            s = line.strip()
            return s.startswith("|") and bool(re.fullmatch(r"[|:\-\s]+", s)) and "-" in s

        def make_table(header, rows):
            ncol = len(header)
            rows = [r + [""] * (ncol - len(r)) if len(r) < ncol else r[:ncol] for r in rows]
            plain = lambda c: re.sub(r"[*`]", "", c)
            maxlens = [max(1, len(plain(header[j])), *(len(plain(r[j])) for r in rows)) for j in range(ncol)]
            tot = sum(maxlens)
            widths = [max(0.42 * inch, USABLE * ml / tot) for ml in maxlens]
            sc = USABLE / sum(widths)
            widths = [w * sc for w in widths]
            data = [[Paragraph(inline(c), cell_hdr) for c in header]]
            data += [[Paragraph(inline(c), cell) for c in r] for r in rows]
            t = Table(data, colWidths=widths, repeatRows=1)
            t.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7ced8")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c6e")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f8")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]))
            return t

        md_dir = os.path.dirname(os.path.abspath(md_path))
        lines = open(md_path).read().splitlines()
        flow = []
        i, n = 0, len(lines)
        while i < n:
            ln = lines[i]
            stripped = ln.strip()
            # markdown table block: a "| ... |" header followed by a |---| separator
            if (stripped.startswith("|") and i + 1 < n and is_sep(lines[i + 1])
                    and not is_sep(ln)):
                header = split_row(ln)
                j = i + 2
                rows = []
                while j < n and lines[j].strip().startswith("|") and not is_sep(lines[j]):
                    rows.append(split_row(lines[j]))
                    j += 1
                flow.append(Spacer(1, 3))
                flow.append(make_table(header, rows))
                flow.append(Spacer(1, 5))
                i = j
                continue
            img = re.match(r"!\[.*?\]\((.+?)\)", stripped)
            if img:
                p = os.path.join(md_dir, img.group(1))
                if os.path.exists(p):
                    iw, ih = ImageReader(p).getSize()           # preserve aspect ratio
                    scale = min(6.9 * inch / iw, 3.6 * inch / ih)
                    flow += [Spacer(1, 4), Image(p, width=iw * scale, height=ih * scale), Spacer(1, 4)]
            elif stripped.startswith("```"):
                pass  # skip code-fence markers
            elif ln.startswith("### "):
                flow.append(Paragraph(inline(ln[4:]), h3))
            elif ln.startswith("## "):
                flow.append(Paragraph(inline(ln[3:]), h2))
            elif ln.startswith("# "):
                flow.append(Paragraph(inline(ln[2:]), h1))
            elif stripped:
                flow.append(Paragraph(inline(ln), body))
            else:
                flow.append(Spacer(1, 3))
            i += 1
        SimpleDocTemplate(pdf_path, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                          leftMargin=0.7 * inch, rightMargin=0.7 * inch).build(flow)
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("PDF render failed: %s", e)
        return False


def _results_from_summary(summary):
    return [TestResult(**t) for t in summary["tests"]]


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--report-only", action="store_true",
                    help="Re-render report/PDF/MultiQC from cached robustness_summary.json (no DE recompute)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_cfg(args.config)
    seed_everything(int(cfg.get("seed", 0)))  # full determinism: same seed -> identical report
    outdir = cfg["outdir"]
    os.makedirs(os.path.join(outdir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    if args.report_only:
        with open(os.path.join(outdir, "robustness_summary.json")) as fh:
            summary = json.load(fh)
        results = _results_from_summary(summary)
        cfg = summary.get("config", cfg)
        cfg.setdefault("n_resamples", cfg.get("n_permutations", 10))  # backward-compat for old summaries
        power = summary.get("power", {})
        fp = summary.get("fingerprint", {})
        # Re-apply current verdict rules to cached metrics (no DE recompute) for tests whose verdict
        # is a pure function of their metrics — keeps --report-only consistent with rule updates.
        rederive = {"test_0": verdict_test_0, "test_4": verdict_test_4, "test_5": verdict_test_5}
        for r in results:
            if r.name in rederive and r.verdict != "SKIP" and r.metrics:
                v, reason, flags = rederive[r.name](r.metrics, cfg)
                r.verdict, r.reason, r.flags = v, reason, flags
        # rewrite summary so cached verdicts stay in sync with the report
        write_summary(results, cfg, power, fp, outdir)
        write_report(results, cfg, power, fp, outdir)
        write_multiqc(results, cfg, outdir)
        ok = render_pdf(os.path.join(outdir, "robustness_report.md"), os.path.join(outdir, "report.pdf"))
        print(f"[report-only] re-rendered report.md + {'report.pdf' if ok else '(PDF FAILED)'} in {os.path.abspath(outdir)}")
        return 0

    log.info("Loading %s", cfg["adata_path"])
    adata = ad.read_h5ad(cfg["adata_path"])
    log.info("Loaded %d cells x %d genes", adata.n_obs, adata.n_vars)

    fp = fingerprint(adata, cfg)
    with open(os.path.join(outdir, "inspect.txt"), "w") as fh:
        fh.write(json.dumps(fp, indent=2, default=str))

    want = set(cfg.get("tests") or
               ["test_0", "test_1", "test_2", "test_3", "test_4", "test_5", "test_6", "composition"])
    results = []
    de_true = None

    # Test 0 runs on RAW counts (before global normalization)
    if "test_0" in want:
        log.info("=== test_0 (injection) ===")
        results.append(test_0_injection(adata, cfg, outdir))

    # pdex expects log-normalized input; match cell-eval run's normalize_total + log1p
    maybe_normalize(adata, cfg)

    if "composition" in want:
        log.info("=== composition ==="); results.append(composition_diagnostic(adata, cfg, outdir))
    if "test_1" in want:
        log.info("=== test_1 ==="); results.append(test_1(adata, cfg, outdir))
    if "test_2" in want:
        log.info("=== test_2 ==="); results.append(test_2(adata, cfg, outdir))
    if "test_3" in want:
        log.info("=== test_3 ==="); r3, de_true = test_3(adata, cfg, outdir); results.append(r3)
    if "test_4" in want:
        log.info("=== test_4 ==="); results.append(test_4(adata, cfg, outdir))
    if "test_5" in want:
        log.info("=== test_5 ==="); results.append(test_5(adata, cfg, outdir))
    if "test_6" in want:
        log.info("=== test_6 ==="); r6, de_true = test_6(adata, cfg, outdir, de_true=de_true); results.append(r6)

    # order results for the report: gates first (0-3), then sensitivity (4-6), then composition
    order = {"test_0": 0, "test_1": 1, "test_2": 2, "test_3": 3, "test_4": 4, "test_5": 5,
             "test_6": 6, "composition": 7}
    results.sort(key=lambda r: order.get(r.name, 99))

    power = {}  # power/sample-size calculation removed

    write_summary(results, cfg, power, fp, outdir)
    write_report(results, cfg, power, fp, outdir)
    write_multiqc(results, cfg, outdir)
    ok = render_pdf(os.path.join(outdir, "robustness_report.md"), os.path.join(outdir, "report.pdf"))
    gv = global_verdict(results)
    print("\n==== RESULTS ====")
    for r in results:
        print(f"  {r.name:11s} {r.verdict:5s} {r.title}")
    print(f"  GLOBAL: {gv}")
    print(f"  report.md + {'report.pdf' if ok else '(PDF FAILED)'} in {os.path.abspath(outdir)}")
    return 1 if gv == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
