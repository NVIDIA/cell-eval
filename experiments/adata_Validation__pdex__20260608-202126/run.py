#!/usr/bin/env python
"""Standalone, reproducible robustness experiment for cell-eval metrics (pdex backend).

Reads ``config.yaml`` + the dataset and runs the six tests from ``TEST_PLAN.md`` using the *real*
cell-eval DE backend (``cell_eval._de_backends.build_de_frame``). Writes per-test tables, QQ plots,
a JSON summary, a markdown report (with all verification p-values, the power/sample-size analysis,
the thresholds table, and the embedded test plan), and renders ``report.pdf``.

For ``de_method: pdex`` the cell-level Wilcoxon test expects log-normalized input. ``cell-eval run``
applies ``normalize_total`` + ``log1p`` inside ``MetricsEvaluator`` before pdex; since this runner
drives ``build_de_frame`` directly (which bypasses that step), it applies the same normalization once
up front when ``normalize_if_raw`` is set and ``.X`` is raw integer counts. normalize_total + log1p
are per-cell operations, so normalizing once and then subsetting is identical to normalizing each
subset. ``allow_discrete: false`` then makes pdex treat the matrix as log1p data (``is_log1p=True``).

Reproducible: a single ``seed`` drives every split/permutation. Re-running with the same config and
seed reproduces the report.

    python run.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field

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
    return cfg


def _looks_raw_integer(adata: ad.AnnData) -> bool:
    """True if .X appears to be raw integer counts (sampled over nonzero values)."""
    X = adata.X
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        return False
    if data.size > 5_000_000:  # subsample for speed; raw/integer is a global property
        rng = np.random.default_rng(0)
        data = data[rng.integers(0, data.size, 5_000_000)]
    return bool(np.allclose(data, np.rint(data)))


def maybe_normalize(adata: ad.AnnData, cfg: dict) -> None:
    """Match cell-eval's normalize_total + log1p when pdex receives raw counts (inplace)."""
    if not cfg.get("normalize_if_raw"):
        return
    if _looks_raw_integer(adata):
        log.info("normalize_if_raw: .X is raw integer counts -> normalize_total + log1p")
        sc.pp.normalize_total(adata, inplace=True)  # normalize to median library size
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
    """frac_sig, mean_lfc, mean_abs_lfc, ks_p_uniform, lambda_gc over a (single-target) DE frame."""
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
    # thin for plotting
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
    direction = float(np.mean(np.sign(la[sa | sb]) == np.sign(lb[sa | sb]))) if union else float("nan")
    # AUC: |LFC_a| ranking predicts membership in S_b
    auc = float("nan")
    if sb.sum() > 0 and (~sb).sum() > 0:
        try:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(sb.astype(int), np.abs(la)))
        except Exception:
            auc = float("nan")
    return {
        "lfc_spearman": spear,
        "sig_jaccard": float(inter / union) if union else float("nan"),
        "direction_agreement": direction,
        "auc_recovery": auc,
        "n_sig_a": int(sa.sum()),
        "n_sig_b": int(sb.sum()),
    }


def stratified_split(obs, strat_cols, seed: int) -> np.ndarray:
    """Balanced A/B labels within each stratum (group by strat_cols present in obs)."""
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


def verdict_from_null(m: dict, cfg: dict) -> tuple[str, list]:
    """Two-sided null verdict (TEST_PLAN.md + skill verdict notes).

    FAIL on anti-conservative behaviour (λ_GC > fail, or frac_sig >> fdr_threshold).
    WARN on mild inflation OR on deflation / non-uniformity (λ_GC < 0.9 or ks_p_uniform < 0.05)
    even when frac_sig ~= 0 — a deflated null is safe but under-powered and must be surfaced.
    """
    flags = []
    lam = m.get("lambda_gc", float("nan"))
    frac = m.get("frac_sig", float("nan"))
    ksp = m.get("ks_p_uniform", float("nan"))
    # ---- anti-conservative -> FAIL ----
    if np.isfinite(lam) and lam > cfg["lambda_gc_fail"]:
        flags.append(f"λ_GC={lam:.3f} > {cfg['lambda_gc_fail']} (anti-conservative)")
        return "FAIL", flags
    if np.isfinite(frac) and frac > 5 * cfg["fdr_threshold"]:
        flags.append(f"frac_sig={frac:.3f} >> fdr_threshold={cfg['fdr_threshold']}")
        return "FAIL", flags
    # ---- mild inflation -> WARN ----
    if (np.isfinite(lam) and lam > cfg["lambda_gc_warn"]) or (np.isfinite(frac) and frac > 2 * cfg["fdr_threshold"]):
        flags.append(f"λ_GC={lam:.3f} / frac_sig={frac:.3f} mildly elevated")
        return "WARN", flags
    # ---- deflation / non-uniformity -> WARN (safe but under-powered) ----
    defl = []
    if np.isfinite(lam) and lam < 0.9:
        defl.append(f"λ_GC={lam:.3f} < 0.9 (deflated null — under-powered)")
    if np.isfinite(ksp) and ksp < 0.05:
        defl.append(f"ks_p_uniform={ksp:.3g} < 0.05 (p-values not Uniform[0,1])")
    if defl:
        return "WARN", defl
    return "PASS", flags


# --------------------------------------------------------------------------- #
# the six tests
# --------------------------------------------------------------------------- #
def perts_in_use(adata, cfg):
    tg = adata.obs[cfg["pert_col"]].astype(str)
    perts = [g for g in sorted(tg.unique()) if g != cfg["control_pert"]]
    if cfg.get("max_conditions"):
        perts = perts[: cfg["max_conditions"]]
    return perts


def test_1(adata, cfg, outdir):
    perts = perts_in_use(adata, cfg)
    pc, mcg = cfg["pert_col"], cfg["min_cells_per_group"]
    rows, pooled_p, skipped = [], [], 0
    for pert in perts:
        sub = adata[adata.obs[pc].astype(str) == pert]
        if sub.n_obs < 2 * mcg:
            skipped += 1
            continue
        for r in range(cfg["n_permutations"]):
            arm = stratified_split(sub.obs, cfg["block_cols"], cfg["seed"] + r)
            if (arm == "A").sum() < mcg or (arm == "B").sum() < mcg:
                continue
            s = sub.copy()
            s.obs["_arm"] = arm
            try:
                de = run_de(s, cfg, groupby="_arm", reference="B")
            except Exception as e:  # noqa: BLE001
                log.warning("test1 %s rep%d: %s", pert, r, e)
                continue
            m = null_metrics(de, cfg)
            m.update({"condition": pert, "rep": r})
            rows.append(m)
            pooled_p.append(de["p_value"].to_numpy().astype(float))
    if not rows:
        return TestResult("test_1", "Within-Condition Direct Split Null", "SKIP",
                          flags=["no condition had enough cells for an A/B split"])
    df = pl.DataFrame(rows)
    df.write_csv(os.path.join(outdir, "tables", "test_1__per_split.csv"))
    agg = {k: float(df[k].mean()) for k in ("frac_sig", "mean_lfc", "mean_abs_lfc", "lambda_gc", "ks_p_uniform")}
    agg["lambda_gc_sd"] = float(df["lambda_gc"].std()) if df.height > 1 else 0.0
    p = np.concatenate(pooled_p)
    png = os.path.join(outdir, "plots", "test_1_qq.png")
    qq_plot(p, png, "Test 1 — within-condition null")
    verdict, flags = verdict_from_null(agg, cfg)
    if np.isfinite(agg["lambda_gc_sd"]) and agg["lambda_gc_sd"] > 0.05:
        flags.append(f"split-to-split SD(λ_GC)={agg['lambda_gc_sd']:.3f} > 0.05 (residual confounding?)")
        if verdict == "PASS":
            verdict = "WARN"
    if skipped:
        flags.append(f"{skipped}/{len(perts)} conditions skipped (too few cells)")
        if skipped > 0.5 * len(perts) and verdict == "PASS":
            verdict = "WARN"
    return TestResult("test_1", "Within-Condition Direct Split Null", verdict, metrics=agg, flags=flags,
                      pvalues={"ks_p_uniform_mean": agg["ks_p_uniform"], "lambda_gc": agg["lambda_gc"]},
                      plot="plots/test_1_qq.png")


def test_2(adata, cfg, outdir):
    pc, mcg = cfg["pert_col"], cfg["min_cells_per_group"]
    ctrl = adata[adata.obs[pc].astype(str) == cfg["control_pert"]]
    if ctrl.n_obs < 2 * mcg:
        return TestResult("test_2", "Control-Control Split Null", "SKIP", flags=["too few control cells"])
    rows, pooled_p = [], []
    for r in range(cfg["n_permutations"]):
        arm = stratified_split(ctrl.obs, cfg["block_cols"], cfg["seed"] + 100 + r)
        s = ctrl.copy()
        s.obs["_arm"] = arm
        try:
            de = run_de(s, cfg, groupby="_arm", reference="B")
        except Exception as e:  # noqa: BLE001
            log.warning("test2 rep%d: %s", r, e)
            continue
        m = null_metrics(de, cfg)
        m["rep"] = r
        rows.append(m)
        pooled_p.append(de["p_value"].to_numpy().astype(float))
    if not rows:
        return TestResult("test_2", "Control-Control Split Null", "SKIP", flags=["no valid control split"])
    df = pl.DataFrame(rows)
    df.write_csv(os.path.join(outdir, "tables", "test_2__per_split.csv"))
    agg = {k: float(df[k].mean()) for k in ("frac_sig", "mean_lfc", "mean_abs_lfc", "lambda_gc", "ks_p_uniform")}
    agg["lambda_gc_sd"] = float(df["lambda_gc"].std()) if df.height > 1 else 0.0
    qq_plot(np.concatenate(pooled_p), os.path.join(outdir, "plots", "test_2_qq.png"),
            "Test 2 — control-control null")
    verdict, flags = verdict_from_null(agg, cfg)
    if np.isfinite(agg["lambda_gc_sd"]) and agg["lambda_gc_sd"] > 0.05:
        flags.append(f"split-to-split SD(λ_GC)={agg['lambda_gc_sd']:.3f} > 0.05 (residual confounding?)")
        if verdict == "PASS":
            verdict = "WARN"
    return TestResult("test_2", "Control-Control Split Null", verdict, metrics=agg, flags=flags,
                      pvalues={"ks_p_uniform_mean": agg["ks_p_uniform"], "lambda_gc": agg["lambda_gc"]},
                      plot="plots/test_2_qq.png")


def _signal_metric(de: pl.DataFrame, cfg: dict) -> float:
    """Mean per-target frac_sig — the DE signal that should collapse under permutation."""
    lfc = de["log2_fold_change"].to_numpy().astype(float)
    fdr = de["fdr"].to_numpy().astype(float)
    sig = _sig_mask(lfc, fdr, cfg)
    tgt = de["target"].to_numpy().astype(str)
    fr = []
    for t in np.unique(tgt):
        msk = tgt == t
        fr.append(sig[msk].sum() / max(msk.sum(), 1))
    return float(np.mean(fr)) if fr else float("nan")


def test_3(adata, cfg, outdir, de_true=None):
    pc = cfg["pert_col"]
    perts = perts_in_use(adata, cfg)
    keep = adata[adata.obs[pc].astype(str).isin(perts + [cfg["control_pert"]])].copy()
    if de_true is None:
        de_true = run_de(keep, cfg, groupby=pc, reference=cfg["control_pert"])
    true_metric = _signal_metric(de_true, cfg)
    rng = np.random.default_rng(cfg["seed"] + 3)
    blocks = [c for c in cfg["block_cols"] if c in keep.obs.columns]
    perm_metrics = []
    labels0 = keep.obs[pc].astype(str).to_numpy()
    is_ctrl = labels0 == cfg["control_pert"]
    for r in range(cfg["n_permutations"]):
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
        except Exception as e:  # noqa: BLE001
            log.warning("test3 perm%d: %s", r, e)
    if len(perm_metrics) < 2:
        return TestResult("test_3", "Label Permutation Null", "SKIP", flags=["not enough permutations"]), de_true
    pm = np.array(perm_metrics, dtype=float)
    sep = (true_metric - pm.mean()) / pm.std(ddof=1) if pm.std(ddof=1) > 0 else float("inf")
    perm_p = float((pm >= true_metric).mean())
    pl.DataFrame({"perm_signal": pm}).write_csv(os.path.join(outdir, "tables", "test_3__permutations.csv"))
    if not np.isfinite(sep):
        verdict = "PASS" if true_metric > pm.max() else "WARN"
    elif sep > 2:
        verdict = "PASS"
    elif sep > 1:
        verdict = "WARN"
    else:
        verdict = "FAIL"
    return TestResult("test_3", "Label Permutation Null", verdict,
                      metrics={"true_signal": true_metric, "perm_mean": float(pm.mean()),
                               "perm_sd": float(pm.std(ddof=1)), "separation_z": float(sep)},
                      flags=[], pvalues={"perm_p": perm_p, "separation_z": float(sep)}), de_true


def test_4(adata, cfg, outdir):
    pc, sg, mcg = cfg["pert_col"], cfg.get("sgrna_col"), cfg["min_cells_per_group"]
    if not sg:
        return TestResult("test_4", "Same-sgRNA Split Reproducibility", "SKIP", flags=["no sgrna_col"])
    perts = perts_in_use(adata, cfg)
    tg = adata.obs[pc].astype(str)
    guides = sorted(adata.obs[sg].astype(str)[tg.isin(perts)].unique())
    rows = []
    for guide in guides:
        gmask = (adata.obs[sg].astype(str) == guide).to_numpy()
        cmask = (tg == cfg["control_pert"]).to_numpy()
        if gmask.sum() < 2 * mcg:
            continue
        gobs = adata.obs[gmask]
        arm = stratified_split(gobs, cfg["block_cols"], cfg["seed"] + 4)
        if (arm == "A").sum() < mcg or (arm == "B").sum() < mcg:
            continue
        des = {}
        for a in ("A", "B"):
            keepmask = cmask.copy()
            sel = np.where(gmask)[0][arm == a]
            keepmask[sel] = True
            s = adata[keepmask].copy()
            lab = np.where((s.obs[pc].astype(str) == cfg["control_pert"]).to_numpy(), cfg["control_pert"], guide)
            s.obs["_g"] = lab
            try:
                des[a] = run_de(s, cfg, groupby="_g", reference=cfg["control_pert"])
            except Exception as e:  # noqa: BLE001
                log.warning("test4 %s arm%s: %s", guide, a, e)
        if "A" in des and "B" in des:
            cmp = compare_signatures(des["A"], des["B"], cfg)
            if cmp:
                cmp["guide"] = guide
                rows.append(cmp)
    if not rows:
        return TestResult("test_4", "Same-sgRNA Split Reproducibility", "SKIP",
                          flags=["no guide had enough cells for an A/B split"])
    df = pl.DataFrame(rows)
    df.write_csv(os.path.join(outdir, "tables", "test_4__per_guide.csv"))
    med_rho = float(df["lfc_spearman"].median())
    return TestResult("test_4", "Same-sgRNA Split Reproducibility", "PASS" if med_rho >= 0.2 else "WARN",
                      metrics={"n_guides": df.height, "median_lfc_spearman": med_rho,
                               "median_jaccard": float(df["sig_jaccard"].median()),
                               "median_direction": float(df["direction_agreement"].median())},
                      flags=[] if med_rho >= 0.2 else [f"median LFC ρ={med_rho:.2f} < 0.2 (LOW REPRODUCIBILITY)"])


def test_5(adata, cfg, outdir):
    pc, sg, mcg = cfg["pert_col"], cfg.get("sgrna_col"), cfg["min_cells_per_group"]
    if not sg:
        return TestResult("test_5", "Same-Gene Independent sgRNA Reproducibility", "SKIP", flags=["no sgrna_col"])
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
                          flags=["no gene with >=2 guides among selected conditions"])
    allg = sorted({g for gs in gene2guides.values() for g in gs})
    cmask = (tg == cfg["control_pert"]).to_numpy()
    de_by_guide = {}
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
        except Exception as e:  # noqa: BLE001
            log.warning("test5 %s: %s", guide, e)
    same, bg = [], []
    for gene, gs in gene2guides.items():
        gs = [g for g in gs if g in de_by_guide]
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                c = compare_signatures(de_by_guide[gs[i]], de_by_guide[gs[j]], cfg)
                if c:
                    same.append(c["lfc_spearman"])
    glist = list(de_by_guide)
    rng = np.random.default_rng(cfg["seed"] + 5)
    guide_gene = {g: gene for gene, gs in gene2guides.items() for g in gs}
    tries = 0
    while len(bg) < max(len(same), 3) and tries < 200 and len(glist) >= 2:
        tries += 1
        a, b = rng.choice(glist, size=2, replace=False)
        if guide_gene.get(a) == guide_gene.get(b):
            continue
        c = compare_signatures(de_by_guide[a], de_by_guide[b], cfg)
        if c:
            bg.append(c["lfc_spearman"])
    if not same or not bg:
        return TestResult("test_5", "Same-Gene Independent sgRNA Reproducibility", "SKIP",
                          flags=["insufficient same-gene or background pairs"])
    same, bg = np.array(same), np.array(bg)
    sep = (same.mean() - bg.mean()) / bg.std(ddof=1) if bg.std(ddof=1) > 0 else float("inf")
    pl.DataFrame({"same_gene_rho": np.pad(same, (0, max(0, len(bg) - len(same))), constant_values=np.nan)[:len(bg)]
                  if len(bg) >= len(same) else same}).write_csv(os.path.join(outdir, "tables", "test_5__pairs.csv"))
    verdict = "PASS" if (not np.isfinite(sep) and same.mean() > bg.mean()) or sep > 1.5 else (
        "WARN" if (np.isfinite(sep) and sep > 1.0) else "FAIL")
    return TestResult("test_5", "Same-Gene Independent sgRNA Reproducibility", verdict,
                      metrics={"n_genes": len(gene2guides), "n_same_pairs": int(len(same)),
                               "same_gene_mean_rho": float(same.mean()),
                               "background_mean_rho": float(bg.mean()), "separation_z": float(sep)},
                      flags=[], pvalues={"separation_z": float(sep)})


def test_6(adata, cfg, outdir, de_true=None):
    pc = cfg["pert_col"]
    perts = perts_in_use(adata, cfg)
    if de_true is None:
        keep = adata[adata.obs[pc].astype(str).isin(perts + [cfg["control_pert"]])].copy()
        de_true = run_de(keep, cfg, groupby=pc, reference=cfg["control_pert"])
    genes_present = set(de_true["feature"].cast(str).unique().to_list())
    # map condition -> target gene (here pert_col IS the gene symbol)
    rows = []
    for pert in perts:
        target = pert  # target_gene_col == pert_col for this dataset
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
        return TestResult("test_6", "Target Gene Knockdown Recovery", "SKIP", flags=["no target genes found"]), de_true
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
    return TestResult("test_6", "Target Gene Knockdown Recovery", verdict,
                      metrics={"n_targets": df.height, "recovery_rate": rec, "direction_rate": dirr,
                               "median_pval_rank": float(df["pval_rank"].median()),
                               "median_lfc_rank": float(df["lfc_rank"].median())},
                      flags=[], pvalues={"median_padj_target": float(df["padj_target"].median())}), de_true


# --------------------------------------------------------------------------- #
# power / sample-size analysis (TEST_PLAN.md §7) — approximate
# --------------------------------------------------------------------------- #
def power_analysis(adata, cfg, results, de_true):
    gates = [r for r in results if r.name in ("test_1", "test_2", "test_3")]
    if any(r.verdict == "FAIL" for r in gates):
        return {"status": "not computed — a validity gate FAILED (pipeline not calibrated)"}
    lams = [r.metrics.get("lambda_gc") for r in results if r.name in ("test_1", "test_2")
            and np.isfinite(r.metrics.get("lambda_gc", float("nan")))]
    lam = float(np.mean(lams)) if lams else 1.0
    # alpha calibration (TEST_PLAN.md §7.1): only deflate alpha when lam>=1; never inflate above fdr_threshold
    if lam >= 1.0:
        alpha_emp = cfg["fdr_threshold"] / lam
        alpha_note = "alpha_empirical = fdr_threshold / lambda_gc (lambda_gc >= 1)"
    else:
        alpha_emp = cfg["fdr_threshold"]
        alpha_note = ("lambda_gc < 1 (deflation) -> alpha_empirical = fdr_threshold (NOT inflated); "
                      "investigate covariate overcorrection / under-powered null")
    # approximate dispersion phi from control counts (method-of-moments), labelled approximate.
    # NOTE: pdex is a cell-level Wilcoxon test (no NB dispersion fit). phi here is an estimate from
    # the control count matrix only, so the power numbers are planning estimates, not DESeq2-derived.
    ctrl = adata[adata.obs[cfg["pert_col"]].astype(str) == cfg["control_pert"]]
    X = ctrl.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    mu = X.mean(0) + 1e-9
    var = X.var(0)
    phi = np.median(np.clip((var - mu) / (mu ** 2), 1e-3, None))
    # effect tiers from test_true LFCs (detected)
    dvals = []
    if de_true is not None:
        d = de_true
        lfc = d["log2_fold_change"].to_numpy().astype(float)
        fdr = d["fdr"].to_numpy().astype(float)
        det = np.isfinite(fdr) & (fdr <= cfg["fdr_threshold"]) & (np.abs(lfc) >= cfg["lfc_threshold"])
        dvals = np.abs(lfc[det])
    if len(dvals) >= 10:
        d_floor = float(np.percentile(dvals, 10))
        d_typ = float(np.median(dvals))
        d_ceil = float(np.percentile(dvals, 90))
    else:
        d_floor = d_typ = d_ceil = float("nan")
    # reproducibility ceiling rho from test_4
    rho = float("nan")
    for r in results:
        if r.name == "test_4":
            rho = r.metrics.get("median_lfc_spearman", float("nan"))
    mu_bar = float(np.median(mu))
    z_a = stats.norm.isf(alpha_emp / 2)
    z_b = stats.norm.isf(0.20)  # 80% power

    def n_required(delta):
        if not np.isfinite(delta) or delta <= 0:
            return float("nan")
        se_unit = np.sqrt(2.0 / mu_bar + phi)
        return float(((z_a + z_b) / (delta / se_unit)) ** 2)

    d_eff = d_typ * np.sqrt(rho) if (np.isfinite(d_typ) and np.isfinite(rho) and rho > 0) else float("nan")
    return {
        "lambda_gc_mean_gates": lam,
        "alpha_empirical": float(alpha_emp),
        "alpha_note": alpha_note,
        "phi_median_approx": float(phi),
        "delta_floor": d_floor, "delta_typical": d_typ, "delta_ceiling": d_ceil,
        "rho_reproducibility": float(rho),
        "delta_effective": float(d_eff),
        "n_required_typical_80pct": n_required(d_typ),
        "n_required_effective_80pct": n_required(d_eff),
        "n_required_floor_80pct": n_required(d_floor),
        "note": "pdex is cell-level Wilcoxon (no NB dispersion fit); phi approximated by "
                "method-of-moments on control counts. Power is a planning estimate, not DESeq2-derived.",
    }


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def global_verdict(results):
    gates = {r.name: r.verdict for r in results if r.name in ("test_1", "test_2", "test_3")}
    if any(v == "FAIL" for v in gates.values()):
        return "FAIL"
    if any(v == "WARN" for v in gates.values()):
        return "WARN"
    return "PASS"


VERDICT_EMOJI = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}


def write_report(results, cfg, power, outdir):
    gv = global_verdict(results)
    scale = "full" if not cfg.get("max_conditions") else f"max_conditions={cfg['max_conditions']}"
    L = [f"# Robustness report — {os.path.basename(os.path.abspath(outdir))}", ""]
    L += [f"- **dataset**: `{cfg['adata_path']}`",
          f"- **DE backend**: `{cfg['de_method']}` (cell-level Wilcoxon; data normalize_total+log1p, is_log1p=True)",
          f"- **pert_col / control**: `{cfg['pert_col']}` / `{cfg['control_pert']}`  |  **sgrna_col**: `{cfg.get('sgrna_col')}`  |  **replicate_col**: `{cfg.get('replicate_col')}`",
          f"- **scale**: {scale}, n_permutations={cfg['n_permutations']}, seed={cfg['seed']}",
          f"- **GLOBAL VERDICT**: {VERDICT_EMOJI.get(gv,'')} **{gv}**  "
          f"(validity gates tests 1–3 must PASS before interpreting tests 4–6)", ""]
    if gv == "FAIL":
        L.append("> A validity gate FAILED — **do not interpret the biology** until the pipeline is fixed.")
        L.append("")
    elif gv == "WARN":
        L.append("> A validity gate WARNed — results are usable but read the flagged caveats (often a "
                 "deflated/under-powered null) before interpreting tests 4–6.")
        L.append("")
    # thresholds table
    L += ["## Verification parameters & thresholds", "",
          "| parameter | value | source |", "|---|---|---|",
          f"| de_method | {cfg['de_method']} | config |",
          f"| allow_discrete / normalize_if_raw | {cfg['allow_discrete']} / {cfg.get('normalize_if_raw')} | config |",
          f"| fdr_threshold | {cfg['fdr_threshold']} | config |",
          f"| lfc_threshold | {cfg['lfc_threshold']} | config |",
          f"| lambda_gc_warn / fail | {cfg['lambda_gc_warn']} / {cfg['lambda_gc_fail']} | TEST_PLAN.md |",
          "| lambda_gc deflation WARN | < 0.90 | skill verdict notes |",
          "| ks_p_uniform WARN | < 0.05 | skill verdict notes |",
          f"| n_permutations | {cfg['n_permutations']} | config |",
          f"| min_cells_per_group | {cfg['min_cells_per_group']} | config |",
          f"| seed | {cfg['seed']} | config |",
          f"| max_conditions | {cfg.get('max_conditions')} | config |",
          f"| block_cols (stratify) | {cfg.get('block_cols')} | config |",
          "| test-3 separation PASS / WARN | > 2 / 1–2 | TEST_PLAN.md |",
          "| test-4 LOW-REPRODUCIBILITY flag | median LFC ρ < 0.2 | TEST_PLAN.md |",
          "| test-5 separation PASS / WARN | > 1.5 / 1.0–1.5 | TEST_PLAN.md |",
          "| test-6 recovery / direction PASS | > 0.5 / > 0.8 | TEST_PLAN.md |", ""]
    # summary table
    L += ["## Test summary", "", "| test | verdict | headline |", "|---|---|---|"]
    for r in results:
        head = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.metrics.items())
        L.append(f"| {r.name} — {r.title} | {VERDICT_EMOJI.get(r.verdict,'')} {r.verdict} | {head} |")
    L.append("")
    # per-test detail
    for r in results:
        L.append(f"## {r.name} — {r.title}  {VERDICT_EMOJI.get(r.verdict,'')} {r.verdict}")
        L.append("")
        if r.metrics:
            for k, v in r.metrics.items():
                L.append(f"- `{k}` = {v}")
        if r.pvalues:
            L.append("")
            L.append("**p-values (verification):** " + ", ".join(f"`{k}` = {v:.4g}" if isinstance(v, float) else f"`{k}` = {v}" for k, v in r.pvalues.items()))
        if r.flags:
            L.append("")
            for f in r.flags:
                L.append(f"- ⚠️ {f}")
        if r.plot:
            L.append("")
            L.append(f"![{r.name} QQ]({r.plot})")
        L.append("")
        L.append(f"Tables: see `tables/{r.name}__*.csv`")
        L.append("")
    # power
    L += ["## Power & sample-size analysis (TEST_PLAN.md §7)", ""]
    if power.get("status"):
        L.append(f"- {power['status']}")
    else:
        L += ["### §7.6 Realized parameters table", "",
              "| parameter | source | realized value |", "|---|---|---|",
              f"| α_empirical | Tests 1–2 mean λ_GC | {power['alpha_empirical']:.4g} |",
              f"| λ_GC (mean of gates) | Tests 1–2 | {power['lambda_gc_mean_gates']:.4g} |",
              f"| φ_median (approx) | control counts MoM | {power['phi_median_approx']:.4g} |",
              f"| δ_floor | true DE 10th pct |LFC| | {power['delta_floor']:.4g} |",
              f"| δ_typical | true DE median |LFC| | {power['delta_typical']:.4g} |",
              f"| δ_ceiling | true DE 90th pct |LFC| | {power['delta_ceiling']:.4g} |",
              f"| ρ_reproducibility | Test 4 median LFC ρ | {power['rho_reproducibility']:.4g} |",
              f"| δ_effective = δ_typ·√ρ | derived | {power['delta_effective']:.4g} |",
              f"| n_required (δ_typical, 80%) | power formula | {power['n_required_typical_80pct']:.4g} |",
              f"| n_required (δ_effective, 80%) | power formula | {power['n_required_effective_80pct']:.4g} |",
              f"| n_required (δ_floor, 80%) | power formula | {power['n_required_floor_80pct']:.4g} |", ""]
        L.append(f"- **α note:** {power['alpha_note']}")
        L.append(f"- **caveat:** {power['note']}")
    L.append("")
    # embed test plan
    L += ["## Appendix — embedded TEST_PLAN.md", ""]
    plan = os.path.join(outdir, "TEST_PLAN.md")
    if os.path.exists(plan):
        with open(plan) as fh:
            for ln in fh.read().splitlines():
                L.append(("#" + ln) if ln.startswith("#") else ln)  # demote headings one level
    md = "\n".join(L)
    with open(os.path.join(outdir, "robustness_report.md"), "w") as fh:
        fh.write(md)
    # JSON summary
    summary = {"config": cfg, "global_verdict": gv, "power": power,
               "tests": [{"name": r.name, "title": r.title, "verdict": r.verdict,
                          "metrics": r.metrics, "pvalues": r.pvalues, "flags": r.flags} for r in results]}
    with open(os.path.join(outdir, "robustness_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return md


def render_pdf(md_path, pdf_path):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer
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
        md_dir = os.path.dirname(os.path.abspath(md_path))
        flow = []
        for ln in open(md_path).read().splitlines():
            img = re.match(r"!\[.*?\]\((.+?)\)", ln.strip())
            esc = (ln.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            esc = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc)
            esc = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", esc)
            if img:
                p = os.path.join(md_dir, img.group(1))
                if os.path.exists(p):
                    flow.append(Spacer(1, 4)); flow.append(Image(p, width=3.2 * inch, height=3.2 * inch)); flow.append(Spacer(1, 4))
            elif ln.startswith("# "):
                flow.append(Paragraph(esc[2:], h1))
            elif ln.startswith("## "):
                flow.append(Paragraph(esc[3:], h2))
            elif ln.strip():
                flow.append(Paragraph(esc, body))
            else:
                flow.append(Spacer(1, 3))
        SimpleDocTemplate(pdf_path, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                          leftMargin=0.7 * inch, rightMargin=0.7 * inch).build(flow)
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("PDF render failed: %s", e)
        return False


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_cfg(args.config)
    outdir = cfg["outdir"]
    os.makedirs(os.path.join(outdir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    log.info("Loading %s", cfg["adata_path"])
    adata = ad.read_h5ad(cfg["adata_path"])
    log.info("Loaded %d cells x %d genes", adata.n_obs, adata.n_vars)

    # pdex expects log-normalized input; match cell-eval run's normalize_total + log1p (see module docstring)
    maybe_normalize(adata, cfg)

    want = set(cfg.get("tests") or ["test_1", "test_2", "test_3", "test_4", "test_5", "test_6"])
    results = []
    de_true = None

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

    log.info("=== power analysis ===")
    power = power_analysis(adata, cfg, results, de_true)

    write_report(results, cfg, power, outdir)
    ok = render_pdf(os.path.join(outdir, "robustness_report.md"), os.path.join(outdir, "report.pdf"))
    gv = global_verdict(results)
    print("\n==== RESULTS ====")
    for r in results:
        print(f"  {r.name:8s} {r.verdict:5s} {r.title}")
    print(f"  GLOBAL: {gv}")
    print(f"  report.md + {'report.pdf' if ok else '(PDF FAILED)'} in {os.path.abspath(outdir)}")
    return 1 if gv == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
