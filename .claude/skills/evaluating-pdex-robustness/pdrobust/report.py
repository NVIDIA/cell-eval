"""Collect TestResults into CSV tables, a JSON summary, and a markdown report."""

from __future__ import annotations

import json
import os
from typing import Sequence

from .harness import RobustnessConfig
from .tests import TestResult

_VERDICT_EMOJI = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️", "SKIP": "⏭️"}


def _test_order(r: "TestResult") -> tuple[int, str]:
    """Sort key: numeric test order (test_1, test_2, ..., test_10), not lexicographic."""
    import re

    m = re.search(r"(\d+)", r.name)
    return (int(m.group(1)) if m else 9999, r.name)


def write_report(results: Sequence[TestResult], cfg: RobustnessConfig) -> str:
    outdir = cfg.outdir
    tables_dir = os.path.join(outdir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    summary = {
        "config": {
            "adata_path": cfg.adata_path,
            "pert_col": cfg.pert_col,
            "control_pert": cfg.control_pert,
            "replicate_col": cfg.replicate_col,
            "allow_discrete": cfg.allow_discrete,
            "normalize_if_raw": cfg.normalize_if_raw,
            "fdr_threshold": cfg.fdr_threshold,
            "backend": "pdex",
        },
        "tests": [],
    }
    for r in results:
        for tname, df in r.tables.items():
            if df is not None and df.height > 0:
                df.write_csv(os.path.join(tables_dir, f"{r.name}__{tname}.csv"))
        summary["tests"].append(
            {"name": r.name, "title": r.title, "verdict": r.verdict,
             "headline": r.headline, "notes": r.notes,
             "tables": list(r.tables.keys())}
        )

    json_path = os.path.join(outdir, "robustness_summary.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    md = _render_markdown(results, cfg)
    md_path = os.path.join(outdir, "robustness_report.md")
    with open(md_path, "w") as fh:
        fh.write(md)
    return md_path


def _render_markdown(results: Sequence[TestResult], cfg: RobustnessConfig) -> str:
    lines = ["# pdex backend robustness report", ""]
    lines.append(f"- **dataset**: `{cfg.adata_path}`")
    lines.append(f"- **backend**: `pdex` (cell-level Wilcoxon)")
    lines.append(f"- **pert_col / control**: `{cfg.pert_col}` / `{cfg.control_pert}`")
    lines.append(f"- **replicate_col (stratify only)**: `{cfg.replicate_col}`  "
                 f"**allow_discrete**: `{cfg.allow_discrete}`  **normalize_if_raw**: `{cfg.normalize_if_raw}`")
    lines.append(f"- **FDR threshold**: {cfg.fdr_threshold}")
    lines.append("")
    lines.append("| Test | Verdict | Headline |")
    lines.append("|------|---------|----------|")
    for r in sorted(results, key=_test_order):
        head = ", ".join(f"{k}={_fmt(v)}" for k, v in r.headline.items())
        lines.append(f"| {r.name} {r.title} | {_VERDICT_EMOJI.get(r.verdict,'')} {r.verdict} | {head} |")
    lines.append("")
    for r in sorted(results, key=_test_order):
        lines.append(f"## {r.name} — {r.title}  {_VERDICT_EMOJI.get(r.verdict,'')} {r.verdict}")
        lines.append("")
        lines.append(r.notes)
        if r.headline:
            lines.append("")
            for k, v in r.headline.items():
                lines.append(f"- `{k}` = {_fmt(v)}")
        if r.tables:
            lines.append("")
            lines.append("Tables: " + ", ".join(f"`tables/{r.name}__{t}.csv`" for t in r.tables))
        qq_png = os.path.join(cfg.outdir, "plots", f"{r.name}_qq.png")
        if os.path.exists(qq_png):
            lines.append("")
            lines.append(f"![{r.name} null p-value QQ-plot](plots/{r.name}_qq.png)")
        lines.append("")
    lines += _appendix_md()
    lines += _test_plan_md()
    return "\n".join(lines)


def _test_plan_md() -> list[str]:
    """Embed the reference test plan (per-test spec, formulas, and verdict thresholds)
    at the bottom of the report so the scoring details travel with the results."""
    plan_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reference", "TEST_PLAN.md")
    try:
        with open(plan_path) as fh:
            raw = fh.read().splitlines()
    except OSError:
        return []
    out = [
        "## Appendix B — test plan, formulas & verdict thresholds",
        "",
        "Full specification for every test (design, scoring formulas, and the heuristic verdict "
        "thresholds), reproduced from `reference/TEST_PLAN.md` so the details travel with the report.",
        "",
    ]
    # Demote the plan's top-level H1 (keep a single H1 in the report); drop its leading title line.
    for ln in raw:
        if ln.startswith("# "):
            continue
        out.append(ln)
    return out


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _appendix_md() -> list[str]:
    """Mathematical definitions of every test score / metric (backend-agnostic)."""
    return [
        "## Appendix — metric definitions & scoring math",
        "",
        "**Notation.** The DE backend (pdex = cell-level Wilcoxon, LFC = log2 mean-ratio; "
        "PyDESeq2 = pseudobulk negative-binomial Wald) yields, for each perturbation `t` and gene "
        "`g`: a log2 fold-change `LFC(t,g)`, p-value `p(t,g)`, and BH-adjusted FDR `q(t,g)`. "
        "`G` = number of genes. Significant set `S_a(t) = { g : q(t,g) < a }` at FDR level `a`.",
        "",
        "### Null calibration (tests 1–3)",
        "- **frac_sig**(t) = `|S_a(t)| / G`; reported as the mean over conditions. Under a true "
        "null this is ≈ the empirical false-positive fraction (≈0 when FDR control holds).",
        "- **mean_abs_lfc** = mean over finite genes of `|LFC(t,g)|`.",
        "- **KS-uniform** = Kolmogorov–Smirnov statistic of `{p(t,g)}` against `Uniform[0,1]`; a "
        "small KS p-value flags departure from the null.",
        "- **Genomic inflation** `λ_GC = median_g χ²₁⁻¹(1 − p(t,g)) / χ²₁⁻¹(0.5)`, where "
        "`χ²₁⁻¹(0.5) = 0.4549`. `λ_GC ≈ 1` ⇒ calibrated; `< 1` ⇒ conservative (deflated); "
        "`> 1` ⇒ anti-conservative (inflated).",
        "- **QQ-plot** (tests 1 & 2): observed `−log10 p` (sorted) vs expected "
        "`−log10((i−0.5)/G)`; the grey 95% band is the pointwise `Beta(i, G−i+1)` order-statistic "
        "envelope under the uniform null. Points on the `y = x` line ⇒ well-calibrated.",
        "- **Permutation separation** (test 3) = mean over discriminative metrics `M` of "
        "`M(identity) − M(permuted)`, where identity (`pred = real`) is the upper bound and "
        "permuted (labels shuffled within blocks) is the null floor.",
        "",
        "### Reproducibility ceiling (tests 4–5)",
        "- **de_spearman_lfc_sig** = Spearman ρ between real and predicted `LFC` over the "
        "union-significant genes (a cell-eval run-pipeline metric).",
        "- **overlap@N**(t) = `|topN_real ∩ topN_pred| / N`, genes ranked by `|LFC|` among "
        "significant; **precision@N** uses the predicted top-N as denominator.",
        "- **direction_match** = fraction of union-significant genes with "
        "`sign(LFC_real) = sign(LFC_pred)`.",
        "- **Ceiling:** test 4 sets real = split-A, pred = split-B of the same perturbation, so "
        "each metric value is the maximum a model can attain; report model scores as "
        "`score / ceiling`.",
        "- **Same-gene** (test 5): per guide pair, `lfc_spearman`, `lfc_pearson`, and "
        "`sig_jaccard = |S_i ∩ S_j| / |S_i ∪ S_j|`. `spearman_sep = mean_same − mean_unrelated` "
        "(verdict uses `max(spearman_sep, jaccard_sep)`); `pearson_saturated` is flagged when both "
        "same- and unrelated-pair mean Pearson > 0.8 (LFC-Pearson then cannot discriminate).",
        "",
        "### Biological positives (tests 6–8)",
        "- **Knockdown** (6): `rank_pct` = rank of the target gene by `|LFC|` within its DE list "
        "÷ G (lower = stronger); `frac_detected_sig` = fraction of targets with `q < a`; "
        "`frac_negative_lfc` = fraction with `LFC < 0` (CRISPRi/KD expectation).",
        "- **Curated targets** (7): per known relationship — significance (`q < a`), rank "
        "percentile by p-value, and direction agreement vs the expected sign.",
        "- **Pathway/regulon** (8): `AUROC` of `|LFC|` separating expected-set genes from the rest, "
        "computed as `U / (n_pos · n_neg)` (Mann–Whitney U); `> 0.5` ⇒ enrichment.",
        "",
        "### Stability (tests 9–10)",
        "- **lfc_corr_to_full** = Pearson correlation of per-gene `LFC` at a downsample level vs "
        "the full-data `LFC` (mean over targets).",
        "- **rank_corr** = Spearman correlation of per-target `n_sig` vs full-data `n_sig`.",
        "- **min_cells_for_stable_lfc** = smallest grid level with mean `lfc_corr_to_full ≥ "
        "stable_lfc_corr` (default 0.9); `None` ⇒ never stabilizes on the grid.",
        "- **cells_per_pseudobulk** ≈ `level / n_replicates` (PyDESeq2 pseudobulk depth proxy); "
        "**anti_correlated_lfc** flags any level with negative `lfc_corr_to_full`.",
        "",
        "### Verdict thresholds (heuristic triage, not proof)",
        "- Nulls: WARN if mean `frac_sig ≥ 0.02` or `|mean LFC| ≥ 0.10`; FAIL if `frac_sig ≥ 0.05`.",
        "- Permutation (3): PASS if mean discriminative separation `≥ 0.20`.",
        "- Reproducibility (4): PASS if LFC ρ `≥ 0.50` or overlap@N `≥ 0.30`. Same-gene (5): PASS if "
        "Spearman/Jaccard separation `> 0.05`.",
        "- Knockdown (6): PASS if `frac_detected_sig ≥ 0.50` and `frac_negative_lfc ≥ 0.60`. "
        "Pathway (8): PASS if mean AUROC `≥ 0.60`.",
        "- Stability (9–10): PASS once `lfc_corr_to_full` reaches `stable_lfc_corr` (0.9).",
        "",
    ]
