---
name: evaluating-pydeseq2-robustness
description: Evaluate cell-eval's run-pipeline PyDESeq2 DE backend with a metric-robustness battery — null calibration, reproducibility ceilings, biological-positive recovery, and downsampling stress tests. Given a perturbation .h5ad (perturbation labels, control cells, a replicate/sample column, raw counts, and optional sgRNA/target-gene/pathway annotations), fingerprint the dataset, confirm a config, run the ten tests through the real cell-eval PyDESeq2 backend, and emit a scientific report. Use when someone wants to stress-test, calibrate, or audit the pydeseq2 backend of `cell-eval run` (or asks whether its DE metrics reward biology vs statistical artifacts).
---

# Evaluating the cell-eval PyDESeq2 backend for metric robustness

This skill drives **cell-eval's PyDESeq2 DE backend** (the one behind
`cell-eval run --de-methods pydeseq2`) through a battery of ten robustness tests adapted from the
metric-robustness analysis plan. It asks whether the backend + the run-pipeline metrics are:

- **calibrated under null comparisons** (no signal in → no signal out),
- **sensitive to real perturbation biology** (target knockdown, curated targets, pathways),
- **reproducible** (define an empirical ceiling for model scores), and
- **stable** under sampling.

It uses the *actual* cell-eval code paths — `cell_eval._pydeseq2_backend.compute_pydeseq2_de` and
`cell_eval.MetricsEvaluator(..., de_methods=["pydeseq2"])` — so results reflect the real pipeline,
not a reimplementation. **Do not reimplement DE or the metrics; call the package.**

## Components

```
scripts/inspect_dataset.py   # fingerprint an .h5ad -> JSON config skeleton + readiness warnings
scripts/run_robustness.py    # driver: load config, run selected tests, write report
scripts/render_pdf.py        # render robustness_report.md -> PDF (embeds QQ-plots + math appendix)
pdrobust/harness.py          # config, DE/run-pipeline wrappers, DE summaries, signature comparison, obs surgery
pdrobust/tests.py            # the ten tests + curated tests (test_1 .. test_10) and verdict heuristics
pdrobust/plots.py            # null p-value QQ-plots vs Uniform[0,1] + genomic-inflation lambda_GC
pdrobust/report.py           # CSV tables + JSON summary + markdown report (+ metric-definitions appendix)
reference/TEST_PLAN.md       # full spec + expected behavior + verdict thresholds (READ THIS)
reference/config.example.yaml
reference/curated_targets_schema.md      # inputs for tests 7 & 8
reference/counterfactual_companion.md    # red-team / biological-counterfactual companion design
```

Run everything with `uv run` from the cell-eval repo root (the package imports `cell_eval` and
requires the `pydeseq2` extra: `uv sync --all-extras --dev`, or `cell-eval[pydeseq2]`).

## What the dataset needs

- **perturbation labels** (`pert_col`) and a **control label** (`control_pert`).
- a **replicate / sample column** (`replicate_col`) — the PyDESeq2 pseudobulk unit (donor, lane,
  gem group, biological replicate). **Required by the backend.**
- **raw integer counts**: a counts layer (preferred) or raw-integer `.X` with `allow_discrete: true`.
- *recommended, gate biological tests*: `sgrna_col` (tests 4–5), `target_gene_col` matching
  `var_names` (tests 5–6), `block_cols` covariates (tests 1–4), and curated
  target/pathway tables (tests 7–8, see `reference/`).

## Workflow

### 1 — Fingerprint the dataset
```bash
uv run python .claude/skills/evaluating-pydeseq2-robustness/scripts/inspect_dataset.py /path/to/data.h5ad
```
Prints obs columns, raw-counts detection, candidate perturbation/replicate/blocking columns, and a
JSON config skeleton between `BEGIN_JSON`/`END_JSON`, plus readiness warnings (no raw counts? no
replicate column? control label unknown?). Parse the JSON as your starting config.

### 2 — Confirm the configuration (REQUIRED before running)
Use **AskUserQuestion** to confirm anything the inspector left ambiguous — these silently break
PyDESeq2 or invalidate the tests if wrong:
1. **perturbation column & exact control label**.
2. **replicate/sample column** (pseudobulk unit). If the dataset is single-replicate, PyDESeq2
   dispersion is unreliable — say so and consider the pdex backend instead.
3. **counts source**: which layer holds raw counts (or `.X` + `allow_discrete`).
4. **target-gene / sgRNA columns** and whether `target_gene_col` values match `var_names`
   (tests 5–6 silently skip otherwise).
5. **which tests** to run and the **downsample grid / n_repeats** (cost scales with both ×
   number of perturbations — PyDESeq2 refits per level/repeat).

Write the confirmed config to a YAML/JSON file (template: `reference/config.example.yaml`).

### 3 — Run the battery
```bash
uv run python .claude/skills/evaluating-pydeseq2-robustness/scripts/run_robustness.py \
    --config /path/to/config.yaml            # add --tests test_1,test_3,test_6 for a subset
```
Outputs under `<outdir>/`: `robustness_report.md`, `robustness_summary.json`, per-test CSVs in
`tables/`, null-calibration **QQ-plots** in `plots/` (`test_1_qq.png`, `test_2_qq.png` — pooled
null p-values vs Uniform[0,1] with the genomic-inflation `lambda_GC`), and the cell-eval
run-pipeline CSVs for tests 3–4. Each test returns PASS / WARN / FAIL / INFO / SKIP (heuristics in
`reference/TEST_PLAN.md`). The markdown report ends with a **metric-definitions appendix** giving
the scoring math for every test.

Export the report to PDF (embeds the QQ-plots and renders the math appendix):
```bash
uv run python .claude/skills/evaluating-pydeseq2-robustness/scripts/render_pdf.py \
    <outdir>/robustness_report.md <outdir>/report.pdf "Title"
```

### 4 — Write the scientific report
Read `robustness_report.md` + the CSV tables and synthesize a report organized by the plan's four
axes. For each, cite the concrete numbers and interpret them:

- **Null calibration** (tests 1–3): is `frac_sig` near 0 under direct splits and control-control
  splits? Do permuted labels collapse the run-pipeline metrics relative to identity (the
  `mean_separation_discriminative` gap)? A backend that calls many DE genes under a null is
  mis-calibrated (anti-conservative) — flag it.
- **Reproducibility ceiling** (tests 4–5): the same-split LFC correlation / overlap is the empirical
  ceiling for model scores — set-overlap/precision usually ceiling well below 1.0, so report model
  scores **relative to** `tables/test_4__ceiling.csv`, not against 1.0. For same-gene-guide agreement
  (test 5), **do not** judge on whole-transcriptome LFC-Pearson — it saturates when perturbations
  share a global response (`pearson_saturated` flag); use the significant-gene Spearman / Jaccard
  separation the test reports instead.
- **Biological sensitivity** (tests 6–8): target-knockdown detection & direction, curated-target
  recovery, pathway AUROC.
- **Stability** (tests 9–10): use `min_cells_for_stable_lfc` per axis as the recommended floor.
  Watch `anti_correlated_lfc` and `cells_per_pseudobulk` for test 10 — tiny, batch-fragmented control
  pseudobulks make LFC magnitudes anti-track the full data; widen `control_downsample_grid` (default
  `[500,1000,2000,5000]`) or coarsen the replicate grouping if the grid never reaches stability.

State explicitly what was tested vs skipped (and why), what passed/failed against thresholds, and
the recommended minimum cell counts. Heuristic verdicts triage attention — **always ground claims
in the per-test tables**, and tune thresholds (top of `pdrobust/tests.py`) to the dataset's depth.

## Notes
- PyDESeq2 needs ≥2 pseudobulk samples per condition; the battery skips sub-designs that can't form
  them and logs it. Very small perturbations / aggressive downsampling will SKIP rather than lie.
- For the metric red-team / biological-counterfactual extension, see
  `reference/counterfactual_companion.md` (build on `harness.run_pipeline` + `compare_signatures`).
- Cost: tests 3–4 and 9–10 refit DESeq2 multiple times. Start with `max_conditions` set and a small
  `downsample_grid` / `n_repeats`, then scale up.
