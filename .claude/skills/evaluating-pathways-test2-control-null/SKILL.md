---
name: evaluating-pathways-test2-control-null
description: Run or reproduce pathway Test 2 comparing control-control null calibration for reference-coded bioconcord OLS and pdex Mann-Whitney on fixed bioconcord pathway scores. Use to evaluate null p-value uniformity, genomic-inflation lambda, nominal p-value frequency, BH-FDR call frequency, median p-value, and absolute descriptive score differences.
---

# Pathway Test 2 — Control-Control Null Calibration

Read the bundled `pathway-methods-memory.md` before running, reimplementing, or changing this skill. Use `test2_control_null.py` with the local `pathway_utils.py`, which loads scoring and OLS from an official ArcInstitute/bioconcord checkout. Do not import another skill or copy the Bioconcord implementation into this folder.

## Mandatory preflight and run capture

Do not start the executable until the user has explicitly confirmed one fully resolved run configuration.

1. Gather the input `.h5ad`, ask whether `adata.X` contains raw counts or log1p-normalized expression, pathway-definition CSV, official Bioconcord checkout, results output directory, separate run root, methods to compare (`ols`, `pdex_mwu`, or both), perturbation/control/block fields, score layer, thresholds, scoring settings, repeats, seeds, and threads. Inspect the input read-only to resolve unknown columns, labels, layers, and control-cell power. Pass the confirmed state as `--expression-state`.
2. Expand paths and resolve every default. Show one concise preflight summary containing inputs, results directory, run root, methods, data fields/layers, thresholds, workload/concurrency, exact command, log path, and resolved-config destination.
3. Ask for explicit confirmation and stop. Do not launch scoring, inference, or plotting before confirmation.
4. After confirmation, create `<run-root>/logs` and `<run-root>/configs`, pass `--run-root <run-root>`, and capture the complete terminal stream with `2>&1 | tee <run-root>/logs/<workflow>__<dataset>__<UTC-timestamp>.log`.
5. Every invocation writes an immutable timestamped YAML snapshot under `<run-root>/configs`. Report the result, log, and YAML paths on completion.

Every box-and-whisker plot must overlay every finite underlying observation as jittered scatter points. Do not sample, aggregate away, or hide values in the scatter layer.

For every pathway correlation matrix, retain every analytically eligible perturbation or guide. Never reduce the unit set merely to make the plot readable. When more than 40 units are shown, omit only the white diagonal cell-count text; keep the full numeric matrix and all units. Any analysis cap must be an explicit, user-confirmed scientific selection, not an automatic display rule.

## Scientific question

Test calibration when both analyzed arms contain only genuine control cells. Any significant pathway is null by construction, so this is a hard validity gate before interpreting real perturbation discoveries.

## Run

```bash
python .claude/skills/evaluating-pathways-test2-control-null/test2_control_null.py \
  --adata <data.h5ad> \
  --programs <pathway-definitions.csv> \
  --bioconcord-root /path/to/bioconcord \
  --pert-col gene --control non-targeting --block-cols batch \
  --score-layer X --min-genes 5 --ctrl-size 50 --n-bins 25 \
  --n-repeats 20 --min-cells-per-arm 20 \
  --methods ols,pdex_mwu --threads 8 --alpha 0.05 --seed 42 \
  --outdir experiments_pathways/<dataset>__test2
```

This skill does not import or require output from another skill. Install the local
`requirements.txt`, clone `https://github.com/ArcInstitute/bioconcord.git`, and pass its root
with `--bioconcord-root` or `BIOCONCORD_ROOT`. Revision
`ee3a66fc512e9ee0fe87409240e16aa43698dff8` is the tested source version. `--programs` is a
required user-supplied CSV because pathway definitions are not committed with the skill.

## Reproduce the implementation

1. Load and score the complete dataset once using official Bioconcord scoring.
2. Select only rows whose perturbation label equals the requested control.
3. Build composite block labels by string-joining requested `obs` columns with `||`; use one global block when none are given.
4. For repeat `r`, split control indices with `stratified_half` and RNG seed `seed + r`. Fail if either arm is below `min_cells_per_arm`.
5. Concatenate the left arm followed by the right arm. Label left cells with the real control label and right cells `pseudo_perturbation`.
6. Run each method on the fixed scores for these cells, using the real control label as reference. Tag native rows with zero-based repeat.

## Null summaries

For every repeat and method, compute:

- `lambda_gc`: clip finite p-values into `(tiny, 1]`, transform with the one-degree-of-freedom chi-square inverse survival function, and divide the observed median by `chi2.ppf(0.5, 1)`.
- `fraction_p_below_alpha`: fraction of pathways with native `p_value <= alpha`.
- `fraction_fdr_below_alpha`: fraction with within-pseudo-perturbation BH `fdr <= alpha`.
- `median_p`: median native p-value.
- `mean_abs_descriptive_difference`: mean absolute pseudo-perturbation-minus-control pathway-score difference.

A calibrated method should show approximately uniform p-values, lambda near 1, nominal p-value frequency near alpha, and few FDR discoveries. Treat these as empirical diagnostics, not pass/fail constants for a finite dataset.

## Required plots

Match the finalized non-pathway Test 2 visual contract while preserving pathway-native statistics.

Create `plots/pathways_test2_null_diagnostics__<dataset>.png` as a 14 x 5 inch, 300 dpi figure with:

1. A pooled null QQ curve per method using expected versus observed `-log10(p)`, a dashed identity line, and a 95% beta-order-statistic null envelope. Log-tail-downsample to at most 3,000 ordered p-values per method. Label each curve with mean repeat-wise lambda GC.
2. Two semi-transparent, 20-bin density histograms with a dashed uniform-density line at 1.
3. A method-specific boxplot plus deterministic-jitter repeat dots for `lambda_gc`, with a dashed line at 1 and dotted lines at 0.9 and 1.1.

When both methods run, also create `plots/pathways_test2_effect_agreement__<dataset>.png`. Pair rows by repeat, pseudo-target, and program; plot OLS coefficient on x and pdex rank-biserial effect on y. Color points by FDR-call category: neither, OLS only, pdex only, or both. Report pooled Spearman rho for all programs and, when defined, programs called by either method. Use `fdr <= alpha` as the call rule. Do not draw `y=x`: the native effects have different units and scales.

## Output contract

- `tables/pathways_test2_null_results__<dataset>.csv`: every canonical pseudo-perturbation-pathway test tagged by repeat.
- `tables/pathways_test2_null_summary__<dataset>.csv`: one row per repeat and method with five diagnostics: lambda GC, nominal-p fraction, FDR-call fraction, median p-value, and mean absolute descriptive difference.
- `tables/pathways_legend__<dataset>.csv`: retained program definitions.
- `plots/pathways_test2_null_diagnostics__<dataset>.png`: QQ, p-value density, and lambda panels.
- `plots/pathways_test2_effect_agreement__<dataset>.png`: paired native-effect null scatter, emitted only when both methods run.
- `pathways_test2_metadata__<dataset>.json`: shared score and argument metadata.

## Reproduction checks

- Verify that every analyzed cell originally carried the control label.
- Keep the same score values; change only the temporary arm labels.
- Require exactly one non-reference group (`pseudo_perturbation`) per repeat.
- Apply BH across all retained pathways within that pseudo-perturbation.
