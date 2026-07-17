# Pathway-method comparison contract

Read this bundled file before running or changing its enclosing pathway skill. Every standalone skill carries an identical flat copy of this statistical contract.

## Purpose and source ownership

- Compare the underlying methods as implemented, even if either method may be statistically imperfect.
- Reuse the bundled `pathway_utils.py`. It must load scoring and OLS directly from `Src/bioconcord/testGeneProgramsConcordance.py` in an official ArcInstitute/bioconcord checkout; do not copy that implementation into the skill or silently replace either estimator.
- Keep each skill independent of sibling skills: its runner imports only the `pathway_utils.py` beside it. Locate Bioconcord through `--bioconcord-root`, `BIOCONCORD_ROOT`, or a discoverable `bioconcord` checkout.
- Before a runner reuses a dedicated output directory, remove only files owned by that runner's `pathways_<workflow>_` prefix so conditional outputs from an older configuration cannot masquerade as current results.

## Common score matrix

- Select one representative gene set per cluster, preferring the representative with the most genes present in the dataset.
- Compute each cell-pathway value as mean expression of pathway genes minus mean expression of expression-matched background genes.
- Preserve bioconcord settings: 25 expression bins, up to 50 matched controls per pathway gene, and random seed 42.
- `score_adata.X` is a signed floating-point **cell x pathway** matrix. It is not a count matrix and can contain negative values.
- Give both methods the identical score matrix, cells, labels, control definition, and trial split.
- A pivoted effect matrix is **perturbation x pathway**. Long result tables use one row per perturbation-pathway test.

## Inference methods

### Bioconcord OLS

- In every pathway-method comparison, run the official source function `run_program_regression` after scores are computed.
- Preserve the intercept. `const` is the control mean; each perturbation dummy coefficient is the perturbation-minus-control pathway effect. Never report `const` as a perturbation.
- The loader wraps Bioconcord's imported `add_constant` helper so its returned intercept-plus-dummy design matrix is float. This current pandas/statsmodels compatibility fix does not edit the checkout or change the model.

### pdex Mann-Whitney

- Test the per-cell scores of each perturbed group against control-cell scores.
- Call pdex in reference mode with `geometric_mean=False, is_log1p=False`.
- `is_log1p=False` is required because the input is already a signed pathway-score matrix. It does not mean the matrix must contain integers; Mann-Whitney ranks the supplied floats directly. Setting it true would incorrectly apply an `expm1`-style back-transformation to signed scores in pdex's mean summaries.
- Report rank-biserial correlation as pdex's native effect.

For both methods, apply BH-FDR across pathways within each perturbation. Retain perturbation-minus-control mean score difference as a shared descriptive check, but do not substitute it for either native effect.

## Overview requirements

- Include every eligible non-control perturbation in tables and plotted data. Axis tick labels may be thinned for readability, but rows or points must not be dropped.
- The method-comparison plot uses significant-pathway counts, labels every perturbation with a non-overlapping leader arrow, displays called-pathway Jaccard (`J=`), and encodes perturbation cell count by marker size/color with a cell-count legend. Do not use Spearman as the right-hand legend.
- Always plot each method's pathway effects against perturbation cell count and median UMI count. Each perturbation is a box-and-whisker summary across pathways with each pathway overlaid as a dot.
- Always create one MA-style expression-versus-effect figure per method for the perturbations with the least, median, and most cells. Use total raw UMI summed across all cells of the perturbation and all retained genes of the pathway plus 1 on a log x axis; do not divide by cell count or pathway size. Use native method effect on y, gray for non-called pathways, and red for pathways with BH-FDR at or below the configured threshold. Use the identical raw-abundance x coordinate for both methods.

## Test 0 raw-UMI injection

- Use control cells only. For one pathway at a time, split controls into pseudo-perturbed and reference arms within requested block strata.
- In pseudo-perturbed cells, multiply raw UMI counts for every gene in the injected pathway by `2**delta`, round to integer counts, and leave zeros at zero. `delta` is gene-level log2 fold-change: delta 1 is 2x and delta 2 is 4x raw UMI.
- Do not add delta to normalized pathway scores. After each raw-count injection, total-normalize to 10,000, apply log1p, and recompute **all** pathway-minus-background scores.
- Test each retained pathway separately and repeat each pathway 10 times. Reuse a repeat's split across injected pathways for fairness.
- Feed the same recomputed signed score matrix to OLS and pdex. Keep pdex `is_log1p=False`.
- Audit raw pathway-gene UMI totals before and after injection and record the observed multiplier. At delta 2 it must be 4x whenever the baseline total is nonzero.
- `TP count` is the number of repeats (out of 10) in which the injected pathway is FDR-called. `FP count` is the total untouched-pathway FDR calls across those repeats. Report counts, never TP/FP rates.
- Plot TP-count and FP-count panels for each method, one gradient-colored trajectory per pathway, plus a black pathway median and a 10th-90th percentile tube. Use integer y axes.

## Test 1 perturbation split-half reproducibility — finalized

- Treat the enclosing skill's detailed `SKILL.md` as the complete reproducible specification. It records input validation, program selection, scoring, split edge cases, method formulas, FDR scope, metrics, repeat averaging, plot contracts, schemas, expected counts, and validation commands.
- Score every cell once on the full data with the fixed Bioconcord program scorer. Reuse the identical signed cell-by-program matrix for every split, repeat, arm, and method.
- Default to 5 repeats. Repeat `r` uses split seed `42 + r`; the Bioconcord scoring seed remains fixed at 42.
- Split the control and every perturbation independently within requested block strata. Drop a perturbation if either arm has fewer than 20 cells; never silently drop an undersized control.
- Run each arm against its own control arm with both `ols` and `pdex_mwu` unless the caller explicitly requests a subset.
- Compute within-method, per-repeat, per-target A/B effect Spearman, significant-program Jaccard, direction agreement over the significant union, and significant counts. Keep these audit values in `pathways_test1_reproducibility__<dataset>.csv`.
- When both methods run, compute cross-method agreement separately within every repeat and arm.

Final visualization decisions:

- Do not emit the former box/scatter reproducibility-versus-cell-count PNG or retain its plotting code.
- Do not emit per-perturbation zoom plots.
- Do not emit repeat-0-only plots.
- Average each arm's target-program native effects across all repeats and plot mean arm A and mean arm B separately.
- Calculate each repeat's perturbation-by-perturbation A-versus-B Spearman matrix, then average those matrices cell by cell across repeats.
- Calculate every correlation over all aligned retained pathway coefficients. Never select correlation features by FDR, significance, effect size, or pair-specific pathway unions.
- Emit one combined mean correlation map and explicit separate OLS and pdex Mann–Whitney maps.
- Fix pdex rank-biserial heatmap colors to `[-1, 1]`. Scale OLS symmetrically to the observed maximum absolute mean effect in that arm.
- Fix all correlation-map colors to `[-1, 1]` and annotate diagonal cells with total perturbation cell counts.
- Save every correlation map's exact ordered Spearman and Pearson matrices in a same-stem NPZ;
  downstream summaries must consume numeric archives rather than infer values from PNG pixels.

Final Test 1 plot filenames:

```text
pathways_test1_mean_arma__<dataset>.png
pathways_test1_mean_armb__<dataset>.png
pathways_test1_corr_matrix_mean__<dataset>.png
pathways_test1_corr_matrix_mean_ols__<dataset>.png
pathways_test1_corr_matrix_mean_pdex_mwu__<dataset>.png
```

Validated `cell_eval2` run:

- Input: a local `cell_eval2.h5ad` supplied through `--adata`; do not record environment-specific absolute paths in the skill.
- Output: `experiments_pathways/cell_eval2__test1/`.
- Data: 5,114 cells, 25 retained perturbations, 1,784 controls, 48 batches, and 97 retained programs.
- Tables: 48,500 full result rows, 250 within-method reproducibility rows, and 250 cross-method rows.
- Mean repeat-wise diagonal rho: approximately 0.92 for OLS and 0.93 for pdex Mann–Whitney.
- OLS effects match direct target-minus-control mean differences to numerical precision. Observed pdex effects stay within `[-1, 1]`.

Shared OLS pathway-stripe interpretation from `cell_eval2`:

- `C021` is `BRUINS_UVC_RESPONSE_VIA_TP53_GROUP_C`. Despite the name, the source defines Group C as p53-independent UV-response genes; interpret a shared stripe as a generic stress/fitness signature, not direct evidence of TP53 activation. It was positive for 20/25 targets and significant in a majority of split draws for 14/25, not every perturbation.
- `C093` is `REACTOME_COMPLEX_I_BIOGENESIS`. A shared mitochondrial/fitness response is plausible for essential-gene perturbations, but its mean direction was mixed (14 positive, 11 negative targets) and it was significant in a majority of split draws for 15/25.
- A visually prominent OLS column is not equivalent to a consistent FDR call. Verify the long result table before making biological claims.

## Tests 2–5 and 7 suite guardrails

- Test 2 compares genuine control cells against a pseudo-perturbed control half on the fixed score matrix. Its five repeat-level diagnostics are lambda GC, nominal-p fraction, FDR-call fraction, median p-value, and mean absolute descriptive score difference.
- Test 3 uses exactly five global and five within-block label permutations when both modes are requested. Keep permutation-level calls intact and average counts, Jaccard, and target-correlation matrices only after each permutation has been analyzed independently. Do not restore the removed count/separation boxplot.
- Test 4 measures the reliability ceiling of one guide against itself using disjoint cell halves and five repeats. Select multi-guide genes, label guides consistently as `GENE | gN`, keep same-gene guides adjacent, and draw gene boundaries. A diagonal cell is split-A versus split-B correlation, not a mathematical self-correlation and therefore need not equal one.
- Test 5 compares distinct guides targeting the same gene with guides targeting different genes using each guide's full-cell effect vector. Reuse one unrelated-pair sample across methods and reuse guide-map labels in every plot. Test 4 and Test 5 share scoring and inference, but Test 4 measures sampling reliability whereas Test 5 measures cross-guide agreement. Because Test 5 pairs share guides, its pairwise Mann-Whitney p-value is descriptive rather than a confirmatory gene-level test.
- Test 7 measures common scoring separately from method inference. Its process RSS value is a process-lifetime high-water delta and may be zero after an earlier stage set the high-water mark; never interpret it as an exact isolated method peak.

## Interpretation guardrails

- Every square correlation matrix emitted by a pathway skill must report finite mean diagonal and
  off-diagonal correlation separately in its panel title. Additional structured summaries (for
  example within-gene versus cross-gene off-diagonal means) supplement rather than replace these two
  global values.
- OLS coefficients and rank-biserial correlations have different scales; compare calls, direction, calibration, and reproducibility without treating their magnitudes as interchangeable.
- Pathway overlap, matched-background overlap, and total-count normalization can create effects in pathways that were not directly injected. Preserve these behaviors because they are part of the compared pipelines.
- At Test 0 delta 0, every pathway is null; excess calls indicate pipeline calibration behavior.
