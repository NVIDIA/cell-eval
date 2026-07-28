---
name: evaluating-pathways-test1-reproducibility
description: Run or reproduce pathway Test 1 comparing perturbation split-half reproducibility for official ArcInstitute Bioconcord OLS and pdex Mann-Whitney on one fixed Bioconcord score matrix. Use for within-method effect-vector Spearman, significant-pathway Jaccard, direction agreement, cell-count dependence, arm-level cross-method agreement, mean arm effects, and repeat-averaged all-pathway and FDR-union perturbation maps reporting diagonal/off-diagonal Spearman and Pearson correlations.
---

# Pathway Test 1 — Perturbation Split-Half Reproducibility

Read the bundled `pathway-methods-memory.md` before running, reimplementing, or changing this skill. Use `test1_reproducibility.py` with the local `pathway_utils.py`, which loads scoring and OLS from an official ArcInstitute/bioconcord checkout. Do not import another skill or copy the Bioconcord implementation into this folder.

## Mandatory preflight and run capture

Do not start the executable until the user has explicitly confirmed one fully resolved run configuration.

1. Gather the input `.h5ad`, ask whether `adata.X` contains raw counts or log1p-normalized expression, pathway-definition CSV, official Bioconcord checkout, results output directory, separate run root, methods to compare (`ols`, `pdex_mwu`, or both), perturbation/control/block fields, score layer, thresholds, scoring settings, repeats, seeds, and threads. Inspect the input read-only to resolve unknown columns, labels, layers, and eligible perturbations. Pass the confirmed state as `--expression-state`.
2. Expand paths and resolve every default. Show one concise preflight summary containing inputs, results directory, run root, methods, data fields/layers, thresholds, unit scope, workload/concurrency, exact command, log path, and resolved-config destination. Before asking for confirmation, estimate wall time separately for every selected method, shared scoring/preparation/rendering, and the complete run; state the hardware tier, cache assumptions, evidence or throughput basis, uncertainty range, and how watchdog de-escalation could extend it.
3. Ask for explicit confirmation and stop. Do not launch scoring, inference, plotting, or reuse before confirmation.
4. After confirmation, create `<run-root>/logs` and `<run-root>/configs`, pass `--run-root <run-root>`, and capture the complete terminal stream with `2>&1 | tee <run-root>/logs/<workflow>__<dataset>__<UTC-timestamp>.log`.
5. Every invocation writes an immutable timestamped YAML snapshot under `<run-root>/configs`. Report the result, log, and YAML paths on completion.

Every box-and-whisker plot must overlay every finite underlying observation as jittered scatter points. Do not sample, aggregate away, or hide values in the scatter layer.

For every pathway correlation matrix, retain every analytically eligible perturbation or guide. Never reduce the unit set merely to make the plot readable. When more than 40 units are shown, omit only the white diagonal cell-count text; keep the full numeric matrix and all units. Any analysis cap must be an explicit, user-confirmed scientific selection, not an automatic display rule.

## Scientific question

Estimate each method's reproducibility ceiling by comparing pathway-effect vectors from two independent halves of the same perturbation and control cells. Score the full dataset once so this test isolates cell sampling and inference, not score recomputation.

## Run

```bash
python .claude/skills/evaluating-pathways-test1-reproducibility/test1_reproducibility.py \
  --adata <data.h5ad> \
  --programs <pathway-definitions.csv> \
  --bioconcord-root /path/to/bioconcord \
  --pert-col gene --control non-targeting --block-cols batch \
  --score-layer X --min-genes 5 --ctrl-size 50 --n-bins 25 \
  --min-cells-per-arm 20 --n-repeats 5 \
  --methods ols,pdex_mwu --threads 8 --fdr 0.05 --seed 42 \
  --outdir experiments_pathways/<dataset>__test1
```

This skill does not import or require output from another skill. Install the local
`requirements.txt`, clone `https://github.com/ArcInstitute/bioconcord.git`, and pass its root
with `--bioconcord-root` or `BIOCONCORD_ROOT`. Revision
`ee3a66fc512e9ee0fe87409240e16aa43698dff8` is the tested source version. `--programs` is a
required user-supplied CSV because pathway definitions are not committed with the skill.

## Reproduce the implementation

1. Load programs and compute the full `cells x pathways` score matrix once with official Bioconcord scoring.
2. Convert every comma-separated block column to string and join multiple columns with `||`; use one global block if no columns are supplied.
3. For repeat `r`, call `split_groups` with seed `seed + r`:
   - process control first, then sorted non-control groups;
   - split every group independently within blocks using random half splits;
   - alternate singleton blocks between arms and use an unstratified fallback only if an arm is empty;
   - fail if the control has fewer than `min_cells_per_arm` cells in either arm;
   - omit any non-control perturbation that falls below the arm threshold;
   - fail if no perturbation survives.
4. Run each requested method on arm A against control A and separately on arm B against control B. Preserve the canonical native-result schema and tag every row with zero-based `repeat` and `arm` (`A` or `B`).
5. If both methods run, compute the bundled cross-method comparison independently in every arm and repeat.

## Reproducibility metrics

For every method, repeat, and perturbation present in both arms:

- Align pathways by sorted pathway label.
- Compute `effect_spearman` across the two native effect vectors with finite/nonconstant safeguards.
- Define significant sets with `fdr <= --fdr` and compute `sig_jaccard`; return 1 when both sets are empty.
- Over the union of pathways significant in either arm, compute the fraction whose effect signs agree; use missing when the union is empty.
- Record significant counts `n_sig_a` and `n_sig_b`.
- Record `n_cells_total = n_target_arm_A + n_target_arm_B`.

Assess within-method reproducibility before interpreting cross-method agreement.

## Correlation-map construction

For all requested methods together:

1. Intersect perturbations across every method, arm, and repeat. Intersect pathways across the same inputs, then require each pathway effect to be finite for every retained perturbation in every method, arm, and repeat. Report dropped perturbations/pathways. This is the shared complete-case basis for combined and single-method outputs.
2. Build the primary all-pathway family. Within each repeat, each entry `(i, j)` compares every pathway on the shared complete-case basis for perturbation `i` in arm A with perturbation `j` in arm B, once by Spearman and once by Pearson.
3. Build the FDR-union family. Within each repeat and target-pair cell, select the union of pathways called at `fdr <= --fdr` in either compared profile by any requested method. Use that identical cell-specific feature set for every method; require at least three values and apply finite/nonconstant safeguards, leaving underpowered cells missing.
4. For the same FDR-union family, compute a method-specific significant-pathway Jaccard matrix
   within every repeat: cell `(i,j)` is `J(sig_A(target i), sig_B(target j))`. Preserve the existing
   significant-set convention that two empty sets have Jaccard 1.
5. Average each correlation and Jaccard matrix cell by cell across repeats; do not correlate effects after averaging.
6. Interpret the diagonal as within-perturbation reproducibility and off-diagonals as cross-perturbation similarity. Compute the finite mean of all diagonal cells and all off-diagonal cells separately for both correlation metrics.
7. Render the Spearman matrix. In every method-panel title, report `Spearman: diag=<mean>, off=<mean>; Pearson: diag=<mean>, off=<mean>`.
8. Derive one target order from the across-method mean Spearman diagonal and use it for every method panel in that family.
9. Annotate each diagonal cell with the perturbation's mean total cell count across repeats only when at most 40 perturbations are shown. Above that threshold suppress only the white diagonal count text and retain the full perturbation matrix.
10. Use `RdBu_r` centered at zero with fixed limits `[-1, 1]`.
11. Save the exact averaged Spearman, Pearson, and (for the FDR-union family) Jaccard matrices beside every PNG as an NPZ archive. The archive contains the plotted and basis methods, shared targets/pathways, dropped targets/pathways, mean and repeat-level per-cell feature counts, selection family, FDR threshold, plus `targets__<method>`, `programs__<method>`, `spearman__<method>`, `pearson__<method>`, `jaccard__<method>`, and finite-Jaccard-repeat arrays. This makes the common comparison basis auditable without recovering values from raster heatmaps.
12. For the combined FDR-union family, emit a Spearman/Pearson box-and-whisker figure comparing
    every finite diagonal cell with every finite ordered off-diagonal cell. Also emit separate
    diagonal and off-diagonal significant-pathway Jaccard boxplots. Every boxplot must overlay
    every finite repeat-averaged matrix cell as jittered scatter, without sampling; use IQR boxes
    and 5th–95th percentile whiskers.

Also average native effects by `method, arm, target, program` across repeats and produce one overview-style effect heatmap per arm. Keep pdex limits fixed to `[-1, 1]`; use observed symmetric OLS limits.

Use FDR for significant-set Jaccard, direction agreement, significant-count summaries, and the explicitly secondary FDR-union correlation-map family. Keep the all-pathway maps primary.

## Output contract

- `tables/pathways_test1_results__<dataset>.csv`: all canonical native test rows tagged by repeat and arm.
- `tables/pathways_test1_reproducibility__<dataset>.csv`: target-level within-method split metrics with `target, n_cells_total, effect_spearman, sig_jaccard, direction_agreement_sig_union, n_sig_a, n_sig_b, method, repeat`.
- `tables/pathways_test1_crossmethod__<dataset>.csv`: arm- and repeat-specific cross-method comparisons; emit only when both methods run.
- `tables/pathways_legend__<dataset>.csv`: retained program definitions.
- `plots/pathways_test1_corr_matrix_mean__<dataset>.png`: both methods side by side; colors encode repeat-averaged Spearman, and each panel title reports mean diagonal/off-diagonal Spearman and Pearson.
- `plots/pathways_test1_corr_matrix_mean__<dataset>.npz`: exact ordered target labels and averaged
  Spearman/Pearson matrices underlying the all-pathway heatmap. Every other correlation-map PNG has
  a same-stem NPZ archive with the same schema.
- `plots/pathways_test1_corr_matrix_mean_ols__<dataset>.png` and `..._pdex_mwu__<dataset>.png`: single-method versions with the same four title statistics.
- `plots/pathways_test1_corr_matrix_mean_fdr05__<dataset>.png`: both methods using the shared per-cell FDR-union feature sets at the default threshold.
- `plots/pathways_test1_corr_matrix_mean_ols_fdr05__<dataset>.png` and `..._pdex_mwu_fdr05__<dataset>.png`: single-method views of the same shared FDR-union matrices.
- `plots/pathways_test1_corr_matrix_mean_fdr05_correlation_boxplots__<dataset>.png`: combined-method
  Spearman/Pearson diagonal-versus-off-diagonal distributions for the FDR-union matrices, with a
  same-stem summary CSV.
- `plots/pathways_test1_corr_matrix_mean_fdr05_jaccard_diagonal_boxplot__<dataset>.png` and
  `..._jaccard_off_diagonal_boxplot__<dataset>.png`: within-perturbation and cross-perturbation
  significant-pathway Jaccard distributions, each with a same-stem summary CSV and every finite
  repeat-averaged matrix cell rendered as jittered scatter.
- `plots/pathways_test1_mean_arma__<dataset>.png` and `..._armb__<dataset>.png`: mean native-effect heatmaps.
- `pathways_test1_metadata__<dataset>.json`: shared score and argument metadata.

## Reproduction checks

- Reuse the identical precomputed score matrix in every arm and repeat.
- Require disjoint A/B indices within a repeat and group-specific cell totals equal to the retained original cells, except cells dropped by the per-arm threshold.
- Calculate correlation maps per repeat before averaging.
- Calculate both Spearman and Pearson matrices per repeat; average matrix cells first, then summarize diagonal and off-diagonal cells separately. Keep heatmap colors Spearman-only.
- Use every shared complete-case pathway in the primary family. In the secondary family, use the shared cross-method FDR union defined above.
- Assert that every combined and single-method archive in a family has the same targets, pathway basis, finite mask, and order; inspect the archived dropped-pathway diagnostics when inputs are incomplete.
- Never compare OLS and rank-biserial magnitudes directly; reproducibility is within method.
