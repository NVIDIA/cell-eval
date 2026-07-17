---
name: evaluating-pathways-test1-reproducibility
description: Run or reproduce pathway Test 1 comparing perturbation split-half reproducibility for official ArcInstitute Bioconcord OLS and pdex Mann-Whitney on one fixed Bioconcord score matrix. Use for within-method effect-vector Spearman, significant-pathway Jaccard, direction agreement, cell-count dependence, arm-level cross-method agreement, mean arm effects, and repeat-averaged perturbation maps using all aligned pathway coefficients and reporting diagonal/off-diagonal Spearman and Pearson correlations.
---

# Pathway Test 1 — Perturbation Split-Half Reproducibility

Read the bundled `pathway-methods-memory.md` before running, reimplementing, or changing this skill. Use `test1_reproducibility.py` with the local `pathway_utils.py`, which loads scoring and OLS from an official ArcInstitute/bioconcord checkout. Do not import another skill or copy the Bioconcord implementation into this folder.

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

For each method:

1. Intersect perturbations and pathways across both arms and all repeats.
2. Within each repeat, build two square matrices whose entry `(i, j)` compares all aligned pathway coefficients for perturbation `i` in arm A with perturbation `j` in arm B: one Spearman matrix and one Pearson matrix. Apply finite/nonconstant safeguards to both. Never restrict the correlation vectors by FDR, significance, effect size, or a pair-specific pathway union.
3. Average each matrix cell by cell across repeats; do not correlate effects after averaging.
4. Interpret the diagonal as within-perturbation reproducibility and off-diagonals as cross-perturbation similarity. Compute the finite mean of all diagonal cells and all off-diagonal cells separately for both correlation metrics.
5. Render the Spearman matrix. In every method-panel title, report `Spearman: diag=<mean>, off=<mean>; Pearson: diag=<mean>, off=<mean>`.
6. Sort both axes by increasing mean Spearman diagonal value, preserving target alignment.
7. Annotate each diagonal cell with the perturbation's mean total cell count across repeats.
8. Use `RdBu_r` centered at zero with fixed limits `[-1, 1]`.
9. Save the exact averaged Spearman and Pearson matrices beside every PNG as an NPZ archive. The
   archive contains `methods`, plus `targets__<method>`, `spearman__<method>`, and
   `pearson__<method>` arrays. This makes downstream distribution plots reproducible without
   recovering values from raster heatmaps.

Also average native effects by `method, arm, target, program` across repeats and produce one overview-style effect heatmap per arm. Keep pdex limits fixed to `[-1, 1]`; use observed symmetric OLS limits.

Use FDR only for significant-set Jaccard, direction agreement, and significant-count summaries.
Do not use FDR to select pathways for any correlation matrix.

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
- `plots/pathways_test1_mean_arma__<dataset>.png` and `..._armb__<dataset>.png`: mean native-effect heatmaps.
- `pathways_test1_metadata__<dataset>.json`: shared score and argument metadata.

## Reproduction checks

- Reuse the identical precomputed score matrix in every arm and repeat.
- Require disjoint A/B indices within a repeat and group-specific cell totals equal to the retained original cells, except cells dropped by the per-arm threshold.
- Calculate correlation maps per repeat before averaging.
- Calculate both Spearman and Pearson matrices per repeat; average matrix cells first, then summarize diagonal and off-diagonal cells separately. Keep heatmap colors Spearman-only.
- Use every aligned pathway coefficient in every correlation vector; never select correlation features using FDR calls.
- Never compare OLS and rank-biserial magnitudes directly; reproducibility is within method.
