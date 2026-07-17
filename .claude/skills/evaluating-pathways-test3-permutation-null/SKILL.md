---
name: evaluating-pathways-test3-permutation-null
description: Run or reproduce pathway Test 3 comparing real pathway signal with five global and five within-block perturbation-label permutations for unchanged bioconcord OLS and pdex Mann-Whitney. Use to contrast unrestricted and batch-preserving shuffle nulls while averaging significant-pathway null counts, cross-method called-set agreement, and method-specific shuffled-effect correlations across permutations.
---

# Pathway Test 3 — Global and Within-Block Label-Permutation Nulls

Read the bundled `pathway-methods-memory.md` before running, reimplementing, or changing this skill. Use `test3_permutation_null.py` with the local `pathway_utils.py`, which loads scoring and OLS from an official ArcInstitute/bioconcord checkout. Do not import another skill or copy the Bioconcord implementation into this folder.

## Scientific question

Compare pathway discoveries under real perturbation labels with two nulls that destroy label-to-cell association: an unrestricted global shuffle and a within-block shuffle that preserves every block's label composition.

## Run

```bash
python .claude/skills/evaluating-pathways-test3-permutation-null/test3_permutation_null.py \
  --adata <data.h5ad> \
  --programs <pathway-definitions.csv> \
  --bioconcord-root /path/to/bioconcord \
  --pert-col gene --control non-targeting --block-cols batch \
  --score-layer X --min-genes 5 --ctrl-size 50 --n-bins 25 \
  --n-permutations 5 --shuffle-mode both \
  --methods ols,pdex_mwu --threads 8 \
  --fdr 0.05 --seed 42 \
  --outdir experiments_pathways/<dataset>__test3
```

This skill does not import or require output from another skill. Install the local
`requirements.txt`, clone `https://github.com/ArcInstitute/bioconcord.git`, and pass its root
with `--bioconcord-root` or `BIOCONCORD_ROOT`. Revision
`ee3a66fc512e9ee0fe87409240e16aa43698dff8` is the tested source version. `--programs` is a
required user-supplied CSV because pathway definitions are not committed with the skill.

## Reproduce the implementation

1. Load programs and compute the full pathway-score matrix once.
2. Build composite block labels from requested columns.
3. Run each requested method once with the real labels. Tag rows with `state="real"` and `permutation=-1`.
4. Count `n_programs` and pathways with `fdr <= --fdr` (`n_sig`) separately for every perturbation and method.
5. Resolve `--shuffle-mode` as `global`, `within`, or both modes in the order global then within.
6. For permutation `k`, initialize RNG with `seed + k` and permute the complete label vector:
   - `global`: permute labels across all cells, preserving only the dataset-wide label histogram;
   - `within`: independently permute labels inside every block, preserving each block's label histogram.
7. Use exactly 5 permutations per requested shuffle mode; the script rejects any other `--n-permutations` value. Run both methods on each mode-specific permuted label vector. Tag rows with `state="permuted"`, `shuffle_mode`, and the zero-based permutation index, then count discoveries by perturbation and method. Fit the real labels only once and tag those detailed rows with `shuffle_mode="real"`.
8. Retain every permutation-level result for auditability, but average plot-level null call counts, called-set Jaccard values, and correlation matrices across the 5 permutations. Never pool pathway rows before calling significance or calculating a permutation-specific correlation matrix.

## Null and separation summaries

For every shuffle mode, method, and perturbation, aggregate permuted `n_sig` values as:

```text
null_mean = mean(n_sig)
null_sd   = sample standard deviation(n_sig)
null_q95  = empirical 95th percentile(n_sig)
```

Merge these values onto the real count and compute:

```text
separation_z = (real n_sig - null_mean) / null_sd
```

Use missing `separation_z` when `null_sd` is zero. Merge the perturbation-specific separation value back onto all count rows for plotting convenience.

## Required plots

Match the finalized non-pathway Test 3 plot roles using pathway-native effects and calls. Do not emit the former real-versus-null count/separation boxplot.

1. Create `plots/pathways_test3_shuffle_call_comparison__<mode>__<dataset>.png` when both methods run. For each perturbation, average its significant-program count separately for OLS and pdex across that mode's permutations. Plot mean pdex count on x and mean OLS count on y; encode shuffled-group cell count by marker size/color, add an identity line and method-dominance background regions, and label every target with its mean called-set Jaccard across permutations. Use only `fdr <= --fdr` as the pathway call rule; do not introduce a gene-level LFC threshold.
2. Create one mode- and method-specific correlation matrix: `pathways_test3_corr_matrix__<mode>__<method>__<dataset>.png`. Within each permutation, compute target-by-target Spearman correlations across native pathway-effect vectors, force the self-correlation diagonal to one, then average matrices cell by cell across permutations. Use `RdBu_r` fixed to `[-1, 1]` and report finite mean diagonal and off-diagonal rho separately. A calibrated null should show a clean diagonal and near-zero off-diagonal values.

## Output contract

- `tables/pathways_test3_results__<dataset>.csv`: all canonical real and permuted test rows with state, permutation, and `shuffle_mode`; real rows occur once with mode `real`.
- `tables/pathways_test3_counts__<dataset>.csv`: per-mode, method, perturbation, state, and permutation `n_programs`, `n_sig`, and merged separation z. Real count rows are repeated once per requested null mode so every real-null comparison is explicit.
- `tables/pathways_test3_separation__<dataset>.csv`: per-mode real counts with `null_mean`, `null_sd`, `null_q95`, and `separation_z`.
- `tables/pathways_test3_method_comparison__<dataset>.csv`: per-mode mean permuted call counts, mean called-set Jaccard, and cell count per target; emit only when both methods run.
- `tables/pathways_legend__<dataset>.csv`: retained program definitions.
- `plots/pathways_test3_shuffle_call_comparison__<mode>__<dataset>.png`: mode-specific cross-method mean null-call scatter; emit only when both methods run.
- `plots/pathways_test3_corr_matrix__<mode>__ols__<dataset>.png`: mode-specific mean OLS permutation-null correlation map when OLS runs.
- `plots/pathways_test3_corr_matrix__<mode>__pdex_mwu__<dataset>.png`: mode-specific mean pdex permutation-null correlation map when pdex runs.
- `pathways_test3_metadata__<dataset>.json`: shared score and argument metadata.

## Reproduction checks

- Require every permutation to retain the dataset-wide label histogram.
- For within-block permutations, additionally require every block to retain its exact real-label histogram. Do not impose this invariant on global permutations.
- Never rescore or reorder cells between real and permuted runs.
- Apply FDR within each perturbation after every inference run.
- Interpret a large positive separation only after confirming low absolute null discovery counts.
- Confirm every summary and plot filters one `shuffle_mode` before aggregating.
- Confirm correlation matrices use the intersection of targets and programs present in every permutation of that mode.
- Confirm each cross-method scatter averages counts and Jaccard over permutations within that mode rather than pooling pathway rows before calling significance.
