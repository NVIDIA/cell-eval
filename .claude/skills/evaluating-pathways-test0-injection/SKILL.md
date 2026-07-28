---
name: evaluating-pathways-test0-injection
description: Run or reproduce pathway Test 0 comparing bioconcord OLS with pdex Mann-Whitney under one-pathway-at-a-time raw-UMI log2FC injections. Use to multiply pathway-gene UMIs by 2**delta, normalize and rescore all pathways, count injected-pathway true positives and untouched-pathway false positives across 10 repeats, audit the realized multiplier, and plot pathway-specific count trajectories with percentile tubes.
---

# Pathway Test 0 — Raw-UMI Pathway Injection

Read the bundled `pathway-methods-memory.md` completely before running, reimplementing, or changing this skill. Use `test0_injection.py` with the local `pathway_utils.py`, which loads scoring and OLS from an official ArcInstitute/bioconcord checkout. Do not import another skill or copy the Bioconcord implementation into this folder.

## Mandatory preflight and run capture

Do not start the executable until the user has explicitly confirmed one fully resolved run configuration.

1. Gather the input `.h5ad`, ask whether `adata.X` contains raw counts or log1p-normalized expression, pathway-definition CSV, official Bioconcord checkout, results output directory, separate run root, methods to compare (`ols`, `pdex_mwu`, or both), perturbation/control/block fields, raw-count layer, thresholds, pathway scope, deltas, repeats, seeds, and threads. Inspect the input read-only to resolve unknown columns, labels, layers, and feature identifiers. Pass the confirmed state as `--expression-state`.
2. Expand paths and resolve every default. Show one concise preflight summary containing inputs, results directory, run root, methods, data fields/layers, thresholds, pathway scope, workload/concurrency, exact command, log path, and resolved-config destination. Before asking for confirmation, estimate wall time separately for every selected method, shared scoring/preparation/rendering, and the complete run; state the hardware tier, cache assumptions, evidence or throughput basis, uncertainty range, and how watchdog de-escalation could extend it.
3. Ask for explicit confirmation and stop. Do not launch scoring, injection, inference, or plotting before confirmation.
4. After confirmation, create `<run-root>/logs` and `<run-root>/configs`, pass `--run-root <run-root>`, and capture the complete terminal stream with `2>&1 | tee <run-root>/logs/<workflow>__<dataset>__<UTC-timestamp>.log`.
5. Every invocation writes an immutable timestamped YAML snapshot under `<run-root>/configs`. Report the result, log, and YAML paths on completion.

Every box-and-whisker plot must overlay every finite underlying observation as jittered scatter points. Do not sample, aggregate away, or hide values in the scatter layer.

For every pathway correlation matrix, retain every analytically eligible perturbation or guide. Never reduce the unit set merely to make the plot readable. When more than 40 units are shown, omit only the white diagonal cell-count text; keep the full numeric matrix and all units. Any analysis cap must be an explicit, user-confirmed scientific selection, not an automatic display rule.

## Scientific question

Measure how often each inference method recovers a known pathway whose member-gene raw UMIs were increased in pseudo-perturbed control cells, and how many untouched pathways it calls at the same time. Inject one pathway per trial so every retained pathway receives its own power and spillover curve.

## Required inputs

- Supply an `.h5ad` with a raw nonnegative count matrix in `--counts-layer` and a control label in `obs[--pert-col]`. The implementation checks only finiteness and nonnegativity, not integer-valuedness; use genuine integer UMI counts for the exact multiplier audit.
- Supply a compatible pathway-definition CSV through required `--programs`; it is not bundled with the skill.
- Supply comma-separated block columns for stratified control splitting; an empty list means one global block.
- Use every retained pathway for final analysis. Reserve `--pathways` for smoke tests; it accepts comma-separated retained labels such as `C000,C001`, not representative names.
- Use unique `adata.var_names` and unique gene names within each program. The script explicitly rejects violations before injection so gene indexing and UMI audits cannot silently double-count features.

## Run

```bash
python .claude/skills/evaluating-pathways-test0-injection/test0_injection.py \
  --adata <data.h5ad> \
  --programs <pathway-definitions.csv> \
  --bioconcord-root /path/to/bioconcord \
  --pert-col gene --control non-targeting --block-cols batch \
  --counts-layer counts --normalization-target 10000 \
  --min-genes 5 --deltas 0,0.5,1,2 \
  --n-repeats 10 --min-cells-per-arm 20 \
  --methods ols,pdex_mwu --threads 8 --score-jobs 1 \
  --seed 42 --fdr 0.05 --log-file "" \
  --outdir experiments_pathways/<dataset>__test0
```

This skill does not import or require output from another skill. Install the local
`requirements.txt`, clone `https://github.com/ArcInstitute/bioconcord.git`, and pass its root
with `--bioconcord-root` or `BIOCONCORD_ROOT`. Revision
`ee3a66fc512e9ee0fe87409240e16aa43698dff8` is the tested source version. `--programs` is a
required user-supplied CSV because pathway definitions are not committed with the skill.

Leave `--log-file` empty to write `pathways_test0_run__<dataset>.log` in `--outdir`, or pass an explicit path. The logger writes simultaneously to stdout and the file, truncating an existing log at that path when a new run starts.

## Reproduce the implementation

### 1. Load counts and programs

1. Read `adata.X` when `--counts-layer` is `X` or empty; otherwise read the named layer.
2. Reject non-finite or negative values and work in float64 CSR form. Treat the matrix as raw UMI counts.
3. Load one representative program per cluster with the bundled selection rule and build integer column indices for every program gene.
4. Select only cells whose perturbation label equals `--control`. Require at least `2 * min_cells_per_arm` controls.

### 2. Create deterministic pseudo-perturbation splits

For repeat `r`, initialize `numpy.random.default_rng(seed + r)` and split control indices within composite block labels:

- Randomly permute cells independently inside each block.
- Put the first floor-half in the reference arm and the remainder in the injected arm.
- Alternate singleton blocks between arms.
- Fall back to one unstratified random half split if an arm is empty.
- Fail if either final arm has fewer than `--min-cells-per-arm` cells.

Concatenate reference-arm rows first and injected-arm rows second. Label them with the real control label and literal `injected`. Reuse this repeat-specific split for every injected pathway and delta.

### 3. Inject one pathway in raw-count space

For injected pathway `p` and log2FC delta `d`, update only injected-arm rows and pathway-gene columns:

```text
counts_after[cell, gene] = np.rint(counts_before[cell, gene] * 2**d)
```

Keep all other entries unchanged and keep zeros at zero. `np.rint` uses NumPy's ties-to-even rounding. For sparse input, convert to CSC, update each selected gene's stored injected-row values, remove zeros, and return CSR. Interpret delta literally: 0 = 1x, 0.5 = sqrt(2)x before rounding, 1 = 2x, and 2 = 4x raw pathway-gene UMI. Any float delta is accepted when `2**delta` is finite and positive, including negative log2FC values.

Do not add delta to normalized expression or to pathway scores.

### 4. Normalize and recompute all scores after every injection

For every nonzero injection trial:

1. Compute each cell's total UMI count and fail on zero-total cells.
2. Multiply each row by `normalization_target / row_total`.
3. Apply `log1p` to every stored nonzero entry.
4. Recompute all retained pathway scores, not only the injected pathway.

Use the vectorized bioconcord-equivalent scorer:

1. Compute mean normalized-log expression for every gene.
2. Assign genes to `pandas.qcut(..., 25, labels=False, duplicates="drop")` expression bins.
3. Reset RNG seed 42 separately for every pathway.
4. For each pathway gene, sample without replacement up to 50 other genes from its bin.
5. Union and deduplicate all sampled controls for that pathway.
6. Construct a sparse gene-by-pathway weight matrix with `+1/n_pathway_genes` on pathway genes and `-1/n_unique_controls` on background genes.
7. Multiply expression by the weight matrix to obtain `cells x pathways` scores.

On the first baseline split, also call original `bioconcord.score_all_programs` and fail unless maximum absolute disagreement is at most `1e-7`. Reuse each repeat's baseline scores and inference results for every pathway designation at delta 0.

### 5. Run both inference methods

Give both methods the same rescored matrix and labels. Run the bundled reference-coded Bioconcord OLS and pdex reference-mode Mann-Whitney exactly as specified in the local statistical contract. Keep pdex `geometric_mean=False` and `is_log1p=False` because its input is the signed pathway-score matrix.

Suppress pdex's nested stderr progress bar only. Log explicit progress to stdout and the persistent run log.

After every pathway/delta trial, log the completed and total trial counts, percentage, repeat, pathway, delta, elapsed wall time, and throughput-based ETA. Also log startup arguments, workload dimensions, first-split scorer validation, repeat/pathway boundaries, uncaught exception tracebacks, and successful completion. Use timestamped log lines and flush them through standard logging handlers so another process can monitor the file with `tail -f`.

### 6. Classify calls and audit UMI injection

For every repeat, delta, injected pathway, and method:

- Mark every tested pathway `called` when `fdr <= --fdr`.
- Set `tp_called` to 1 when the designated injected pathway is called, otherwise 0.
- Set `false_positive_count` to the number of called pathways other than the injected pathway.
- Sum injected-arm raw UMIs across the injected pathway genes before and after injection.
- Record `observed_umi_multiplier = UMI_after / UMI_before`; use missing when the baseline total is zero.

Aggregate by injected pathway, delta, and method:

```text
TP count = sum(tp_called across repeats)
FP count = sum(false_positive_count across repeats)
```

These are counts, not rates. At 10 repeats, TP count ranges from 0 to 10; FP count can exceed 10 because every repeat can call many untouched pathways.

## Output schemas and expected sizes

Let `P` be the number of retained pathways scored in every trial, `T` the number selected for injection (`T=P` in a final run), `R` repeats, `D` unique deltas, and `M` methods.

- `tables/pathways_test0_per_program__<dataset>.csv`: `T * R * D * M * P` native test rows. After the canonical method columns, append `repeat, delta_log2fc, umi_multiplier, pathway_gene_umi_before, pathway_gene_umi_after, observed_umi_multiplier, injected_program, is_injected, called`.
- `tables/pathways_test0_per_repeat__<dataset>.csv`: `T * R * D * M` rows with columns `repeat, delta_log2fc, umi_multiplier, pathway_gene_umi_before, pathway_gene_umi_after, observed_umi_multiplier, method, injected_program, tp_called, false_positive_count, n_untouched`.
- `tables/pathways_test0_per_pathway__<dataset>.csv`: `T * D * M` rows keyed by `injected_program, delta_log2fc, method`, followed by `umi_multiplier, observed_umi_multiplier_mean, n_repeats, tp_count, fp_count, fp_count_median_per_repeat`.
- `tables/pathways_test0_summary__<dataset>.csv`: `D * M` rows keyed by `delta_log2fc, method`, followed by `n_pathways, tp_count_median, tp_count_q10, tp_count_q90, fp_count_median, fp_count_q10, fp_count_q90`. Compute quantiles with pandas' default linear interpolation.
- `tables/pathways_legend__<dataset>.csv`: retained pathway definitions.
- `pathways_test0_metadata__<dataset>.json`: arguments, score source, injection space/unit/design, rescoring, and vectorized-scorer validation contract.
- `pathways_test0_run__<dataset>.log`: timestamped startup, validation, trial progress/ETA, errors, and completion when `--log-file` is not supplied.

The implementation accumulates all trial frames and writes files after all trials finish; absence of partial CSVs during a run is expected.

## Required plot

Create `plots/pathways_test0_calibration__<dataset>.png` with method rows and exactly two columns:

1. injected-pathway TP count;
2. untouched-pathway FP count.

For every panel:

- Draw one trajectory per injected pathway across delta, using a shared viridis pathway gradient.
- Overlay a black pathway-wise median trajectory.
- Shade the pathway-wise 10th-90th percentile count tube.
- Use integer y-axis ticks. Fix TP limits to `-0.5` through `R + 0.5`; scale FP to the observed maximum.
- Label the color bar as injected pathway and select at most eight displayed labels with `unique(linspace(0, P-1, min(8, P)).astype(int))` over sorted program labels.
- Never plot TP or FP rates.

## Reproduction checks

- Require the first baseline vectorized score comparison to pass `1e-7`.
- Require every injected pathway to have exactly `R` trial rows per delta and method before per-pathway aggregation.
- At delta 0, require requested multiplier 1 and unchanged UMI totals; require observed multiplier 1 only when the baseline pathway-gene UMI total is nonzero. Every designated pathway is null at this delta.
- At delta 2, require observed raw-UMI multiplier exactly 4 for nonzero baseline totals. Delta 0.5 may differ slightly from square root of 2 because counts are rounded.
- Require both methods to receive byte-equivalent score values within each trial.
- Preserve effects in untouched pathways caused by overlap, background selection, and total-count normalization; these are pipeline behavior, not bookkeeping errors.
