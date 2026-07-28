---
name: evaluating-pathways-test5-samegene-sgrna
description: Run or reproduce pathway Test 5 comparing whether independent sgRNAs targeting the same gene have more concordant pathway signatures than unrelated guides under unchanged bioconcord OLS and pdex Mann-Whitney. Use for guide-pair effect-vector Spearman, significant-set Jaccard, one-sided Mann-Whitney separation, probability-of-superiority AUC, gene-grouped guide heatmaps, same-gene zooms, and guide correlation blocks.
---

# Pathway Test 5 — Same-Gene Independent-Guide Concordance

Read the bundled `pathway-methods-memory.md` before running, reimplementing, or changing this skill. Use `test5_samegene_sgrna.py` with the local `pathway_utils.py`, which loads scoring and OLS from an official ArcInstitute/bioconcord checkout. Do not import another skill or copy the Bioconcord implementation into this folder.

## Mandatory preflight and run capture

Do not start the executable until the user has explicitly confirmed one fully resolved run configuration.

1. Gather the input `.h5ad`, ask whether `adata.X` contains raw counts or log1p-normalized expression, pathway-definition CSV, official Bioconcord checkout, results output directory, separate run root, methods to compare (`ols`, `pdex_mwu`, or both), perturbation/guide/control fields, score layer and optional gene-symbol field, thresholds, scoring settings, gene/guide scope, seeds, and threads. Inspect the input read-only to resolve unknown columns, labels, layers, feature identifiers, and powered same-gene groups. Pass the confirmed state as `--expression-state`.
2. Expand paths and resolve every default. Show one concise preflight summary containing inputs, results directory, run root, methods, data fields/layers, thresholds, selected gene/guide scope, workload/concurrency, exact command, log path, and resolved-config destination. Before asking for confirmation, estimate wall time separately for every selected method, shared scoring/preparation/rendering, and the complete run; state the hardware tier, cache assumptions, evidence or throughput basis, uncertainty range, and how watchdog de-escalation could extend it.
3. Ask for explicit confirmation and stop. Do not launch subsetting, scoring, inference, or plotting before confirmation.
4. After confirmation, create `<run-root>/logs` and `<run-root>/configs`, pass `--run-root <run-root>`, and capture the complete terminal stream with `2>&1 | tee <run-root>/logs/<workflow>__<dataset>__<UTC-timestamp>.log`.
5. Every invocation writes an immutable timestamped YAML snapshot under `<run-root>/configs`. Report the result, log, and YAML paths on completion.

Every box-and-whisker plot must overlay every finite underlying observation as jittered scatter points. Do not sample, aggregate away, or hide values in the scatter layer.

Retain every analytically eligible same-gene guide group by default: `--max-genes 0` means unlimited. Never reduce the guide set merely to make a correlation plot readable. When more than 40 guides are shown, omit only the white diagonal cell-count text; keep the full numeric matrix and every guide. A nonzero gene cap is allowed only as an explicit, user-confirmed scientific selection.

## Scientific question

Test whether two independently designed guides assigned to the same target gene produce more similar pathway-effect vectors than guides assigned to different genes.

## Run

```bash
python .claude/skills/evaluating-pathways-test5-samegene-sgrna/test5_samegene_sgrna.py \
  --adata <data.h5ad> \
  --programs <pathway-definitions.csv> \
  --bioconcord-root /path/to/bioconcord \
  --pert-col gene --sgrna-col <guide_column> --control non-targeting \
  --score-layer X [--gene-symbol-col gene_name] \
  --min-genes 5 --ctrl-size 50 --n-bins 25 \
  --min-cells 20 --min-guides 2 --max-genes 0 --max-control 1500 \
  [--normalize-raw] --background-multiplier 10 \
  --methods ols,pdex_mwu --threads 8 --fdr 0.05 --seed 42 \
  --outdir experiments_pathways/<dataset>__test5
```

This skill does not import or require output from another skill. Install the local
`requirements.txt`, clone `https://github.com/ArcInstitute/bioconcord.git`, and pass its root
with `--bioconcord-root` or `BIOCONCORD_ROOT`. Revision
`ee3a66fc512e9ee0fe87409240e16aa43698dff8` is the tested source version. `--programs` is a
required user-supplied CSV because pathway definitions are not committed with the skill.

## Reproduce the implementation

### 1. Select eligible guides

1. Read observations in backed mode before loading expression. Convert target-gene and sgRNA columns to strings and treat case-insensitive `""`, `nan`, `none`, and `na` as invalid guide labels.
2. Count valid guides among non-control cells and retain guides with at least `--min-cells` cells. Assign each guide to its modal non-control target-gene value.
3. Retain genes with at least `--min-guides` powered guides. Rank genes by decreasing guide count, then total selected-guide cells, then gene label. `--max-genes` defaults to zero (unlimited); use a nonzero cap only when the user explicitly confirms that scientific selection.
4. Keep every selected guide cell and sample at most `--max-control` control cells without replacement using `--seed`. Load only this subset into memory.
5. If `var_names` are feature IDs, require `--gene-symbol-col` and replace the in-memory subset's feature index with that column, preserving original IDs in `var['source_var_name']` and making duplicate symbols unique. Never modify the source file.
6. When `--normalize-raw` is supplied, require `--score-layer X`, total-normalize each selected cell's raw counts to 10,000 and apply log1p. Otherwise use the requested already normalized score layer.
7. Score the fixed selected subset once using official Bioconcord scoring. Label controls with the common control label and non-controls by guide. Record selected genes and guides in metadata.

### 2. Construct guide pairs

- Enumerate every unordered pair of eligible guides assigned to the same gene; require at least one such pair.
- Enumerate every unordered pair assigned to different genes as the unrelated pool.
- Let `S` be the number of same-gene pairs. Cap the unrelated sample at:

```text
min(number of all unrelated pairs,
    max(S, S * background_multiplier))
```

- If needed, sample unrelated-pair indices without replacement using RNG seed `--seed`. Preserve all same-gene pairs.

### 3. Infer guide effects and compare pairs

1. Run each pathway method once with guides as perturbation groups and pooled control as reference.
2. Pivot each method's native effects into `guide x pathway` form.
3. For every retained same-gene and sampled unrelated pair, compute:
   - Spearman correlation across aligned native pathway effects;
   - Jaccard similarity of the two guides' `fdr <= --fdr` pathway sets;
   - guide and gene labels;
   - cell counts for both guides.
4. Within each method, drop missing Spearman values and compare same-gene correlations with unrelated correlations using one-sided `scipy.stats.mannwhitneyu(..., alternative="greater")`.
5. Report probability of superiority:

```text
AUC = U / (n_same_gene_pairs * n_unrelated_pairs)
```

An AUC of 0.5 indicates no separation; values above 0.5 favor higher concordance for same-gene pairs.

Because guide pairs share guides, pairwise observations are not statistically independent. Treat the Mann-Whitney p-value as a descriptive screen rather than a gene-level confirmatory test; use the effect distribution, AUC, Test 4 reliability ceiling, and gene-specific plots together.

## Required plots

Create `plots/pathways_test5_samegene__<dataset>.png` with:

1. A method-hued violin plot of pairwise effect Spearman by `same_gene` versus `unrelated`, clipped to observed support and showing quartiles.
2. A method-specific bar plot of `auc_same_gt_unrelated` with a dashed reference at 0.5 and y limits `[0, 1]`.

Also match the finalized guide-level Test 5 diagnostic views using pathway-native effects:

1. `plots/pathways_test5_guide_heatmap_<method>__<dataset>.png`: one heatmap per method with guide rows grouped by target gene and separated by black lines. Label rows `GENE | gN`, so guides tackling the same gene are visibly adjacent and share the same prefix. Order pathways by mean guide effect. Use fixed `[-1, 1]` for pdex rank-biserial and observed symmetric limits for OLS.
2. `plots/pathways_test5_corr_matrix__<dataset>.png`: one panel per method showing guide-by-guide Spearman correlation across all aligned native pathway effects, ordered in the same labeled gene blocks with black horizontal and vertical separators and fixed limits `[-1, 1]`. Report finite mean diagonal and off-diagonal rho, plus the within-gene and cross-gene off-diagonal means, in each panel title.
3. `plots/pathways_test5_zoom__<dataset>_NN.png`: one page per retained gene. Each method row contains a native-effect scatter for the two most-powered guides and a heatmap strip containing every retained guide for that gene. Reuse the exact `GENE | gN` identities from the guide-map table even when the zoom orders guides by power. Color scatter pathways called in either guide separately from non-called pathways.

## Output contract

- `tables/pathways_test5_results__<dataset>.csv`: concatenated canonical guide-pathway native results.
- `tables/pathways_test5_pairs__<dataset>.csv`: one row per method and guide pair with pair type, guide/gene labels, effect Spearman, significant-set Jaccard, and guide cell counts.
- `tables/pathways_test5_summary__<dataset>.csv`: pair counts, median correlations, AUC, and one-sided Mann-Whitney p-value by method.
- `tables/pathways_test5_guide_map__<dataset>.csv`: raw guide, target gene, compact `GENE | gN` plot label, and cell count.
- `tables/pathways_legend__<dataset>.csv`: retained program definitions.
- `plots/pathways_test5_samegene__<dataset>.png`: pair distributions and AUC.
- `plots/pathways_test5_guide_heatmap_ols__<dataset>.png` and `..._pdex_mwu__<dataset>.png`: gene-grouped guide pathway effects.
- `plots/pathways_test5_corr_matrix__<dataset>.png`: combined method guide-correlation blocks.
- `plots/pathways_test5_zoom__<dataset>_NN.png`: gene-specific cross-method zoom pages.
- `pathways_test5_metadata__<dataset>.json`: shared score and argument metadata.

## Reproduction checks

- Require at least two eligible guides for at least one target gene.
- Require at least two retained target genes so an unrelated-guide background exists.
- Require nonempty finite Spearman samples in both pair classes before Mann-Whitney testing.
- Sample unrelated pairs once and reuse the identical pair list for both methods.
- Use pathway FDR only for pairwise Jaccard; use all aligned native effects for Spearman and the separation test.
- Verify the selected subset contains only the retained same-gene guides plus sampled controls.
- Use `--normalize-raw` only for raw-count X; never renormalize an already log-normalized layer.
- Keep guide and gene ordering identical across method heatmaps and correlation panels.
