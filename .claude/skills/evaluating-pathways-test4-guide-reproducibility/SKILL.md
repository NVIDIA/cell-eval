---
name: evaluating-pathways-test4-guide-reproducibility
description: Run or reproduce pathway Test 4 comparing same-guide split-half pathway reproducibility for unchanged bioconcord OLS and pdex Mann-Whitney. Use when an sgRNA column is available to measure guide-level effect-vector Spearman, significant-pathway Jaccard, direction agreement, per-arm cross-method agreement, repeat-averaged arm effects, and guide maps reporting diagonal/off-diagonal Spearman and Pearson correlations.
---

# Pathway Test 4 — Same-Guide Split-Half Reproducibility

Read the bundled `pathway-methods-memory.md` before running, reimplementing, or changing this skill. Use `test4_guide_reproducibility.py` with the local `pathway_utils.py`, which loads scoring and OLS from an official ArcInstitute/bioconcord checkout; all Test 1-style split and correlation helpers are bundled locally. Do not import another skill or copy the Bioconcord implementation into this folder.

## Scientific question

Estimate the reproducibility ceiling of individual sgRNA pathway signatures by splitting cells assigned to each guide into independent halves. This is Test 1 at guide rather than target-gene resolution.

## Run

```bash
python .claude/skills/evaluating-pathways-test4-guide-reproducibility/test4_guide_reproducibility.py \
  --adata <data.h5ad> \
  --programs <pathway-definitions.csv> \
  --bioconcord-root /path/to/bioconcord \
  --pert-col gene --sgrna-col <guide_column> --control non-targeting \
  --block-cols batch --score-layer X [--gene-symbol-col gene_name] \
  --min-genes 5 --ctrl-size 50 --n-bins 25 \
  --min-cells-per-arm 20 --n-repeats 5 \
  --max-guides 24 --max-guides-per-gene 3 --max-control 1200 [--normalize-raw] \
  --methods ols,pdex_mwu --threads 8 --fdr 0.05 --seed 42 \
  --outdir experiments_pathways/<dataset>__test4
```

This skill does not import or require output from another skill. Install the local
`requirements.txt`, clone `https://github.com/ArcInstitute/bioconcord.git`, and pass its root
with `--bioconcord-root` or `BIOCONCORD_ROOT`. Revision
`ee3a66fc512e9ee0fe87409240e16aa43698dff8` is the tested source version. `--programs` is a
required user-supplied CSV because pathway definitions are not committed with the skill.

## Reproduce the implementation

1. Read observations in backed mode before loading expression. Convert target and guide columns to strings and define invalid guides by case-insensitive membership in `""`, `nan`, `none`, or `na`.
2. Require at least `2 * min_cells_per_arm` cells per non-control guide. Map each guide to its modal target gene and discard genes with fewer than two powered guides. Rank eligible genes by decreasing powered-guide count, total cell count, then gene name. Within each gene, rank guides by decreasing cell count then guide label. Retain up to `--max-guides-per-gene` guides per gene (3 by default) without exceeding `--max-guides` overall (24 by default). Never retain a one-guide fragment of a gene group.
3. Retain every selected guide cell and sample at most `--max-control` control cells without replacement using `--seed`. Load only these cells into memory.
4. If `var_names` are feature IDs, require `--gene-symbol-col` and replace the in-memory subset's feature index with that column, preserving original IDs in `var['source_var_name']` and making duplicate symbols unique. Never modify the source file.
5. When `--normalize-raw` is supplied, require `--score-layer X`, total-normalize each selected cell's raw counts to 10,000 and apply log1p. Otherwise use the requested already normalized score layer.
6. Score this fixed guide/control subset once with the official Bioconcord program scorer. Reuse its identical signed cell-by-pathway matrix in every repeat, arm, and method.
7. Keep every selected control cell regardless of guide content. Assign every control cell the common control label and every retained non-control cell its guide string. This produces guide-level inference groups.
8. Build composite block labels from the requested columns.
9. For repeat `r`, split each guide and the pooled control group with the Test 1 `split_groups` procedure and seed `seed + r`. Map local retained indices back to subset indices.
10. Run each method separately on arm A and arm B. Tag canonical native rows with repeat and arm.
11. For each method, calculate Test 1 split metrics by guide: total cells, native-effect Spearman, significant-set Jaccard, sign agreement over the significant union, and arm-specific significant counts.
12. If both methods run, calculate the bundled cross-method comparison within every arm and repeat.
13. Preserve every repeat's split results. Average native effects by `method, arm, guide, program` across repeats and produce one overview-style effect heatmap per arm, with guides grouped by gene and black horizontal gene boundaries.
14. For each method and repeat, compute full split-A guide by split-B guide Spearman and Pearson matrices across shared pathway effects. Average both matrices cell by cell across repeats. Render Spearman colors and report mean diagonal/off-diagonal values for both metrics in every panel title. Order both axes by target gene and within-gene guide number, draw black boundaries between genes, and annotate diagonal cells with mean total guide-cell count. Emit combined and separate method maps with fixed correlation limits `[-1, 1]`.
15. Preserve raw guide constructs in native result tables. For plots, map guides within each modal target gene to compact deterministic labels `GENE | g1`, `GENE | g2`, and so on. Same-gene guides must be adjacent in heatmaps and correlation maps, share an obvious gene prefix, and be separated from the next gene group by a visible boundary. Store the mapping in metadata and a dedicated guide-map table.

## Output contract

- `tables/pathways_test4_results__<dataset>.csv`: all guide-pathway native results tagged by repeat and arm.
- `tables/pathways_test4_reproducibility__<dataset>.csv`: guide-level Test 1 split metrics.
- `tables/pathways_test4_crossmethod__<dataset>.csv`: arm/repeat cross-method comparisons; emit only for both methods.
- `tables/pathways_test4_guide_map__<dataset>.csv`: raw guide, target gene, compact plot label, and selected cell count.
- `tables/pathways_legend__<dataset>.csv`: retained program definitions.
- `plots/pathways_test4_mean_arma__<dataset>.png` and `..._armb__<dataset>.png`: guide-pathway native effects averaged across all repeats for each arm.
- `plots/pathways_test4_corr_matrix_mean__<dataset>.png`: combined repeat-averaged guide correlation maps; colors encode Spearman and titles report diagonal/off-diagonal Spearman and Pearson means.
- `plots/pathways_test4_corr_matrix_mean_ols__<dataset>.png` and `..._pdex_mwu__<dataset>.png`: explicit single-method maps with the same four title statistics.
- `pathways_test4_metadata__<dataset>.json`: shared score and argument metadata.

## Final visualization decisions

Follow finalized pathway Test 1 at guide resolution: do not emit the obsolete reproducibility box/scatter PNG, repeat-0-only plots, or per-guide zoom plots. Average effects and correlation maps across all five repeats. Keep all repeat-level metrics in tables for auditability.

## Reproduction checks

- Require control cells to share one reference label rather than their guide strings.
- Verify the selected subset contains only the chosen guides plus controls and record the selected guide list in metadata.
- Verify every plotted target gene has at least two selected guides; a correlation map containing only `g1` labels is a failed selection, not a valid same-gene comparison.
- Use `--normalize-raw` only for a raw-count X; never renormalize an already log-normalized layer.
- Require each retained non-control guide to meet the per-arm threshold; `split_groups` drops guides that fail it.
- Reuse fixed pathway scores and disjoint guide-specific halves.
- Require correlation maps to intersect guides and programs across both arms and every repeat, then average repeat-specific correlations rather than correlating already averaged effects.
- Keep pdex mean-effect heatmaps fixed to `[-1, 1]`, OLS heatmaps symmetric to their observed arm-specific range, and every correlation map fixed to `[-1, 1]`.
- Interpret low guide reproducibility alongside `n_cells_total`; it is the empirical ceiling for guide-level conclusions.
