---
name: evaluating-pathways-overview
description: Run or reproduce the finalized pathway-method overview comparing official ArcInstitute Bioconcord OLS with pdex Mann-Whitney on identical Bioconcord cell-level pathway scores. Use for full-dataset pathway inference, effect and significant-call comparisons, called-pathway Jaccard, direction agreement, pathway-effect distributions ordered by perturbation cell or UMI count, and MA-style raw-expression versus effect clouds for least-, median-, and most-cell perturbations with FDR calls highlighted.
---

# Pathway Method Overview

Read the bundled `pathway-methods-memory.md` completely before running, reimplementing, or changing this skill. Treat that file and this specification as the statistical contract. Keep the implementation in this folder:

- `overview.py`: orchestration and file output.
- `pathway_utils.py`: local program loading, scoring, inference, split/permutation, summary, metadata, and plotting utilities.
- An external official ArcInstitute/bioconcord checkout supplies the scoring and OLS routines; no Bioconcord implementation is copied into this skill.
- `requirements.txt`: standalone Python dependencies.

## Mandatory preflight and run capture

Do not start the executable until the user has explicitly confirmed one fully resolved run configuration.

1. Gather the input `.h5ad`, ask whether `adata.X` contains raw counts or log1p-normalized expression, pathway-definition CSV, official Bioconcord checkout, results output directory, separate run root, methods to compare (`ols`, `pdex_mwu`, or both), perturbation/control and QC fields, score/count layers, thresholds, scoring settings, seeds, and threads. Inspect the input read-only to resolve unknown columns, labels, layers, and feature identifiers. Pass the confirmed state as `--expression-state`.
2. Expand paths and resolve every default. Show one concise preflight summary containing inputs, results directory, run root, methods, data fields/layers, thresholds, workload/concurrency, exact command, log path, and resolved-config destination. Before asking for confirmation, estimate wall time separately for every selected method, shared scoring/preparation/rendering, and the complete run; state the hardware tier, cache assumptions, evidence or throughput basis, uncertainty range, and how watchdog de-escalation could extend it.
3. Ask for explicit confirmation and stop. Do not launch scoring, inference, plotting, or cache reuse before confirmation.
4. After confirmation, create `<run-root>/logs` and `<run-root>/configs`, pass `--run-root <run-root>`, and capture the complete terminal stream with `2>&1 | tee <run-root>/logs/<workflow>__<dataset>__<UTC-timestamp>.log`.
5. Every invocation writes an immutable timestamped YAML snapshot under `<run-root>/configs`. Report the result, log, and YAML paths on completion.

Every box-and-whisker plot must overlay every finite underlying observation as jittered scatter points. Do not sample, aggregate away, or hide values in the scatter layer.

For every pathway correlation matrix, retain every analytically eligible perturbation or guide. Never reduce the unit set merely to make the plot readable. When more than 40 units are shown, omit only the white diagonal cell-count text; keep the full numeric matrix and all units. Any analysis cap must be an explicit, user-confirmed scientific selection, not an automatic display rule.

## Required inputs

- Supply an `.h5ad` file whose `obs` contains the perturbation column, control label, and per-cell UMI column.
- Supply `--programs <pathway-definitions.csv>`. The CSV is not bundled and must contain `cluster`, `representative`, and whitespace-delimited `genes` columns.
- Supply log-normalized expression in `adata.X` or `--score-layer`. Never use raw counts as the overview scoring matrix.
- Supply finite, nonnegative raw UMI counts in `--counts-layer` for expression-versus-effect plots. This matrix is used only for the plot x coordinate, never for pathway scoring or inference.
- Keep `--ctrl-size 50`, `--n-bins 25`, and seed 42. `load_and_score` rejects other scoring settings.

## Run

```bash
python .claude/skills/evaluating-pathways-overview/overview.py \
  --adata <data.h5ad> \
  --programs <pathway-definitions.csv> \
  --bioconcord-root /path/to/bioconcord \
  --pert-col gene --control non-targeting \
  --score-layer X --counts-layer counts --umi-col UMI_count \
  --min-genes 5 --ctrl-size 50 --n-bins 25 \
  --methods ols,pdex_mwu --threads 8 --fdr 0.05 \
  --outdir experiments_pathways/<dataset>__overview
```

This skill does not import sibling skills. Install its dependencies, clone the official source,
and either pass its root as above or set `BIOCONCORD_ROOT`:

```bash
python -m pip install -r .claude/skills/evaluating-pathways-overview/requirements.txt
git clone https://github.com/ArcInstitute/bioconcord.git /path/to/bioconcord
git -C /path/to/bioconcord checkout ee3a66fc512e9ee0fe87409240e16aa43698dff8
```

The pinned revision records the tested source version. The loader also discovers a `bioconcord`
checkout in the current directory or an ancestor directory. `--programs` is required because
pathway definitions are deliberately not committed with the skill.

`--threads 0` delegates thread choice to pdex. `--methods` accepts `ols`, `pdex_mwu`, or both; cross-method outputs require both.

Every pathway runner removes files beginning with its own output prefix from the requested root, `plots/`, and `tables/` directories before starting. This prevents stale conditional plots or tables from surviving when an existing dedicated output directory is reused; unrelated prefixes are untouched.

## Reproduce the implementation

### 1. Select retained programs

1. Read the program CSV and group rows by `cluster` in sorted order.
2. For every candidate row, split `genes` on whitespace and retain only genes present in `adata.var_names`.
3. Within a cluster, select the candidate with the largest number of present genes. Preserve first-row order on a tie.
4. Drop the cluster if fewer than `--min-genes` genes remain.
5. Label retained clusters as `C{cluster:03d}` and preserve the selected representative name and filtered gene list.

### 2. Compute the common cell-by-pathway score matrix

1. Read `adata.X`, or copy the requested layer into `X` because Bioconcord reads only `X`.
2. Reject missing matrices and non-finite values.
3. Call official `score_all_programs` once with the retained gene sets and `n_jobs=1` unless explicitly changed.
4. Preserve bioconcord scoring: 25 mean-expression bins, up to 50 same-bin background genes per pathway gene, and random state 42.
5. Extract `score_adata.obs[program_labels]` as a float64 matrix shaped `n_cells x n_programs`.

Guide-level scenario skills may select their documented powered guide/control subset before expression is loaded, normalize a raw-count subset to 10,000 counts per cell plus log1p when explicitly requested, and call `score_anndata` instead of `load_and_score`. This changes only the analyzed cell subset; program selection, matched-background scoring, random state, and downstream method contract remain identical.

Each score is:

```text
mean(log-normalized expression of pathway genes)
  - mean(log-normalized expression of matched background genes)
```

The result is signed floating-point data, not counts.

### 3. Run reference-coded bioconcord OLS

1. Build a temporary AnnData whose `obs` contains one score column per pathway and `pathway_group` labels.
2. Call `bioconcord.run_program_regression` with the requested control as `referenceLevel`.
3. Exclude the `const` row from test results. Interpret `const` as the control mean and every remaining coefficient as perturbation minus control.
4. For each perturbation-pathway pair, emit the coefficient and p-value. Independently compute `mean_difference = mean(score_perturbed) - mean(score_control)` as a check.
5. Reconstruct the signed t statistic from the two-sided p-value using residual degrees of freedom `n_cells - n_groups`; derive standard error as `abs(coefficient / statistic)` when finite and nonzero.
6. Apply BH correction to p-values separately within every perturbation.

The loader's float cast around Bioconcord's imported `add_constant` helper is a current pandas/statsmodels compatibility fix. It neither edits the official checkout nor changes the model.

### 4. Run pdex Mann-Whitney

1. Wrap the identical score matrix in AnnData with pathway names as variables and `pathway_group` in `obs`.
2. Call `pdex(..., groupby="pathway_group", mode="ref", reference=control, geometric_mean=False, is_log1p=False, threads=threads)`.
3. Test the per-cell scores of each perturbed group against the control-cell scores.
4. Compute the native reported effect from pdex's U statistic:

```text
rank_biserial = 2 * U / (n_target * n_reference) - 1
```

5. Retain pdex's p-value and within-perturbation FDR. Compute the shared descriptive mean difference as `target_mean - ref_mean`.

Keep `is_log1p=False`: it prevents an inappropriate score back-transformation. Signed float scores are valid Mann-Whitney input.

### 5. Use the canonical result schema

Emit one row per perturbation-pathway test with:

```text
method, target, program, effect, mean_difference, statistic,
standard_error, p_value, n_target, n_reference, fdr
```

`standard_error` is unavailable and therefore missing for pdex. Pivot with `index=target`, `columns=program`, and `values=effect` to obtain a perturbation-by-pathway effect matrix.

### 6. Compute overview comparisons

For every perturbation shared by both methods:

- Count pathways with `fdr <= --fdr` as `n_sig_ols` and `n_sig_pdex_mwu`.
- Compute called-set Jaccard as intersection size divided by union size; return 1 when both sets are empty.
- Compute descriptive Spearman across all common pathway effect values when at least three finite, nonconstant pairs exist.
- Compute direction agreement only over the union of pathways called by either method.
- Record `n_cells` from `n_target`.

Summarize perturbation covariates from `obs`: cell count, median UMI per cell, mean UMI per cell, and total UMI. Reject missing or non-finite UMI values.

### 7. Construct expression-versus-effect clouds

1. Sort tested perturbations by `(n_cells, target)` and select three distinct rows: first, integer-middle (`len // 2`), and last. Label them `least_cells`, `median_cells`, and `most_cells`.
2. Read the raw matrix from `--counts-layer` (`X` or empty selects `adata.X`) and reject negative or non-finite values.
3. For every selected perturbation and retained pathway, sum raw UMIs over all combinations of that perturbation's cells and the pathway's retained genes. Do not divide by cell count or pathway size.
4. Use `pathway_umi_sum + 1` as the log-scaled x coordinate.
5. Merge the common abundance coordinate with each method's native `effect`, `mean_difference`, `p_value`, and `fdr` by perturbation and pathway.
6. Mark a pathway/program as called only when its method-specific BH-adjusted `fdr <= --fdr`. Do not add an effect-size threshold.

## Required plots

Always produce:

1. **Effect heatmaps:** one panel per method using perturbations as rows and pathways as columns. Use `RdBu_r`, zero center, fixed `[-1, 1]` for rank-biserial effects, and observed symmetric limits for OLS. Include every perturbation in the matrix and explicitly set every y-axis label.
2. **Effects by cell count:** order perturbations by `n_cells`, draw one box-and-whisker distribution across pathway effects per perturbation, and overlay every pathway as a deterministically jittered dot.
3. **Effects by UMI count:** repeat the preceding display ordered by median UMI per cell.
4. **Expression-versus-effect clouds:** create one three-panel figure per method. Show least-, median-, and most-cell perturbations as columns; plot one pathway per point with total raw UMI summed across all perturbation cells and retained pathway genes plus 1 on a log x axis and native method effect on the y axis. Draw non-called pathways in gray and FDR-called pathways in red, and report both counts in each panel legend. Add a dashed zero-effect line. Share a symmetric y range across panels, fixed to `[-1, 1]` for pdex and observed maximum absolute effect for OLS.

When both methods run, also produce the significant-call scatter:

- x = pdex significant-pathway count; y = OLS count.
- Add a dashed identity line.
- Set marker area and marker color from perturbation cell count.
- Use only `cell count per perturbation` for the right-hand color legend.
- Label every point as `<target> J=<called-set Jaccard>` using alternating left/right leader arrows and deterministic vertical spacing to reduce overlap.
- Do not encode effect Spearman in this plot.

## Output contract

- `tables/pathways_overview_ols__<dataset>.csv`: all OLS tests.
- `tables/pathways_overview_pdex_mwu__<dataset>.csv`: all pdex tests.
- `tables/pathways_overview_comparison__<dataset>.csv`: per-perturbation call counts and agreement metrics; emit only when both methods run.
- `tables/pathways_overview_perturbation_covariates__<dataset>.csv`: cell and UMI summaries.
- `tables/pathways_overview_expression_effects__<dataset>.csv`: method, representative role, target, cell count, pathway, retained gene count, `pathway_umi_sum`, `pathway_umi_sum_plus_one`, count source, native effect, mean difference, p-value, FDR, and call indicator.
- `tables/pathways_legend__<dataset>.csv`: program label, representative name, and retained gene count.
- `plots/pathways_overview_effects__<dataset>.png`: native-effect heatmaps.
- `plots/pathways_overview_effects_by_cell_count__<dataset>.png`: boxes and pathway dots ordered by cells.
- `plots/pathways_overview_effects_by_umi_count__<dataset>.png`: boxes and pathway dots ordered by median UMI.
- `plots/pathways_overview_expression_effects_ols__<dataset>.png`: OLS raw-expression versus effect clouds for three representative perturbations.
- `plots/pathways_overview_expression_effects_pdex_mwu__<dataset>.png`: pdex raw-expression versus effect clouds for the same three perturbations.
- `plots/pathways_overview_method_comparison__<dataset>.png`: significant-count comparison; emit only for both methods.
- `pathways_run_metadata__<dataset>.json`: arguments, dimensions, score source/formula, method definitions, and FDR scope.

## Reproduction checks

- Require score shape `n_cells x n_programs` and effect-matrix shape `n_noncontrol_perturbations x n_programs`.
- Require the OLS coefficient and independently computed mean difference to agree up to numerical fitting error in this intercept-plus-dummy model.
- Require all eligible non-control perturbations in output tables and heatmap rows; visual tick thinning must never remove data.
- Require exactly three distinct representative perturbations and one point per retained pathway in every method/panel.
- Require both methods to use identical raw-abundance x coordinates; only native y effects and FDR highlighting may differ.
- Preserve native effect scales. Do not directly compare the magnitude of an OLS coefficient with rank-biserial correlation.
