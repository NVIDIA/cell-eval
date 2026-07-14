---
name: evaluating-test01-overview
description: Generate a scatter plot comparing DE gene counts per perturbation between pyDESeq2 and pdex. x = pdex n_sig, y = pyDESeq2 n_sig, one point per perturbation, sized by cell count, diagonal = equal calling. Saved to plots/test01_overview.png. Use when you want a quick method-comparison overview of how many genes each backend calls per perturbation.
type: skill
---

# DE Gene Count Overview: pyDESeq2 vs pdex

**Standalone scatter plot — runs both DE backends on the full dataset and plots n_sig per perturbation against each other.**

## What it shows

Scatter plot with:
- **x-axis**: # significant DE genes per perturbation — pdex (cell-level Wilcoxon)
- **y-axis**: # significant DE genes per perturbation — pyDESeq2 (pseudobulk DESeq2 Wald)
- **dot size**: cell count for that perturbation
- **colour**: cell count (viridis scale)
- **dashed diagonal**: equal calling — points above = pyDESeq2 more liberal; below = pdex more liberal
- **labelled**: every perturbation is named, with a **leader arrow** to its dot and its **`J=` Jaccard**
  = |DE_pdex ∩ DE_pyDESeq2| / |DE_pdex ∪ DE_pyDESeq2| (overlap of the two backends' significant gene
  sets for that perturbation; 1 = identical DE sets, 0 = disjoint)

Significance threshold: **FDR < `fdr_threshold`** (default 0.05) **AND** |LFC| ≥ `lfc_threshold` (default 0.1).

## Run it

```bash
H5AD=<path/to/adata.h5ad>
RUN_DIR="experiments/$(basename "$H5AD" .h5ad)__overview__$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
cp .claude/skills/evaluating-test01-overview/overview.py "$RUN_DIR/"
cp .claude/skills/evaluating-test01-overview/config.example.yaml "$RUN_DIR/config.yaml"
# Edit config.yaml: adata_path, pert_col, control_pert, replicate_col
cd "$RUN_DIR" && uv run python overview.py --config config.yaml
# Re-render without rerunning DE:
uv run python overview.py --config config.yaml --plot-only
```

## Parameters

| parameter | what it controls | default |
|-----------|-----------------|---------|
| `fdr_threshold` | FDR cutoff to call a gene significant | 0.05 |
| `lfc_threshold` | \|LFC\| cutoff to call a gene significant | 0.1 |
| `replicate_col` | pseudobulk unit column (required for pydeseq2) | batch |
| `min_cells_per_group` | minimum cells per group | 20 |
| `num_threads` | DE computation threads | 8 |
| `--plot-only` | skip DE, load cached tables and re-render PNG | — |

## Outputs

Every output filename ends with `__<dataset>` (the input `.h5ad` basename) so runs on different
datasets never overwrite each other.

| file | description |
|------|-------------|
| `plots/test01_overview__<dataset>.png` | scatter plot (n_sig per perturbation: pyDESeq2 vs pdex) |
| `plots/test01_corr_matrix__<dataset>.png` | **perturbation × perturbation Spearman-LFC correlation, one panel per backend** (pdex \| pyDESeq2), over the union DE genes. Diagonal = 1 (self); **off-diagonal = cross-perturbation signature similarity** — dim ⇒ perturbation-specific, bright ⇒ a shared program (less specific). |
| `plots/test01_ma_pydeseq2__<dataset>.png` | **MA scatter for pyDESeq2**: 1×3 panel showing mean raw count (x) vs log2 LFC (y) for least / median / most-cell perturbations. DE genes (FDR < threshold, \|LFC\| ≥ threshold) highlighted in red. |
| `plots/test01_ma_pdex__<dataset>.png` | **MA scatter for pdex**: same 1×3 layout, x = mean raw count (same scale as pyDESeq2 for direct comparison), y = log2 LFC. |
| `tables/overview_pydeseq2_full__<dataset>.csv` | pydeseq2 DE (all genes × all perturbations) |
| `tables/overview_pdex_full__<dataset>.csv` | pdex DE (all genes × all perturbations) |
| `tables/overview_cell_counts__<dataset>.csv` | cell count per perturbation |
| `tables/overview_jaccard__<dataset>.csv` | per-perturbation n_sig (pdex, pyDESeq2) + Jaccard of their DE-gene sets |
| `tables/overview_ma_pydeseq2__<dataset>.csv` | gene-level table: feature, perturbation, mean_expr (raw count), log2_fold_change, fdr — for the 3 representative perturbations (pyDESeq2) |
| `tables/overview_ma_pdex__<dataset>.csv` | same as above for pdex |

## Interpretation

- **Points on diagonal**: both methods agree on the number of DE genes.
- **Points above diagonal** (pyDESeq2 >> pdex): pseudobulk inflating calls — often low-cell-count perturbations where few cells spread across many batches collapse variance. Check with Test 2 (control-null).
- **Points below diagonal** (pdex >> pyDESeq2): cell-level Wilcoxon more sensitive — expected for perturbations with high per-cell variability; may reflect pseudoreplication (Squair et al. 2021).
- **Small dots far from diagonal**: low cell count is driving the disagreement — interpret cautiously.

### MA scatter interpretation

Both methods use **mean raw count** on the x-axis so the plots are directly comparable.

- **Trumpet / funnel shape**: DE genes concentrate at low expression — the "noise floor" region where small absolute changes produce large LFC.
- **pyDESeq2 inflation at low cell count**: The least-cells panel often shows far more red points than pdex for the same perturbation, revealing that DESeq2's Wald test becomes anti-conservative with few pseudobulk replicates.
- **pdex spread across the x-axis**: Cell-level Wilcoxon calls DE genes at all expression levels, not just the low end — more balanced but more susceptible to pseudoreplication at high cell counts.
- **Representative perturbations**: Least / median / most cells are chosen from the intersection of both DE frames (control excluded), sorted ascending by cell count.
