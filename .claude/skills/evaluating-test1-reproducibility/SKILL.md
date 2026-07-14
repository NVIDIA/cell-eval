---
name: evaluating-test1-reproducibility
description: Run ONLY Test 1 (within-condition reproducibility) from the cell-eval metric-robustness battery and emit a single-test report. Splits each perturbation's cells into halves A/B (controls also split) and asks whether the two independent DE signatures agree (split-half Spearman LFC rho, DEG Jaccard, direction), plus rho-vs-cell-count and a difference-is-null QQ. Use when someone wants the reproducibility ceiling of a perturbation DE metric without running the whole battery.
type: skill
---

# Test 1 — Within-Condition Reproducibility

**One test from the cell-eval metric-robustness battery, run on its own.** This skill runs **only
`test_1`** and produces a self-contained report for it. Role: **validity gate (reproducibility ceiling)** — Sets the empirical ceiling: no model can score higher on a perturbation than the data agrees with itself.

Runs **both pdex and pydeseq2** on the same splits in one pass via `reproducibility_heatmap.py`.

## What it asks
If a perturbation's cells are split in half and each half is run against control independently, do the two DE signatures agree (DE_A ~= DE_B)? Disagreement caps how well any model can score on that perturbation. Also: does agreement depend on cell count (undersampling vs a genuine method limitation)?

## How DE is computed
Cell-eval DE per half vs split control; both backends on the same A/B splits for a direct comparison. Needs: `--pert-col` + `--control`.

## Run it (standalone)

```bash
RUN_DIR="experiments_all/$(basename $H5AD .h5ad)__test_1__$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
python .claude/skills/evaluating-test1-reproducibility/reproducibility_heatmap.py \
  --adata <h5ad> --methods pdex,pydeseq2 \
  --pert-col <pert_col> --control non-targeting \
  --replicate-col <batch_col> --block-cols <batch_col> \
  --zoom-per-page 1 --max-genes 400 \
  --outdir "$RUN_DIR"
```

## Parameters
| flag | what it controls | default |
|---|---|---|
| `--min-cells` | min cells per arm; a perturbation needs ≥2x to be split | 20 |
| `--block-cols` | covariate the A/B split is balanced within | batch |
| `--fdr / --lfc` | DEG-calling cutoffs | 0.05 / 0.1 |
| `--max-genes` | cap heatmap columns for readability | 400 |
| `--zoom-per-page` | perturbations per zoom PNG | 1 |
| `--seed` | RNG seed for every split | 0 |

## Reproducibility visualisations — `reproducibility_heatmap.py`

A standalone diagnostic (reuses the shared runner's exact Test-1 helpers `stratified_split` / `_de_two`
/ `maybe_normalize`, so the split-half DE matches the real test). It runs one A/B split per
perturbation (controls also split; **one split**, not the multi-draw median — the ρ shown is that
split's) and renders three layers. It computes **both backends on the same splits** so they compare
apples-to-apples.

```bash
python .claude/skills/evaluating-test1-reproducibility/reproducibility_heatmap.py \
  --adata <h5ad> --methods pdex,pydeseq2 \
  --pert-col gene --control non-targeting --replicate-col batch --block-cols batch \
  --zoom-per-page 1 --max-genes 400 --outdir <out>
```

Every output filename ends with `__<dataset>` (the input `.h5ad` basename) so runs on different
datasets never overwrite each other.

- **Layer 1 — `test1_heatmap_<method>__<dataset>.png`** (one per backend): rows = perturbations sorted
  by split-half ρ (low ρ at top), columns = union DE genes sorted by split-A mean LFC, two panels
  **split A | split B**. Diverging blue–white–red, centred 0, capped ±2 log2FC (white = 0).
- **Layer 2 — `test1_zoom__<dataset>_NN.png`** (one PNG per perturbation, sorted by ρ): **one row per
  method** (e.g. a **pdex row and a pydeseq2 row**), each = LFC_A-vs-LFC_B scatter (grey = non-DE, dark
  blue = DE in A∪B) + that method's split-A/B heatmap strips over a shared gene order.
  `--zoom-per-page N` puts N perturbations per PNG.
- **Layer 3 — `test1_corr_matrix__<dataset>.png`** (one panel per backend): split-A × split-B signature
  correlation matrix — entry (i,j) = Spearman(pert i split-A, pert j split-B) over the union DE genes.
  The **diagonal is within-perturbation A/B reproducibility** (bright = reproducible); off-diagonal =
  cross-perturbation (should be dim if signatures are perturbation-specific). **Each diagonal cell is
  labelled with that perturbation's cell count**, so reproducibility can be read against power.
- `test1_rho_<method>__<dataset>.csv` — per-perturbation split-half ρ.

Colour spec: diverging blue–white–red centred at 0, capped ±2 log2FC — genes near 0 in both splits
appear white, so colour in split B where split A is white (or vice versa) is irreproducibility made
visible.
