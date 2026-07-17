---
name: evaluating-test4-guide-reproducibility
description: Run ONLY Test 4 (same-sgRNA split reproducibility) from the cell-eval metric-robustness battery and emit a single-test report. This is Test 1 run at the sgRNA guide level, splitting each guide's cells into halves A and B versus split control and measuring split-half agreement with Spearman, Pearson, DEG Jaccard, and direction. Use when someone wants the per-guide reproducibility ceiling and cross-guide specificity without running the whole battery. Needs an sgRNA column.
---

# Test 4 — Same-sgRNA Split Reproducibility (guide-level Test 1)

**One test from the cell-eval metric-robustness battery, run on its own.** This skill runs **only
`test_4`** and produces a self-contained report for it. Role: **sensitivity diagnostic (per-guide reproducibility ceiling)** — The empirical reproducibility ceiling at the resolution per-guide downstream metrics operate.

Runs **both pdex and pydeseq2** on the same splits in one pass via `guide_split_reproducibility.py`.

## What it asks
Same as Test 1 but the unit is a single **guide**: split each guide's cells in half (controls also split) and ask whether the two DE signatures agree. This is the empirical ceiling for any per-guide downstream metric, and its rho-vs-cell-count separates undersampling from a genuine method limitation.

## How DE is computed
Cell-eval DE per guide-half vs split control; both backends on the same A/B splits for an
apples-to-apples comparison. Needs: `--sgrna-col` (+ `--pert-col` / `--control`).

## Run it (standalone)

```bash
RUN_DIR="experiments_all/$(basename $H5AD .h5ad)__test_4__$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
python .claude/skills/evaluating-test4-guide-reproducibility/guide_split_reproducibility.py \
  --adata <h5ad> --methods pdex,pydeseq2 \
  --pert-col <pert_col> --control non-targeting \
  --sgrna-col <sgrna_col> --target-gene-col <pert_col> \
  --replicate-col <batch_col> --block-cols <batch_col> \
  --max-guides 24 --max-control 1200 --zoom-per-page 1 \
  --outdir "$RUN_DIR"
```

## Parameters
| flag | what it controls | default |
|---|---|---|
| `--sgrna-col` | obs column holding the guide (the unit of this test) | required |
| `--min-cells` | min cells per arm; a guide needs ≥2x to be included | 20 |
| `--max-guides` | cap #guides (most-powered first) for readable heatmap | None |
| `--max-control` | subsample control cells (speed) | None |
| `--fdr / --lfc` | DEG-calling cutoffs | 0.05 / 0.1 |
| `--seed` | RNG seed | 0 |

## Guide-level reproducibility visualisations — `guide_split_reproducibility.py`

The same heatmap / zoom / correlation views as
[`evaluating-test1-reproducibility`](../evaluating-test1-reproducibility/) — **at the guide level**. It
**reuses Test 1's plotting layers directly** (imports `reproducibility_heatmap.py` and calls them with
`unit='guide'` / `title_prefix='Test-4'`, so there is no forked copy), and the split-half DE uses the
shared runner's exact helpers (`stratified_split` / `_de_two` / `maybe_normalize`). The unit is an
individual sgRNA (labeled `GENE.gN`): each guide's cells are split into halves A/B (control also split),
`DE_A = guide_A vs ctrl_A` and `DE_B = guide_B vs ctrl_B`, and the split-half Spearman LFC ρ measures
within-guide reproducibility. It computes **both backends on the same A/B splits**. Every output filename
is suffixed with the input `.h5ad` basename.

```bash
python .claude/skills/evaluating-test4-guide-reproducibility/guide_split_reproducibility.py \
  --adata <h5ad> --methods pdex,pydeseq2 \
  --pert-col gene --control non-targeting --sgrna-col <guide_col> --target-gene-col gene \
  --replicate-col <batch/gem_group> --block-cols <batch/gem_group> \
  --max-guides 24 --max-control 1200 --zoom-per-page 1 --outdir <out>
```

- **Layer 1 — `test4_guide_heatmap_<method>__<dataset>.png`** (one per backend): rows = guides (`GENE.gN`)
  sorted by split-half ρ (low ρ at top), columns = union DE genes sorted by split-A mean LFC, two panels
  **split A | split B**. Diverging blue–white–red, capped ±2 log2FC.
- **Layer 2 — `test4_zoom__<dataset>_NN.png`** (one PNG per guide, sorted by ρ): **one row per method**
  (pdex row + pydeseq2 row), each = LFC_A-vs-LFC_B scatter + that method's split-A/B heatmap strips.
- **Layer 3 — `test4_corr_matrix__<dataset>.png`** (one panel per backend): split-A × split-B **guide**
  correlation matrix — **diagonal = within-guide A/B reproducibility** (bright = reproducible),
  off-diagonal = cross-guide. **Each diagonal cell is labelled with that guide's cell count** (n cells
  before the A/B split), so you can read reproducibility against power at a glance. Every panel title
  reports finite diagonal and off-diagonal means for both Spearman and Pearson.
- `test4_rho_<method>__<dataset>.csv` — per-guide split-half ρ.

**This differs from `evaluating-test5-samegene-sgrna`'s `samegene_guide_heatmap.py`:** Test 4 splits ONE
guide's cells in half (within-guide reproducibility ceiling); Test 5 compares DIFFERENT guides of the
same gene (does the metric reward shared biology). Levers: `--max-guides` (most-powered first) and
`--max-control` bound runtime; `--methods pdex` skips the slower backend.
