---
name: evaluating-test3-permutation-null
description: Run ONLY Test 3 (label permutation null) from the cell-eval metric-robustness battery and emit a single-test report. Shuffles perturbation labels within block strata, recomputes DE, and scores the separation (z) of the real signal from the permuted null, plus a cell-count-stratified real-vs-shuffled p-value diagnostic (ECDF/QQ). Use when someone wants a permutation/label-shuffle null or signal-vs-noise separation check without running the whole battery.
type: skill
---

# Test 3 — Label Permutation Null

**One test from the cell-eval metric-robustness battery, run on its own.** This skill runs **only
`test_3`** and produces a self-contained report for it. Role: **validity gate (signal vs noise)** — If real signal is not clearly outside the permuted null, the metric cannot tell signal from noise.

Runs **both pdex and pydeseq2** in one pass via `shuffle_de_comparison.py`, which compares
per-perturbation DE gene counts on a perturbed-vs-perturbed null. Uses `--config config.yaml`
(see `config.example.yaml`).

## What it asks
If the perturbation labels are randomly **shuffled** (within batch) and DE is run on fake-pert X vs
random fake-pert Y, does the method correctly call ~0 DE genes? Generates two variants: global
shuffle (across all batches) and within-batch shuffle.

## How DE is computed
Shuffled-label fake-pert X vs random fake-pert Y; both backends on the same shuffled null for
a direct comparison. Needs: `pert_col` + `control_pert` + `block_cols`.

## Run it (standalone)

```bash
RUN_DIR="experiments_all/$(basename $H5AD .h5ad)__test_3__$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
cp .claude/skills/evaluating-test3-permutation-null/config.example.yaml "$RUN_DIR/config.yaml"
# Edit config.yaml: adata_path, pert_col, control_pert, replicate_col, block_cols
python .claude/skills/evaluating-test3-permutation-null/shuffle_de_comparison.py \
  --config "$RUN_DIR/config.yaml" --outdir "$RUN_DIR"
```

## Parameters (in `config.yaml`)
| key | what it controls | default |
|---|---|---|
| `adata_path` | path to the h5ad | required |
| `pert_col` | obs column with perturbation labels | required |
| `control_pert` | control label in pert_col | required |
| `replicate_col` | pseudobulk unit for pydeseq2 | batch |
| `block_cols` | columns used for within-batch shuffle variant | [batch] |
| `fdr_threshold / lfc_threshold` | DEG-calling cutoffs | 0.05 / 0.1 |
| `seed` | RNG seed | 0 |

## Outputs (in `--outdir/plots/`)
- `test_3_shuffle_de_comparison__global.png` — scatter: n_sig_pydeseq2 vs n_sig_pdex, global shuffle
- `test_3_shuffle_de_comparison__within.png` — same, within-batch shuffle
