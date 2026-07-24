---
name: evaluating-test3-permutation-null
description: Run ONLY Test 3 (label permutation null) from the DE metric-robustness battery and emit a single-test report. Shuffles perturbation labels within block strata, recomputes DE, and scores the separation (z) of the real signal from the permuted null, plus a cell-count-stratified real-vs-shuffled p-value diagnostic (ECDF/QQ). Use when someone wants a permutation/label-shuffle null or signal-vs-noise separation check without running the whole battery.
---

# Test 3 — Label Permutation Null

**One test from the DE metric-robustness battery, run on its own.** This skill runs **only
`test_3`** and produces a self-contained report for it. Role: **validity gate (signal vs noise)** — If real signal is not clearly outside the permuted null, the metric cannot tell signal from noise.

Runs **both pdex and pydeseq2** in one pass via `shuffle_de_comparison.py`, which compares
per-perturbation DE gene counts on a perturbed-vs-perturbed null. Uses `--config config.yaml`
(see `config.example.yaml`).

`de_backends.py` is bundled with this skill and calls the upstream `pdex` and
`pydeseq2` packages directly. Do not import project-private DE backend modules.

## Mandatory preflight and run capture

Do not start the executable until the user has explicitly confirmed one fully resolved run configuration.

1. Gather the input `.h5ad`, ask whether `adata.X` contains raw counts or log1p-normalized expression, results output directory, separate run root, methods to compare, non-parametric engine (`pdex` or `rsc`) when `pdex` is selected, required observation columns and control label, replicate/block columns, count or score layers, thresholds, seeds, shuffle modes, repeats, and worker/thread settings. Inspect the input read-only to resolve unknown columns, labels, and layers. Pass the confirmed state as `--expression-state`.
2. Expand paths and resolve every CLI and YAML default. Show one concise preflight summary containing the input, results directory, run root, methods/engine, data fields, thresholds, workload/concurrency, exact command, log path, cache/replot behavior, and resolved-config destination.
3. Ask for explicit confirmation and stop. Do not launch computation, replotting, or archive reuse before confirmation.
4. After confirmation, create `<run-root>/logs` and `<run-root>/configs`, pass `--run-root <run-root>`, and capture the complete terminal stream with `2>&1 | tee <run-root>/logs/<workflow>__<dataset>__<UTC-timestamp>.log`.
5. Every invocation writes an immutable timestamped YAML snapshot under `<run-root>/configs`, including replot runs. Report the result, log, and YAML paths on completion.

Every box-and-whisker plot must overlay every finite underlying observation as jittered scatter points. Do not sample, aggregate away, or hide values in the scatter layer.

Keep `pdex` as the stable internal/table schema key, but label every plot with the actual selected engine: `pdex` for Arc pdex and `RSC` for RAPIDS GPU Wilcoxon. Never display an RSC result as pdex.

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

`--comparison-workers N --n-threads 1` runs independent shuffled comparisons concurrently for CPU
PyDESeq2. Do not combine comparison workers with the RSC non-parametric engine.
| key | what it controls | default |
|---|---|---|
| `adata_path` | path to the h5ad | required |
| `pert_col` | obs column with perturbation labels | required |
| `control_pert` | control label in pert_col | required |
| `replicate_col` | pseudobulk unit for pydeseq2 | batch |
| `block_cols` | columns used for within-batch shuffle variant | [batch] |
| `fdr_threshold / lfc_threshold` | DEG-calling cutoffs | 0.05 / 0.1 |
| `seed` | RNG seed | 0 |
| `non_parametric_engine` / `--non-parametric-engine` | `pdex` for Arc pdex or `rsc` for RAPIDS GPU Wilcoxon | pdex |

## Outputs (in `--outdir/plots/`)
- `test_3_shuffle_de_comparison__global.png` — scatter: n_sig_pydeseq2 vs n_sig_pdex, global shuffle
- `test_3_shuffle_de_comparison__within.png` — same, within-batch shuffle
- `test_3_corr_matrix__<mode>__<method>.png` — fake-perturbation Spearman correlation map; every
  title reports finite mean diagonal and off-diagonal rho separately.
- `test3_lfc_vectors_<mode>_<method>.parquet` — long-form shuffled-comparison LFC vectors with one
  row per feature and execution-engine provenance. Version-5
  `test_3_lfc_matrix__<mode>.npz` archives record `non_parametric_engine=pdex|rsc` for fast
  correlation-matrix replotting; reject and recompute older `cpu`-labeled archives.
