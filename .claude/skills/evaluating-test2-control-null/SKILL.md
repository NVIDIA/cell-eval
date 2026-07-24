---
name: evaluating-test2-control-null
description: "Run ONLY Test 2 (control-control split null) from the DE metric-robustness battery and emit a single-test report. Splits control cells into A/B (a true null) and checks the DE null is calibrated: p-values Uniform[0,1], genomic-inflation lambda_GC ~= 1, frac_sig ~= alpha. Output includes a 3-panel diagnostic plot: QQ overlaid across splits | p-value histogram | lambda_GC boxplot. Use when someone wants to check null calibration / lambda_GC / uniformity of a DE metric without running the whole battery."
---

# Test 2 — Control-Control Split Null

**One test from the DE metric-robustness battery, run on its own.** This skill runs **only
`test_2`** and produces a self-contained report for it. Role: **validity gate (null calibration)** — Anti-conservative behaviour here means false positives downstream — a hard gate.

Runs **both pdex and pydeseq2** in one pass via `control_null_diagnostics.py`.

`de_backends.py` is bundled with this skill and calls the upstream `pdex` and
`pydeseq2` packages directly. Do not import project-private DE backend modules.

## Mandatory preflight and run capture

Do not start the executable until the user has explicitly confirmed one fully resolved run configuration.

1. Gather the input `.h5ad`, ask whether `adata.X` contains raw counts or log1p-normalized expression, results output directory, separate run root, methods to compare, non-parametric engine (`pdex` or `rsc`) when `pdex` is selected, required observation columns and control label, replicate/block columns, count or score layers, thresholds, seeds, repeats, and worker/thread settings. Inspect the input read-only to resolve unknown columns, labels, and layers. Pass the confirmed state as `--expression-state`.
2. Expand paths and resolve every default. Show one concise preflight summary containing the input, results directory, run root, methods/engine, data fields, thresholds, workload/concurrency, exact command, log path, and resolved-config destination.
3. Ask for explicit confirmation and stop. Do not launch computation or plotting before confirmation.
4. After confirmation, create `<run-root>/logs` and `<run-root>/configs`, pass `--run-root <run-root>`, and capture the complete terminal stream with `2>&1 | tee <run-root>/logs/<workflow>__<dataset>__<UTC-timestamp>.log`.
5. Every invocation writes an immutable timestamped YAML snapshot under `<run-root>/configs`. Report the result, log, and YAML paths on completion.

Every box-and-whisker plot must overlay every finite underlying observation as jittered scatter points. Do not sample, aggregate away, or hide values in the scatter layer.

Keep `pdex` as the stable internal/table schema key, but label every plot with the actual selected engine: `pdex` for Arc pdex and `RSC` for RAPIDS GPU Wilcoxon. Never display an RSC result as pdex.

## What it asks
When you split **control cells** into a pseudo-perturbation vs a reference (same population, so a true null), does the pipeline correctly find nothing? A calibrated null gives Uniform[0,1] p-values (lambda_GC ~= 1, frac_sig ~= alpha); inflation means false positives, deflation means it is under-powered.

## How DE is computed
Control-only A/B stratified splits, both backends on the same splits for a direct comparison. Needs: `control_pert`.

## Run it (standalone)

```bash
RUN_DIR="experiments_all/$(basename $H5AD .h5ad)__test_2__$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
python .claude/skills/evaluating-test2-control-null/control_null_diagnostics.py \
  --adata <h5ad> --methods pdex,pydeseq2 \
  --pert-col <pert_col> --control non-targeting \
  --replicate-col <batch_col> --block-cols <batch_col> \
  --n-resamples 10 --outdir "$RUN_DIR"
```

## Parameters

For exact CPU PyDESeq2 scheduling, use `--threads 1 --resample-workers N` to run independent null
splits concurrently. This changes scheduling only; each split still uses the same PyDESeq2 model.
| flag | what it controls | default |
|---|---|---|
| `--n-resamples` | number of independent control A/B splits | 10 |
| `--block-cols` | covariate the split is balanced within | batch |
| `--fdr` | expected frac_sig under a true null (alpha) | 0.05 |
| `--seed` | RNG seed for every split | 0 |
| `--methods` | comma-sep backends (run both for comparison) | pdex,pydeseq2 |
| `--non-parametric-engine` | `pdex` or numerically matched RAPIDS GPU Wilcoxon (`rsc`) | pdex |

## Outputs (in `--outdir`)
- `plots/test_2_pvalue_diagnostics__<dataset>.png` — 3-panel: QQ per method | p-value histogram | λ_GC boxplot
- `plots/test_2_lfc_agreement__<dataset>.png` — per-gene LFC scatter on the null (pdex vs pydeseq2)
- `tables/test2_lfc_vectors_<method>__<dataset>.parquet` — split/feature LFC and FDR vectors,
  tagged with method and execution engine

## Null-calibration comparison — `control_null_diagnostics.py`

A standalone diagnostic that runs the control-control null for **both backends on the same A/B splits**
(same seeds as the runner, `seed + 100 + r`) and renders **one 3-panel figure comparing pdex vs
pydeseq2** — so you can see, side by side, whether each backend is calibrated / inflated / deflated on a
true null. It reuses the shared runner's exact Test-2 helpers (`stratified_split` / `run_de` /
`null_metrics` / `lambda_gc` / `maybe_normalize`), so the null matches the real test.

```bash
python .claude/skills/evaluating-test2-control-null/control_null_diagnostics.py \
  --adata <h5ad> --methods pdex,pydeseq2 \
  --pert-col gene --control non-targeting --replicate-col batch --block-cols batch \
  --n-resamples 10 --outdir <out>
```

Produces `plots/test_2_pvalue_diagnostics__<dataset>.png` (14×5 in, 300 dpi; the filename is suffixed
with the input `.h5ad` basename so runs on different datasets never overwrite each other):
- **Panel 1 — QQ**: expected vs observed −log₁₀(p), one curve per method (colorblind blue = pdex,
  orange = pydeseq2), grey `y=x` diagonal, 95% beta null envelope; each curve labelled with its mean
  λ_GC. **Above the diagonal ⇒ anti-conservative (inflated); below ⇒ conservative.**
- **Panel 2 — p-value histogram**: 20 bins, density (Uniform = flat at 1.0), two semi-transparent
  overlaid histograms, dashed line at density 1.0. Left spike ⇒ inflated; right skew ⇒ conservative.
- **Panel 3 — λ_GC stability**: boxplot + per-split strip of λ_GC per method; dashed red line at 1.0,
  dotted ±0.1 band at 0.9 / 1.1.

It also writes `plots/test_2_lfc_agreement__<dataset>.png` — a per-gene **LFC-agreement scatter** on the
null: **x = pdex log2FC, y = pydeseq2 log2FC**, one point per gene per split (both backends share the
same A/B split for each seed, so points are paired), colored by DE-call category (grey = DE in neither ·
blue = pdex only · orange = pydeseq2 only · green = both), with `y=x` and, in the title, the Spearman ρ
(all genes and DE-in-either) **and the DE-calling threshold used** (`DE call = FDR ≤ <fdr> AND |log2FC| ≥
<lfc>`, from the `--fdr`/`--lfc` args). Since this is a true null, the cloud should sit at the origin and **any
off-origin colored point is a false positive** — the plot shows whether the two backends' false
positives coincide (points on the diagonal) or are independent noise (points on the axes). On a
well-calibrated null both DE counts are ≈ 0 and the two backends' LFC estimates agree tightly (e.g. ρ ≈
0.92 on the K562 essential subset, with 0 DE calls from either backend).

The reusable `plot_test2_lfc_agreement(de_pdex, de_pydeseq2, output_path, fdr, lfc)` takes the per-gene
per-split `de_long` frames (the 3rd return value of `control_null_pvalues`).

The reusable `plot_test2_diagnostics(pvalues_pdex, pvalues_pydeseq2, lambda_gc_per_split_pdex,
lambda_gc_per_split_pydeseq2, alpha, n_resamples, output_path)` takes already-pooled arrays (genes ×
splits p-values + one λ_GC per split, both returned by `control_null_pvalues`), so it can be called
directly on p-values from any source. **Key message it makes visible:** the two backends usually miss
calibration in *opposite* directions — cell-level pdex tends toward anti-conservative (λ_GC ≳ 1,
curves up) while pseudobulk pydeseq2 tends toward conservative/deflated (λ_GC < 1, curves down).
