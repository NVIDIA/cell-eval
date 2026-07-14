---
name: evaluating-test2-control-null
description: Run ONLY Test 2 (control-control split null) from the cell-eval metric-robustness battery and emit a single-test report. Splits control cells into A/B (a true null) and checks the DE null is calibrated: p-values Uniform[0,1], genomic-inflation lambda_GC ~= 1, frac_sig ~= alpha. Output includes a 3-panel diagnostic plot: QQ overlaid across splits | p-value histogram | lambda_GC boxplot. Use when someone wants to check null calibration / lambda_GC / uniformity of a DE metric without running the whole battery.
type: skill
---

# Test 2 — Control-Control Split Null

**One test from the cell-eval metric-robustness battery, run on its own.** This skill runs **only
`test_2`** and produces a self-contained report for it. Role: **validity gate (null calibration)** — Anti-conservative behaviour here means false positives downstream — a hard gate.

Runs **both pdex and pydeseq2** in one pass via `control_null_diagnostics.py`.

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
| flag | what it controls | default |
|---|---|---|
| `--n-resamples` | number of independent control A/B splits | 10 |
| `--block-cols` | covariate the split is balanced within | batch |
| `--fdr` | expected frac_sig under a true null (alpha) | 0.05 |
| `--seed` | RNG seed for every split | 0 |
| `--methods` | comma-sep backends (run both for comparison) | pdex,pydeseq2 |

## Outputs (in `--outdir`)
- `plots/test_2_pvalue_diagnostics__<dataset>.png` — 3-panel: QQ per method | p-value histogram | λ_GC boxplot
- `plots/test_2_lfc_agreement__<dataset>.png` — per-gene LFC scatter on the null (pdex vs pydeseq2)

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
0.92 on cell_eval2, with 0 DE calls from either backend).

The reusable `plot_test2_lfc_agreement(de_pdex, de_pydeseq2, output_path, fdr, lfc)` takes the per-gene
per-split `de_long` frames (the 3rd return value of `control_null_pvalues`).

The reusable `plot_test2_diagnostics(pvalues_pdex, pvalues_pydeseq2, lambda_gc_per_split_pdex,
lambda_gc_per_split_pydeseq2, alpha, n_resamples, output_path)` takes already-pooled arrays (genes ×
splits p-values + one λ_GC per split, both returned by `control_null_pvalues`), so it can be called
directly on p-values from any source. **Key message it makes visible:** the two backends usually miss
calibration in *opposite* directions — cell-level pdex tends toward anti-conservative (λ_GC ≳ 1,
curves up) while pseudobulk pydeseq2 tends toward conservative/deflated (λ_GC < 1, curves down).
