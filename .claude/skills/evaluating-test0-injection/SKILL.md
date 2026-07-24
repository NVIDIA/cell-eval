---
name: evaluating-test0-injection
description: Run ONLY Test 0 (controlled effect-size injection / calibration curve) from the DE metric-robustness battery and emit a single-test report. Spikes a known log2FC into ~a dozen genes in a control-vs-control split at several delta tiers (incl. delta=0) and measures the null false-positive rate and the TPR-vs-effect-size curve (smallest resolvable delta), plus compositional-coupling of untouched genes. Use when someone wants to calibrate a DE metric's false-positive rate / resolving power, or check injection/FPR/TPR calibration, without running the whole battery.
---

# Test 0 — Controlled Effect-Size Injection / Calibration Curve

**One test from the DE metric-robustness battery, run on its own.** This skill runs **only
`test_0`** and produces a self-contained report for it. Role: **validity gate (calibration + power)** — A FAIL here means the pipeline mis-calls untouched genes — do not interpret any biology until it is fixed.

Runs **both pdex and pydeseq2** in one pass via `inject_and_count_de.py`. Both datasets
(k562_essential_sub + adata_Validation) can be run into the same output directory — output filenames are
suffixed with the `.h5ad` basename so they never collide.

`de_backends.py` is bundled with this skill and calls the upstream `pdex` and
`pydeseq2` packages directly. Do not import project-private DE backend modules.

## Mandatory preflight and run capture

Do not start the executable until the user has explicitly confirmed one fully resolved run configuration.

1. Gather the input `.h5ad`, ask whether `adata.X` contains raw counts or log1p-normalized expression, results output directory, separate run root, methods to compare, non-parametric engine (`pdex` or `rsc`) when `pdex` is selected, required observation columns and control label, replicate/block columns, count or score layers, thresholds, seeds, repeats, and worker/thread settings. Inspect the input read-only to resolve unknown columns, labels, and layers. Pass the confirmed state as `--expression-state`.
2. Expand paths and resolve every default. Show one concise preflight summary containing the input, results directory, run root, methods/engine, data fields, thresholds, workload/concurrency, exact command, log path, and resolved-config destination.
3. Ask for explicit confirmation and stop. Do not launch computation, plotting, replotting, or cache reuse before confirmation.
4. After confirmation, create `<run-root>/logs` and `<run-root>/configs`, pass `--run-root <run-root>`, and capture the complete terminal stream with `2>&1 | tee <run-root>/logs/<workflow>__<dataset>__<UTC-timestamp>.log`.
5. Every invocation writes an immutable timestamped YAML snapshot under `<run-root>/configs`, including plot-only or cache-driven runs. Report the result, log, and YAML paths on completion.

Every box-and-whisker plot must overlay every finite underlying observation as jittered scatter points. Do not sample, aggregate away, or hide values in the scatter layer.

Keep `pdex` as the stable internal/table schema key, but label every plot with the actual selected engine: `pdex` for Arc pdex and `RSC` for RAPIDS GPU Wilcoxon. Never display an RSC result as pdex.

## What it asks
If we inject a **known** log2FC into a small number of genes (~a dozen, the scale of a real perturbation) in a **control-vs-control** split, what false-positive rate does the metric produce at the null (delta=0), and what is the smallest effect size it can resolve (TPR vs delta)? Does the FPR among *untouched* genes climb with delta (compositional coupling from library-size renormalization)?

## How DE is computed
Injects on **raw counts** (before normalization); runs both backends (cell-level Wilcoxon pdex and
pseudobulk pydeseq2 Wald) on the same injection splits for a direct comparison. Needs: raw `.X`
or a `counts` layer.

## Run it (standalone)

```bash
RUN_DIR="experiments_all/test0_injection__$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
SCRIPT=".claude/skills/evaluating-test0-injection/inject_and_count_de.py"

# K562 essential subset — 100 draws
python "$SCRIPT" \
  --adata /path/to/k562_essential_sub.h5ad \
  --pert-col gene --control non-targeting --replicate-col batch \
  --counts-layer counts --n-repeats 100 --out "$RUN_DIR/injection_recovery.csv"

# adata_Validation — 20 draws (no counts layer, uses .X)
python "$SCRIPT" \
  --adata /path/to/adata_Validation.h5ad \
  --pert-col target_gene --control non-targeting --replicate-col batch \
  --n-repeats 20 --out "$RUN_DIR/injection_recovery_validation.csv"
```

## Parameters

`--pydeseq-workers N --threads 1` parallelizes independent injection tasks for CPU PyDESeq2. The
injection implementation keeps the count matrix sparse, including while modifying anchor columns.
| flag | what it controls | default |
|---|---|---|
| `--deltas` | log2FC tiers injected (comma-sep; 0 = null FPR baseline) | `0,0.5,1,2` |
| `--n-genes` | anchor genes injected per draw | 12 |
| `--n-repeats` | draws pooled for stable rates (K562 subset uses 100, adata_Validation 20) | 100 |
| `--fdr / --lfc` | DEG-calling cutoffs | 0.05 / 0.1 |
| `--replicate-col` | pseudobulk unit (required for pydeseq2 within each repeat) | batch |
| `--counts-layer` | raw-counts layer; falls back to `.X` if absent | auto |
| `--non-parametric-engine` | `pdex` or numerically matched RAPIDS GPU Wilcoxon (`rsc`) | pdex |
| `--methods` | comma-separated backends; useful for engine-parity runs | pdex,pydeseq2 |

## Outputs (in `$RUN_DIR`)
- `injection_recovery__<dataset>__recovered_box.png` — # anchors recovered (TPR) vs δ, pdex vs pydeseq2
- `injection_recovery__<dataset>__false_pos_box.png` — # false-positive DE genes vs δ, pdex vs pydeseq2
- `injection_recovery__<dataset>.csv` — pooled mean±SD per (δ, method)
- `injection_recovery__<dataset>__per_repeat.csv` — raw per-draw rows
