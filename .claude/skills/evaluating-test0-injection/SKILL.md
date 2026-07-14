---
name: evaluating-test0-injection
description: Run ONLY Test 0 (controlled effect-size injection / calibration curve) from the cell-eval metric-robustness battery and emit a single-test report. Spikes a known log2FC into ~a dozen genes in a control-vs-control split at several delta tiers (incl. delta=0) and measures the null false-positive rate and the TPR-vs-effect-size curve (smallest resolvable delta), plus compositional-coupling of untouched genes. Use when someone wants to calibrate a DE metric's false-positive rate / resolving power, or check injection/FPR/TPR calibration, without running the whole battery.
type: skill
---

# Test 0 — Controlled Effect-Size Injection / Calibration Curve

**One test from the cell-eval metric-robustness battery, run on its own.** This skill runs **only
`test_0`** and produces a self-contained report for it. Role: **validity gate (calibration + power)** — A FAIL here means the pipeline mis-calls untouched genes — do not interpret any biology until it is fixed.

Runs **both pdex and pydeseq2** in one pass via `inject_and_count_de.py`. Both datasets
(cell_eval2 + adata_Validation) can be run into the same output directory — output filenames are
suffixed with the `.h5ad` basename so they never collide.

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

# cell_eval2 — 100 draws
python "$SCRIPT" \
  --adata /data2/shared/perturb_seq/k562/Replogle_K562_Essential_2022/cell_eval2.h5ad \
  --pert-col gene --control non-targeting --replicate-col batch \
  --counts-layer counts --n-repeats 100 --out "$RUN_DIR/injection_recovery.csv"

# adata_Validation — 20 draws (no counts layer, uses .X)
python "$SCRIPT" \
  --adata /workspace/cell-eval/adata_Validation.h5ad \
  --pert-col target_gene --control non-targeting --replicate-col batch \
  --n-repeats 20 --out "$RUN_DIR/injection_recovery_validation.csv"
```

## Parameters
| flag | what it controls | default |
|---|---|---|
| `--deltas` | log2FC tiers injected (comma-sep; 0 = null FPR baseline) | `0,0.5,1,2` |
| `--n-genes` | anchor genes injected per draw | 12 |
| `--n-repeats` | draws pooled for stable rates (cell_eval2 uses 100, adata_Validation 20) | 100 |
| `--fdr / --lfc` | DEG-calling cutoffs | 0.05 / 0.1 |
| `--replicate-col` | pseudobulk unit (required for pydeseq2 within each repeat) | batch |
| `--counts-layer` | raw-counts layer; falls back to `.X` if absent | auto |

## Outputs (in `$RUN_DIR`)
- `injection_recovery__<dataset>__recovered_box.png` — # anchors recovered (TPR) vs δ, pdex vs pydeseq2
- `injection_recovery__<dataset>__false_pos_box.png` — # false-positive DE genes vs δ, pdex vs pydeseq2
- `injection_recovery__<dataset>.csv` — pooled mean±SD per (δ, method)
- `injection_recovery__<dataset>__per_repeat.csv` — raw per-draw rows
