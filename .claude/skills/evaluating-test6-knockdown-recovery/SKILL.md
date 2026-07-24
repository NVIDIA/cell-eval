---
name: evaluating-test6-knockdown-recovery
description: Run ONLY Test 6 (target-gene knockdown recovery) from the DE metric-robustness battery and emit a single-test report. For each perturbation, checks whether the targeted gene itself is detected as differentially expressed in the expected (knock-down) direction, reporting recovery and direction rates. Use when someone wants to check target-gene knockdown / on-target recovery of a DE pipeline without running the whole battery. Needs a target-gene column matching var_names.
---

# Test 6 — Target Gene Knockdown Recovery

**One test from the DE metric-robustness battery, run on its own.** This skill runs **only
`test_6`** and produces a self-contained report for it. Role: **sensitivity diagnostic (assay/guide quality)** — High recovery is reassuring but partly reflects assay and guide quality, not the metric alone.

Runs **both pdex and pydeseq2** in one pass via `knockdown_recovery.py`.

`de_backends.py` is bundled with this skill and calls the upstream `pdex` and
`pydeseq2` packages directly. Do not import project-private DE backend modules.

## Mandatory preflight and run capture

Do not start the executable until the user has explicitly confirmed one fully resolved run configuration.

1. Gather the input `.h5ad`, ask whether `adata.X` contains raw counts or log1p-normalized expression, results output directory, separate run root, methods to compare, non-parametric engine (`pdex` or `rsc`) when `pdex` is selected, perturbation/control/replicate fields, count layer, thresholds, and worker/thread settings. Inspect the input read-only to resolve unknown columns, labels, layers, and target-gene matching. Pass the confirmed state as `--expression-state`.
2. Expand paths and resolve every default. Show one concise preflight summary containing the input, results directory, run root, methods/engine, data fields, thresholds, workload/concurrency, exact command, log path, and resolved-config destination.
3. Ask for explicit confirmation and stop. Do not launch computation or plotting before confirmation.
4. After confirmation, create `<run-root>/logs` and `<run-root>/configs`, pass `--run-root <run-root>`, and capture the complete terminal stream with `2>&1 | tee <run-root>/logs/<workflow>__<dataset>__<UTC-timestamp>.log`.
5. Every invocation writes an immutable timestamped YAML snapshot under `<run-root>/configs`. Report the result, log, and YAML paths on completion.

Every box-and-whisker plot must overlay every finite underlying observation as jittered scatter points. Do not sample, aggregate away, or hide values in the scatter layer.

Keep `pdex` as the stable internal/table schema key, but label every plot with the actual selected engine: `pdex` for Arc pdex and `RSC` for RAPIDS GPU Wilcoxon. Never display an RSC result as pdex.

## What it asks
Is the **targeted gene itself** knocked down — significant and in the expected (negative for CRISPRi/KO) direction — in each perturbation's DE? Reported per method (pdex / pydeseq2): LFC, p-value, FDR, and rank within that perturbation's DE result.

## How DE is computed
One DE per perturbation vs control; both backends for a direct comparison. Needs: `--pert-col` where the perturbation name matches a `var_name` (gene in the count matrix).

## Run it (standalone)

```bash
RUN_DIR="experiments_all/$(basename $H5AD .h5ad)__test_6__$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
python .claude/skills/evaluating-test6-knockdown-recovery/knockdown_recovery.py \
  --adata <h5ad> --methods pdex,pydeseq2 \
  --pert-col <pert_col> --control non-targeting \
  --replicate-col <batch_col> \
  --fdr 0.05 --outdir "$RUN_DIR"
```

## Parameters

`--threads N` controls CPU inference for the single shared multi-contrast PyDESeq2 model.
| flag | what it controls | default |
|---|---|---|
| `--pert-col` | obs column with perturbation labels (must match var_names) | gene |
| `--control` | control label in pert_col | non-targeting |
| `--replicate-col` | pseudobulk unit for pydeseq2 | batch |
| `--counts-layer` | raw-counts layer (auto-detects 'counts') | auto |
| `--fdr` | FDR cutoff to call the target significant | 0.05 |
| `--methods` | comma-sep backends | pdex,pydeseq2 |
| `--non-parametric-engine` | `pdex` or numerically matched RAPIDS GPU Wilcoxon (`rsc`) | pdex |

## Outputs (in `--outdir`)
- `test6_knockdown_recovery_crossmethod__<dataset>.png` — matplotlib table: one row per perturbation, columns = LFC / p-value / FDR / rank stats per method; rows where methods disagree highlighted
- `test6_knockdown_recovery_crossmethod__<dataset>.csv` — full table
- `test6_knockdown_recovery_crossmethod__<dataset>.md` — rounded markdown
