---
name: evaluating-test6-knockdown-recovery
description: Run ONLY Test 6 (target-gene knockdown recovery) from the cell-eval metric-robustness battery and emit a single-test report. For each perturbation, checks whether the targeted gene itself is detected as differentially expressed in the expected (knock-down) direction, reporting recovery and direction rates. Use when someone wants to check target-gene knockdown / on-target recovery of a DE pipeline without running the whole battery. Needs a target-gene column matching var_names.
type: skill
---

# Test 6 — Target Gene Knockdown Recovery

**One test from the cell-eval metric-robustness battery, run on its own.** This skill runs **only
`test_6`** and produces a self-contained report for it. Role: **sensitivity diagnostic (assay/guide quality)** — High recovery is reassuring but partly reflects assay and guide quality, not the metric alone.

Runs **both pdex and pydeseq2** in one pass via `knockdown_recovery.py`.

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
| flag | what it controls | default |
|---|---|---|
| `--pert-col` | obs column with perturbation labels (must match var_names) | gene |
| `--control` | control label in pert_col | non-targeting |
| `--replicate-col` | pseudobulk unit for pydeseq2 | batch |
| `--counts-layer` | raw-counts layer (auto-detects 'counts') | auto |
| `--fdr` | FDR cutoff to call the target significant | 0.05 |
| `--methods` | comma-sep backends | pdex,pydeseq2 |

## Outputs (in `--outdir`)
- `test6_knockdown_recovery_crossmethod__<dataset>.png` — matplotlib table: one row per perturbation, columns = LFC / p-value / FDR / rank stats per method; rows where methods disagree highlighted
- `test6_knockdown_recovery_crossmethod__<dataset>.csv` — full table
- `test6_knockdown_recovery_crossmethod__<dataset>.md` — rounded markdown
