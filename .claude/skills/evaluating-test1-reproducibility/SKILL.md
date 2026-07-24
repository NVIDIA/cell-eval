---
name: evaluating-test1-reproducibility
description: Run ONLY Test 1 (within-condition reproducibility) from the DE metric-robustness battery and emit a single-test report. Splits each perturbation's cells into halves A/B (controls also split) and asks whether independent DE signatures agree using split-half Spearman, Pearson, DEG Jaccard, and direction, including diagonal/off-diagonal correlation summaries, rho-vs-cell-count, and a difference-is-null QQ. Use when someone wants the reproducibility ceiling and cross-perturbation specificity of a DE metric without running the whole battery.
---

# Test 1 — Within-Condition Reproducibility

**One test from the DE metric-robustness battery, run on its own.** This skill runs **only
`test_1`** and produces a self-contained report for it. Role: **validity gate (reproducibility ceiling)** — Sets the empirical ceiling: no model can score higher on a perturbation than the data agrees with itself.

Runs **both pdex and pydeseq2** on the same splits in one pass via `reproducibility_heatmap.py`.

`de_backends.py` is bundled with this skill and calls the upstream `pdex` and
`pydeseq2` packages directly. Do not import project-private DE backend modules.

## Mandatory preflight and run capture

Do not start the executable until the user has explicitly confirmed one fully resolved run configuration.

1. Gather the input `.h5ad`, ask whether `adata.X` contains raw counts or log1p-normalized expression, results output directory, separate run root, methods to compare, non-parametric engine (`pdex` or `rsc`) when `pdex` is selected, required observation columns and control label, replicate/block columns, count or score layers, thresholds, seeds, repeats, and worker/thread settings. Inspect the input read-only to resolve unknown columns, labels, and layers. Pass the confirmed state as `--expression-state`.
2. Expand paths and resolve every default. Show one concise preflight summary containing the input, results directory, run root, methods/engine, data fields, thresholds, workload/concurrency, exact command, log path, cache behavior, and resolved-config destination.
3. Ask for explicit confirmation and stop. Do not launch computation, plotting, resume mode, or cache reuse before confirmation.
4. After confirmation, create `<run-root>/logs` and `<run-root>/configs`, pass `--run-root <run-root>`, and capture the complete terminal stream with `2>&1 | tee <run-root>/logs/<workflow>__<dataset>__<UTC-timestamp>.log`.
5. Every invocation writes an immutable timestamped YAML snapshot under `<run-root>/configs`. Report the result, log, and YAML paths on completion.

Every box-and-whisker plot must overlay every finite underlying observation as jittered scatter points. Do not sample, aggregate away, or hide values in the scatter layer.

Keep `pdex` as the stable internal/table schema key, but label every plot with the actual selected engine: `pdex` for Arc pdex and `RSC` for RAPIDS GPU Wilcoxon. Never display an RSC result as pdex.

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
  --zoom-per-page 1 --max-genes 2000 --n-repeats 5 \
  --outdir "$RUN_DIR"
```

## Parameters
| flag | what it controls | default |
|---|---|---|
| `--min-cells` | min cells per arm; a perturbation needs ≥2x to be split | 20 |
| `--block-cols` | covariate the A/B split is balanced within | batch |
| `--fdr / --lfc` | DEG-calling cutoffs | 0.05 / 0.1 |
| `--max-genes` | fixed DE-feature panel size; use the most variable genes from the shared cross-method DE union | 2000 |
| `--zoom-per-page` | perturbations per zoom PNG | 1 |
| `--seed` | RNG seed for every split | 0 |
| `--n-repeats` | independently seeded A/B splits averaged in each Layer-3 matrix | 5 |
| `--threads` | worker threads supplied to each DE backend fit; affects speed, not estimates | 8 |
| `--non-parametric-engine` | `pdex` or numerically matched RAPIDS GPU Wilcoxon (`rsc`) | pdex |
| `--parallel-repeats` | POSIX fork workers for independent repeats; increase only when RAM/CPU permit | 1 |
| `--perturbation-workers` | within-repeat CPU PyDESeq2 workers; use with `--threads 1` and `--parallel-repeats 1` | 1 |
| `--pairwise-workers` | POSIX fork workers used after DE to build pair-specific DE-union correlation rows | 8 |
| `--signature-cache-dir` | optional directory for atomic partial-repeat checkpoints | none |
| `--resume-signatures` | load strictly matching local checkpoints and compute only missing repeats/methods | off |

## Reproducibility visualisations — `reproducibility_heatmap.py`

A standalone diagnostic (reuses the shared runner's exact Test-1 helpers `stratified_split` / `_de_two`
/ `maybe_normalize`, so the split-half DE matches the real test). By default it runs **five A/B
split repeats** per perturbation, using seeds `seed + repeat` (controls are split independently in
every repeat). It computes **both backends on the same splits** so they compare apples-to-apples.

Layer 3 is the multi-repeat result: construct the complete Spearman and Pearson A-versus-B matrices
separately in each repeat, then average corresponding matrix cells across the five repeats. Its
diagonal/off-diagonal numbers are calculated from those averaged matrices. Layers 1 and 2 remain
explicitly labelled **repeat-0 diagnostics** because arithmetic averaging of FDR values does not
define an aggregated DEG call.

On POSIX, repeat fits may run concurrently with `fork`, so workers inherit the loaded AnnData
copy-on-write. Do not select concurrency by hand from core count alone: the runner automatically
caps requested repeat workers by actual CPU affinity and available/cgroup memory. CPU use is held
to at most 75% after accounting for backend threads. Memory uses a conservative 1.5x input-file
estimate per worker and at most 50% of currently available memory. If memory cannot be estimated,
the runner defaults to one repeat worker. After loading AnnData, it performs a second cap using the
actual dense/sparse storage of `X`, every layer, and `raw.X`; this catches compressed files whose
on-disk size understates worker memory. These caps affect speed, not seeds or estimates.

Before NumPy, SciPy, or anndata is imported, the runner forces OpenMP, OpenBLAS, MKL, NumExpr,
BLIS, and Accelerate hidden pools to one thread. Backend parallelism still follows `--threads`.
Set Numba's startup maximum to that same requested value because pdex changes its active Numba
thread count at runtime; setting `NUMBA_NUM_THREADS=1` while passing pdex a larger value makes every
fit fail.
The runner re-executes itself once with those environment variables set so the limits also precede
Python startup/site hooks. This ordering is mandatory: setting variables merely before the local
NumPy import was insufficient on this environment and created 78 plotting threads; without any
guard, roughly 97 threads per repeat worker was observed. Limits cover Numba, Polars/Rayon,
Tokio, async-std, and jemalloc background pools in addition to BLAS/OpenMP.

On POSIX, the runner also holds a non-blocking advisory lock in `--outdir` for its lifetime. A
second Test-1 process targeting the same output directory exits immediately with a clear error
instead of racing to overwrite plots or multiplying hidden resource pools. Run separate backends
sequentially into one cache directory; partial-repeat resume prevents this serialization from
wasting completed work.

For long runs, set `--signature-cache-dir <dir> --resume-signatures`. Immediately after each repeat
finishes, the skill atomically rewrites a partial checkpoint containing every completed repeat plus
method, dataset, repeat-count, seed, perturbation, control, and block metadata. A later run loads
only an exact metadata match and computes missing repeats or backends. The current cache records
`non_parametric_engine=pdex|rsc`; stale pdex caches using the former `cpu` label are rejected.
PyDESeq2-only version-4/5 caches remain readable because this selector does not affect that backend.
Checkpoints are trusted local pickle files produced by this skill; never resume
from an untrusted file. If a process pool fails, completed repeats remain checkpointed and missing
ones fall back to sequential execution. If POSIX fork is unavailable, all repeats run sequentially.

The runner deliberately creates at most one repeat-level fork pool per process. If multiple
uncached methods are requested with `--parallel-repeats > 1`, the first computed backend uses the
pool and later backends automatically use safe sequential repeats. This guard fixes a pdex deadlock
caused by creating a second fork pool after a threaded pyDESeq2 pool. To parallelize every backend,
run one method per fresh invocation into the same checkpoint directory, then make a final
`--resume-signatures` run with both methods.

```bash
python .claude/skills/evaluating-test1-reproducibility/reproducibility_heatmap.py \
  --adata <h5ad> --methods pdex,pydeseq2 \
  --pert-col gene --control non-targeting --replicate-col batch --block-cols batch \
  --zoom-per-page 1 --max-genes 2000 --n-repeats 5 --outdir <out>
```

Every output filename ends with `__<dataset>` (the input `.h5ad` basename) so runs on different
datasets never overwrite each other.

The effective CPU and memory caps are printed before loading the data. This reserves capacity for
the operating system, plotting, and the interactive session and avoids an OOM-prone worker count.

- **Layer 1 — `test1_heatmap_<method>__<dataset>.png`** (one per backend, explicitly repeat 0): rows = perturbations sorted
  by split-half ρ (low ρ at top), columns = union DE genes sorted by split-A mean LFC, two panels
  **split A | split B**. Diverging blue–white–red, centred 0, capped ±2 log2FC (white = 0).
- **Layer 2 — `test1_zoom__<dataset>_NN.png`** (repeat 0; one PNG per perturbation, sorted by ρ): **one row per
  method** (e.g. a **pdex row and a pydeseq2 row**), each = LFC_A-vs-LFC_B scatter (grey = non-DE, dark
  blue = DE in A∪B) + that method's split-A/B heatmap strips over a shared gene order.
  `--zoom-per-page N` puts N perturbations per PNG.
- **Layer 3 — `test1_corr_matrix__<dataset>.png`** (one panel per backend): split-A × split-B signature
  correlation matrix. Entry (i,j) is computed separately per repeat as Spearman(pert i split-A,
  pert j split-B) over the union DE genes, then averaged cell-by-cell across `--n-repeats`.
  The **diagonal is within-perturbation A/B reproducibility** (dark red = higher positive
  correlation); off-diagonal =
  cross-perturbation (should be dim if signatures are perturbation-specific). **Each diagonal cell is
  labelled with that perturbation's cell count**, so reproducibility can be read against power. The
  heatmap colors encode Spearman. Every method-panel title reports the finite means of matching and
  nonmatching cells for both metrics as `Spearman: diag=<mean>, off=<mean>; Pearson: diag=<mean>,
  off=<mean>`. Pearson uses the same per-repeat aligned union-DE-gene vectors and
  finite/nonconstant safeguards, and its matrices are averaged by the same rule.
- Every Layer-3 PNG has a same-stem `.npz` archive containing `methods`, the ordered
  `targets__<method>` labels, the exact `features` used for correlation, and exact
  `spearman__<method>` and `pearson__<method>` averaged matrices. Use these numeric archives—not
  rasterized heatmaps—for downstream boxplots or tests.
- The primary fixed-panel Layer-3 matrix also emits `test1_corr_matrix_boxplots__<dataset>.png`
  and a same-stem CSV. Each method gets Spearman and Pearson panels comparing diagonal
  (within-perturbation) with off-diagonal (cross-perturbation) correlations. Boxes use the IQR,
  whiskers use the 5th–95th percentiles, and the rasterized scatter layer contains **every finite
  matrix value**, never a sample. The CSV records n, mean, standard deviation, minimum, 5th/25th/
  50th/75th/95th percentiles, and maximum for every method × metric × group.
- A one-method cache-building run writes method-suffixed matrices such as
  `test1_corr_matrix_pydeseq2__<dataset>.png`; it must not overwrite the canonical unsuffixed
  two-method plot because its union-DE gene set is method-specific. Only a run loading both method
  caches writes the two canonical matrices: `test1_corr_matrix__<dataset>.png` and
  `test1_corr_matrix_all_genes__<dataset>.png`.
- Make the canonical two-method `test1_corr_matrix__<dataset>.png` the **only DE correlation
  matrix**. Build one shared cross-method union of genes passing FDR/LFC in either split, method,
  perturbation, or repeat; restrict it to genes with finite split-A and split-B effects everywhere;
  rank by cross-perturbation split-A LFC variance; and retain at most `--max-genes` (normally 2,000).
  Evaluate both methods on this exact same ordered panel for every perturbation, A×B matrix cell,
  and repeat. Never perform pairwise feature deletion or reselection. Do not emit a separate
  method-specific "own DE genes" matrix or a duplicate `test1_corr_matrix_shared_union` file.
- Make `test1_corr_matrix_all_genes__<dataset>.png` the **only all-gene correlation matrix**. Use
  one complete-case gene panel shared by both methods and finite in every perturbation, split, and
  repeat. Evaluate both methods on that exact panel. Do not emit method-specific all-gene,
  `test1_corr_matrix_all_genes_shared`, or lowest-expression correlation matrices.
- Treat the shared DE-2,000 and shared all-gene matrices as the two default correlation outputs.
- `test1_corr_matrix_pair_specific_de_union__<dataset>.png` is a separate feature-selection
  sensitivity analysis. For each method and repeat, cell `(i,j)` correlates split-A perturbation
  `i` with split-B perturbation `j` over
  `DE_A(i) ∪ DE_B(j)`. Calls require that perturbation's CPM eligibility plus the configured
  FDR/absolute-LFC thresholds, and retained genes must have finite effects in both compared vectors.
  Compute every repeat independently and average corresponding correlation cells. Keep one common
  perturbation set/order across all methods and repeats, but do not force the cell-specific gene
  panels to be identical across methods: each backend's own DE calls define its sensitivity panel.
  This output complements rather than replaces the canonical fixed-panel matrix.
- The pair-specific matrix has a same-stem NPZ containing both averaged correlation matrices,
  mean per-cell panel sizes, finite-repeat counts, thresholds, ordered perturbations, and the exact
  panel-definition string. It also emits
  `test1_corr_matrix_pair_specific_de_union_boxplots__<dataset>.png` plus a same-stem all-values
  summary CSV using the same box/whisker/scatter contract as the fixed-panel matrices.
- `test1_rho_<method>__<dataset>.csv` — per-repeat, per-perturbation split-half ρ, including repeat,
  seed, and cell count.
- `test1_lfc_vectors_<method>__<dataset>.parquet` — long-form per-repeat, per-perturbation LFC
  vectors with one row per feature, separate split-A / split-B values, and execution-engine provenance.

Colour spec: diverging blue–white–red centred at 0, capped ±2 log2FC — genes near 0 in both splits
appear white, so colour in split B where split A is white (or vice versa) is irreproducibility made
visible.
