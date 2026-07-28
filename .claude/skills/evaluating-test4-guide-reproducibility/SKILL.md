---
name: evaluating-test4-guide-reproducibility
description: Run ONLY Test 4 (same-sgRNA split reproducibility) from the DE metric-robustness battery and emit a single-test report. This is Test 1 run at the sgRNA guide level, splitting each guide's cells into halves A and B versus split control and measuring split-half agreement with Spearman, Pearson, DEG Jaccard, and direction. Use when someone wants the per-guide reproducibility ceiling and cross-guide specificity without running the whole battery. Needs an sgRNA column.
---

# Test 4 — Same-sgRNA Split Reproducibility (guide-level Test 1)

**One test from the DE metric-robustness battery, run on its own.** This skill runs **only
`test_4`** and produces a self-contained report for it. Role: **sensitivity diagnostic (per-guide reproducibility ceiling)** — The empirical reproducibility ceiling at the resolution per-guide downstream metrics operate.

Runs **both pdex and pydeseq2** on the same splits in one pass via `guide_split_reproducibility.py`.

`de_backends.py` is bundled with this skill and calls the upstream `pdex` and
`pydeseq2` packages directly. Do not import project-private DE backend modules.

Before executing, read and follow
[`hardware-execution-contract.md`](hardware-execution-contract.md). Use the bundled
`run_with_watchdog.py` for every inference attempt; worker value `0` means
hardware-adaptive selection.

## Mandatory preflight and run capture

Do not start the executable until the user has explicitly confirmed one fully resolved run configuration.

1. Gather the input `.h5ad`, ask whether `adata.X` contains raw counts or log1p-normalized expression, results output directory, separate run root, methods to compare, non-parametric engine (`pdex` or `rsc`) when `pdex` is selected, perturbation/guide/target-gene/control fields, replicate/block columns, count layer, thresholds, seeds, guide limits, repeats, and worker/thread settings. Inspect the input read-only to resolve unknown columns, labels, layers, and guide counts. Pass the confirmed state as `--expression-state`.
2. Expand paths and resolve every default. Show one concise preflight summary containing the input, results directory, run root, methods/engine, data fields, thresholds, selected guide scope, workload/concurrency, exact command, log path, cache behavior, and resolved-config destination. Before asking for confirmation, estimate wall time separately for every selected method, shared preparation/rendering, and the complete run; state the hardware/tier, cache assumptions, evidence or throughput basis, uncertainty range, and how watchdog de-escalation could extend it.
3. Ask for explicit confirmation and stop. Do not launch computation, plotting, or cache reuse before confirmation.
4. After confirmation, create `<run-root>/logs` and `<run-root>/configs`, pass `--run-root <run-root>`, and capture the complete terminal stream with `2>&1 | tee <run-root>/logs/<workflow>__<dataset>__<UTC-timestamp>.log`.
5. Every invocation writes an immutable timestamped YAML snapshot under `<run-root>/configs`. Report the result, log, and YAML paths on completion.

Every box-and-whisker plot must overlay every finite underlying observation as jittered scatter points. Do not sample, aggregate away, or hide values in the scatter layer.

Keep `pdex` as the stable internal/table schema key, but label every plot with the actual selected engine: `pdex` for Arc pdex and `RSC` for RAPIDS GPU Wilcoxon. Never display an RSC result as pdex.

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
| `--non-parametric-engine` | `pdex` for Arc pdex or `rsc` for numerically matched RAPIDS GPU Wilcoxon | pdex |
| `--pairwise-workers` | POSIX fork workers used after DE to build pair-specific DE-union correlation rows | 8 |
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
  --max-guides 24 --max-control 1200 --threads 1 --guide-workers 8 \
  --zoom-per-page 1 --outdir <out>
```

- **Layer 1 — `test4_guide_heatmap_<method>__<dataset>.png`** (one per backend): rows = guides (`GENE.gN`)
  sorted by split-half ρ (low ρ at top), columns = union DE genes sorted by split-A mean LFC, two panels
  **split A | split B**. Diverging blue–white–red, capped ±2 log2FC.
- **Layer 2 — `test4_zoom__<dataset>_NN.png`** (one PNG per guide, sorted by ρ): **one row per method**
  (pdex row + pydeseq2 row), each = LFC_A-vs-LFC_B scatter + that method's split-A/B heatmap strips.
- **Layer 3 — `test4_corr_matrix__<dataset>.png`** (one panel per backend): split-A × split-B **guide**
  correlation matrix — **diagonal = within-guide A/B reproducibility** (dark red = higher positive
  correlation),
  off-diagonal = cross-guide. **Each diagonal cell is labelled with that guide's cell count** (n cells
  before the A/B split), so you can read reproducibility against power at a glance. Every panel title
  reports finite diagonal and off-diagonal means for both Spearman and Pearson.
- The primary fixed-panel Layer-3 matrix also emits `test4_corr_matrix_boxplots__<dataset>.png`
  and a same-stem CSV. Each method gets Spearman and Pearson panels comparing diagonal (within-guide)
  with off-diagonal (cross-guide) correlations. Boxes use the IQR, whiskers use the 5th–95th
  percentiles, and the rasterized scatter layer contains **every finite matrix value**, never a
  sample. The CSV records n, mean, standard deviation, minimum, 5th/25th/50th/75th/95th
  percentiles, and maximum for every method × metric × group.
- Whenever those primary correlation boxplots are rendered, also emit
  `test4_corr_matrix_de_jaccard_diagonal_boxplot__<dataset>.png` and
  `test4_corr_matrix_de_jaccard_off_diagonal_boxplot__<dataset>.png`, each with a same-stem
  summary CSV. Define matrix cell `(i,j)` as `J(DE_A(guide i), DE_B(guide j))`; both DE sets
  require the configured FDR/absolute-LFC thresholds and guide-specific CPM eligibility. The
  diagonal plot shows same-guide split reproducibility and the off-diagonal plot shows
  cross-guide DE-set similarity. Treat an empty union as undefined—not Jaccard 1. Use IQR
  boxes, 5th–95th percentile whiskers, and jittered scatter containing **every finite matrix
  cell**, never a sample. Test 4 has one split repeat, so no cross-repeat averaging is needed.
- `test4_corr_matrix_pair_specific_de_union__<dataset>.png` is a separate feature-selection
  sensitivity analysis. For each method, cell `(i,j)` correlates split-A guide `i` with split-B guide
  `j` over `DE_A(i) ∪ DE_B(j)`. Calls require the guide's CPM eligibility plus the configured
  FDR/absolute-LFC thresholds, and retained genes must have finite effects in both compared vectors.
  Keep one common guide set/order across methods, while allowing each backend's own DE calls to
  define its cell-specific panels. This complements rather than replaces the canonical fixed-panel
  matrix.
- The pair-specific matrix has a same-stem NPZ containing Spearman and Pearson matrices, per-cell
  panel sizes, thresholds, ordered guides, and the exact panel-definition string. It also emits
  `test4_corr_matrix_pair_specific_de_union_boxplots__<dataset>.png` plus a same-stem all-values
  summary CSV using the same box/whisker/scatter contract as the fixed-panel matrices.
- `test4_rho_<method>__<dataset>.csv` — per-guide split-half ρ.
- `test4_lfc_vectors_<method>__<dataset>.parquet` — long-form per-guide LFC vectors with one row per
  feature and separate split-A / split-B values.
- `test4_de_masks_<method>__<dataset>.npz` — version-2 compact cache of bit-packed per-guide
  split-A DE, split-B DE, CPM-eligibility, and feature-presence masks. The archive records its feature
  universe, guide metadata, `pdex|rsc` execution engine, FDR/LFC thresholds, CPM normalization,
  minimum CPM, counts layer, and bit order. Use this cache with the LFC Parquet for exact pair-specific DE-union correlations without
  rerunning DE. Reject a cache when its recorded thresholds or CPM/count-layer semantics do not match
  the requested analysis.

**This differs from `evaluating-test5-samegene-sgrna`'s `samegene_guide_heatmap.py`:** Test 4 splits ONE
guide's cells in half (within-guide reproducibility ceiling); Test 5 compares DIFFERENT guides of the
same gene (does the metric reward shared biology). Levers: `--max-guides` (most-powered first) and
`--max-control` bound runtime; `--methods pdex` skips the slower backend.
With a CUDA-capable RAPIDS environment, `--methods pdex --non-parametric-engine rsc` retains every guide while
moving the Wilcoxon/FDR/LFC calculation to the GPU.
For CPU PyDESeq2, `--guide-workers N --threads 1` runs independent guide fits concurrently without
changing the per-guide PyDESeq2 model or numerical results. The default `0`
selects a CPU/RAM-safe worker count from the measured machine.
