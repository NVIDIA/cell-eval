# Test 7 — DE Method Benchmark

## What it does

Benchmarks pdex vs pydeseq2 (and any other registered DE backend) on the real dataset,
measuring per DE call:

- **Wall time** (seconds)
- **Peak CPU RAM delta** (MB) — RSS before vs after each call via `/proc/self/status`
- **Peak traced memory** (MB) — via `tracemalloc` (heap allocations only)
- **GPU VRAM delta** (MB) — via `nvidia-smi`; 0 if no GPU or method doesn't use one

## Key findings (plates dataset, 39k cells × 18k genes, one perturbation)

| | pdex | pydeseq2 |
|---|---|---|
| Time per call | ~30 s | ~48 s |
| Peak extra RAM | **+8.3 GB** | **+2.9 GB** |
| GPU VRAM | 0 | 0 |
| Data used | Full log-norm cell × gene matrix | Pseudobulk (3 plate replicates × gene) |

**pdex is faster but uses ~3× more RAM** because it operates at cell level (Wilcoxon
across 39k cells × 18k genes). pydeseq2 first aggregates to pseudobulk (3 replicates × 18k
genes) before fitting DESeq2, so its working matrix is tiny.

For the shuffle null on CE2 (smaller, 5k cells, 70-96 pseudobulk samples):
both methods take ~23 s per call — the pseudobulk compression benefit disappears when
there are many replicates per group.

## Output files

```
<outdir>/test7_benchmark_summary__<dataset>.csv    — one row per (method, perturbation)
<outdir>/test7_benchmark_report__<dataset>.md      — summary table
<outdir>/plots/test7_time__<dataset>.png           — wall time boxplot
<outdir>/plots/test7_ram__<dataset>.png            — RSS delta boxplot
<outdir>/plots/test7_gpu__<dataset>.png            — GPU VRAM delta boxplot
```

## Usage

```bash
# CE2 dataset (batch replicate)
python benchmark_de_methods.py \
    --adata /data2/shared/perturb_seq/k562/Replogle_K562_Essential_2022/cell_eval2.h5ad \
    --pert-col gene --control non-targeting \
    --replicate-col batch --counts-layer counts \
    --n-perts 10 --outdir .

# Plates dataset (plate replicate)
python benchmark_de_methods.py \
    --adata /workspace/cell-eval/adata_Validation_plates.h5ad \
    --pert-col target_gene --control non-targeting \
    --replicate-col plate \
    --n-perts 10 --outdir .
```

## Notes

- `--n-perts` controls how many randomly sampled perturbations to benchmark (default 10).
  Increase for more stable estimates; 5 is sufficient for a quick check.
- pydeseq2 requires a `--replicate-col` for pseudobulk aggregation. With only 3 replicates
  (e.g. 3 plates) the dispersion estimate may not converge — this is expected.
- Neither method uses GPU; the GPU plot will be flat at 0 on most systems.
- The RAM delta measured via RSS is approximate — it includes OS-level memory management
  effects. `peak_traced_mb` (tracemalloc) captures Python heap allocations more precisely
  but excludes C extensions (numpy, scipy internals).
