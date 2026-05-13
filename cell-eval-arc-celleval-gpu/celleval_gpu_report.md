# cell-eval (gpu backend) — Arc benchmark

- Generated: `2026-05-13 17:26:51 CEST`
- Inputs: `/home/sdicks/git/rapids-singlecell-notebooks/adata_real.h5ad` + `/home/sdicks/git/rapids-singlecell-notebooks/arc/adata_pred.h5ad`
- `pert_col`: `drugname_drugconc` · `control`: `[('DMSO_TF', 0.0, 'uM')]`
- Backend: `gpu` (pdex via `cell_eval.pdex._rsc`, metrics via `cell_eval.metrics._anndata._gpu`)
- **Bench wall (cell-eval only)**: **1m 13.3s**
- Total wall incl. h5ad load + sparsify: 1m 52.4s

## Stages

| Stage | Wall time |
|---|---:|
| `real:read_h5ad` | 16.95s |
| `real:dense_to_csr` | 2.03s |
| `pred:read_h5ad` | 18.03s |
| `pred:dense_to_csr` | 1.99s |
| `setup_total_io_and_sparsify` | 39.10s |
| `metrics_evaluator_init_inc_de` | 16.40s |
| `compute_all_metrics` | 56.89s |
| `bench_wall_clock` | 1m 13.3s |
| `wall_clock_total` | 1m 52.4s |

## Aggregated metrics (mean over perturbations)

| Metric | Mean |
|---|---:|
| `overlap_at_N` | 0.743499 |
| `overlap_at_50` | 0.266748 |
| `overlap_at_100` | 0.395265 |
| `overlap_at_200` | 0.519864 |
| `overlap_at_500` | 0.664353 |
| `precision_at_N` | 0.749101 |
| `precision_at_50` | 0.266460 |
| `precision_at_100` | 0.394951 |
| `precision_at_200` | 0.519549 |
| `precision_at_500` | 0.664039 |
| `de_spearman_sig` | 0.931832 |
| `de_direction_match` | 0.910207 |
| `de_spearman_lfc_sig` | 0.822859 |
| `de_sig_genes_recall` | 0.850969 |
| `de_nsig_counts_real` | 1004.138776 |
| `de_nsig_counts_pred` | 1131.949660 |
| `pr_auc` | 0.886796 |
| `roc_auc` | 0.880728 |
| `pearson_delta` | 0.976591 |
| `mse` | 0.000101 |
| `mae` | 0.003981 |
| `mse_delta` | 0.000091 |
| `mae_delta` | 0.003554 |
| `discrimination_score_l1` | 0.999354 |
| `discrimination_score_l2` | 0.999930 |
| `discrimination_score_cosine` | 0.999978 |
| `pearson_edistance` | 0.955619 |
| `clustering_agreement` | 0.711679 |
