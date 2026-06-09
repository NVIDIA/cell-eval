# Robustness experiment — `adata_Validation` × pdex

Stress-tests cell-eval's metrics for **calibration & robustness** on
`adata_Validation.h5ad` using the **real** cell-eval DE backend
`cell_eval._de_backends.build_de_frame` with `de_method = pdex` (cell-level
Wilcoxon rank-sum — the test behind `cell-eval run --de-methods pdex`).

This folder is self-contained and reproducible: `run.py` reads only
`config.yaml` + the dataset, and a single `seed` drives every split/permutation.

## Dataset
- 98,927 cells × 18,080 genes, raw integer counts in `.X` (no layers).
- `target_gene` (50 perturbations + `non-targeting` control; all 50 match `var_names`),
  `guide_id` (85 guides), `batch` (48 batches).
- Only **4** target genes have ≥2 guides → Test 5 runs on a small set.

## Normalization (pdex)
pdex expects log-normalized input. `cell-eval run` normalizes inside
`MetricsEvaluator`; this runner drives `build_de_frame` directly, so it applies
the same `normalize_total` + `log1p` **once up front** (`normalize_if_raw: true`)
and then passes `allow_discrete: false` so pdex treats the matrix as log1p data
(`is_log1p=True`). normalize_total + log1p are per-cell, so normalizing once and
subsetting is identical to normalizing each subset.

## Tests (see `TEST_PLAN.md`)
- **Validity gates (must PASS first):** 1 within-condition null, 2 control-control
  null, 3 label-permutation null.
- **Sensitivity diagnostics:** 4 same-sgRNA reproducibility (empirical ceiling),
  5 same-gene cross-sgRNA reproducibility, 6 target-gene knockdown recovery.
- Global verdict: if ANY of tests 1–3 FAIL → **FAIL** (don't interpret biology).

## Run / re-run (exact command)
```bash
cd experiments/adata_Validation__pdex__20260608-202126
source ../../.venv/bin/activate
python run.py --config config.yaml
```
Environment: the repo's `.venv` (has `cell_eval` + `pdex` + `reportlab`).
Same `seed` ⇒ identical report.

## Outputs
- `robustness_report.md`, `report.pdf` — the report (verdicts, all verification
  p-values, thresholds table, power/sample-size §7, embedded `TEST_PLAN.md`).
- `robustness_summary.json` — machine-readable results.
- `tables/` — per-split / per-guide / per-target CSVs incl. p-value/FDR/LFC.
- `plots/` — QQ PNGs (Beta(i, G−i+1) 95% envelope) for tests 1–2.
- `run.log` — full run log.
