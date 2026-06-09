---
name: evaluating-metrics-robustness
description: Stress-test cell-eval's metrics for stability & robustness on a perturbation .h5ad (e.g. adata_Validation.h5ad). Acts as a metaskill — fingerprint the dataset, confirm a configuration with the user in dialog, then INSTANTIATE a self-contained, reproducible experiment in a per-run experiment folder (frozen config.yaml + standalone run.py + a copy of the test plan + README) that runs cell-eval's real DE backend (pdex or pydeseq2) through the test plan and outputs a robustness report (markdown + PDF) with all p-values, power/sample-size analysis, thresholds, and the embedded test plan. Use when someone wants to audit/calibrate cell-eval metrics, validate a DE pipeline's calibration before interpreting biology, or produce a reproducible robustness report for a dataset.
type: skill
---

# Evaluating cell-eval metrics for stability & robustness (metaskill)

This is the **metaskill**. Given a perturbation dataset in `.h5ad` format
(e.g. `/home/yangzhang/code/cell-eval/adata_Validation.h5ad`), you work in two phases:

**Phase A — nail down the configuration (interactive).**
1. **create a fresh experiment folder** for this run,
2. **fingerprint** the dataset,
3. **confirm the configuration with the user** in dialog (do not guess silently).

**Phase B — instantiate a concrete, reproducible experiment.**
4. **instantiate** the confirmed config as a self-contained, runnable experiment *in the experiment
   folder* — a frozen `config.yaml`, a standalone `run.py`, a copy of `TEST_PLAN.md`, and a `README.md`
   with the exact command — so the user can run (and re-run) the concrete experiment for their dataset,
5. **run it** to produce a **robustness report** (markdown + PDF) reproducible from the folder alone.

The metaskill decides *what* to run via dialog; the experiment folder *is* the concrete, re-runnable
experiment. Keep `run.py` self-contained (it reads only `config.yaml` + the dataset) so the folder is
portable and the result reproduces under the same `seed`.

`TEST_PLAN.md` (in this directory) is the canonical specification of *what* the tests are, their
formulas, and verdict rules. **Do not reimplement DE or the metrics — call the real `cell_eval` code.**
Run commands from the repo root `/home/yangzhang/code/cell-eval` in the **Python environment the user
confirms in Phase A** — either `uv run python …`, or `source .venv/bin/activate` then `python …`.

## Experiment folder (one per run)

Every time the skill is carried out, create a new, self-contained **experiment folder** so runs never
clobber each other and each result is fully reproducible. Convention:

```
experiments/<dataset-stem>__<de_method>__<UTC-timestamp>/
```
e.g. `experiments/adata_Validation__pydeseq2__20260604-141530/`, created under the repo root
`/home/yangzhang/code/cell-eval/` (make the parent `experiments/` if missing). Create it first:

```bash
RUN_DIR="experiments/$(basename <h5ad> .h5ad)__<de_method>__$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
```

The experiment folder is the **instantiated, self-contained experiment** — everything to run it and
to reproduce it lives in `$RUN_DIR`.

Instantiated by the metaskill (Phase B):
- `config.yaml` — the frozen, user-confirmed configuration (with `outdir: .`),
- `run.py` — the standalone runner (reads only `config.yaml` + the dataset; implements the test plan
  via cell-eval; writes the outputs below and renders the PDF),
- `TEST_PLAN.md` — a copy of the test plan used (provenance + embedded in the report),
- `README.md` — what this experiment is and the **exact command to run / re-run it**,
- `inspect.txt` — saved fingerprint of the dataset.

Produced by running `run.py`:
- `robustness_report.md`, `report.pdf` — the robustness report (see "Report contents (REQUIRED)"),
- `robustness_summary.json`, `tables/`, `plots/` — machine-readable results + per-gene CSVs + QQ PNGs.

Report the absolute path of `$RUN_DIR` **and the run command** to the user at the end.

## Use the real cell-eval code (do not reimplement DE or the metrics)

Drive cell-eval's actual code paths so the report reflects the real pipeline:

- **Direct DE** — `build_de_frame(...)` returns the canonical DE frame
  `[target, feature, log2_fold_change, p_value, fdr]` for the configured backend:
  ```python
  from cell_eval._de_backends import build_de_frame
  de = build_de_frame(
      mode="real", adata=adata, control_pert=control_pert, pert_col=groupby,
      num_threads=n, allow_discrete=allow_discrete, de_method=de_method,
      de_kwargs=de_kwargs or None, counts_layer=counts_layer, replicate_col=replicate_col,
  )  # -> polars DataFrame
  ```
- **Run-pipeline metrics** — `cell_eval.MetricsEvaluator(...).compute(...)` is exactly
  `cell-eval run --de-methods <de_method>`:
  ```python
  from cell_eval import MetricsEvaluator
  ev = MetricsEvaluator(
      adata_pred=pred, adata_real=real, control_pert=control_pert, pert_col=pert_col,
      num_threads=n, outdir=RUN_DIR, prefix="testX", allow_discrete=allow_discrete,
      de_methods=[de_method], counts_layer=counts_layer, replicate_col=replicate_col,
      de_kwargs={de_method: de_kwargs} if de_kwargs else None,
  )
  perturbation_results, agg_results = ev.compute(profile="full", write_csv=True, basename="results.csv")
  ```
- **Backend dispatch & validation** — `cell_eval._de_backends.normalize_de_methods(de_method)`
  validates the name (`pdex` or `pydeseq2`). `MetricsEvaluator` is re-exported at the top level
  (`from cell_eval import MetricsEvaluator`). The null-split tests MUST use the **same `de_method` as the intended real
  analysis** (TEST_PLAN.md §"Critical").

## Test plan → what to run

`TEST_PLAN.md` defines six tests. Tests 1–3 are **validity gates**; tests 4–6 are **sensitivity
diagnostics**. Implement each exactly as specified there (formulas, expected behaviour, verdict rules):

| TEST_PLAN.md test | role | how to run it |
|---|---|---|
| 1 — Within-Condition Direct Split Null | gate | per condition: `n_permutations` stratified A/B splits → `build_de_frame` A-vs-B → null metrics (`frac_sig`, `mean_lfc`, `mean_abs_lfc`, `ks_p_uniform`, `λ_GC`, QQ) |
| 2 — Control-Control Split Null | gate | split control cells into pseudo-pert vs reference → same null metrics + check downstream cell-metrics collapse to chance |
| 3 — Label Permutation Null | gate | true metrics vs `n_permutations` within-`batch_cols` label shuffles → separation z-score / empirical p |
| 4 — Same-sgRNA Split Reproducibility | ceiling | split each guide's cells A/B, each vs control → compare signatures (Spearman LFC, Jaccard, direction, AUC) |
| 5 — Same-Gene Independent sgRNA Reproducibility | sensitivity | same-gene guide pairs vs unrelated-pair background → separation score |
| 6 — Target Gene Knockdown Recovery | sensitivity | per guide vs control → target-gene `lfc/padj/rank/direction`; recovery & direction rates |

**Global verdict gate:** if ANY of tests 1–3 FAIL, the global verdict is **FAIL** — report it and tell
the user not to interpret the biology until the pipeline is fixed; tests 4–6 are only informative once
1–3 PASS. (Full global logic + the "realized parameters → real analysis" mapping is in TEST_PLAN.md
§"Global Verdict Logic" and §7.)

### Verdict implementation notes (encode these in `run.py`)
- **Null tests (1–2) are two-sided.** FAIL only on *anti-conservative* behaviour — `λ_GC >
  lambda_gc_fail` (1.10) or `frac_sig ≫ fdr_threshold`. But also **WARN on deflation / non-uniformity**
  — `λ_GC < 0.9` **or** `ks_p_uniform < 0.05` (p-values not Uniform[0,1]) — *even when `frac_sig ≈ 0`*.
  A strongly deflated null (e.g. pseudobulk DESeq2 with heavy shrinkage) is safe (no false positives)
  but under-powered, and must be **surfaced, not silently passed**. Do not let `frac_sig ≈ 0` alone
  earn a clean PASS when `λ_GC` / KS say the null is mis-shaped.
- **α calibration (TEST_PLAN.md §7.1):** compute `α_empirical = fdr_threshold / λ_GC` **only when
  `λ_GC ≥ 1`**. **When `λ_GC < 1` (deflation), do NOT inflate α** — set `α_empirical = fdr_threshold`
  and flag the deflation (investigate covariate overcorrection). Never report an α larger than
  `fdr_threshold`.
- **Dispersion `φ`:** prefer the DESeq2-fitted dispersion; if the backend wrapper does not expose it,
  approximate via method-of-moments on control counts and **label the power numbers as estimates**.

## Report contents (REQUIRED)

The generated report (`$RUN_DIR/robustness_report.md` and its rendered `report.pdf`) MUST include,
in addition to the per-test verdicts and headline numbers:

1. **All p-values used for verification.** For every test, report the p-values that drive its verdict
   — e.g. `ks_p_uniform` (tests 1–2), the empirical permutation p `perm_p` and separation z-scores
   (test 3), per-target `padj_target` (test 6), and the pooled null Wald/Wilcoxon p-value summary.
   Include a **`p_values` table per test** and reference the full per-gene p-value/FDR CSVs written in
   `$RUN_DIR/tables/` (so nothing is hidden behind a summary statistic).
2. **Power & sample-size analysis** (TEST_PLAN.md §7). Report the realized **`α_empirical`**
   (`= fdr_threshold / λ_GC` when `λ_GC ≥ 1`; **`= fdr_threshold` when `λ_GC < 1`, with the deflation
   flagged** — never inflate α above `fdr_threshold`, per §7.1),
   the dispersion `φ_median` from the null-split DESeq2 fit, the effect-size tiers
   (`δ_floor` / `δ_typical` / `δ_ceiling`), the reproducibility-attenuated `δ_effective = δ·√ρ`, the
   **power at each δ tier**, and **`n_required`** (80% power) per tier — as the §7.6 realized-parameters
   table. (Compute these only when the validity gates pass; otherwise state that power is not reported
   because the pipeline failed calibration.)
3. **Every threshold used for verification**, listed explicitly with its value and source, so the
   verdicts are reproducible. At minimum: `fdr_threshold`, `lfc_threshold`, `lambda_gc_warn` (1.05),
   `lambda_gc_fail` (1.10), `n_permutations`, `min_cells_per_group`, `seed`, the test-3 separation
   thresholds (PASS > 2, WARN 1–2), test-4 reproducibility tiers, test-5 separation (> 1.5), and
   test-6 `recovery_rate` / `direction_rate` / rank thresholds. Render this as a **"Verification
   parameters & thresholds" table** at the top of the report.
4. **A full copy of the test plan.** Embed the complete `TEST_PLAN.md` (design, formulas, verdict
   rules) as an appendix in both the markdown report and the PDF, so the report is self-describing.

Put the thresholds table and parameters near the top (right after the global verdict), the per-test
p-value/power details within each test's section, and the embedded test plan as the final appendix.

## Procedure

### Step 0 — Create the experiment folder (Phase A)
As above: `mkdir -p "$RUN_DIR"` and put everything there.

### Step 1 — Fingerprint the dataset (Phase A)
By default **load the full `.h5ad`** and inspect obs columns, `.X` scale (raw-integer vs
log-normalized, checked over ALL stored values), candidate perturbation/control/replicate/sgRNA
columns, and whether `target_gene` values match `var_names`; save the output to `$RUN_DIR/inspect.txt`.
For a very large file, confirm the full load is OK first (Step 2) and otherwise fall back to
`backed="r"` + sparse `.data`. Use the environment confirmed in Step 2 (`uv run python …` **or** an
activated venv `python …`):
```bash
python - "$ADATA" <<'PY' | tee "$RUN_DIR/inspect.txt"   # or: uv run python - "$ADATA"
import sys, anndata as ad, numpy as np, scipy.sparse as sp
a = ad.read_h5ad(sys.argv[1])                       # full load (default)
print("shape:", a.n_obs, "x", a.n_vars, "| layers:", list(a.layers))
for c in a.obs.columns:
    try: print(f"  {c:24s} dtype={str(a.obs[c].dtype):10s} n_unique={a.obs[c].nunique()}")
    except TypeError: pass
X = a.X; data = X.data if sp.issparse(X) else np.asarray(X).ravel()
data = data[np.isfinite(data)]
print("X raw-integer (ALL values):", bool(np.allclose(data, np.rint(data))),
      "| min/max/mean: %.3f/%.3f/%.4f" % (data.min(), data.max(), data.mean()))
PY
```

### Step 2 — Confirm the configuration with the user (Phase A, REQUIRED)
Use **AskUserQuestion** to confirm — never assume:
0. **execution environment & data loading**: run via **`uv run python …`** or via an **activated
   project venv** (`source .venv/bin/activate` then `python …`)? And **is it OK to load the full
   `.h5ad` into memory** (default: yes; for a very large file choose backed/subsampled instead)?
1. **`de_method`**: `pydeseq2` (pseudobulk DESeq2 **Wald**, recommended by TEST_PLAN.md for raw counts)
   or `pdex` (cell-level Wilcoxon). Must match the intended real-analysis DE method.
2. **`pert_col`** and exact **`control_pert`** label.
3. **expression / counts**: `pydeseq2` → `allow_discrete: true`, `normalize_if_raw: false`, raw
   `counts_layer` (or raw `.X`); `pdex` → log-normalized `.X` (raw auto-normalized if `normalize_if_raw`).
4. **`replicate_col`** (REQUIRED for `pydeseq2`; optional for `pdex`) and **`block_cols`** covariates
   used in the split design formula / stratification.
5. **`target_gene_col`** / **`sgrna_col`** (gate tests 4–6; `target_gene_col` must match `var_names`).
6. **knobs**: `n_permutations` (TEST_PLAN.md default 10), `min_cells_per_group` (20),
   `fdr_threshold` (0.05), `lfc_threshold` (0.1), `seed`, and `max_conditions` (cap for smoke runs).

Record the confirmed values — in Phase B (Step 3) they are frozen into **`$RUN_DIR/config.yaml`**
with `outdir: .` so the experiment folder is portable.

Example `$RUN_DIR/config.yaml` for `adata_Validation.h5ad` (pydeseq2 / DESeq2 Wald):
```yaml
adata_path: /home/yangzhang/code/cell-eval/adata_Validation.h5ad
de_method: pydeseq2
pert_col: target_gene
control_pert: non-targeting
replicate_col: batch
block_cols: []
allow_discrete: true
normalize_if_raw: false
target_gene_col: target_gene
fdr_threshold: 0.05
lfc_threshold: 0.1
n_permutations: 10
min_cells_per_group: 20
seed: 0
max_conditions: null          # null = all perturbations; set an int for a quick smoke run
outdir: .                     # outputs land in this experiment folder (portable)
tests: [test_1, test_2, test_3, test_4, test_5, test_6]
```

### Step 3 — Instantiate the experiment package (Phase B)
Materialize a self-contained, runnable experiment in `$RUN_DIR`:

- **`config.yaml`** — the frozen confirmed config from Step 2 (`outdir: .`).
- **`TEST_PLAN.md`** — `cp .claude/skills/evaluating-metrics-robustness/TEST_PLAN.md "$RUN_DIR/"`.
- **`run.py`** — a standalone runner that loads `config.yaml` + the dataset and, for each selected
  test, implements the recipe from `TEST_PLAN.md` using `build_de_frame` / `MetricsEvaluator` (see
  "Use the real cell-eval code"). Write shared helpers once and reuse (TEST_PLAN.md §Implementation
  Notes): `stratified_split(...)` (tests 1,2,4), `compare_signatures(de_A, de_B)` (tests 4,5),
  `null_metrics(de)` (`frac_sig`, `mean_lfc`, `mean_abs_lfc`, `ks_p_uniform`, `λ_GC` + QQ PNG with the
  Beta(i, G−i+1) envelope → `plots/`). SKIP any condition with `< 2 × min_cells_per_group` cells (log
  it; >50% SKIP → WARN). Use the fixed `seed` for every split/permutation. It emits per-test
  `TestResult` + an aggregate `RobustnessReport` (schema in TEST_PLAN.md §"Output Schema") and writes
  `robustness_summary.json`, per-test CSVs in `tables/` (incl. the full per-gene p-value/FDR/LFC
  tables), a `robustness_report.md` satisfying **"Report contents (REQUIRED)"**, and **renders
  `report.pdf` itself** (reportlab + a Unicode font like DejaVu; embed QQ PNGs + the full
  `TEST_PLAN.md` appendix) — so one command reproduces the entire report.
- **`README.md`** — one screen: what this experiment is (dataset, `de_method`, tests), the **exact run
  command**, the environment needed (`uv`, `cell_eval`, the `pydeseq2` extra if used), and a map of
  the outputs.

Keep `run.py` **config-driven and self-contained** — no hard-coded dataset/columns — so `$RUN_DIR` is
portable and re-runnable in isolation.

### Step 4 — Run the experiment → robustness report (Phase B, reproducible)
The instantiated folder is now runnable by the user (or by you on their behalf) with one command:
```bash
cd "$RUN_DIR" && uv run python run.py --config config.yaml
```
This produces `robustness_report.md`, `report.pdf`, `robustness_summary.json`, `tables/`, and
`plots/` inside `$RUN_DIR`. The same `seed` ⇒ an identical report, so the experiment reproduces from
the folder alone. Smoke-test first with a small `max_conditions` / `n_permutations` (pydeseq2 refits
DESeq2 per condition × permutation), confirm it runs clean, then run at full size.

### Step 5 — Summarize for the user
Grounded in the per-test tables and the global verdict gate, report:
- the **global verdict** (did validity gates 1–3 pass? if not, stop and flag the pipeline),
- per-test verdicts with concrete numbers (`frac_sig`, `λ_GC`, KS p, separation z-scores, recovery /
  direction rates, reproducibility Spearman / Jaccard),
- the empirical ceilings (tests 4–5) and what they imply for downstream model scores,
- what was SKIPped and why,
- the absolute path of `$RUN_DIR` and its `report.pdf`.

## Conventions & guardrails
- The DE method for the null-split tests **must equal** the intended real-analysis DE method.
- **Validity gates first:** never present tests 4–6 as meaningful if tests 1–3 FAIL.
- Everything for a run lives in its **experiment folder**; never write outputs elsewhere.
- The experiment folder is the **deliverable**: a frozen `config.yaml` + self-contained `run.py` +
  `TEST_PLAN.md` + `README.md` that the user can run and re-run independently of this metaskill.
- Reproducibility: a single fixed `seed` drives all splits/permutations; record it in `config.yaml`,
  and keep `run.py` config-driven so the same folder reproduces the same report.
- Ground every claim in the per-test tables; thresholds come from `TEST_PLAN.md` (`lambda_gc_warn`
  1.05 / `lambda_gc_fail` 1.10, `frac_sig ≈ fdr_threshold`, separation > 2, etc.) — report deviations.
