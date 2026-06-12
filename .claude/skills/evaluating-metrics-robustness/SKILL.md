---
name: evaluating-metrics-robustness
description: Stress-test cell-eval's metrics for stability & robustness on a perturbation .h5ad (e.g. adata_Validation.h5ad). Acts as a metaskill — fingerprint the dataset, confirm a configuration with the user in dialog, then INSTANTIATE a self-contained, reproducible experiment in a per-run experiment folder (frozen config.yaml + standalone run.py + a copy of the test plan + README) that runs cell-eval's real DE backend (pdex or pydeseq2) through the test plan and outputs a robustness report (markdown + PDF) with all p-values, thresholds, per-test tier explanations, and the embedded test plan. Use when someone wants to audit/calibrate cell-eval metrics, validate a DE pipeline's calibration before interpreting biology, or produce a reproducible robustness report for a dataset.
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

`TEST_PLAN.md` defines the tests. Tests **0–3** are **validity gates**; tests **4–6** plus the
**composition** diagnostic are **sensitivity diagnostics**. **Read TEST_PLAN.md §"Read first: unit of
analysis & DE method" before anything else** — whether the null-split tests behave depends entirely
on the DE method's unit of analysis (cell-level `pdex`/Wilcoxon is subject to *pseudoreplication*,
Squair et al. 2021; only pseudobulk/mixed-model nulls are calibrated). Implement each test exactly as
specified there (formulas, expected behaviour, verdict rules):

| TEST_PLAN.md test | role | how to run it |
|---|---|---|
| 0 — Controlled Effect-Size Injection | gate | on control cells: split A/B, spike known log2FC into a known fraction of genes in B at several δ tiers (incl. δ=0) → same DE → FPR-at-null + TPR-vs-δ curve; flag compositional coupling (FPR rising at large δ) |
| 1 — Within-Condition Direct Split Null | gate | per condition: `n_resamples` stratified A/B splits → `build_de_frame` A-vs-B → null metrics (`frac_sig`, `mean_lfc`, `mean_abs_lfc`, `ks_p_uniform`, `λ_GC`, QQ) |
| 2 — Control-Control Split Null | gate | split control cells into pseudo-pert vs reference → same null metrics + check downstream cell-metrics collapse to chance |
| 3 — Label Permutation Null | gate | true metrics vs `n_resamples` within-`block_cols` label shuffles → separation z-score / empirical p |
| 4 — Same-sgRNA Split Reproducibility | ceiling | split each guide's cells A/B, each vs control → compare signatures (Spearman LFC, Jaccard, direction, AUC) |
| 5 — Same-Gene Independent sgRNA Reproducibility | sensitivity | same-gene guide pairs vs unrelated-pair background → separation score (expect modest concordance; power-limited with few guides) |
| 6 — Target Gene Knockdown Recovery | sensitivity | per guide vs control → target-gene `lfc/padj/rank/direction`; recovery & direction rates (also reflects assay/guide quality) |
| Composition diagnostic | confounder | if a cell-type/cluster/cell-cycle column exists, TVD of per-pert proportions vs control; else SKIP with guidance |

**Global verdict gate:** if ANY of tests 0–3 FAIL, the global verdict is **FAIL** — report it and tell
the user not to interpret the biology until the pipeline is fixed; tests 4–6 are only informative once
0–3 PASS. (Full global logic + the "realized parameters → real analysis" mapping is in TEST_PLAN.md
§"Global Verdict Logic" and §7.)

**Evaluate corrected AND uncorrected data.** Batch / cell-cycle are often regressed out before DE; a
good metric should be insensitive to this but real ones are not. Record `covariate_correction` in the
config and, when feasible, run both states and compare. **Validate across ≥2 datasets/modalities**
before trusting thresholds.

### Verdict implementation notes (encode these in `run.py`)
- **Null tests (1–2) are two-sided.** FAIL only on *anti-conservative* behaviour — `λ_GC >
  lambda_gc_fail` (1.10) or `frac_sig ≫ fdr_threshold`. But also **WARN on deflation / non-uniformity**
  — `λ_GC < 0.9` **or** `ks_p_uniform < 0.05` (p-values not Uniform[0,1]) — *even when `frac_sig ≈ 0`*.
  A strongly deflated null (e.g. pseudobulk DESeq2 with heavy shrinkage) is safe (no false positives)
  but under-powered, and must be **surfaced, not silently passed**. Do not let `frac_sig ≈ 0` alone
  earn a clean PASS when `λ_GC` / KS say the null is mis-shaped.
- **Test 0 (injection) is a gate.** Inject on **raw counts** (before normalization), at δ tiers
  including `δ=0`. FAIL if the δ=0 FPR is anti-conservative (> 2×`fdr_threshold`); WARN if the metric
  is under-powered (`max_TPR < 0.5`) or shows **compositional coupling** (FPR among *untouched* genes
  rising steeply at large δ — a real artifact of library-size renormalization that ties back to the
  composition confounder). Report the FPR-at-null, TPR-vs-δ curve, and the minimum resolvable δ.
- **Unit of analysis must be reported.** Derive it from `de_method` (`pdex` → individual cell /
  Wilcoxon → pseudoreplication caveat; `pydeseq2` → pseudobulk replicate) and state it in the report's
  "How DE was computed" section.
- **Composition diagnostic.** If `obs` has a cell-type / cluster / cell-cycle column (auto-detect or
  `celltype_cols`), compute per-perturbation TVD of proportions vs control and WARN on TVD > 0.10;
  otherwise SKIP with explicit guidance that composition cannot be assessed.

## Report contents (REQUIRED)

The generated report (`$RUN_DIR/robustness_report.md` and its rendered `report.pdf`) is **the
deliverable** — a domain expert must be able to make sense of it on first read, without reverse-
engineering the code. Structure it **context-first** (see TEST_PLAN.md §"Report & presentation
conventions"); a buried, summary-only report is a failure even if every number is correct.

**Lead the report with, in this order:**

- **A. Global verdict** with a one-line reason, and the per-gate (0–3) verdicts each with their own
  one-line reason.
- **B. Dataset & experimental context** — file, #cells/#genes, #perturbations, control cell count,
  cells-per-perturbation (min/median/max), #guides and #genes-with-≥2-guides, the full obs-column
  list, and an explicit **"what is NOT available"** line (cell type, cell cycle, donor/patient, sex,
  tissue, timepoint) so the reader never has to guess.
- **C. How DE was computed** — the **DE backend, unit of analysis** (cell-level vs pseudobulk), and
  the **covariate-correction state** of `.X`, followed by the **pseudoreplication caveat** (cite
  Squair et al. 2021) when the unit is the cell, and a note that the metric should ideally be checked
  on both corrected and uncorrected data.
- **D. At-a-glance table** — every test with verdict + its **local one-line reason**.

Then the **per-test detail** sections, each opening with a **1–3 sentence plain-language "what this
tests / how to read it"** (reuse the "In one sentence" lines from TEST_PLAN.md), then the local
verdict reason, the rounded metric table, the verification p-values, flags, and the plot.

- **Test 0 MUST render the full per-δ calibration curve as an inline table in the report body** —
  one row per injected effect size `δ` (including `δ=0`) with columns **TPR (injected genes)**,
  **FPR (untouched genes)**, **median observed LFC**, and **λ_GC** — not just summary scalars or a
  CSV/PNG. This is the headline evidence (e.g. for cell-level `pdex` the FPR and λ_GC climb steeply
  with δ, exposing the compositional-coupling artifact; for pseudobulk they stay controlled), so it
  belongs in the text where the reader sees it. Read the curve from `tables/test_0__calibration_curve.csv`
  at render time so it appears in both full and `--report-only` output.
- **Test 0 injects an EXPRESSION-stratified anchor set; reports TPR by expression tier + FPR by gene
  class.** The injected (anchor) genes are chosen as **equal shares from the low / mid / high
  mean-expression tertiles** (detectability is dominated by expression level, so this makes the
  TPR-vs-δ curve readable per tier — the min resolvable δ for sparse vs typical vs abundant genes —
  rather than an artifact of the mix). Report **TPR broken down by injected-gene expression tier**
  (high/mid/low, with min resolvable δ each) AND the **untouched-gene FPR per gene class** — AnchorCorr
  (genes correlated with the anchors), HighlyExpr, LowlyExpr, HouseKeeping, Marker (N/A without
  cell-type labels), HighlyVarG, Random — both as inline tables + plot panels. All expression tiers
  and gene-class memberships are computed **deterministically over ALL control cells** (a fixed
  function of the h5ad, not of the cell subsample). Expect FPR coupling to surface first in
  HighlyExpr/AnchorCorr/HighlyVarG and spare LowlyExpr; HouseKeeping flags memorization. Do NOT inject
  only HVG / only HEG / only LEG.
- **Test 0 MUST also state its injection design in plain prose** so the section is self-describing:
  define **what an "arm" is** (one side of the two-group split; arm B receives the injected effect,
  arm A is the reference, DE = B vs A), and give the concrete setup — **how many genes were injected**
  (count + fraction + of how many *expressed* and *total* genes), **cells per arm** (and that this is
  capped by `injection_max_cells_per_arm` for speed/memory, so it's below the control-cell total and
  slightly uneven due to the stratified split), the **δ tiers**, and which population the cells came
  from. A reader must not have to ask "how many genes were injected?" or "what is an arm?".

**Formatting rules:** round all numbers to 4–6 significant figures (`0.1952`, never
`0.19523047652509917`); keep each test's failure/caveat reason **local to that test**; put heavy
material (full thresholds dump, embedded TEST_PLAN.md) **last**. **In the PDF, render every markdown
table as a real typeset table** (reportlab `Table`/`TableStyle` — shaded header row, grid lines,
wrapping cells, alternating row backgrounds), NOT as raw `| … |` pipe text: the `render_pdf` helper
must detect markdown table blocks (a `| header |` row followed by a `|---|` separator) and convert
them to `Table` flowables (skip code-fence ``` lines too). Include a **Glossary** (λ_GC,
ks_p_uniform, frac_sig, separation z, TPR/FPR, max_TPR / min resolvable δ, reproducibility ρ, "unit of analysis", **"arm"**, **"stratified split / block_cols"**) and a
**Known limitations & next steps** section (single dataset, cell-level unit,
composition not assessed, corrected-vs-uncorrected, why enrichment is excluded).

**Also emit (optional but encouraged):** a **MultiQC** custom-content file
(`$RUN_DIR/multiqc/*_mqc.txt`, https://github.com/MultiQC/MultiQC) so the verdicts render as an
interactive HTML table that integrates with downstream single-cell (scdownstream) pipelines.

In addition to the above, the report MUST include:

1. **All p-values used for verification.** For every test, report the p-values that drive its verdict
   — e.g. `ks_p_uniform` (tests 1–2), the empirical permutation p `perm_p` and separation z-scores
   (test 3), per-target `padj_target` (test 6), and the pooled null Wald/Wilcoxon p-value summary.
   Include a **`p_values` table per test** and reference the full per-gene p-value/FDR CSVs written in
   `$RUN_DIR/tables/` (so nothing is hidden behind a summary statistic).
2. **Every threshold used for verification**, listed explicitly with its value and source, so the
   verdicts are reproducible. At minimum: `fdr_threshold`, `lfc_threshold`, `lambda_gc_warn` (1.05),
   `lambda_gc_fail` (1.10), the λ_GC deflation/ks WARN thresholds (< 0.90 / < 0.05), the test-0
   injection δ tiers and null-FPR FAIL threshold (> 2×α), `n_resamples`, `min_cells_per_group`,
   `seed`, the test-1/test-4 reproducibility tiers (ρ 0.6/0.3, Jaccard 0.3/0.1, direction 0.7/0.6),
   the test-3 separation thresholds (PASS > 2, WARN 1–2), test-5 separation (> 1.5), and test-6
   `recovery_rate` / `direction_rate` thresholds. Render this as a **"Verification parameters &
   thresholds" table** — placed *after* the context and per-test sections (it is reference material,
   not a first impression). Each test's own metric **tiers** are explained inline in its result section.
3. **A full copy of the test plan.** Embed the complete `TEST_PLAN.md` (design, formulas, verdict
   rules) as an appendix in both the markdown report and the PDF, so the report is self-describing.

> **No power/sample-size calculation.** The skill does **not** compute a power / `n_required` /
> dispersion analysis (it was removed). Do not add a power section to the report.

Ordering: **global verdict → dataset/experimental context → how DE was computed (unit of analysis) →
at-a-glance → per-test detail (with plain-language + per-test tier explanations + local reasons) →
thresholds → glossary → limitations → embedded TEST_PLAN.md appendix.** Heavy reference material goes
last so the report's first screen is interpretable.

## Procedure

### Step 0 — Create the experiment folder (Phase A)
As above: `mkdir -p "$RUN_DIR"` and put everything there.

### Step 1 — Fingerprint the dataset (Phase A)
By default **load the full `.h5ad`** and capture the **experimental context the report must lead with**:
shape, layers, obs columns (with n_unique), `.X` scale (raw-integer vs log-normalized, over ALL
values), perturbation/control accounting (control cell count, cells-per-perturbation min/median/max),
guides and **#genes-with-≥2-guides**, whether `target_gene` matches `var_names`, and — critically —
**which structural/correctable covariates are present vs ABSENT** (cell type, cluster, cell-cycle
phase, donor/patient, sex, tissue, timepoint). Save to `$RUN_DIR/inspect.txt`. For a very large file,
confirm the full load is OK first (Step 2) and otherwise fall back to `backed="r"`.
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
HINTS = ("cell_type","celltype","cluster","leiden","phase","cell_cycle","donor","patient",
         "sex","gender","tissue","timepoint","sample","replicate")
present = [c for c in a.obs.columns if any(h in c.lower() for h in HINTS)]
print("covariate-like columns present:", present or "NONE beyond the perturbation/guide/batch keys")
print("MISSING (assume absent; report it):",
      [h for h in ("cell_type","cell_cycle","donor","patient","sex","tissue","timepoint")
       if not any(h in c.lower() for c in a.obs.columns)])
PY
```
The runner's own `fingerprint()` re-captures this into `inspect.txt` and the report's context section,
so the report never forces the reader to guess the experimental setup.

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
6. **`covariate_correction`** — what (if anything) was regressed out of `.X` (`none` = uncorrected).
   Confirm whether to evaluate uncorrected, corrected, or **both** (recommended); record the state so
   the report can label it. Also **`celltype_cols`** for the composition diagnostic (empty if none).
7. **knobs**: `n_resamples` (TEST_PLAN.md default 10), `min_cells_per_group` (20),
   `fdr_threshold` (0.05), `lfc_threshold` (0.1), `seed`, `max_conditions` (cap for smoke runs), and
   the **Test 0 injection** params (`injection_deltas` e.g. `[0,0.25,0.5,1,2]`, `injection_frac_genes`
   e.g. `0.10`, `injection_max_cells_per_arm` e.g. `3000`).

Record the confirmed values — in Phase B (Step 3) they are frozen into **`$RUN_DIR/config.yaml`**
with `outdir: .` so the experiment folder is portable.

Example `$RUN_DIR/config.yaml` for `adata_Validation.h5ad` (pydeseq2 / DESeq2 Wald):
```yaml
adata_path: /home/yangzhang/code/cell-eval/adata_Validation.h5ad
de_method: pydeseq2
pert_col: target_gene
control_pert: non-targeting
replicate_col: batch
block_cols: [batch]
allow_discrete: true
normalize_if_raw: false
target_gene_col: target_gene
sgrna_col: guide_id
covariate_correction: none    # what was regressed out of .X; evaluate uncorrected AND corrected
celltype_cols: []             # obs cols for composition control; [] -> composition SKIPs with guidance
fdr_threshold: 0.05
lfc_threshold: 0.1
n_resamples: 10
min_cells_per_group: 20
seed: 0
max_conditions: null          # null = all perturbations; set an int for a quick smoke run
# Test 0 — effect-size injection / calibration curve
injection_deltas: [0.0, 0.25, 0.5, 1.0, 2.0]   # log2FC tiers; 0.0 = null FPR baseline
injection_frac_genes: 0.10
injection_max_cells_per_arm: 3000
emit_multiqc: true            # also emit a MultiQC custom-content table
outdir: .                     # outputs land in this experiment folder (portable)
tests: [test_0, composition, test_1, test_2, test_3, test_4, test_5, test_6]
```
(For `pdex`: set `de_method: pdex`, `allow_discrete: false`, `normalize_if_raw: true` so raw counts
get `normalize_total`+`log1p` and pdex runs with `is_log1p=True`.)

### Step 3 — Instantiate the experiment package (Phase B)
Materialize a self-contained, runnable experiment in `$RUN_DIR`:

- **`config.yaml`** — the frozen confirmed config from Step 2 (`outdir: .`).
- **`TEST_PLAN.md`** — `cp .claude/skills/evaluating-metrics-robustness/TEST_PLAN.md "$RUN_DIR/"`.
- **`run.py`** — **start from the reference implementation `run_template.py` in this skill directory**
  (`cp .claude/skills/evaluating-metrics-robustness/run_template.py "$RUN_DIR/run.py"`). It is the
  battle-tested runner that already satisfies everything below — both backends (`pdex`/`pydeseq2`),
  all tests (0, 1–6, composition), the context-first report, inline Test-0 calibration curve +
  injection-design prose, glossary, MultiQC export, proper PDF tables, and `--report-only`. It is
  fully config-driven, so for a new dataset you normally only change `config.yaml`; edit `run.py`
  only to add a genuinely new test or backend. The template loads `config.yaml` + the dataset and, for
  each selected test, drives the recipe from `TEST_PLAN.md` via `build_de_frame` / `MetricsEvaluator`
  (see "Use the real cell-eval code"). Its shared helpers (TEST_PLAN.md §Implementation Notes):
  `fingerprint(...)` (dataset/experimental context + missing-covariate detection),
  `stratified_split(...)` (tests 0,1,2,4), `compare_signatures(de_A, de_B)` (tests 4,5),
  `null_metrics(de)` (`frac_sig`, `mean_lfc`, `mean_abs_lfc`, `ks_p_uniform`, `λ_GC` + QQ PNG with the
  Beta(i, G−i+1) envelope → `plots/`), `test_0_injection(...)` (inject on **raw** counts before
  normalization; TPR/FPR curve PNG), and `composition_diagnostic(...)`. Each `TestResult` carries a
  one-line local `reason`. SKIP any condition with `< 2 × min_cells_per_group` cells (log it; >50%
  SKIP → WARN). Use the fixed `seed` for every split/permutation/injection. It writes
  `robustness_summary.json` (incl. the fingerprint + per-test `reason`), per-test CSVs in `tables/`
  (incl. the full per-gene p-value/FDR/LFC tables and the Test-0 calibration curve), a
  `robustness_report.md` satisfying **"Report contents (REQUIRED)"** (context-first, plain-language,
  glossary, local reasons, rounded numbers), an optional **MultiQC** custom-content file under
  `multiqc/`, and **renders `report.pdf` itself** (reportlab + a Unicode font like DejaVu; embed the
  plots + the full `TEST_PLAN.md` appendix). Provide a **`--report-only`** mode that re-renders the
  report/PDF/MultiQC from the cached `robustness_summary.json` **without recomputing DE**, so report
  formatting can be iterated cheaply after the (expensive) compute has run once.
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
the folder alone. Smoke-test first with a small `max_conditions` / `n_resamples` (pydeseq2 refits
DESeq2 per condition × permutation), confirm it runs clean, then run at full size.

### Step 5 — Summarize for the user
Grounded in the per-test tables and the global verdict gate, report:
- the **global verdict** (did validity gates 0–3 pass? if not, stop and flag the pipeline) and the
  **DE method + unit of analysis** caveat (cell-level ⇒ ranking, not calibrated FDR),
- per-test verdicts with concrete numbers (Test 0 null-FPR + min resolvable δ, `frac_sig`, `λ_GC`,
  KS p, separation z-scores, recovery / direction rates, reproducibility Spearman / Jaccard),
- the empirical ceilings (tests 4–5) and what they imply for downstream model scores,
- what was SKIPped and why (e.g. composition: no cell-state column),
- the absolute path of `$RUN_DIR` and its `report.pdf`.

## Conventions & guardrails
- The DE method for the null-split tests **must equal** the intended real-analysis DE method, and the
  report **must state the unit of analysis** (cell-level ⇒ pseudoreplication caveat, Squair 2021).
- **Validity gates first:** never present tests 4–6 as meaningful if tests 0–3 FAIL.
- **Context-first, readable report:** lead with dataset/experimental context + how DE was computed;
  plain-language per test; local verdict reasons; rounded numbers; heavy appendices last.
- Everything for a run lives in its **experiment folder**; never write outputs elsewhere.
- The experiment folder is the **deliverable**: a frozen `config.yaml` + self-contained `run.py` +
  `TEST_PLAN.md` + `README.md` that the user can run and re-run independently of this metaskill.
- **Determinism (REQUIRED):** the run must be **completely seeded** and **bit-for-bit reproducible** —
  same dataset + `config.yaml` + `seed` ⇒ identical report and tables every time. A single `seed`
  drives every split, permutation, and the Test-0 injection sampling via
  `np.random.default_rng(seed + offset)` (no unseeded RNG); call `seed_everything(seed)` at startup
  (pins `PYTHONHASHSEED`, `random`, `np.random`, `scanpy.settings.seed`) and pass `random_state=seed`
  to any PCA/HVG. Record `seed` in `config.yaml`; verify by running twice and diffing
  `robustness_summary.json` (`run_template.py` already does all of this).
- Ground every claim in the per-test tables; thresholds come from `TEST_PLAN.md` (`lambda_gc_warn`
  1.05 / `lambda_gc_fail` 1.10, `frac_sig ≈ fdr_threshold`, separation > 2, etc.) — report deviations.
