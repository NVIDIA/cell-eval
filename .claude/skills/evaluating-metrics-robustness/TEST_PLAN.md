# Cell Metrics Robustness & Stability Test Plan

> **Purpose.** Validate that a cell-metrics evaluation pipeline produces calibrated, unbiased results before interpreting biological findings. Tests **0–3** are **validity gates** — all must pass before proceeding. Tests **4–6** are **sensitivity diagnostics** — they characterise how much signal the data contains and set empirical ceilings for downstream metrics.

---

## Read first: unit of analysis & DE method (the single most important framing)

**The null-split tests (0, 1, 2, 4) split *cells* and run DE directly. Whether the null looks null depends entirely on the DE method's *unit of analysis*.**

- **Cell-level tests** (e.g. `pdex` = Wilcoxon rank-sum, or a t-test) treat every cell as an independent observation. Cells from the same sample/animal/batch are correlated, so this **pseudoreplication** inflates the effective *n* and intra-sample correlation alone produces "significant" genes under a true null. See **Squair et al. 2021, *Nat Commun* 12:5692** (https://www.nature.com/articles/s41467-021-25960-2). For such tests, expect the null-split p-values to be **mis-shaped** (deflated *or* inflated, λ_GC ≠ 1, non-uniform) even with zero real effect — and read the output as a **gene ranking, not a calibrated FDR**.
- **Pseudobulk / mixed-model tests** (e.g. DESeq2/edgeR/limma on replicate-level pseudobulk, or a GLMM) aggregate to the true unit of replication first. Only under these does the expected null behaviour (λ_GC ≈ 1, Uniform p-values, `frac_sig ≈ α`) actually hold.

> **The report MUST state the DE method and unit of analysis up front** and interpret every null in that light. The null-split tests **must use the same DE method as the intended real analysis** — they calibrate *that* pipeline, not DE in the abstract.

### Corrected vs uncorrected data

It is common to regress out **batch** and **cell-cycle** (and sometimes other covariates) before DE. An ideal perturbation metric would be insensitive to this; real metrics are not. **Evaluate the metric on both corrected and uncorrected data** for the major covariates and report which state each run used. State explicitly what (if anything) was regressed out of `.X`.

### Scope & validation

- This battery calibrates DE-based metrics. It deliberately **excludes gene-set / pathway enrichment** as a gate: enrichment is an unsolved problem (databases disagree; many real effects only appear under multi-gene perturbation), so it is too noisy to act as a pass/fail. It may be reported as informational only.
- Metric behaviour is partly experiment-dependent. **Validate across ≥2 datasets / modalities** (e.g. a CRISPRi Perturb-seq screen *and* a chemical/Tahoe-style screen) before trusting thresholds.

---

## Report & presentation conventions (the deliverable must be readable)

The generated report is the deliverable; a domain expert must be able to make sense of it without reverse-engineering the code. Required:

1. **Lead with dataset & experimental context** — what experiment, #cells/#genes, #perturbations, #samples/replicates/batches, cells-per-perturbation, guides-per-gene, and **which covariates are present vs absent** (cell type, cell cycle, donor/patient, sex, tissue, timepoint). Do not make the reader guess. State the **DE method, unit of analysis, and corrected/uncorrected state** in this opening section.
2. **Plain-language per test** — one to three sentences on *what the test asks and how to read it*, alongside the numbers.
3. **Local verdict reasons** — each test states its own one-line PASS/WARN/FAIL reason next to its result; do not make the reader hunt in a global flag list.
4. **Round numbers** — 4–6 significant figures (`0.1952`, not `0.19523047652509917`).
5. **Glossary** — define λ_GC, ks_p_uniform, frac_sig, separation z, TPR/FPR, reproducibility ρ, "unit of analysis".
6. **Don't bury context at the bottom** — verdict + context + per-test detail first; heavy appendices (full test plan, parameter dumps) last.
7. **Interactive-friendly output** — optionally emit **MultiQC** custom-content files (https://github.com/MultiQC/MultiQC) so the results render as an interactive HTML report that integrates with downstream single-cell pipelines. (Ideal future state: hover tooltips explaining λ_GC etc.)

---

## Inputs

### Required

| Field | Type | Description |
|---|---|---|
| `adata` | `AnnData` | Cells × genes count matrix |
| `perturbation_col` | `str` | `obs` column containing sgRNA / perturbation labels |
| `control_label` | `str` | Value in `perturbation_col` identifying control cells |

### Recommended

| Field | Type | Description |
|---|---|---|
| `target_gene_map` | `dict[str, str]` | sgRNA → target gene name |
| `batch_cols` | `list[str]` | e.g. `['batch', 'donor', 'replicate', 'cell_type']` |
| `cell_cycle_col` | `str` | `obs` column for cell-cycle phase |
| `guide_efficacy` | `dict[str, float]` | sgRNA → knockdown efficiency estimate |
| `baseline_expression` | `dict[str, float]` | Gene → baseline expression level |

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_resamples` | `10` | Resampling iterations for Tests 2 and 3 |
| `min_cells_per_group` | `20` | Minimum cells per split arm; skip condition if not met |
| `fdr_threshold` | `0.05` | BH-adjusted p-value cutoff for significance calls |
| `lfc_threshold` | `0.1` | Minimum \|LFC\| for a gene to be called significant |
| `lambda_gc_warn` | `1.05` | λ_GC threshold above which a WARN is issued |
| `lambda_gc_fail` | `1.10` | λ_GC threshold above which a FAIL is issued |

---

## Shared Notation

| Symbol | Definition |
|---|---|
| G | Number of genes tested |
| p(g) | Raw p-value for gene g |
| q(g) | BH-adjusted p-value for gene g |
| LFC(g) | Log₂ fold change for gene g |
| S | Set of significant genes: `{g : q(g) ≤ fdr_threshold AND \|LFC(g)\| ≥ lfc_threshold}` |
| F̂(x) | Empirical CDF of `{p(g)}_g` evaluated at x |
| χ²₁⁻¹(p) | Inverse CDF of the χ²(1) distribution at probability p |

---

## Shared Formulas

All six tests report the following metrics where DE results are available.

### Fraction Significant

```
frac_sig = |S| / G
```

Reported as the mean over conditions. Approximates the empirical false-positive rate under a true null. Expected value ≈ `fdr_threshold` when the null holds.

### Mean and Mean Absolute LFC

```
mean_lfc     = (1/G) Σ_g LFC(g)
mean_abs_lfc = (1/G) Σ_g |LFC(g)|
```

`mean_lfc` tests for **directional bias**; it can cancel to zero even with large symmetric LFCs.
`mean_abs_lfc` tests for **magnitude inflation**. Both must be reported together.

### KS Uniformity Statistic

```
D = sup_x | F̂(x) − x |
```

`F̂` is the empirical CDF of `{p(g)}_g`. The reported `ks_p_uniform` is the two-sided KS p-value against Uniform[0,1]. A small `ks_p_uniform` (< 0.05) indicates departure from the expected null p-value distribution.

### Genomic Inflation Factor

```
λ_GC = median_g [ χ²₁⁻¹(1 − p(g)) ] / χ²₁⁻¹(0.5)
```

where `χ²₁⁻¹(0.5) = 0.4549` (the median of a χ²(1) distribution).

| λ_GC range | Interpretation |
|---|---|
| ≈ 1.00 | Well-calibrated |
| < 1.00 | Conservative / deflated |
| 1.00–1.05 | Acceptable |
| 1.05–1.10 | **WARN** — mild inflation |
| > 1.10 | **FAIL** — anti-conservative; do not proceed |

### QQ-Plot Envelope

Plot sorted observed `−log10 p₍ᵢ₎` (i = 1…G) against expected `−log10((i − 0.5)/G)`.

The 95% pointwise confidence envelope uses the exact Beta order-statistic distribution:

```
lower visual bound = −log10[ B₀.₉₇₅(i, G−i+1) ]
upper visual bound = −log10[ B₀.₀₂₅(i, G−i+1) ]
```

where `Bₚ(α, β)` denotes the p-th quantile of the Beta(α, β) distribution. **Note:** the −log10 transform reverses the interval, so the lower Beta quantile (`B₀.₀₂₅`) maps to the upper visual bound and vice versa. Points lying on y = x indicate a well-calibrated test. This envelope is exact (not a normal approximation) because the i-th order statistic of Uniform[0,1] follows exactly Beta(i, G−i+1).

### Wald Test Statistic

The recommended DE test for count data is the **DESeq2 Wald test**:

```
W = β̂ / SE(β̂)
```

where `β̂` is the MLE log₂ fold change estimated from the negative binomial model, and `SE(β̂)` is derived from the Fisher Information at the MLE:

```
SE(β̂) = sqrt[ I(β̂)⁻¹ ]
```

`W` is the signal-to-noise ratio of the effect estimate: how many standard errors the estimated LFC sits from zero. Under H₀ (β = 0), asymptotic normality of the MLE gives `W ~ N(0, 1)`. The p-value is `p = 2Φ(−|W|)` where Φ is the standard normal CDF.

**Why the Wald test:** MLEs are asymptotically normally distributed, so dividing the estimate by its standard error yields a statistic with a known null distribution. DESeq2 additionally applies empirical Bayes shrinkage to both `β̂` and dispersions to stabilise estimates in the low-count, high-noise regime typical of RNA-seq.

**Design formula:** always include blocking variables as covariates:

```
~ replicate + batch + donor + split_label
```

This ensures the split coefficient is tested cleanly. Omitting covariates can produce false inflation even under a true null.

**Alternative tests (use whichever matches the real analysis):**

| Test | Statistic | Notes |
|---|---|---|
| edgeR QLF | F-statistic | Quasi-likelihood; more conservative; good for small n |
| limma-voom | moderated t | Best for normalised log-CPM input |
| Wilcoxon rank-sum | W | Non-parametric; ignores replicate structure; use only if others unavailable |

> **Critical:** the null split tests must use the **same DE method** as the real analysis. They are a calibration check on the pipeline, not a standalone test.

---

## Test 0 — Controlled Effect-Size Injection / Calibration Curve

> **In one sentence.** Spike a known log2 fold-change into a **small number of genes (~a dozen — the scale of a real perturbation)** in a control-vs-control split, pooled over several random gene-draws, then measure how many injected genes are recovered (TPR) and how many untouched genes are wrongly called (FPR) across effect sizes — separating "null = null" from "pipeline is dead" and quantifying the smallest effect the metric can resolve.

> **Why a small NUMBER, not a fraction.** Library-size renormalization redistributes counts: if you perturb a large *fraction* of genes the per-cell total shifts, so untouched genes get renormalized and acquire spurious DE (compositional coupling). A real perturbation moves only ~a dozen genes, which barely moves the library size — so injecting a small fixed **number** is the faithful test, and the legacy "fraction of genes" mode is an artificial worst case that unfairly penalizes library-size-normalized methods (e.g. cell-level Wilcoxon). To keep the TPR/FPR rates stable despite few genes per draw, repeat the injection over `injection_n_repeats` independent random draws and **pool the per-gene counts**. (The fraction mode is retained as an explicit large-fraction *stress* test via `injection_n_genes: null` + `injection_frac_genes`.)

### Question

When a real effect of known size is present, does the pipeline detect it (and only it)? And under no effect, is the false-positive rate controlled? A null that produces nothing is indistinguishable from a broken pipeline unless you also show it *can* detect a planted signal.

### Design

On control cells only (so the only DE is what we inject):

1. Stratified-split control cells into reference arm A and pseudo-perturbation arm B (as in Test 2), **once**. Cap cells/arm for speed if needed.
2. Repeat over `injection_n_repeats` independent random draws (pooled). On each draw, pick a **small fixed number** `injection_n_genes` (e.g. **12** — the scale of a real perturbation) of genes as the **anchors**, **stratified across the expression (detection) spectrum** — equal shares from the **low / mid / high mean-expression tertiles**. Detectability is dominated by expression level, so injecting evenly across tiers makes the TPR-vs-δ curve interpretable **per tier** (the smallest resolvable δ for sparse vs typical vs abundant genes) instead of an artifact of the injected mix. Do NOT inject a large *fraction* of genes (artificially induces compositional coupling — see above), nor only housekeeping/marker/anchor-correlated genes (biased, circular), nor only HEG (optimistic) / only LEG (pessimistic).
3. For each effect size `δ` in `injection_deltas` (log2 fold-change tiers; **include `δ = 0` as the FPR-at-null baseline**), multiply the **raw counts** of the anchor genes in arm B by `2^δ`, then normalize (`normalize_total` + `log1p`) and run the **same DE method** as the real analysis, B vs A.
4. **Pool the per-gene significance counts across all draws** and record, per `δ`: **TPR** = fraction of injected/anchor genes called significant; **FPR** = fraction of *untouched* genes called significant; the median observed LFC of anchors; and λ_GC (over pooled p-values).
5. **Break the injected-gene TPR down by EXPRESSION TIER** (low/mid/high) — report the min resolvable δ (TPR≥0.5) per tier — and **break the untouched-gene FPR down by gene class**: **AnchorCorr** (genes most correlated with the anchor signature — easiest/upper-bound, most exposed to coupling), **HighlyExpr** (abundant — high signal), **LowlyExpr** (sparse — hard, zero-dominated), **HouseKeeping** (constitutive — should be predictable; watch memorization), **Marker** (cell-type identity — N/A without a cell-type annotation), **HighlyVarG** (high-variance complement to the anchors), **Random** (unbiased baseline). All expression tiers and gene-class memberships must be computed **deterministically over all control cells** — a fixed function of the h5ad, independent of the DE cell subsample.

> **Determinism.** Anchor selection, gene-class assignment, and the cell subsample are all seeded; given the same h5ad + seed the calibration curve (and every per-class FPR) is bit-for-bit reproducible.

### Reported metrics

| Metric | Meaning |
|---|---|
| `null_FPR` | FPR at `δ = 0` — the empirical false-positive rate under a true null. Should be ≈ `fdr_threshold`. |
| `max_TPR` | Highest TPR across the injected tiers — can the metric resolve a strong effect at all? |
| `min_resolvable_delta` | Smallest `δ` reaching TPR ≥ 0.5 — the effect size the metric can actually detect. |
| `FPR_at_max_delta` | FPR among untouched genes at the largest `δ`. If this rises far above `null_FPR`, strong perturbation of a few genes is inducing **false DE in untouched genes via library-size renormalization** — a **compositional coupling** artifact (related to the composition confounder; mitigate with median-of-ratios / spike-in normalization or pseudobulk). |

**The report MUST show the full per-`δ` calibration curve inline** — one row per `δ` (including
`δ=0`) with `TPR` (injected genes), `FPR` (untouched genes), median observed LFC, and `λ_GC` — and
write it to `tables/test_0__calibration_curve.csv`. The per-tier ramp (e.g. FPR/λ_GC rising sharply
with `δ`) is the headline evidence and must not be hidden behind summary scalars or a plot alone.

### Verdict Rules

| Condition | Verdict |
|---|---|
| `null_FPR` controlled (≤ ~2×`fdr_threshold`) and TPR rises with δ to ≥ 0.5 | **PASS** |
| `null_FPR` clean but `FPR_at_max_delta` ≫ `null_FPR` (compositional coupling), or `max_TPR < 0.5` (under-powered) | **WARN** |
| `null_FPR` > 2×`fdr_threshold` (anti-conservative) | **FAIL** |

---

## Test 1 — Within-Condition Reproducibility

> **In one sentence.** Split each perturbation's cells into two halves and compare each half to control (`DE_A`, `DE_B`) — both carry the *real* perturbation signal (NOT a uniform null; that's Test 2) — and measure whether the two independent half-signatures **agree** (`DE_A ≈ `DE_B`); high split-half LFC correlation = a reproducible signature and the empirical ceiling for downstream metrics.

### Question

When a perturbation's cells are split in half and each half is profiled against control, do the two halves recover the **same** differential-expression signature?

### Design

For each perturbation with ≥ `2 × min_cells_per_group` cells:

- Split the perturbed cells into A and B (stratified by `batch_cols`); split the **control** cells into two halves `ctrl_A`, `ctrl_B` (each = half of all controls); compute `DE_A = A vs ctrl_A` and `DE_B = B vs ctrl_B` with cell-eval's DE. **No 1:1 cell matching** — each half-perturbation is compared to an independent half of the controls.

Two statistics:
1. **(PRIMARY) reproducibility** — agreement of `DE_A` and `DE_B`: **median split-half Spearman LFC ρ** (+ DEG Jaccard, direction agreement), over perturbations.
2. **(secondary) difference-is-null** — the direct `A_pert vs B_pert` contrast (same perturbation ⇒ should be ≈null): `frac_sig` / `λ_GC` / `ks_p_uniform`, as a sanity check.

Average over `test1_n_resamples` seeded splits.

**Reproducibility vs cell count (method vs undersampling).** Splitting a condition into halves starves small conditions of cells, so low ρ could reflect *undersampling* rather than a poor method. To separate the two, plot **one point per perturbation** — its total cell count (x, log) vs its median split-half ρ (y) — and report `Spearman(cell count, ρ)` plus the median ρ for **well-powered** (≥ `test1_wellpowered_min_cells`, default 200 total ≈ ≥100 per half) vs **under-powered** perturbations. Interpretation: points that **climb with cell count and plateau** ⇒ the low scores are power-limited; points that **stay low even at high cell counts** (well-powered ρ ≈ overall) ⇒ a **genuine method limitation**. Written to `tables/test_1__repro_vs_ncells_no_match.csv` and `plots/test_1_undersampling.png`. (Caveat: in datasets where almost all perturbations are already well-powered there is little dynamic range; a *controlled downsampling* sweep is the way to map the full power curve.)

### Test Statistic

`ρ = Spearman(LFC_A, LFC_B)` over genes, per perturbation; the test reports the **median ρ across perturbations**. (The secondary difference-is-null uses the same null statistics as Test 2 on the `A_pert`-vs-`B_pert` contrast.)

### Expected Behaviour

| Perturbation strength | median split-half LFC ρ | direction agreement |
|---|---|---|
| Strong / reproducible | > 0.6 | > 0.8 |
| Moderate | 0.3–0.6 | 0.6–0.8 |
| Low / non-reproducible | < 0.3 | ≈ 0.5 |

The secondary difference-is-null should be ≈null (λ_GC≈1, p uniform) by construction.

### Verdict Rules

Applied to the **median split-half Spearman LFC ρ** across perturbations (the verdict driver).

| Condition | Verdict |
|---|---|
| median ρ > 0.6 | **PASS** — strong, reproducible signatures |
| median ρ ∈ [0.3, 0.6] | **WARN** — moderate reproducibility (downstream metrics bounded by this) |
| median ρ < 0.3 | **FAIL** — low; the two half-signatures disagree, so the data cannot support reliable downstream comparisons |

Report the result, the thresholds, and the per-perturbation tier counts (how many strong / moderate / low) so the ceiling is explicit.

### Failure Diagnostics

| Pattern | Likely Cause |
|---|---|
| p-value histogram spikes near 0 | Unabsorbed batch structure leaking into split |
| LFCs skewed in one direction | Normalization bias or unbalanced split |
| QQ inflation (points above diagonal) | Missing covariates; overdispersion underestimated |
| QQ deflation (points below diagonal) | Overcorrection; too many covariates |

---

## Test 2 — Control-Control Split Null

> **In one sentence.** Same as Test 1 but splitting only control cells, so any detected signal is pure noise/batch — this confirms the null; uncorrected batch structure leaks in here.

### Question

Do metrics remain near null when pseudo-perturbations are created by splitting control cells?

### Design

1. Take all cells where `perturbation_col == control_label`.
2. Perform `n_resamples` independent **stratified** splits into groups A and B (random A/B balanced within `batch_cols`). No 1:1 control-to-control matching mode: both arms are the *same* population (controls), so a stratified split already balances batch/depth and matching controls-to-controls is uninformative. (The batch-controlled 1:1-matched design belongs to **Test 1**, where two *different* populations — perturbed vs control — are compared.)
3. In each split, treat group A as a pseudo-perturbation and group B as the reference.
4. Aggregate the DE null-metrics: mean ± SD across splits.

### Test Statistic

Same Wald test as Test 1; the DE null-metrics (`frac_sig`, `mean_lfc`, `mean_abs_lfc`, `ks_p_uniform`, `λ_GC`) should all collapse to null.

### Expected Behaviour

| Metric | Expected |
|---|---|
| `frac_sig` | ≈ `fdr_threshold` |
| `mean_lfc`, `mean_abs_lfc` | Near zero |
| `ks_p_uniform` | > 0.05 |
| `λ_GC` | ≈ 1.00 |
| Split-to-split SD of `λ_GC` | Low (< 0.05) |
| QQ-plot | Points hug y = x diagonal within envelope; consistent across splits |

### Verdict Rules

Same thresholds as Test 1, applied to the mean over splits.

---

## Test 3 — Label Permutation Null

> **In one sentence.** Shuffle the perturbation labels (within batch) and recompute; real biological signal should sit far outside the shuffled distribution (high separation z), otherwise the metric cannot tell signal from noise.

### Question

Do metrics collapse when perturbation labels are broken?

### Design

1. Compute all metrics under true perturbation labels → `true_metrics`.
2. For each of `n_resamples` iterations:
   a. Shuffle `perturbation_col` values within each stratum defined by `batch_cols`.
   b. Recompute DE → `perm_metrics[i]`.
3. Compute separation score between true and permuted distributions.

### Cell-count-stratified p-value diagnostic

In addition to the scalar separation score, collect the **per-gene p-values per perturbation** for both the real (unshuffled) DE and every shuffled DE, together with each perturbation's cell count, and write them tidily to `tables/test_3__pvalues_by_cellcount.csv` (per-perturbation, per-kind ∈ {real, shuffled}: `n_cells`, `cellcount_stratum`, `frac_sig`, `frac_p_lt_05`, `lambda_gc`, `ks_p_uniform`) plus a deterministically subsampled pooled vector in `tables/test_3__pooled_pvalues.csv`. The diagnostic plot `plots/test_3_pvalue_diagnostics.png` has four panels:

- **(A) pooled p-value ECDF** (real vs shuffled): real should bow **above** the diagonal (small-p excess = signal); shuffled should track the diagonal (Uniform[0,1] = calibrated null).
- **(B) QQ vs Uniform** (−log10): real points rise above y=x; shuffled hug it (or rise above it if anti-conservative even with no signal).
- **(C) per-perturbation fraction of genes with p<0.05 vs cell count** (log-x): real well above shuffled; shuffled near the 0.05 reference at all cell counts.
- **(D) p-value ECDF faceted by cell-count tertile** (configurable via `test3_n_cellcount_strata`): checks that the shuffled null stays Uniform and the real small-p excess holds for **small- vs large-cell-count** perturbations.

All subsampling is seeded (`test3_pooled_pval_subsample` cap) so the diagnostic is bit-for-bit reproducible.

### Separation Score

```
separation = (true_metric − mean(perm_metrics)) / SD(perm_metrics)
```

This is equivalent to a z-score of the true metric relative to the permutation null. Also compute an empirical p-value:

```
perm_p = |{i : perm_metrics[i] ≥ true_metric}| / n_resamples
```

### Expected Behaviour

| Metric | Expected |
|---|---|
| Permuted `frac_sig`, `λ_GC` | Indistinguishable from Test 1/2 null |
| Separation score for strong metrics | > 2.0 (true signal clearly outside null) |
| `perm_p` for strong metrics | < 0.05 |
| Real pooled p-values | small-p excess (ECDF above diagonal / QQ above line); frac(p<0.05) ≫ 0.05 |
| Shuffled pooled p-values | ≈ Uniform[0,1]: frac(p<0.05) ≈ 0.05, λ_GC ≈ 1, across all cell-count strata |

### Verdict Rules

| Condition | Verdict |
|---|---|
| True metrics sit clearly outside permutation null (separation > 2) | **PASS** |
| Separation borderline (1 < separation ≤ 2) | **WARN** |
| True metrics indistinguishable from permuted (separation ≤ 1) | **FAIL** — metrics cannot detect real signal |

---

## Test 4 — Same-sgRNA Split Reproducibility (guide-level Test 1)

> **In one sentence.** **Test 1's within-condition reproducibility design run at the sgRNA (guide) level** — split each guide's cells in two (controls also split) and compare each half vs control; agreement between the halves is the **reproducibility ceiling** at the resolution per-guide downstream metrics operate, and no model can score higher than the data agrees with itself.

### Question

If a single sgRNA's cells are split into two halves and each is run vs control, are the resulting DE signatures reproducible?

### Design

Identical to Test 1, but the **unit is one guide** rather than a perturbation/gene. For each sgRNA with ≥ `2 × min_cells_per_group` cells, split its cells into halves A/B (stratified by `block_cols`), split the controls into halves, run `DE_A = A vs ctrl_half_A` and `DE_B = B vs ctrl_half_B` (**no 1:1 cell matching**) plus the difference-is-null `DE_AB = A vs B`, repeated for `test1_n_resamples` draws (each guide's reported value is the median over draws).

### Reproducibility Metrics

Same as Test 1: Spearman LFC ρ over **all genes** (primary) and over **DE genes**, DEG-set Jaccard, and directional agreement (see Test 1 §Test Statistic). Also reported, exactly as in Test 1: ρ-vs-cell-count **per guide** (separates a low-reproducibility method from undersampled guides) and the secondary `A`-vs-`B` difference-is-null QQ. Emits the **same two plots** as Test 1 (`plots/test_4_reproducibility.png`, `plots/test_4_undersampling.png`).

### Expected Behaviour & Verdict Rules

| Strength | LFC ρ | DEG Jaccard | Directional agreement | Verdict (on median split-half LFC ρ) |
|---|---|---|---|---|
| Strong | > 0.6 | > 0.3 | > 0.7 | **PASS** |
| Moderate | 0.3–0.6 | 0.1–0.3 | 0.6–0.7 | **WARN** |
| Weak | < 0.3 | < 0.1 | ≈ 0.5 (chance) | **FAIL** |

> **Note:** Same tiers as Test 1 (harmonised). This test sets the **empirical ceiling** for any per-guide downstream metric — a guide with low reproducibility here cannot be expected to score well on cross-model or cross-condition comparisons.

---

## Test 5 — Same-Gene Independent sgRNA Reproducibility

> **In one sentence.** Do two different guides for the same gene agree more than two unrelated guides? **Expect only modest concordance** — guides for one gene often differ substantially in knockdown efficacy — and treat a low score with few guide pairs as **power-limited, not proof of off-target activity.**

### Question

Do independent sgRNAs targeting the same gene recover similar perturbation signatures?

### Design

For each target gene G with ≥ 2 sgRNAs:

1. Compute DE for each sgRNA versus control.
2. For all same-gene guide pairs (i, j): compute reproducibility metrics from Test 4.
3. For matched unrelated guide pairs (random pairs from different target genes): compute the same metrics → background distribution.
4. Optionally run leave-one-guide-out (LOGO) analysis:
   - Reference signature: consensus of all sgRNAs targeting G except guide i.
   - Query signature: guide i alone.
   - Compare LOGO scores against same-gene pair scores.

### Primary statistic — AUC (probability of superiority), gated by Mann-Whitney significance

The primary metric is the **DEG-restricted LFC ρ** (the all-genes ρ is diluted to ≈0 by thousands of non-responsive genes for *both* same-gene and background pairs, so it cannot discriminate gene-specific biology — the signal lives in the DE genes). The headline separation statistic is the **Mann-Whitney common-language effect size**:

```
AUC = U / (n_same_gene_pairs · n_background_pairs)
    = P(a random same-gene pair is more concordant than a random unrelated pair)
```

`AUC = 0.5` ⇒ no separation; `1.0` ⇒ perfect. AUC is **rank-based** (robust to the wide, skewed background distribution that essential-gene screens produce) and **not inflated by sample size** (unlike a raw p-value). It is the effect-size companion to the one-sided **Mann-Whitney U p-value** (significance that same-gene > background), so both are reported: AUC for magnitude, p for significance. The **separation z = (mean_same − mean_bg)/SD(bg)** is still reported as a secondary number, but it is **not** the verdict driver — it is variance-sensitive and understates a real separation whenever the background spread is large.

### Expected Behaviour

| Metric | Expected |
|---|---|
| AUC P(same-gene > unrelated), DEG LFC ρ | > 0.5, ideally ≥ 0.65 for clear discrimination |
| Same-gene DEG LFC ρ / Jaccard | Higher than background (Mann-Whitney p significant) |
| Separation z (secondary) | Higher is better, but deflated by a wide background |

### Verdict Rules

| Condition | Verdict |
|---|---|
| **Underpowered: < ~5 genes with ≥2 guides (< ~5 same-gene pairs)** | **WARN (uninformative — cannot conclude)** — too few pairs to carry statistical weight; **never FAIL.** Do NOT read a low score as "guides off-target": same-gene guides genuinely differ in knockdown efficacy, so modest concordance is expected, and with few pairs neither AUC nor the separation z is meaningful. |
| **AUC ≥ 0.65 and Mann-Whitney p < α** (default 0.05) | **PASS** — same-gene pairs clearly out-concord unrelated pairs |
| **AUC ≥ 0.55 and Mann-Whitney p < α** | **WARN** — a real but **modest** separation (the metric rewards shared biology only weakly; expected when single-guide signatures are noisy) |
| **AUC ≈ 0.5 or not significant** (with enough pairs to conclude) | **FAIL** — guides for the same gene are indistinguishable from unrelated guides (possible off-target / inefficacy, or the metric does not capture the shared biology) |

> **Note (effect size vs significance).** With many pairs the Mann-Whitney p can be tiny even when AUC is only ~0.6 — a *real but modest* separation. Reporting AUC alongside p prevents reading "p ≪ 0.05" as a strong result. Conversely a high AUC with a non-significant p (few pairs) is the underpowered case above.

---

## Test 6 — Target Gene Knockdown Recovery

> **In one sentence.** Is the targeted gene itself knocked down (negative LFC, significant)? High recovery is reassuring, but it **partly measures assay and guide quality, not the metric alone.**

### Question

Is the target gene itself detected as differentially expressed in the expected direction?

### Design

For each sgRNA with a known target gene G (from `target_gene_map`):

1. Run DE: perturbed cells vs control.
2. Extract result for gene G.
3. Record the following per-sgRNA outputs.

### Per-sgRNA Output Fields

| Field | Formula / Source | Description |
|---|---|---|
| `lfc_target` | `LFC(G)` from DE | Log₂ fold change of target gene |
| `padj_target` | `q(G)` from DE | BH-adjusted p-value of target gene |
| `pval_rank` | `rank(p(g))[G] / G` | Percentile rank of target gene by raw p-value (lower = more significant) |
| `lfc_rank` | `rank(\|LFC(g)\|)[G] / G` | Percentile rank by absolute LFC |
| `correct_direction` | `LFC(G) < 0` for CRISPRi/KO | Whether LFC is in the expected direction |
| `detected` | `q(G) ≤ fdr_threshold` | Binary: is target gene significant? |

### Aggregate Recovery Metrics

Computed over all sgRNAs with a mapped target gene:

```
recovery_rate       = |{sgRNAs : detected = True}| / N_sgrnas
direction_rate      = |{sgRNAs : correct_direction = True}| / N_sgrnas
median_pval_rank    = median over sgRNAs of pval_rank
median_lfc_rank     = median over sgRNAs of lfc_rank
```

### Expected Behaviour

| Metric | Expected (good data) | Warning threshold |
|---|---|---|
| `recovery_rate` | > 0.5 | < 0.2 |
| `direction_rate` | > 0.8 (CRISPRi/KO) | < 0.6 |
| `median_pval_rank` | < 0.10 (top 10% by p-value) | > 0.25 |
| `median_lfc_rank` | < 0.15 | > 0.30 |

> **Note for CRISPR KO:** reduced transcript abundance is expected for many genes but is not guaranteed due to nonsense-mediated decay variability. `direction_rate` thresholds may need adjustment based on the gene and perturbation system.

### Verdict Rules

| Condition | Verdict |
|---|---|
| `recovery_rate` > 0.5 and `direction_rate` > 0.8 | **PASS** |
| `recovery_rate` ∈ [0.2, 0.5] or `direction_rate` ∈ [0.6, 0.8] | **WARN** — check guide efficacy estimates |
| `recovery_rate` < 0.2 or `direction_rate` < 0.6 | **FAIL** — poor guide efficacy or expression artefact |

---

## Output Schema

```python
@dataclass
class TestResult:
    verdict:  Literal["PASS", "WARN", "FAIL", "SKIP"]
    metrics:  dict[str, float]       # all numeric outputs for this test
    flags:    list[str]              # human-readable failure reasons
    reason:   str                    # one-line LOCAL verdict reason (shown next to the result)
    details:  dict                   # per-condition or per-sgRNA breakdown

@dataclass
class RobustnessReport:
    test0: TestResult   # Effect-size injection / calibration curve   (GATE)
    test1: TestResult   # Within-condition null                       (GATE)
    test2: TestResult   # Control-control null                        (GATE)
    test3: TestResult   # Label permutation                          (GATE)
    test4: TestResult   # Same-sgRNA reproducibility
    test5: TestResult   # Cross-sgRNA reproducibility
    test6: TestResult   # Target knockdown recovery

    global_verdict: Literal["PASS", "WARN", "FAIL"]
    blocking_flags: list[str]   # reasons for FAIL at global level
    info_flags:     list[str]   # non-blocking WARNs
```

---

## Global Verdict Logic

Validity gates are Tests **0, 1, 2, 3**.

| Rule | Global Verdict |
|---|---|
| Tests 0–3 all PASS | Proceed to Tests 4–6 |
| Any of Tests 0–3 is FAIL (e.g. anti-conservative null, or injected-effect FPR > 2×α) | **FAIL** — do not interpret real results; fix pipeline first |
| Any of Tests 0–3 WARN (e.g. deflated/under-powered cell-level null, compositional coupling) | **WARN** — results usable but read the caveats; with a cell-level test, treat output as ranking, not calibrated FDR |
| Gates PASS/WARN but Test 6 WARN | **WARN** — results valid but guide efficacy may limit sensitivity |
| Tests 4–5 show no separation from background | **WARN** — empirical ceiling is low; downstream metrics may not be informative |

---

## Implementation Notes

1. **Shared split function.** Tests 1, 2, and 4 all require stratified A/B splits. Implement a single `stratified_split(cells, stratify_cols, min_per_group)` utility used by all three.

2. **Shared reproducibility function.** Tests 4 and 5 both compare two DE result objects. Implement a single `compare_signatures(de_A, de_B) -> ReproducibilityMetrics` used by both.

3. **SKIP conditions.** If a condition has fewer than `2 × min_cells_per_group` cells, mark that condition as SKIP and exclude from aggregation. If > 50% of conditions are SKIP, escalate to WARN.

4. **Consistency requirement.** All six tests must use the same DE function as the real analysis pipeline. Pass the DE function as a callable argument to each test runner.

5. **Determinism (REQUIRED — completely seeded).** The entire run must be **bit-for-bit reproducible**: the same dataset + `config.yaml` + `seed` must produce an identical report and identical per-test tables every time. A single `seed` drives **every** stochastic step — all stratified splits, label permutations, and the Test-0 injection gene-selection/subsampling — via `np.random.default_rng(seed + fixed_offset)` (never an unseeded `default_rng()` / global `np.random`). Call a `seed_everything(seed)` at startup that pins `PYTHONHASHSEED`, `random`, `np.random`, and `scanpy.settings.seed`, and pass `random_state=seed` to any PCA/HVG/neighbors. The DE backends are deterministic given fixed input (pdex Wilcoxon; pydeseq2 DESeq2). Validate by running twice and diffing `robustness_summary.json` / `tables/` — they must be identical.

