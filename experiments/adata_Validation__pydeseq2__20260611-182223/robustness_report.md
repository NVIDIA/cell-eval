# Robustness report — adata_Validation__pydeseq2__20260611-182223

## Global verdict: ❌ **FAIL**

Validity gates (Tests 0–3) must hold before the sensitivity diagnostics (Tests 4–6) are interpretable.

- ✅ **test_0** — PASS: null FPR=0 controlled; resolves δ≥0.5 log2FC
- ❌ **test_1** — FAIL [matched]: median split-half LFC ρ=0.2163 (low reproducibility; PASS>0.6/WARN≥0.3/FAIL<0.3); Jaccard=0.4539, direction=0.97; (2°) difference-is-null λ_GC=0.3787, frac_sig=1.24e-06
- ⚠️ **test_2** — WARN: deflated/under-powered null (λ_GC=0.4727<0.9, ks_p_uniform=1.06e-276); no false positives but p-values are not calibrated
- ✅ **test_3** — PASS: true signal 0.1526 vs shuffled 0.0133 (separation z=245.403, perm_p=0)

> ❌ A validity gate FAILED — **do not interpret the biology** until the pipeline is fixed. Failure mode(s): non-uniform null p-values (test_1); deflated/under-powered cell-level null (test_2: λ_GC=0.4727).

## 1. Dataset & experimental context

- **file**: `/home/yangzhang/code/cell-eval/adata_Validation.h5ad`
- **size**: 98,927 cells × 18,080 genes; layers=none
- **perturbations**: 50 + control `non-targeting` (38,176 control cells)
- **cells per perturbation**: min 161, median 1090, max 2925
- **guides**: 85 total; **4 gene(s) have ≥2 guides** (limits Test 5)
- **obs columns**: `target_gene`(n=51), `guide_id`(n=85), `batch`(n=48)
- **.X**: raw-integer=True, min/max/mean=1/1620/6.5528

- **structural covariates present**: only the columns above
- **NOT available in this dataset** (assumed absent, interpret accordingly): cell type / cluster, cell-cycle phase, donor, patient, sex, tissue, timepoint

## 2. How DE was computed

- **DE backend**: `pydeseq2`
- **unit of analysis**: **pseudobulk replicate / sample (DESeq2 Wald)**
- **covariate correction of .X**: **none** (this run evaluates the metric on this state of the data)
- **input scaling**: normalize_if_raw=False, allow_discrete=True (pdex is_log1p=False)
- **stratification / blocking**: ['batch']
- **scale**: full (all conditions), n_resamples=10, seed=0

> **Unit-of-analysis caveat.** Counts are aggregated to replicate-level pseudobulk samples before testing, which respects the true unit of replication and avoids pseudoreplication.

> **Correction note.** Many groups regress out batch / cell-cycle before DE. An ideal perturbation metric is insensitive to this, but real ones are not — evaluate on **both corrected and uncorrected** data. This run is **none**; set `covariate_correction` and re-run to compare.

## 3. At-a-glance

**What each test is for** (validity gates 0–3 must hold before the sensitivity diagnostics 4–6 are interpretable):

| test | role | the question it answers |
|---|---|---|
| test_0 | gate — calibration + power | Inject a known effect: what false-positive rate at the null, and what effect size can the metric resolve (TPR vs δ)? |
| test_1 | **reproducibility** | Split a perturbation in half vs control — do the two half-signatures agree (DE_A≈DE_B)? Is it improved by 1:1 batch-matched controls? |
| test_2 | gate — null calibration | Split control cells into A/B (stratified, true null) — are the p-values Uniform[0,1] (λ_GC≈1, frac_sig≈α)? |
| test_3 | gate — signal vs noise | Shuffle perturbation labels — is real signal far outside the permuted null (separation z)? |
| test_4 | sensitivity — ceiling | Same-guide split reproducibility — the empirical ceiling for any downstream metric. |
| test_5 | sensitivity | Do independent guides for the same gene agree more than unrelated guides? (power-limited with few guides) |
| test_6 | sensitivity | Is the targeted gene itself knocked down in the right direction? (also reflects assay/guide quality) |
| composition | confounder | Do perturbations shift cell-state proportions vs control? (needs a cell-type column) |

| test | verdict | what it tells you (local reason) |
|---|---|---|
| **test_0** — Effect-Size Injection / Calibration Curve | ✅ PASS | PASS: null FPR=0 controlled; resolves δ≥0.5 log2FC |
| **test_1** — Within-Condition Reproducibility | ❌ FAIL | FAIL [matched]: median split-half LFC ρ=0.2163 (low reproducibility; PASS>0.6/WARN≥0.3/FAIL<0.3); Jaccard=0.4539, direction=0.97; (2°) difference-is-null λ_GC=0.3787, frac_sig=1.24e-06 |
| **test_2** — Control-Control Split Null | ⚠️ WARN | WARN: deflated/under-powered null (λ_GC=0.4727<0.9, ks_p_uniform=1.06e-276); no false positives but p-values are not calibrated |
| **test_3** — Label Permutation Null | ✅ PASS | PASS: true signal 0.1526 vs shuffled 0.0133 (separation z=245.403, perm_p=0) |
| **test_4** — Same-sgRNA Split Reproducibility | ✅ PASS | PASS: same-sgRNA reproducibility ceiling — median LFC ρ=0.7378 (strong), Jaccard=0.7454 (strong), direction=0.9652 (strong) |
| **test_5** — Same-Gene Independent sgRNA Reproducibility | ⚠️ WARN | WARN (underpowered, cannot conclude): same-gene ρ=0.569 trends higher than background ρ=0.5123 on only 4 pair(s) |
| **test_6** — Target Gene Knockdown Recovery | ✅ PASS | PASS: recovery_rate=1, direction_rate=1 over 50 target genes (one DE per perturbation/gene, not per guide; also reflects assay/guide quality, not the metric alone) |
| **composition** — Composition Control | ⏭️ SKIP | SKIP: no cell-state column to assess composition shift |

_This verdict table is also exported as a **MultiQC** custom-content file (`multiqc/cell_eval_robustness_mqc.txt`) — run `multiqc .` in this folder to render an interactive HTML report that integrates with downstream single-cell pipelines._

## 4. Per-test detail

### test_0 — Effect-Size Injection / Calibration Curve  ✅ PASS

*What this tests.* Spike a known log2 fold-change into a known fraction of genes in a control-vs-control split, then measure how many injected genes we recover (TPR) and how many untouched genes are wrongly called (FPR) across effect sizes. This separates a genuinely calibrated null from a pipeline that simply finds nothing, and tells you the smallest effect the metric can resolve.

**Tiers** — null FPR (δ=0): ≤ α PASS · > 2α FAIL (anti-conservative). max TPR across δ: > 0.5 = resolves effects · < 0.5 WARN (under-powered). FPR-by-class at high δ: ≈ null FPR = clean · ≫ null FPR = compositional coupling (worst in HighlyExpr/AnchorCorr, spares LowlyExpr).

*Injection design (ground truth).* An **arm** is one side of the two-group split that differential expression compares (term borrowed from clinical trials: a reference arm vs a treatment arm). Here **both arms are control cells** (so they are truly identical) — **arm A** is the reference and **arm B** is the pretend-perturbation that receives the injected effect; DE is run as *B vs A*.

From the **38,176 `non-targeting` control cells**, a random ~6,000 were subsampled and split into the two arms of **~2,988 cells each**, stratified within `['batch']` (the per-arm count is capped at `injection_max_cells_per_arm`=3000 for speed/memory — DE is re-run for every δ tier — and is slightly under the cap because the split halves each batch with integer rounding; raise the cap to use more of the available control cells). **1,782 genes (10% of the 17,841 expressed genes; 18,080 genes total)** were chosen as the *injected* (**anchor**) set, **stratified across the expression (detection) spectrum** — equal shares from the low / mid / high mean-expression tertiles (**594 low · 594 mid · 594 high**) — because detectability is dominated by expression level, so this makes the TPR-vs-δ curve readable *per tier* (the smallest resolvable δ for sparse vs typical vs abundant genes) rather than an artifact of the mix. In arm B their raw counts were multiplied by 2^δ for each effect size **δ ∈ [0.0, 0.25, 0.5, 1.0, 2.0]** (log2 fold-change; δ=0 = untouched null). The other 16,059 expressed genes are the *untouched* set used to measure the false-positive rate. TPR is recovery of the injected/anchor set (reported per expression tier); FPR is false calls among the untouched set (reported per gene class).

*Gene-class FPR breakdown.* The untouched-gene FPR is broken down into gene classes — **AnchorCorr** (genes most correlated with the injected anchors — the easiest/upper-bound case and the one most exposed to compositional coupling), **HighlyExpr** (abundant, high signal), **LowlyExpr** (sparse, zero-dominated, hard), **HouseKeeping** (constitutive — should be predictable; watch for memorization), **Marker** (cell-type identity genes — the biologically interesting case; N/A without a cell-type annotation), **HighlyVarG** (high-variance complement to the anchors), and **Random** (unbiased baseline). All class memberships are computed **deterministically over all control cells** (a fixed function of the h5ad), independent of the DE cell subsample.

**Verdict reason:** PASS: null FPR=0 controlled; resolves δ≥0.5 log2FC

| metric | value |
|---|---|
| `null_FPR` | 0 |
| `max_TPR` | 0.8305 |
| `min_resolvable_delta_log2fc` | 0.5 |
| `FPR_at_max_delta` | 0.0081 |
| `max_delta_log2fc` | 2 |
| `n_injected_genes` | 1782 |
| `n_injected_low_expr` | 594 |
| `n_injected_mid_expr` | 594 |
| `n_injected_high_expr` | 594 |
| `injection_stratified_by` | expression tertile |
| `n_expressed_genes` | 17841 |
| `n_total_genes` | 18080 |
| `frac_genes_injected` | 0.1 |
| `control_cells_used` | 6000 |
| `cells_per_arm` | 2988 |
| `FPR_AnchorCorr_at_max_delta` | 0.0095 |
| `FPR_HighlyExpr_at_max_delta` | 0 |
| `FPR_LowlyExpr_at_max_delta` | 0 |
| `FPR_HouseKeeping_at_max_delta` | 0 |
| `FPR_Marker_at_max_delta` | nan |
| `FPR_HighlyVarG_at_max_delta` | 0.0122 |
| `FPR_Random_at_max_delta` | 0.0079 |
| `TPR_lowexpr_at_max_delta` | 0.4916 |
| `TPR_midexpr_at_max_delta` | 1 |
| `TPR_highexpr_at_max_delta` | 1 |
| `min_resolvable_delta_lowexpr` | nan |
| `min_resolvable_delta_midexpr` | 0.5 |
| `min_resolvable_delta_highexpr` | 0.25 |

Calibration curve (injected δ in log2FC; δ=0 is the null FPR baseline):

| δ (log2FC) | TPR (injected) | FPR (untouched) | median obs LFC | λ_GC |
|---|---|---|---|---|
| 0 | 0 | 0 | -0.000441 | 0.4696 |
| 0.25 | 0.4714 | 0.0017 | 0.1609 | 0.8398 |
| 0.5 | 0.5898 | 0.0028 | 0.3655 | 1.2993 |
| 1 | 0.7828 | 0.0073 | 0.9603 | 1.9874 |
| 2 | 0.8305 | 0.0081 | 1.9584 | 2.1518 |

_FPR and λ_GC climb steeply as δ grows — the compositional-coupling artifact (false DE in untouched genes from library-size renormalization under strong, widespread injected effects)._

FPR by gene class (false calls among **untouched** genes; anchors = injected):

| δ (log2FC) | AnchorCorr | HighlyExpr | LowlyExpr | HouseKeeping | Marker | HighlyVarG | Random |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | nan | 0 | 0 |
| 0.25 | 0.0032 | 0 | 0 | 0 | nan | 0.0033 | 0.0012 |
| 0.5 | 0.0044 | 0 | 0 | 0 | nan | 0.0055 | 0.0024 |
| 1 | 0.0095 | 0 | 0 | 0 | nan | 0.0111 | 0.0061 |
| 2 | 0.0095 | 0 | 0 | 0 | nan | 0.0122 | 0.0079 |

_Compare classes: AnchorCorr/HighlyExpr/HighlyVarG should inflate first under coupling; LowlyExpr stays low (zero-dominated); Random is the baseline; HouseKeeping flags memorization; Marker is N/A without cell-type labels._

TPR by **injected-gene expression tier** (recovery of the anchors; detection power is dominated by expression level):

| δ (log2FC) | high expr | mid expr | low expr |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 0.25 | 1 | 0.4141 | 0 |
| 0.5 | 1 | 0.7694 | 0 |
| 1 | 1 | 1 | 0.3485 |
| 2 | 1 | 1 | 0.4916 |

_min resolvable δ (TPR≥0.5): high-expr=0.25, mid=0.5, low=nan. Abundant genes are easiest to recover; sparse (low-expr) genes are the conservative floor — the real limit on what the metric can resolve._

**Verification p-values:** `null_FPR`=0

- ⚠️ null FPR=0 ≤ α — conservative (no false positives)

![test_0](plots/test_0_injection.png)

_Full per-gene/per-split numbers: `tables/test_0__*.csv`_

### test_1 — Within-Condition Reproducibility  ❌ FAIL

*What this tests.* REPRODUCIBILITY. Split each perturbation's cells into halves A and B and run each vs control (DE_A, DE_B) — both carry the real perturbation signal. The question is whether the two independent half-signatures AGREE (DE_A ≈ DE_B): median split-half Spearman LFC ρ (PASS>0.6 strong / 0.3–0.6 moderate / <0.3 low). Run under two control-assignment scenarios — no_match (split control halves) vs matched (1:1 batch-matched controls) — to see whether batch matching improves reproducibility. (Secondary sanity check: the direct A_pert-vs-B_pert contrast, same perturbation, should be ≈null.)

**Tiers** (per metric, strong/moderate/low) — LFC Spearman ρ (verdict driver): > 0.6 PASS · 0.3–0.6 WARN · < 0.3 FAIL. DEG Jaccard: > 0.3 · 0.1–0.3 · < 0.1. Direction agreement: > 0.7 · 0.6–0.7 · ≈ 0.5. Low ρ = the two half-signatures disagree → ceiling for downstream metrics. (Secondary difference-is-null should be ≈null: λ_GC≈1, p uniform.)

**Verdict reason:** FAIL [matched]: median split-half LFC ρ=0.2163 (low reproducibility; PASS>0.6/WARN≥0.3/FAIL<0.3); Jaccard=0.4539, direction=0.97; (2°) difference-is-null λ_GC=0.3787, frac_sig=1.24e-06

**What this compares.** Split each perturbation's cells into two halves A and B and run each half against control (`DE_A`, `DE_B`). The **primary** question is *reproducibility* — do the two independent half-signatures agree (`DE_A ≈ DE_B`)? — under two control-assignment scenarios: **no_match** (1.a — controls split into two halves, cell-eval pdex, no 1:1 matching) vs **matched** (2.a — perturbed cells 1:1 batch-matched to controls, then split into perturb–control pairs). The comparison answers: *does 1:1 batch matching improve reproducibility?* A **secondary** difference-is-null stat (direct `A_pert vs B_pert`, the same perturbation ⇒ should be ≈null) is reported as a sanity check.

| statistic | no_match | matched |
|---|---|---|
| **reproducibility** — median Spearman LFC ρ (PRIMARY) | 0.7168 (strong) | 0.2163 (low) |
| reproducibility — median DEG Jaccard | 0.6477 (strong) | 0.4539 (strong) |
| reproducibility — median direction agreement | 0.957 (strong) | 0.97 (strong) |
| # perturbations strong (ρ>0.6) | 70 | 24 |
| # moderate (0.3–0.6) | 30 | 14 |
| # low (<0.3) | 0 | 62 |
| (2°) difference-is-null λ_GC | 0.3783 | 0.3787 |
| (2°) difference-is-null frac_sig | 0 | 1.24e-06 |
| (2°) difference-is-null ks_p_uniform | 0 | 0 |
| # perturbations tested | 50 | 50 |

_Tiers (strong/moderate/low) per metric: LFC ρ (verdict driver): >0.6 strong/PASS, 0.3–0.6 moderate/WARN, <0.3 low/FAIL. DEG Jaccard: >0.3 strong, 0.1–0.3 moderate, <0.1 low. Direction agreement: >0.7 strong, 0.6–0.7 moderate, ≈0.5 low. Low = the two half-signatures disagree, so downstream metrics cannot be trusted above this ceiling. The matched (batch-controlled) scenario's LFC ρ drives the verdict. difference-is-null: A and B are the same perturbation, so a calibrated pipeline calls ≈α (λ_GC≈1, p uniform)._

- ⚠️ median split-half LFC ρ — matched (batch-controlled)=0.2163 vs no-match=0.7168 (Δ=-0.5004)
- ⚠️ per-perturbation reproducibility tiers (matched): 24 strong / 14 moderate / 62 low (of 50)
- ⚠️ thresholds — LFC ρ (verdict driver): >0.6 strong/PASS, 0.3–0.6 moderate/WARN, <0.3 low/FAIL. DEG Jaccard: >0.3 strong, 0.1–0.3 moderate, <0.1 low. Direction agreement: >0.7 strong, 0.6–0.7 moderate, ≈0.5 low. Low = the two half-signatures disagree, so downstream metrics cannot be trusted above this ceiling.
- ⚠️ secondary difference-is-null (A_pert vs B_pert, same perturbation ⇒ should be ≈null): λ_GC=0.3787, frac_sig=1.24e-06, ks_p_uniform=0

![test_1](plots/test_1_reproducibility.png)

_Full per-gene/per-split numbers: `tables/test_1__*.csv`_

### test_2 — Control-Control Split Null  ⚠️ WARN

*What this tests.* CALIBRATION / NULL. Split only control cells into A and B (a stratified random split balanced within batch; same population ⇒ true null) and run DE between them: a calibrated pipeline should produce Uniform[0,1] p-values (λ_GC≈1, frac_sig≈α). Inflation (λ_GC>1) = false positives; deflation (λ_GC<1) = under-powered (cell-level pseudoreplication).

**Tiers** — λ_GC: ≈ 1 calibrated · 1.05–1.10 or < 0.9 WARN · > 1.10 FAIL (anti-conservative). frac_sig: ≈ α good · ≫ α FAIL. ks_p_uniform: > 0.05 good · < 0.05 WARN (p-values not Uniform[0,1]). Arms are a stratified control-control split (true null).

**Verdict reason:** WARN: deflated/under-powered null (λ_GC=0.4727<0.9, ks_p_uniform=1.06e-276); no false positives but p-values are not calibrated

| metric | value |
|---|---|
| `frac_sig` | 0 |
| `mean_lfc` | 0.000218 |
| `mean_abs_lfc` | 0.0528 |
| `lambda_gc` | 0.4727 |
| `ks_p_uniform` | 1.06e-276 |
| `lambda_gc_sd` | 0.0204 |

**Verification p-values:** `ks_p_uniform_mean`=1.06e-276, `lambda_gc`=0.4727

- ⚠️ λ_GC=0.4727 < 0.9 (deflated null — under-powered)
- ⚠️ ks_p_uniform=1.06e-276 < 0.05 (p-values not Uniform[0,1])

![test_2](plots/test_2_qq.png)

_Full per-gene/per-split numbers: `tables/test_2__*.csv`_

### test_3 — Label Permutation Null  ✅ PASS

*What this tests.* Shuffle the perturbation labels (within batch) and recompute. Real biological signal should sit far outside the shuffled distribution (high separation z); if not, the metric cannot tell signal from noise.

**Tiers** — separation z (true signal vs permuted null): > 2 PASS · 1–2 WARN · ≤ 1 FAIL (metric cannot tell signal from noise).

**Verdict reason:** PASS: true signal 0.1526 vs shuffled 0.0133 (separation z=245.403, perm_p=0)

| metric | value |
|---|---|
| `true_signal` | 0.1526 |
| `perm_mean` | 0.0133 |
| `perm_sd` | 0.000568 |
| `separation_z` | 245.403 |

**Verification p-values:** `perm_p`=0, `separation_z`=245.403

_Full per-gene/per-split numbers: `tables/test_3__*.csv`_

### test_4 — Same-sgRNA Split Reproducibility  ✅ PASS

*What this tests.* Split each guide's cells in two and compare each half vs control. Agreement between the halves is the reproducibility ceiling — no model can score higher on a perturbation than the data agrees with itself.

**Tiers** (same as Test 1) — LFC Spearman ρ: > 0.6 strong/PASS · 0.3–0.6 moderate/WARN · < 0.3 low/FAIL. DEG Jaccard: > 0.3 · 0.1–0.3 · < 0.1. Direction: > 0.7 · 0.6–0.7 · ≈ 0.5. This is the empirical reproducibility ceiling for downstream metrics.

**Verdict reason:** PASS: same-sgRNA reproducibility ceiling — median LFC ρ=0.7378 (strong), Jaccard=0.7454 (strong), direction=0.9652 (strong)

| metric | value |
|---|---|
| `n_guides` | 54 |
| `median_lfc_spearman` | 0.7378 |
| `median_jaccard` | 0.7454 |
| `median_direction` | 0.9652 |

- ⚠️ tiers — LFC ρ (verdict driver): >0.6 strong/PASS, 0.3–0.6 moderate/WARN, <0.3 low/FAIL. DEG Jaccard: >0.3 strong, 0.1–0.3 moderate, <0.1 low. Direction agreement: >0.7 strong, 0.6–0.7 moderate, ≈0.5 low. Low = the two half-signatures disagree, so downstream metrics cannot be trusted above this ceiling.

_Full per-gene/per-split numbers: `tables/test_4__*.csv`_

### test_5 — Same-Gene Independent sgRNA Reproducibility  ⚠️ WARN

*What this tests.* Do two different guides for the same gene agree more than two unrelated guides? Note that guides for the same gene often differ in knockdown efficacy, so only modest concordance is expected; a low score with few guide pairs is power-limited, not proof of off-target activity.

**Tiers** — separation (same-gene vs background guide agreement): > 1.5 PASS · 1.0–1.5 WARN · < 1.0 FAIL. With < ~5 guide pairs the result is power-limited ⇒ WARN (cannot conclude), never FAIL.

**Verdict reason:** WARN (underpowered, cannot conclude): same-gene ρ=0.569 trends higher than background ρ=0.5123 on only 4 pair(s)

| metric | value |
|---|---|
| `n_genes` | 4 |
| `n_same_pairs` | 4 |
| `same_gene_mean_rho` | 0.569 |
| `background_mean_rho` | 0.5123 |
| `separation_z` | 0.5328 |

**Verification p-values:** `separation_z`=0.5328

- ⚠️ UNINFORMATIVE / power-limited: only 4 gene(s) with ≥2 guides (4 same-gene pair(s)). Separation z=0.5328 carries essentially no statistical weight, so this is NOT evidence that guides are off-target or inefficacious. Same-gene guides also genuinely differ in knockdown efficacy, so only modest concordance is expected even with more pairs.

_Full per-gene/per-split numbers: `tables/test_5__*.csv`_

### test_6 — Target Gene Knockdown Recovery  ✅ PASS

*What this tests.* Is the targeted gene itself knocked down (negative LFC, significant)? Computed per target gene (one DE per perturbation, pooling that gene's guides — not per individual guide). High recovery is reassuring, but it partly reflects assay and guide quality, not the metric alone.

**Tiers** — recovery_rate (targets detected as DE): > 0.5 PASS · 0.2–0.5 WARN · < 0.2 FAIL. direction_rate (knocked down in expected direction): > 0.8 PASS · 0.6–0.8 WARN · < 0.6 FAIL. (Also reflects assay/guide quality, not the metric alone.)

**Verdict reason:** PASS: recovery_rate=1, direction_rate=1 over 50 target genes (one DE per perturbation/gene, not per guide; also reflects assay/guide quality, not the metric alone)

| metric | value |
|---|---|
| `n_targets` | 50 |
| `recovery_rate` | 1 |
| `direction_rate` | 1 |
| `median_pval_rank` | 8.3e-05 |
| `median_lfc_rank` | 0.1046 |

**Verification p-values:** `median_padj_target`=0

_Full per-gene/per-split numbers: `tables/test_6__*.csv`_

### composition — Composition Control  ⏭️ SKIP

*What this tests.* Do perturbations shift cell-type / cluster / cell-cycle proportions vs control? Such shifts make cross-population DE read composition change as expression change. Requires a cell-state annotation in obs.

**Tiers** — per-perturbation total-variation distance (TVD) of cell-state proportions vs control: > 0.10 ⇒ flagged (its DE may be partly compositional). SKIP when no cell-state column exists.

**Verdict reason:** SKIP: no cell-state column to assess composition shift

- ⚠️ no cell-type / cluster / cell-cycle column in obs — composition shift cannot be assessed. Only `batch` is present. A perturbation that changes proliferation or differentiation will alter cell-state proportions, and cross-population DE reads that as expression change. Recommend annotating cell type (and cell-cycle phase) and re-running, or interpreting strong hits with this confounder in mind.

_Full per-gene/per-split numbers: `tables/composition__*.csv`_

## 5. Verification parameters & thresholds

| parameter | value | source |
|---|---|---|
| de_method / unit | pydeseq2 / pseudobulk replicate / sample (DESeq2 Wald) | config |
| covariate_correction | none | config |
| fdr_threshold | 0.05 | config |
| lfc_threshold | 0.1 | config |
| lambda_gc_warn / fail | 1.05 / 1.1 | TEST_PLAN.md |
| λ_GC deflation WARN / ks_p_uniform WARN | < 0.90 / < 0.05 | skill notes |
| injection δ tiers (log2FC) | [0.0, 0.25, 0.5, 1.0, 2.0] | config |
| injection fraction of genes | 0.1 | config |
| n_resamples / min_cells_per_group / seed | 10 / 20 / 0 | config |
| max_conditions / block_cols | None / ['batch'] | config |
| test-0 null FPR FAIL | > 2×α | skill notes |
| test-3 separation PASS / WARN | > 2 / 1–2 | TEST_PLAN.md |
| test-4 LOW-REPRODUCIBILITY flag | median LFC ρ < 0.2 | TEST_PLAN.md |
| test-5 separation PASS / WARN | > 1.5 / 1.0–1.5 | TEST_PLAN.md |
| test-6 recovery / direction PASS | > 0.5 / > 0.8 | TEST_PLAN.md |

## 6. Glossary

- **unit of analysis** — Whether each *cell* (pdex/Wilcoxon) or each *replicate/sample* (pseudobulk DESeq2) is one observation. Cell-level tests treat correlated cells from one sample as independent (pseudoreplication), distorting null p-values — see Squair et al. 2021, Nat Commun 12:5692 (https://www.nature.com/articles/s41467-021-25960-2).
- **arm** — One side of the two-group split that DE compares (clinical-trial term: reference arm vs treatment arm). In the null/injection tests both arms are drawn from the same cells; arm B is the pretend-perturbation (and receives the injected effect in Test 0), arm A is the reference, and DE is run B vs A.
- **stratified split (block_cols)** — Cells are divided into arms separately within each level of block_cols (e.g. each batch) and then pooled, so both arms have the same batch composition. This prevents technical structure (batch effects) from masquerading as signal; it is why the per-arm counts are slightly uneven (odd-sized batches can't halve exactly).
- **λ_GC (genomic inflation)** — Median observed χ² statistic / its null median. ≈1 well-calibrated; >1 inflated (false positives); <1 deflated (conservative, under-powered).
- **ks_p_uniform** — KS-test p-value that the per-gene p-values are Uniform[0,1] as a true null requires. Small (<0.05) ⇒ the p-value distribution is mis-shaped.
- **frac_sig** — Fraction of genes called significant (FDR≤threshold AND |LFC|≥threshold). Under a true null this is the empirical false-positive rate; it should be ≈α.
- **mean_abs_lfc** — Mean |log2 fold-change| across genes — magnitude inflation under the null.
- **TPR / FPR (Test 0)** — Of injected genes, the fraction recovered (true-positive rate); of untouched genes, the fraction wrongly called (false-positive rate).
- **max_TPR / min resolvable δ (Test 0)** — max_TPR = the highest TPR across the injected δ tiers — the sensitivity ceiling (best-case recovery of true effects). min resolvable δ = the smallest injected log2FC that reaches TPR ≥ 0.5, i.e. the smallest real effect the metric reliably detects. A max_TPR well below 1 means even strong real effects are partly missed (false negatives).
- **separation z (Tests 3,5)** — (observed − null mean) / null SD. >2 means the true signal is clearly outside the permuted/background null.
- **LFC Spearman ρ / Jaccard (Tests 1, 4)** — Rank correlation of fold-changes / overlap of significant gene sets between two split halves — the empirical reproducibility ceiling.

## 7. Known limitations & recommended next steps

- **Single dataset / modality.** Metric behaviour is partly experiment-dependent; validate on ≥2 datasets/modalities (e.g. a CRISPRi Perturb-seq screen and a chemical/Tahoe-style screen) before trusting thresholds.
- **Cell-level unit of analysis.** With pdex, null p-values are not calibrated (see §2). Re-run with `de_method: pydeseq2` for a pseudobulk comparison.
- **Composition not assessed** when no cell-state column is present (see the Composition test). Proliferation/differentiation shifts can masquerade as expression change.
- **Corrected vs uncorrected.** Evaluate the metric on both; this run is `none`.
- Gene-set / pathway enrichment metrics are intentionally **not** included: enrichment is unsolved (databases disagree; many effects need multi-gene perturbations) and would be a noisy gate.

## Appendix — embedded TEST_PLAN.md

## Cell Metrics Robustness & Stability Test Plan

> **Purpose.** Validate that a cell-metrics evaluation pipeline produces calibrated, unbiased results before interpreting biological findings. Tests **0–3** are **validity gates** — all must pass before proceeding. Tests **4–6** plus the **composition** diagnostic are **sensitivity diagnostics** — they characterise how much signal the data contains and set empirical ceilings for downstream metrics.

---

### ⚠️ Read first: unit of analysis & DE method (the single most important framing)

**The null-split tests (0, 1, 2, 4) split *cells* and run DE directly. Whether the null looks null depends entirely on the DE method's *unit of analysis*.**

- **Cell-level tests** (e.g. `pdex` = Wilcoxon rank-sum, or a t-test) treat every cell as an independent observation. Cells from the same sample/animal/batch are correlated, so this **pseudoreplication** inflates the effective *n* and intra-sample correlation alone produces "significant" genes under a true null. See **Squair et al. 2021, *Nat Commun* 12:5692** (https://www.nature.com/articles/s41467-021-25960-2). For such tests, expect the null-split p-values to be **mis-shaped** (deflated *or* inflated, λ_GC ≠ 1, non-uniform) even with zero real effect — and read the output as a **gene ranking, not a calibrated FDR**.
- **Pseudobulk / mixed-model tests** (e.g. DESeq2/edgeR/limma on replicate-level pseudobulk, or a GLMM) aggregate to the true unit of replication first. Only under these does the expected null behaviour (λ_GC ≈ 1, Uniform p-values, `frac_sig ≈ α`) actually hold.

> **The report MUST state the DE method and unit of analysis up front** and interpret every null in that light. The null-split tests **must use the same DE method as the intended real analysis** — they calibrate *that* pipeline, not DE in the abstract.

#### Corrected vs uncorrected data

It is common to regress out **batch** and **cell-cycle** (and sometimes other covariates) before DE. An ideal perturbation metric would be insensitive to this; real metrics are not. **Evaluate the metric on both corrected and uncorrected data** for the major covariates and report which state each run used. State explicitly what (if anything) was regressed out of `.X`.

#### Composition confounder

A perturbation that changes **proliferation or differentiation** shifts cell-type/state proportions. Cross-population DE then reads that composition change as expression change. When a cell-type / cluster / cell-cycle annotation exists, run the **Composition diagnostic** (below); when it does not, say so and flag that strong hits may be composition-driven.

#### Scope & validation

- This battery calibrates DE-based metrics. It deliberately **excludes gene-set / pathway enrichment** as a gate: enrichment is an unsolved problem (databases disagree; many real effects only appear under multi-gene perturbation), so it is too noisy to act as a pass/fail. It may be reported as informational only.
- Metric behaviour is partly experiment-dependent. **Validate across ≥2 datasets / modalities** (e.g. a CRISPRi Perturb-seq screen *and* a chemical/Tahoe-style screen) before trusting thresholds.

---

### Report & presentation conventions (the deliverable must be readable)

The generated report is the deliverable; a domain expert must be able to make sense of it without reverse-engineering the code. Required:

1. **Lead with dataset & experimental context** — what experiment, #cells/#genes, #perturbations, #samples/replicates/batches, cells-per-perturbation, guides-per-gene, and **which covariates are present vs absent** (cell type, cell cycle, donor/patient, sex, tissue, timepoint). Do not make the reader guess. State the **DE method, unit of analysis, and corrected/uncorrected state** in this opening section.
2. **Plain-language per test** — one to three sentences on *what the test asks and how to read it*, alongside the numbers.
3. **Local verdict reasons** — each test states its own one-line PASS/WARN/FAIL reason next to its result; do not make the reader hunt in a global flag list.
4. **Round numbers** — 4–6 significant figures (`0.1952`, not `0.19523047652509917`).
5. **Glossary** — define λ_GC, ks_p_uniform, frac_sig, separation z, TPR/FPR, reproducibility ρ, "unit of analysis".
6. **Don't bury context at the bottom** — verdict + context + per-test detail first; heavy appendices (full test plan, parameter dumps) last.
7. **Interactive-friendly output** — optionally emit **MultiQC** custom-content files (https://github.com/MultiQC/MultiQC) so the results render as an interactive HTML report that integrates with downstream single-cell pipelines. (Ideal future state: hover tooltips explaining λ_GC etc.)

---

### Inputs

#### Required

| Field | Type | Description |
|---|---|---|
| `adata` | `AnnData` | Cells × genes count matrix |
| `perturbation_col` | `str` | `obs` column containing sgRNA / perturbation labels |
| `control_label` | `str` | Value in `perturbation_col` identifying control cells |

#### Recommended

| Field | Type | Description |
|---|---|---|
| `target_gene_map` | `dict[str, str]` | sgRNA → target gene name |
| `batch_cols` | `list[str]` | e.g. `['batch', 'donor', 'replicate', 'cell_type']` |
| `cell_cycle_col` | `str` | `obs` column for cell-cycle phase |
| `guide_efficacy` | `dict[str, float]` | sgRNA → knockdown efficiency estimate |
| `baseline_expression` | `dict[str, float]` | Gene → baseline expression level |

#### Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_resamples` | `10` | Resampling iterations for Tests 2 and 3 |
| `min_cells_per_group` | `20` | Minimum cells per split arm; skip condition if not met |
| `fdr_threshold` | `0.05` | BH-adjusted p-value cutoff for significance calls |
| `lfc_threshold` | `0.1` | Minimum \|LFC\| for a gene to be called significant |
| `lambda_gc_warn` | `1.05` | λ_GC threshold above which a WARN is issued |
| `lambda_gc_fail` | `1.10` | λ_GC threshold above which a FAIL is issued |

---

### Shared Notation

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

### Shared Formulas

All six tests report the following metrics where DE results are available.

#### Fraction Significant

```
frac_sig = |S| / G
```

Reported as the mean over conditions. Approximates the empirical false-positive rate under a true null. Expected value ≈ `fdr_threshold` when the null holds.

#### Mean and Mean Absolute LFC

```
mean_lfc     = (1/G) Σ_g LFC(g)
mean_abs_lfc = (1/G) Σ_g |LFC(g)|
```

`mean_lfc` tests for **directional bias**; it can cancel to zero even with large symmetric LFCs.
`mean_abs_lfc` tests for **magnitude inflation**. Both must be reported together.

#### KS Uniformity Statistic

```
D = sup_x | F̂(x) − x |
```

`F̂` is the empirical CDF of `{p(g)}_g`. The reported `ks_p_uniform` is the two-sided KS p-value against Uniform[0,1]. A small `ks_p_uniform` (< 0.05) indicates departure from the expected null p-value distribution.

#### Genomic Inflation Factor

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

#### QQ-Plot Envelope

Plot sorted observed `−log10 p₍ᵢ₎` (i = 1…G) against expected `−log10((i − 0.5)/G)`.

The 95% pointwise confidence envelope uses the exact Beta order-statistic distribution:

```
lower visual bound = −log10[ B₀.₉₇₅(i, G−i+1) ]
upper visual bound = −log10[ B₀.₀₂₅(i, G−i+1) ]
```

where `Bₚ(α, β)` denotes the p-th quantile of the Beta(α, β) distribution. **Note:** the −log10 transform reverses the interval, so the lower Beta quantile (`B₀.₀₂₅`) maps to the upper visual bound and vice versa. Points lying on y = x indicate a well-calibrated test. This envelope is exact (not a normal approximation) because the i-th order statistic of Uniform[0,1] follows exactly Beta(i, G−i+1).

#### Wald Test Statistic

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

### Test 0 — Controlled Effect-Size Injection / Calibration Curve

> **In one sentence.** Spike a known log2 fold-change into a known fraction of genes in a control-vs-control split, then measure how many injected genes are recovered (TPR) and how many untouched genes are wrongly called (FPR) across effect sizes — separating "null = null" from "pipeline is dead" and quantifying the smallest effect the metric can resolve.

#### Question

When a real effect of known size is present, does the pipeline detect it (and only it)? And under no effect, is the false-positive rate controlled? A null that produces nothing is indistinguishable from a broken pipeline unless you also show it *can* detect a planted signal.

#### Design

On control cells only (so the only DE is what we inject):

1. Stratified-split control cells into reference arm A and pseudo-perturbation arm B (as in Test 2). Cap cells/arm for speed if needed.
2. Pick the injected gene set (the **anchors**), a fraction `injection_frac_genes` (e.g. 10%) of expressed genes, **stratified across the expression (detection) spectrum** — equal shares from the **low / mid / high mean-expression tertiles**. Detectability is dominated by expression level, so injecting evenly across tiers makes the TPR-vs-δ curve interpretable **per tier** (the smallest resolvable δ for sparse vs typical vs abundant genes) instead of an artifact of the injected mix. Do NOT inject only housekeeping/marker/anchor-correlated genes (biased, circular), or only HEG (optimistic) / only LEG (pessimistic).
3. For each effect size `δ` in `injection_deltas` (log2 fold-change tiers; **include `δ = 0` as the FPR-at-null baseline**), multiply the **raw counts** of the anchor genes in arm B by `2^δ`, then normalize (`normalize_total` + `log1p`) and run the **same DE method** as the real analysis, B vs A.
4. Record, per `δ`: **TPR** = fraction of injected/anchor genes called significant; **FPR** = fraction of *untouched* genes called significant; the median observed LFC of anchors; and λ_GC.
5. **Break the injected-gene TPR down by EXPRESSION TIER** (low/mid/high) — report the min resolvable δ (TPR≥0.5) per tier — and **break the untouched-gene FPR down by gene class**: **AnchorCorr** (genes most correlated with the anchor signature — easiest/upper-bound, most exposed to coupling), **HighlyExpr** (abundant — high signal), **LowlyExpr** (sparse — hard, zero-dominated), **HouseKeeping** (constitutive — should be predictable; watch memorization), **Marker** (cell-type identity — N/A without a cell-type annotation), **HighlyVarG** (high-variance complement to the anchors), **Random** (unbiased baseline). All expression tiers and gene-class memberships must be computed **deterministically over all control cells** — a fixed function of the h5ad, independent of the DE cell subsample.

> **Determinism.** Anchor selection, gene-class assignment, and the cell subsample are all seeded; given the same h5ad + seed the calibration curve (and every per-class FPR) is bit-for-bit reproducible.

#### Reported metrics

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

#### Verdict Rules

| Condition | Verdict |
|---|---|
| `null_FPR` controlled (≤ ~2×`fdr_threshold`) and TPR rises with δ to ≥ 0.5 | **PASS** |
| `null_FPR` clean but `FPR_at_max_delta` ≫ `null_FPR` (compositional coupling), or `max_TPR < 0.5` (under-powered) | **WARN** |
| `null_FPR` > 2×`fdr_threshold` (anti-conservative) | **FAIL** |

---

### Test 1 — Within-Condition Reproducibility

> **In one sentence.** Split each perturbation's cells into two halves and compare each half to control (`DE_A`, `DE_B`) — both carry the *real* perturbation signal (NOT a uniform null; that's Test 2) — and measure whether the two independent half-signatures **agree** (`DE_A ≈ `DE_B`); high split-half LFC correlation = a reproducible signature and the empirical ceiling for downstream metrics.

#### Question

When a perturbation's cells are split in half and each half is profiled against control, do the two halves recover the **same** differential-expression signature? And does **1:1 batch-matched** control assignment improve that reproducibility over a plain split?

#### Design

For each perturbation with ≥ `2 × min_cells_per_group` cells, in **two scenarios**:

- **Scenario 1.a — no_match (baseline, no batch control):** split the perturbed cells into A and B (stratified by `batch_cols`); split the **control** cells into two halves `ctrl_A`, `ctrl_B` (each = half of all controls); compute `DE_A = A vs ctrl_A` and `DE_B = B vs ctrl_B` with cell-eval's DE (**no 1:1 matching**).
- **Scenario 2.a — matched (batch-controlled):** **1:1-match** the perturbed cells to control cells *within batch* on QC features (`total_counts`, `n_genes_by_counts`) using `scmetrics` (inline fallback); split the matched **perturb–control pairs** into A and B (stratified by batch); compute `DE_A = A_pert vs A_matched_ctrl` and `DE_B = B_pert vs B_matched_ctrl` (Wilcoxon).

Two statistics per scenario:
1. **(PRIMARY) reproducibility** — agreement of `DE_A` and `DE_B`: **median split-half Spearman LFC ρ** (+ DEG Jaccard, direction agreement), over perturbations.
2. **(secondary) difference-is-null** — the direct `A_pert vs B_pert` contrast (same perturbation ⇒ should be ≈null): `frac_sig` / `λ_GC` / `ks_p_uniform`, as a sanity check.

Average over `test1_n_resamples` seeded splits; report both scenarios side by side (does matching raise reproducibility?).

#### Test Statistic

`ρ = Spearman(LFC_A, LFC_B)` over genes, per perturbation; the test reports the **median ρ across perturbations**. (The secondary difference-is-null uses the same null statistics as Test 2 on the `A_pert`-vs-`B_pert` contrast.)

#### Expected Behaviour

| Perturbation strength | median split-half LFC ρ | direction agreement |
|---|---|---|
| Strong / reproducible | > 0.6 | > 0.8 |
| Moderate | 0.3–0.6 | 0.6–0.8 |
| Low / non-reproducible | < 0.3 | ≈ 0.5 |

The matched scenario should reach ρ ≥ the no_match scenario (batch/depth matching removes a nuisance source of disagreement). The secondary difference-is-null should be ≈null (λ_GC≈1, p uniform) by construction.

#### Verdict Rules

Applied to the **median split-half Spearman LFC ρ** of the **matched** scenario (verdict driver); both scenarios reported.

| Condition | Verdict |
|---|---|
| median ρ > 0.6 | **PASS** — strong, reproducible signatures |
| median ρ ∈ [0.3, 0.6] | **WARN** — moderate reproducibility (downstream metrics bounded by this) |
| median ρ < 0.3 | **FAIL** — low; the two half-signatures disagree, so the data cannot support reliable downstream comparisons |

Report the result, the thresholds, and the per-perturbation tier counts (how many strong / moderate / low) so the ceiling is explicit.

#### Failure Diagnostics

| Pattern | Likely Cause |
|---|---|
| p-value histogram spikes near 0 | Unabsorbed batch structure leaking into split |
| LFCs skewed in one direction | Normalization bias or unbalanced split |
| QQ inflation (points above diagonal) | Missing covariates; overdispersion underestimated |
| QQ deflation (points below diagonal) | Overcorrection; too many covariates |

---

### Test 2 — Control-Control Split Null

> **In one sentence.** Same as Test 1 but splitting only control cells, so any detected signal is pure noise/batch — this confirms the null and that downstream cell-metrics collapse to chance; uncorrected batch structure leaks in here.

#### Question

Do metrics remain near null when pseudo-perturbations are created by splitting control cells?

#### Design

1. Take all cells where `perturbation_col == control_label`.
2. Perform `n_resamples` independent **stratified** splits into groups A and B (random A/B balanced within `batch_cols`). No 1:1 control-to-control matching mode: both arms are the *same* population (controls), so a stratified split already balances batch/depth and matching controls-to-controls is uninformative. (The batch-controlled 1:1-matched design belongs to **Test 1**, where two *different* populations — perturbed vs control — are compared.)
3. In each split, treat group A as a pseudo-perturbation and group B as the reference.
4. Run the full cell-metrics workflow for each split.
5. Aggregate metrics: mean ± SD across splits.

#### Test Statistic

Same Wald test as Test 1. Additionally compute all downstream cell-metrics outputs (overlap, AUC, delta correlations) to verify they also collapse to null.

#### Expected Behaviour

| Metric | Expected |
|---|---|
| `frac_sig` | ≈ `fdr_threshold` |
| `mean_lfc`, `mean_abs_lfc` | Near zero |
| `ks_p_uniform` | > 0.05 |
| `λ_GC` | ≈ 1.00 |
| Split-to-split SD of `λ_GC` | Low (< 0.05) |
| QQ-plot | Points hug y = x diagonal within envelope; consistent across splits |
| DEG overlap / precision / AUC | Near chance level |
| Delta correlations | Near zero |
| AnnData-level deltas | Near zero |

#### Verdict Rules

Same thresholds as Test 1, applied to the mean over splits. Additionally: if cell-metrics scores (overlap, AUC) are substantially above chance across splits, verdict is **FAIL**.

---

### Test 3 — Label Permutation Null

> **In one sentence.** Shuffle the perturbation labels (within batch) and recompute; real biological signal should sit far outside the shuffled distribution (high separation z), otherwise the metric cannot tell signal from noise.

#### Question

Do metrics collapse when perturbation labels are broken?

#### Design

1. Compute all metrics under true perturbation labels → `true_metrics`.
2. For each of `n_resamples` iterations:
   a. Shuffle `perturbation_col` values within each stratum defined by `batch_cols`.
   b. Recompute DE and all cell-metrics → `perm_metrics[i]`.
3. Compute separation score between true and permuted distributions.

#### Separation Score

```
separation = (true_metric − mean(perm_metrics)) / SD(perm_metrics)
```

This is equivalent to a z-score of the true metric relative to the permutation null. Also compute an empirical p-value:

```
perm_p = |{i : perm_metrics[i] ≥ true_metric}| / n_resamples
```

#### Expected Behaviour

| Metric | Expected |
|---|---|
| Permuted `frac_sig`, `λ_GC` | Indistinguishable from Test 1/2 null |
| Separation score for strong metrics | > 2.0 (true signal clearly outside null) |
| `perm_p` for strong metrics | < 0.05 |

#### Verdict Rules

| Condition | Verdict |
|---|---|
| True metrics sit clearly outside permutation null (separation > 2) | **PASS** |
| Separation borderline (1 < separation ≤ 2) | **WARN** |
| True metrics indistinguishable from permuted (separation ≤ 1) | **FAIL** — metrics cannot detect real signal |

---

### Test 4 — Same-sgRNA Split Reproducibility

> **In one sentence.** Split each guide's cells in two and compare each half vs control; agreement between the halves is the **reproducibility ceiling** — no model can score higher on a perturbation than the data agrees with itself.

#### Question

If the same sgRNA's cells are split into two groups and each is compared to control, are the resulting DE signatures reproducible?

#### Design

For each sgRNA with ≥ `2 × min_cells_per_group` perturbed cells:

1. Split perturbed cells into arms A and B (stratified by `batch_cols`).
2. Run DE: A vs control → `de_A`.
3. Run DE: B vs control → `de_B`.
4. Compare `de_A` and `de_B` using reproducibility metrics below.

#### Reproducibility Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| LFC correlation | `ρ = Spearman(LFC_A, LFC_B)` over all G genes | Overall signature agreement |
| DEG overlap | `Jaccard(S_A, S_B) = \|S_A ∩ S_B\| / \|S_A ∪ S_B\|` | Significant gene set overlap |
| Directional agreement | `(1/G) Σ_g 𝟙[sign(LFC_A(g)) = sign(LFC_B(g))]` | Fraction of genes agreeing in direction |
| AUC recovery | AUROC of S_A ranks predicting S_B membership | Rank-based reproducibility |

#### Expected Behaviour

| Perturbation strength | LFC correlation | DEG Jaccard | Directional agreement |
|---|---|---|---|
| Strong | > 0.6 | > 0.3 | > 0.7 |
| Moderate | 0.3–0.6 | 0.1–0.3 | 0.6–0.7 |
| Weak | < 0.3 | < 0.1 | ≈ 0.5 (chance) |

> **Note:** This test sets the **empirical ceiling** for all downstream evaluation metrics. A perturbation with low reproducibility here cannot be expected to score well on cross-model or cross-condition comparisons.

#### Verdict Rules

Reproducibility is not a pass/fail gate but an empirical characterisation. Flag perturbations with LFC correlation < 0.2 as **LOW REPRODUCIBILITY** and exclude them from sensitivity benchmarks.

---

### Test 5 — Same-Gene Independent sgRNA Reproducibility

> **In one sentence.** Do two different guides for the same gene agree more than two unrelated guides? **Expect only modest concordance** — guides for one gene often differ substantially in knockdown efficacy — and treat a low score with few guide pairs as **power-limited, not proof of off-target activity.**

#### Question

Do independent sgRNAs targeting the same gene recover similar perturbation signatures?

#### Design

For each target gene G with ≥ 2 sgRNAs:

1. Compute DE for each sgRNA versus control.
2. For all same-gene guide pairs (i, j): compute reproducibility metrics from Test 4.
3. For matched unrelated guide pairs (random pairs from different target genes): compute the same metrics → background distribution.
4. Optionally run leave-one-guide-out (LOGO) analysis:
   - Reference signature: consensus of all sgRNAs targeting G except guide i.
   - Query signature: guide i alone.
   - Compare LOGO scores against same-gene pair scores.

#### Separation Score (same-gene vs background)

```
separation = (mean_same_gene_metric − mean_background_metric) / SD(background_metric)
```

#### Expected Behaviour

| Metric | Expected |
|---|---|
| Same-gene LFC correlation | Substantially higher than background |
| Same-gene DEG Jaccard | Substantially higher than background |
| Separation score | > 1.5 for reliable target gene recovery |

#### Verdict Rules

| Condition | Verdict |
|---|---|
| **Underpowered: < ~5 genes with ≥2 guides (< ~5 same-gene pairs)** | **WARN (uninformative — cannot conclude)** — too few pairs to carry statistical weight; **never FAIL.** Do NOT read a low score as "guides off-target": same-gene guides genuinely differ in knockdown efficacy, so modest concordance is expected, and with few pairs the separation z is meaningless. |
| Same-gene pairs score clearly above background (separation > 1.5) | **PASS** |
| Marginal separation (1.0 < separation ≤ 1.5) | **WARN** — possible off-target effects or weak perturbation |
| Same-gene pairs indistinguishable from background **and enough pairs to conclude** | **FAIL** — guides may be off-target or inefficacious |

---

### Test 6 — Target Gene Knockdown Recovery

> **In one sentence.** Is the targeted gene itself knocked down (negative LFC, significant)? High recovery is reassuring, but it **partly measures assay and guide quality, not the metric alone.**

#### Question

Is the target gene itself detected as differentially expressed in the expected direction?

#### Design

For each sgRNA with a known target gene G (from `target_gene_map`):

1. Run DE: perturbed cells vs control.
2. Extract result for gene G.
3. Record the following per-sgRNA outputs.

#### Per-sgRNA Output Fields

| Field | Formula / Source | Description |
|---|---|---|
| `lfc_target` | `LFC(G)` from DE | Log₂ fold change of target gene |
| `padj_target` | `q(G)` from DE | BH-adjusted p-value of target gene |
| `pval_rank` | `rank(p(g))[G] / G` | Percentile rank of target gene by raw p-value (lower = more significant) |
| `lfc_rank` | `rank(\|LFC(g)\|)[G] / G` | Percentile rank by absolute LFC |
| `correct_direction` | `LFC(G) < 0` for CRISPRi/KO | Whether LFC is in the expected direction |
| `detected` | `q(G) ≤ fdr_threshold` | Binary: is target gene significant? |

#### Aggregate Recovery Metrics

Computed over all sgRNAs with a mapped target gene:

```
recovery_rate       = |{sgRNAs : detected = True}| / N_sgrnas
direction_rate      = |{sgRNAs : correct_direction = True}| / N_sgrnas
median_pval_rank    = median over sgRNAs of pval_rank
median_lfc_rank     = median over sgRNAs of lfc_rank
```

#### Expected Behaviour

| Metric | Expected (good data) | Warning threshold |
|---|---|---|
| `recovery_rate` | > 0.5 | < 0.2 |
| `direction_rate` | > 0.8 (CRISPRi/KO) | < 0.6 |
| `median_pval_rank` | < 0.10 (top 10% by p-value) | > 0.25 |
| `median_lfc_rank` | < 0.15 | > 0.30 |

> **Note for CRISPR KO:** reduced transcript abundance is expected for many genes but is not guaranteed due to nonsense-mediated decay variability. `direction_rate` thresholds may need adjustment based on the gene and perturbation system.

#### Verdict Rules

| Condition | Verdict |
|---|---|
| `recovery_rate` > 0.5 and `direction_rate` > 0.8 | **PASS** |
| `recovery_rate` ∈ [0.2, 0.5] or `direction_rate` ∈ [0.6, 0.8] | **WARN** — check guide efficacy estimates |
| `recovery_rate` < 0.2 or `direction_rate` < 0.6 | **FAIL** — poor guide efficacy or expression artefact |

---

### Composition Diagnostic

> **In one sentence.** Do perturbations shift cell-type / cluster / cell-cycle proportions versus control? Such shifts make cross-population DE read composition change as expression change; this requires a cell-state annotation in `obs`.

#### Design

If a categorical cell-state column exists (`cell_type` / `cluster` / `leiden` / cell-cycle `phase`, configurable via `celltype_cols`):

1. Compute the control's cell-state proportion vector.
2. For each perturbation, compute its proportion vector and the **total-variation distance (TVD)** from control: `TVD = ½ Σ_k |p_pert(k) − p_ctrl(k)|`.
3. Flag perturbations with `TVD > 0.10` — their DE may partly reflect composition.

If no such column exists, **SKIP with an explicit message**: composition cannot be assessed, only the available structural columns are present, and strong hits should be interpreted with this confounder in mind. Recommend annotating cell type / cell-cycle and re-running.

#### Verdict Rules

| Condition | Verdict |
|---|---|
| No perturbation shifts proportions (`TVD ≤ 0.10` for all) | **PASS** |
| One or more perturbations with `TVD > 0.10` | **WARN** — list them; their DE is partly compositional |
| No cell-state column available | **SKIP** (with guidance) |

---

### Output Schema

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
    composition: TestResult  # Composition diagnostic

    global_verdict: Literal["PASS", "WARN", "FAIL"]
    blocking_flags: list[str]   # reasons for FAIL at global level
    info_flags:     list[str]   # non-blocking WARNs
```

---

### Global Verdict Logic

Validity gates are Tests **0, 1, 2, 3**.

| Rule | Global Verdict |
|---|---|
| Tests 0–3 all PASS | Proceed to Tests 4–6 |
| Any of Tests 0–3 is FAIL (e.g. anti-conservative null, or injected-effect FPR > 2×α) | **FAIL** — do not interpret real results; fix pipeline first |
| Any of Tests 0–3 WARN (e.g. deflated/under-powered cell-level null, compositional coupling) | **WARN** — results usable but read the caveats; with a cell-level test, treat output as ranking, not calibrated FDR |
| Gates PASS/WARN but Test 6 WARN | **WARN** — results valid but guide efficacy may limit sensitivity |
| Tests 4–5 show no separation from background | **WARN** — empirical ceiling is low; downstream metrics may not be informative |
| Composition diagnostic WARN | informational — flag affected perturbations |

---

### Implementation Notes

1. **Shared split function.** Tests 1, 2, and 4 all require stratified A/B splits. Implement a single `stratified_split(cells, stratify_cols, min_per_group)` utility used by all three.

2. **Shared reproducibility function.** Tests 4 and 5 both compare two DE result objects. Implement a single `compare_signatures(de_A, de_B) -> ReproducibilityMetrics` used by both.

3. **SKIP conditions.** If a condition has fewer than `2 × min_cells_per_group` cells, mark that condition as SKIP and exclude from aggregation. If > 50% of conditions are SKIP, escalate to WARN.

4. **Consistency requirement.** All six tests must use the same DE function as the real analysis pipeline. Pass the DE function as a callable argument to each test runner.

5. **Determinism (REQUIRED — completely seeded).** The entire run must be **bit-for-bit reproducible**: the same dataset + `config.yaml` + `seed` must produce an identical report and identical per-test tables every time. A single `seed` drives **every** stochastic step — all stratified splits, label permutations, and the Test-0 injection gene-selection/subsampling — via `np.random.default_rng(seed + fixed_offset)` (never an unseeded `default_rng()` / global `np.random`). Call a `seed_everything(seed)` at startup that pins `PYTHONHASHSEED`, `random`, `np.random`, and `scanpy.settings.seed`, and pass `random_state=seed` to any PCA/HVG/neighbors. The DE backends are deterministic given fixed input (pdex Wilcoxon; pydeseq2 DESeq2). Validate by running twice and diffing `robustness_summary.json` / `tables/` — they must be identical.

---

---

### Power / sample-size analysis — REMOVED

A power / `n_required` / dispersion calculation is intentionally **not** part of this skill (it was removed). Report calibration (Tests 0–3), reproducibility (Tests 1, 4), and recovery (Test 6) directly; do not derive power, `α_empirical`, or required sample sizes.