# Robustness test plan — cell-eval pdex backend

This is the worked specification the battery implements. Each test maps to a function in
`pdrobust/tests.py`. The unit under test is cell-eval's pdex DE backend (the default cell-level
Wilcoxon test) as used by the run pipeline (`cell-eval run --de-methods pdex`), exercised two ways:

- **DE-backend-direct** (`run_pdex_de` → `cell_eval._pdex_backend.compute_pdex_de`):
  cell-level Wilcoxon contrasts of every level vs the reference; we read the canonical DE frame
  `[target, feature, log2_fold_change, p_value, fdr]` and summarize calibration / effect recovery.
- **Run-pipeline** (`run_pipeline` → `cell_eval.MetricsEvaluator(..., de_methods=["pdex"])`):
  the full real-vs-pred metric suite, so we can ask whether the *metrics* (overlap_at_N,
  de_spearman_lfc_sig, roc_auc, pr_auc, de_direction_match, de_sig_genes_recall,
  de_nsig_counts_*, pearson_delta, …) are calibrated / sensitive / stable.

Core distinction (from the plan):
- **Pure nulls** — should produce no signal (tests 1–3).
- **Reproducibility controls** — should recover the same signal; define an empirical ceiling (tests 4–5).
- **Biological positives** — anchored to known biology (tests 6–8).
- **Stress tests** — degrade the procedure in predictable ways (tests 9–10).

## pdex prerequisites
- **Log-normalized `.X`**. pdex sets `is_log1p = not allow_discrete`, so keep `allow_discrete: false`
  and supply log1p(library-normalized) expression. If `.X` is raw integer counts, the harness
  normalizes once at load (`normalize_total(1e4)+log1p`) when `normalize_if_raw: true`, so pdex and
  the AnnData-level metrics both see lognorm input.
- **Cell-level — NOT pseudobulk.** pdex compares per-cell expression between groups with a Wilcoxon
  rank-sum test. There is **no replicate / pseudobulk-sample requirement**. `replicate_col` is
  optional and used only to stratify A/B splits and subsampling; when absent, splits stratify by
  `block_cols` (or sample randomly). The only design constraint is having enough cells per side, so
  the battery keeps a minimum-cells guard and SKIPs (rather than lies) when a condition is too small.
- High power caveat: with thousands of cells, cell-level Wilcoxon is high-powered and may call many
  genes; interpret null `frac_sig` alongside LFC magnitudes, not in isolation.

---

## Notation
The DE backend yields, for each perturbation `t` and gene `g`: a log2 fold-change `LFC(t,g)`,
a p-value `p(t,g)`, and a BH-adjusted FDR `q(t,g)`. `G` = number of genes tested. The significant
set at FDR level `a` is `S_a(t) = { g : q(t,g) < a }` (default `a = fdr_threshold = 0.05`).
`sign(x) ∈ {−1, 0, +1}`. `1[·]` is the indicator (1 if true, else 0). `TopN_x(t)` = the N genes of
the largest `|LFC|` among the significant genes of side `x ∈ {real, pred}`. Run-pipeline metric
values are averaged over perturbations unless noted; `T` = number of perturbations scored.

---

## Test 1 — Within-Condition Direct Split Null  (`test_1_within_split_null`)
Split each condition's cells into A/B, stratified by `replicate_col` (if set) + `block_cols`, andrun A-vs-B directly. Expect few significant genes, LFCs ≈ 0, uniform-ish p-values. Use Wald test on the LFC of A vs B split, with replicate/block as covariates. The test statistic W = β̂/SE(β̂) should be approximately standard normal, p-values uniform, and LFCs near zero — any deviation flags pipeline miscalibration.

**Verdict**: Null Split Verdict
After running DE on the within-condition A/B split, evaluate calibration by inspecting four diagnostics: (1) the p-value histogram should be uniform across [0,1] with no spike near zero; (2) the proportion of significant genes at FDR 5% should be at or below 5%; (3) the LFC distribution should be symmetric and centered at zero; and (4) the QQ-plot of the Wald statistic should hug the N(0,1) diagonal without inflation or deflation. If all four hold, verdict is PASS — the pipeline is well calibrated and real comparisons can proceed. If any diagnostic fails, verdict is FAIL — do not proceed; instead flag the specific pattern (e.g. p-value spike, LFC skew, QQ inflation) and trace it back to likely causes such as unabsorbed batch structure, missing covariates in the design formula, incorrect stratification during the split, or a misspecified dispersion model. A marginal case — mild QQ inflation with few significant genes — should be flagged as WARN with a note that small sample sizes may cause asymptotic assumptions to break down. The verdict should always use the same DE method as the real analysis, since the null split is a calibration check on the pipeline itself, not an independent test.

**Formulas.**
- `frac_sig(t) = |S_a(t)| / G, S_a(t) = {g : q(a,g) ≤ t} where q is the BH-adjusted p-value`  (reported as the mean over conditions; ≈ empirical false-positive
  rate under a true null).
- `mean_lfc(t) = (1/G) Σ_g LFC(t,g)`,  `mean_abs_lfc(t) = (1/G) Σ_g |LFC(t,g)|`.
- KS-uniform statistic `D = sup_x |F̂(x) − x|`, where `F̂` is the empirical CDF of `{p(t,g)}_g`;
  the reported `ks_p_uniform` is its p-value (small ⇒ departure from `Uniform[0,1]`).
- Genomic inflation `λ_GC = median_g χ²₁⁻¹(1 − p(t,g)) / χ²₁⁻¹(0.5)`, with `χ²₁⁻¹(0.5) = 0.4549`
  (the median of a 1-df χ²). `λ_GC ≈ 1` ⇒ calibrated; `< 1` ⇒ conservative (deflated); `> 1` ⇒
  anti-conservative (inflated). conventional warning threshold λ_GC > 1.05.
- QQ-plot: sorted observed `−log10 p₍ᵢ₎` (i = 1…G) vs expected `−log10((i − 0.5)/G)`; the
  grey 95% pointwise envelope is `[−log10 Bₚ(α,β), −log10 Bₚ(α,β)]` where `Bₚ(α,β)` denotes
  the p-th quantile of the Beta(α,β) distribution, with bounds
  `[−log10 B₀.₉₇₅(i, G−i+1), −log10 B₀.₀₂₅(i, G−i+1)]`; note the −log10 transform
  reverses the interval so the upper visual bound comes from the lower Beta quantile.
  Points on `y = x` ⇒ well-calibrated.

## Test 2 — Control-Control Split Null  (`test_2_control_control_null`)
Partition control cells into pseudo-perturbation vs pseudo-reference (stratified), run DE,
repeat `n_repeats` times. Expect ~0 significant genes and near-zero LFC.
**Verdict**: same null thresholds as test 1 (averaged over repeats).

**Formulas.** Same as Test 1 (`frac_sig`, `mean_lfc`, `mean_abs_lfc`, KS, `λ_GC`, QQ-plot), pooled
over the `n_repeats` control pseudo-splits instead of over conditions.

## Test 3 — Label Permutation Null  (`test_3_label_permutation_null`)
Run the run pipeline twice: (a) **identity** `pred == real` (metrics should max out), and
(b) **permuted** `pred` = real with perturbation labels shuffled within `block_cols`
(controls untouched). Strong metrics should *separate* the two.
**Verdict**: PASS if the mean identity-minus-permuted gap on discriminative metrics
(`overlap_at_N`, `de_spearman_lfc_sig`, `de_spearman_sig`, `de_direction_match`, `roc_auc`,
`pr_auc`, `de_sig_genes_recall`, `precision_at_N`) ≥ 0.20; WARN below.

**Formulas.**
- For each metric `M`: `gap(M) = M(identity) − M(permuted)`.
- `mean_separation = (1/|D|) Σ_{M ∈ D} gap(M)`, where `D` is the discriminative-metric set above.
- identity (`pred = real`) is the attainable upper bound; permuted (labels shuffled within blocks,
  controls fixed) is the null floor — a genuine metric should show `gap(M) >> 0`.

## Test 4 — Same-Perturbation Split Reproducibility  (`test_4_same_pert_split`)
Split each perturbation's cells into halves A and B; `real` = controls+A, `pred` = controls+B;
run the pipeline. This is the **empirical ceiling** for model scores. The per-metric ceiling is
written to `tables/test_4__ceiling.csv`. **Set-overlap/precision ceilings are usually well below
1.0** (sampling noise in the DE gene set), so downstream model scores should be reported
**relative to this ceiling** (e.g. `model_overlap_at_N / ceiling_overlap_at_N`), not against 1.0.
**Verdict**: PASS if median LFC correlation ≥ 0.50 or mean `overlap_at_N` ≥ 0.30; else WARN
(a low ceiling caps what any model can achieve and is itself a finding).

**Formulas.** Let `U(t) = { g : g ∈ S_a^real(t) ∪ S_a^pred(t) }` be the union-significant set.
- `de_spearman_lfc_sig(t) = ρ_Spearman( LFC_real(t,·), LFC_pred(t,·) )` over `g ∈ U(t)`.
- `overlap@N(t) = |TopN_real(t) ∩ TopN_pred(t)| / N`;
  `precision@N(t) = |TopN_real(t) ∩ TopN_pred(t)| / |TopN_pred(t)|`.
- `direction_match(t) = (1/|U(t)|) Σ_{g ∈ U(t)} 1[ sign LFC_real(t,g) = sign LFC_pred(t,g) ]`.
- **Ceiling interpretation:** with `real = split A`, `pred = split B` of the *same* perturbation,
  each metric value is the maximum a model can attain on this data; report model scores as
  `score / ceiling`, not against 1.0.

## Test 5 — Same-Gene Independent sgRNA Reproducibility  (`test_5_same_gene_guides`)
Requires `sgrna_col` + `target_gene_col`. DE per guide vs control; compare same-gene guide pairs
to matched unrelated guide pairs.
**Metric caveat:** whole-transcriptome **LFC-Pearson saturates** when perturbations share a global
response (e.g. essential-gene knockdowns drive a common stress/proliferation axis), so it barely
discriminates same-gene from unrelated pairs and is reported only as a flagged diagnostic
(`pearson_saturated`). The verdict instead uses **rank/set-based separation** (significant-gene
Spearman and sig-Jaccard), which carry the gene-level signal.
**Verdict**: PASS if same-gene exceeds unrelated by > 0.05 on Spearman or sig-Jaccard separation.

**Formulas.** For a guide pair `(i, j)` over their common genes:
- `lfc_pearson = r(LFC_i, LFC_j)`,  `lfc_spearman = ρ_Spearman(LFC_i, LFC_j)`.
- `sig_jaccard = |S_a(i) ∩ S_a(j)| / |S_a(i) ∪ S_a(j)|`.
- Separation of metric `M`: `Δ_M = mean_{same-gene pairs} M − mean_{unrelated pairs} M`.
- Verdict statistic = `max(Δ_lfc_spearman, Δ_sig_jaccard)`.
- `pearson_saturated = 1[ mean_same lfc_pearson > 0.8  ∧  mean_unrelated lfc_pearson > 0.8 ]`
  (then LFC-Pearson cannot discriminate target identity and is reported only as a diagnostic).

## Test 6 — Target Gene Knockdown Recovery  (`test_6_target_knockdown`)
Requires `target_gene_col` (values must match `var_names`). DE per perturbation vs control; for
each, locate the target gene and record LFC, FDR, rank by p / signed LFC / |LFC|, and direction.
**Verdict**: PASS if ≥50% of targets are significant **and** ≥60% have negative LFC
(CRISPRi/KD expectation); WARN otherwise. For CRISPR-KO, expect weaker transcript reduction —
relax expectations and read the per-target table.

**Formulas.** For perturbation `t` with target gene `g*` (ranks taken within `t`'s DE list, `n` genes):
- `rank_p_pct = rank_↑(p(t,g*)) / n`  (ascending p; small ⇒ top),
  `rank_abs_lfc_pct = rank_↓(|LFC(t,g*)|) / n`  (descending |LFC|; small ⇒ top).
- `frac_detected_sig = (1/T) Σ_t 1[ q(t,g*) < a ]`.
- `frac_negative_lfc = (1/T) Σ_t 1[ LFC(t,g*) < 0 ]`.

## Test 7 — Curated Gene-to-Target Recovery  (`test_7_curated_targets`)
Requires `curated_targets_csv` (schema in `curated_targets_schema.md`). For each curated
perturbation→target relationship, record significance, rank, and direction agreement.
**Verdict**: PASS if ≥40% significant and (where directions given) ≥60% direction-correct.
SKIP without the CSV.

**Formulas.** For each curated relationship `(pert p → target g)`:
- `is_sig = 1[ q(p,g) < a ]`,  `rank_p_pct = rank_↑(p(p,g)) / n_p`.
- `direction_ok = 1[ sign LFC(p,g) = expected_sign ]` (only when an expected direction is given).
- `frac_sig = mean is_sig`;  `direction_accuracy = mean direction_ok` over relationships with a
  defined expected direction.

## Test 8 — Pathway / Regulon Recovery  (`test_8_pathway_recovery`)
Requires `gene_sets_json` = `{"<pert_gene>": ["geneA", ...]}`. For each perturbation, compute the
AUROC of |LFC| separating expected-set genes from the rest.
**Verdict**: PASS if mean AUROC ≥ 0.60. SKIP without the JSON.

**Formulas.** Score genes by `s_g = |LFC(t,g)|`; let positives = expected-set members
(`n_pos`), negatives = the rest (`n_neg`). With the Mann–Whitney `U`-statistic
`U = Σ_{i ∈ pos} Σ_{j ∈ neg} ( 1[s_i > s_j] + ½·1[s_i = s_j] )`,
`AUROC = U / (n_pos · n_neg)`. `AUROC > 0.5` ⇒ set genes are enriched among top-|LFC| genes.

## Test 9 — Cell-Count Downsampling  (`test_9_cell_downsampling`)
Downsample perturbed cells across `downsample_grid` (× `n_repeats`), holding controls fixed;
recompute DE; compare each level to full-data DE (LFC correlation, n_sig) and the rank
correlation of per-perturbation n_sig to full-data. The perturbed pool is thinned proportionally
across perturbations, so each level reports `cells_per_condition` (≈ level / n_perturbations).
**Verdict**: PASS once LFC-correlation-to-full reaches `stable_lfc_corr` (default 0.9). Headline
`min_cells_for_stable_lfc` is the smallest level meeting it. Negative correlation at any level
(`anti_correlated_lfc`) and shallow levels (< `min_cells_per_group` cells per condition) are
flagged in the note.

**Formulas.** At downsample level `ℓ` (mean over `n_repeats`), comparing to the full-data DE:
- `lfc_corr_to_full(ℓ) = mean_t  r( LFC_ℓ(t,·), LFC_full(t,·) )` (per-target Pearson, then averaged).
- `rank_corr(ℓ) = ρ_Spearman( n_sig_ℓ(·), n_sig_full(·) )` over the shared targets, where
  `n_sig(t) = |S_a(t)|`.
- `min_cells_for_stable_lfc = min { ℓ : lfc_corr_to_full(ℓ) ≥ c }`, `c = stable_lfc_corr` (default
  0.9); `None` if the grid never reaches `c`.
- `cells_per_condition(ℓ) ≈ ℓ / T` (the perturbed pool is thinned proportionally across the `T`
  perturbations).
- `anti_correlated_lfc = 1[ ∃ ℓ : lfc_corr_to_full(ℓ) < 0 ]`.

## Test 10 — Control-Count Downsampling  (`test_10_control_downsampling`)
Same engine as test 9 but holds perturbed cells fixed and downsamples controls, using
`control_downsample_grid` (finer/larger by default — pdex compares against the control cell pool,
so a small/depleted control set destabilizes the per-cell distributions). Yields the recommended
**minimum control-cell count** and surfaces the failure mode where LFC magnitudes anti-track the
full-data estimates under tiny control sets (check `anti_correlated_lfc` and `cells_per_condition`,
which for controls equals the level). If the grid never reaches `stable_lfc_corr`, raise its upper
bound.

**Formulas.** Same `lfc_corr_to_full(ℓ)`, `rank_corr(ℓ)`, `min_cells_for_stable_lfc`, and
`anti_correlated_lfc` as Test 9, but the downsampled pool is the controls; the depth proxy is
`cells_per_condition(ℓ) = ℓ` (controls are a single cell-level group for pdex).

---

## Companion: biological counterfactual / red-team layer
See `counterfactual_companion.md`. The battery's signature comparison + summary primitives are
the substrate for generating matched transcriptome variants (plausible counterfactuals vs
statistically-matched-but-impossible corruptions) and checking that metrics rank them

## Reading verdicts
PASS/WARN/FAIL/INFO/SKIP are heuristic flags to triage attention, **not** proof of (in)correctness.
Always read the per-test CSV tables under `<outdir>/tables/`. Thresholds live at the top of
`pdrobust/tests.py` and should be tuned to the dataset's depth and replicate structure.
