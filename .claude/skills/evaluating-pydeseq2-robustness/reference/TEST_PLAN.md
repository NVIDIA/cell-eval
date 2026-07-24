# Robustness test plan — cell-eval PyDESeq2 backend

This is the worked specification the battery implements. Each test maps to a function in
`pdrobust/tests.py`. The unit under test is cell-eval's PyDESeq2 DE backend as used by the
run pipeline (`cell-eval run --de-methods pydeseq2`), exercised two ways:

- **DE-backend-direct** (`run_pydeseq2_de` → `cell_eval._pydeseq2_backend.compute_pydeseq2_de`):
  pseudobulk DESeq2 contrasts; we read the canonical DE frame `[target, feature,
  log2_fold_change, p_value, fdr]` and summarize calibration / effect recovery.
- **Run-pipeline** (`run_pipeline` → `cell_eval.MetricsEvaluator(..., de_methods=["pydeseq2"])`):
  the full real-vs-pred metric suite, so we can ask whether the *metrics* (overlap_at_N,
  de_spearman_lfc_sig, roc_auc, pr_auc, de_direction_match, de_sig_genes_recall,
  de_nsig_counts_*, pearson_delta, …) are calibrated / sensitive / stable.

Core distinction (from the plan):
- **Pure nulls** — should produce no signal (tests 1–3).
- **Reproducibility controls** — should recover the same signal; define an empirical ceiling (tests 4–5).
- **Biological positives** — anchored to known biology (tests 6–8).
- **Stress tests** — degrade the procedure in predictable ways (tests 9–10).

## PyDESeq2 prerequisites (enforced by the backend)
- **Raw integer counts** in `counts_layer` (or `.X` with `allow_discrete: true`). Fractional values are rejected.
- **`replicate_col`** identifying the pseudobulk unit. The backend sums counts per
  `(replicate × perturbation)` group into one pseudobulk sample. A design with **<2 pseudobulk
  samples per condition** only warns, but DESeq2 dispersion estimation is unreliable there — the
  battery skips sub-designs that cannot form ≥2 replicates per side and logs it.

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
Split each condition's cells into A/B, stratified by `replicate_col` + `block_cols`, and run DE
A-vs-B directly. Expect few significant genes, LFCs ≈ 0, uniform-ish p-values.
**Verdict**: PASS if mean `frac_sig` < 0.02 and |mean LFC| < 0.10; WARN/FAIL above.

**Formulas.**
- `frac_sig(t) = |S_a(t)| / G`  (reported as the mean over conditions; ≈ empirical false-positive
  rate under a true null).
- `mean_lfc(t) = (1/G) Σ_g LFC(t,g)`,  `mean_abs_lfc(t) = (1/G) Σ_g |LFC(t,g)|`.
- KS-uniform statistic `D = sup_x |F̂(x) − x|`, where `F̂` is the empirical CDF of `{p(t,g)}_g`;
  the reported `ks_p_uniform` is its p-value (small ⇒ departure from `Uniform[0,1]`).
- Genomic inflation `λ_GC = median_g χ²₁⁻¹(1 − p(t,g)) / χ²₁⁻¹(0.5)`, with `χ²₁⁻¹(0.5) = 0.4549`
  (the median of a 1-df χ²). `λ_GC ≈ 1` ⇒ calibrated; `< 1` ⇒ conservative (deflated); `> 1` ⇒
  anti-conservative (inflated).
- QQ-plot: the sorted observed `−log10 p₍ᵢ₎` (i = 1…G) vs the expected `−log10((i − 0.5)/G)`; the
  grey 95% band is the pointwise order-statistic envelope `[−log10 B₀.₉₇₅(i, G−i+1),
  −log10 B₀.₀₂₅(i, G−i+1)]` where `Bᵖ(α,β)` is the Beta(α,β) p-quantile. Points on `y = x` ⇒
  well-calibrated.

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
correlation of per-perturbation n_sig to full-data. Each level reports `cells_per_pseudobulk`.
**Verdict**: PASS once LFC-correlation-to-full reaches `stable_lfc_corr` (default 0.9). Headline
`min_cells_for_stable_lfc` is the smallest level meeting it. Negative correlation at any level
(`anti_correlated_lfc`) and shallow pseudobulks (< `min_pseudobulk_cells`) are flagged in the note.

**Formulas.** At downsample level `ℓ` (mean over `n_repeats`), comparing to the full-data DE:
- `lfc_corr_to_full(ℓ) = mean_t  r( LFC_ℓ(t,·), LFC_full(t,·) )` (per-target Pearson, then averaged).
- `rank_corr(ℓ) = ρ_Spearman( n_sig_ℓ(·), n_sig_full(·) )` over the shared targets, where
  `n_sig(t) = |S_a(t)|`.
- `min_cells_for_stable_lfc = min { ℓ : lfc_corr_to_full(ℓ) ≥ c }`, `c = stable_lfc_corr` (default
  0.9); `None` if the grid never reaches `c`.
- `cells_per_pseudobulk(ℓ) ≈ ℓ / R`, where `R` = number of pseudobulk replicates the perturbed
  pool is summed into.
- `anti_correlated_lfc = 1[ ∃ ℓ : lfc_corr_to_full(ℓ) < 0 ]`.

## Test 10 — Control-Count Downsampling  (`test_10_control_downsampling`)
Same engine as test 9 but holds perturbed cells fixed and downsamples controls, using
`control_downsample_grid` (finer/larger by default — small, batch-fragmented control pseudobulks
destabilize LFCs). Yields the recommended **minimum control-cell count** and surfaces the known
failure mode where LFC magnitudes anti-track the full-data estimates under tiny control sets
(check `anti_correlated_lfc` and `cells_per_pseudobulk`). If the grid never reaches
`stable_lfc_corr`, raise its upper bound or coarsen the replicate grouping.

**Formulas.** Same `lfc_corr_to_full(ℓ)`, `rank_corr(ℓ)`, `min_cells_for_stable_lfc`, and
`anti_correlated_lfc` as Test 9, but the downsampled pool is the controls; `cells_per_pseudobulk(ℓ)`
≈ `ℓ / R_ctrl`, the control pseudobulk depth (small, batch-fragmented control pseudobulks are the
known destabilizer).

---

## Companion: biological counterfactual / red-team layer
See `counterfactual_companion.md`. The battery's signature comparison + summary primitives are
the substrate for generating matched transcriptome variants (plausible counterfactuals vs
statistically-matched-but-impossible corruptions) and checking that metrics rank them

## Reading verdicts
PASS/WARN/FAIL/INFO/SKIP are heuristic flags to triage attention, **not** proof of (in)correctness.
Always read the per-test CSV tables under `<outdir>/tables/`. Thresholds live at the top of
`pdrobust/tests.py` and should be tuned to the dataset's depth and replicate structure.
