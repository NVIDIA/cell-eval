---
name: evaluating-test5-samegene-sgrna
description: Run ONLY Test 5 (same-gene independent-sgRNA reproducibility) from the cell-eval metric-robustness battery and emit a single-test report. Asks whether independent guides targeting the SAME gene produce more concordant signatures than unrelated guide pairs, using the Mann-Whitney effect size AUC = P(same>unrelated) on DEG-restricted LFC rho, gated by significance. Use when someone wants to check whether a DE metric rewards shared biology / same-gene concordance without running the whole battery. Needs an sgRNA column and genes with >=2 guides.
type: skill
---

# Test 5 — Same-Gene Independent sgRNA Reproducibility

**One test from the cell-eval metric-robustness battery, run on its own.** This skill runs **only
`test_5`** and produces a self-contained report for it. Role: **sensitivity diagnostic (biology reward)** — Power-limited with few guides per gene; WARN-only (never FAIL) when under ~5 same-gene pairs.

Runs **both pdex and pydeseq2** in one pass via `samegene_guide_heatmap.py`.

## What it asks
Do two different guides for the **same gene** produce more similar signatures than two **unrelated** guides? The headline statistic is **AUC = P(a random same-gene pair is more concordant than a random unrelated pair)** on DEG-restricted LFC rho, gated by a Mann-Whitney p; a cells-per-guide trend shows whether weak concordance is biology or undersampling.

## How DE is computed
Cell-eval DE per guide vs control; both backends on the same guides for a direct comparison. Needs: `--sgrna-col` + genes with ≥2 guides.

## Run it (standalone)

```bash
RUN_DIR="experiments_all/$(basename $H5AD .h5ad)__test_5__$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
python .claude/skills/evaluating-test5-samegene-sgrna/samegene_guide_heatmap.py \
  --adata <h5ad> --methods pdex,pydeseq2 \
  --pert-col <pert_col> --control non-targeting \
  --sgrna-col <sgrna_col> --target-gene-col <pert_col> \
  --replicate-col <batch_col> --block-cols <batch_col> \
  --min-guides 2 --max-control 1200 --zoom-per-page 1 \
  --outdir "$RUN_DIR"
```

## Parameters
| flag | what it controls | default |
|---|---|---|
| `--sgrna-col` | obs column holding the guide | required |
| `--min-guides` | genes need ≥ this many qualifying guides | 2 |
| `--min-cells` | min cells for a guide's DE to be computed | 20 |
| `--fdr / --lfc` | DEG cutoff + concordance threshold | 0.05 / 0.1 |
| `AUC tiers` | P(same>unrelated) cutoffs -> PASS / WARN, each gated by MWU p < alpha | >= 0.65 / >= 0.55 |
| `seed` | RNG seed for background-pair sampling | 0 |

## Verdict strategy (report part 6, stated before the verdict)
primary = **DEG-restricted LFC rho** (all-genes rho is ~0 for both and can't discriminate). **PASS AUC >= 0.65 & MWU p < alpha / WARN AUC >= 0.55 & significant / FAIL otherwise**. With < ~5 genes-with->=2-guides or < ~5 same-gene pairs => **WARN-only (never FAIL)**.

## Outputs (in `$RUN_DIR`)
- `robustness_report.md` + `report.pdf` — a **single-test report**: global verdict → dataset context →
  how DE was computed → the `test_5` 7-part section (question · data · parameters · method · results ·
  verdict strategy · verdict) → glossary → known limitations.
- `robustness_summary.json` — machine-readable metrics + fingerprint + verdict.
- `tables/test_5__same_gene_pairs.csv, test_5__background_pairs.csv` (per-gene / per-split CSVs).
- `plots/test_5_separation.png`, `plots/test_5_vs_ncells.png` — plot(s) for this test.
- `multiqc/*_mqc.txt` — MultiQC custom-content verdict tile (optional).

## Same-gene guide visualisations — `samegene_guide_heatmap.py`

A standalone diagnostic that renders the **Test-1 heatmap/correlation views at the guide level** — the
unit is an individual sgRNA (labeled `GENE-guide`), and the reproducibility question is Test 5's: *do
independent guides of the same gene agree?* It reuses the shared runner's exact DE path (`maybe_normalize`
/ `_de_two` / `run_de`) — each guide's signature is its cells vs control — and computes **both backends
on the same cells** for an apples-to-apples comparison. It only keeps genes with `≥ --min-guides`
qualifying guides (the same-gene block structure is the whole point), and guides are compared against the
**same (full) control**, so a uniform positive baseline is expected — the informative signal is the
**contrast** between within-gene blocks and the cross-gene background. Every output filename is suffixed
with the input `.h5ad` basename.

```bash
python .claude/skills/evaluating-test5-samegene-sgrna/samegene_guide_heatmap.py \
  --adata <h5ad> --methods pdex,pydeseq2 \
  --pert-col gene --control non-targeting --sgrna-col <guide_col> --target-gene-col gene \
  --replicate-col <batch/gem_group> --block-cols <batch/gem_group> \
  --min-guides 2 --max-genes 12 --max-control 1500 --zoom-per-page 1 --outdir <out>
```

- **Layer 1 — `test5_guide_heatmap_<method>__<dataset>.png`** (one per backend): rows = guides labeled
  `GENE-guide`, **grouped by gene** (same-gene guides adjacent, black separators between genes),
  columns = union DE genes sorted by mean LFC. Matching rows within a gene block = reproducible
  on-target effect; a guide whose row doesn't match its gene-mates is off-target / inefficacious.
- **Layer 2 — `test5_zoom__<dataset>_NN.png`** (one PNG per gene with ≥2 guides): **one row per method**
  = LFC(guide_i) vs LFC(guide_j) scatter (grey = non-DE, dark blue = DE in either) + a heatmap strip
  with **one row per guide of that gene** over the shared gene order. Title carries the guide-pair ρ and
  (for >2 guides) the mean pairwise ρ across the gene's guides. This is "one guide's signature vs another
  guide targeting the same perturbation."
- **Layer 3 — `test5_corr_matrix__<dataset>.png`** (one panel per backend): **guide × guide** Spearman-LFC
  correlation over the union DE genes, guides ordered by gene with gene-block separators. **Bright
  within-gene blocks = same-gene guides agree (on-target, reproducible); bright cross-gene off-diagonal =
  a shared program / low specificity.** Each panel's title reports **within-gene ρ̄ vs cross-gene ρ̄** —
  the gap between them is the same-gene concordance signal that the `test_5` AUC verdict quantifies.

Levers: `--max-genes` caps #genes-with-guides (by #guides desc) for readability/runtime, `--max-control`
subsamples control cells for speed, `--methods pdex` alone skips the (slower) pydeseq2 backend.
