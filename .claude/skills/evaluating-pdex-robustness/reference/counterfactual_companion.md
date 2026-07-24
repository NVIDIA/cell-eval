# Companion: biological counterfactual / metric red-team layer

This is the proposed companion project from the analysis plan. It is **not auto-run** by the
battery, but the battery's primitives are its substrate. Documented here so the skill can scaffold
it when asked.

## Why
VCC showed metrics can be optimized by exploiting statistical structure rather than learning
biology. The red-team layer stress-tests each metric directly: does it reward biology, or can it
be satisfied by variance inflation, pseudo-bulk shortcuts, DE inflation, or broken regulatory
structure?

## Core idea
For each perturbation, generate matched transcriptome variants and feed them as the **predicted**
AnnData to `run_pipeline` (real = observed ground truth):

1. **observed ground truth** (positive anchor)
2. **biologically plausible counterfactuals** — perturb within pathway/GRN/PPI constraints
3. **biologically impossible but statistically matched corruptions** — preserve simple statistics,
   break biology: shuffled gene–gene covariance, TF→target sign inversion, pathway-incoherent DE,
   variance amplification
4. **obvious statistical corruptions / metric hacks**

A metric is *biologically aligned* if it ranks them:
`ground-truth ≈ plausible > impossible-matched > obvious-hack`.

## How to build it on this skill
- Use `harness.run_pipeline` to score each variant against the real AnnData (PyDESeq2 backend).
- Use `harness.compare_signatures` / `de_summary` to verify a corruption preserved the statistic it
  was supposed to (e.g. same marginal LFC distribution) while breaking structure.
- Constraints come from the same resources as tests 7–8 (pathways, GRNs/regulons, PPIs).
- Corruption generators operate on the predicted count matrix / DE frame:
  - *shuffle covariance*: permute each gene's values across cells independently (kills co-expression)
  - *sign inversion*: flip LFC sign of TF targets relative to the TF
  - *pathway-incoherent DE*: redistribute significant calls to random genes preserving the count
  - *variance amplification*: scale per-gene variance without moving means

## Outputs
- Per-metric vulnerability report (which metrics fail to separate plausible vs impossible).
- Biological plausibility profiles per generated variant.
- Alignment scores: plausible-vs-impossible AUC; biological adversarial sensitivity.

## Agentic loop (optional)
Claude Code (or LangChain/DeepAgents) can automate: retrieve evidence → propose
counterfactuals/corruptions → run cell-eval → critique metric behavior → suggest metric hardening.
Keep every generated variant + constraint auditable so failures are explainable.
