# Curated resources for biological-positive tests (7 & 8)

## Test 7 — `curated_targets_csv`
A CSV of known perturbation→target relationships. One row per relationship.

| column              | required | meaning                                                                 |
|---------------------|----------|-------------------------------------------------------------------------|
| `pert_gene`         | yes      | perturbation label; must be a value in `obs[pert_col]`                  |
| `target_gene`       | yes      | expected downstream gene; must be a value in `var_names`                |
| `expected_direction`| no       | `up`/`down` (also `+`/`-`, `pos`/`neg`); enables direction-accuracy     |
| `confidence`        | no       | free tier label (e.g. `high`/`medium`/`low`); carried into the report   |
| `relationship_type` | no       | e.g. direct-TF, pathway, complex, marker (informational)                |
| `source`            | no       | citation / DB (informational)                                           |

Example:
```csv
pert_gene,target_gene,expected_direction,confidence,relationship_type,source
GATA1,HBB,up,high,direct-TF,literature
GATA1,KLF1,up,high,direct-TF,literature
MYC,CDKN1A,down,medium,pathway,reactome
```

Recovery is scored per relationship: significant (`fdr < fdr_threshold`), rank percentile of
the target in the perturbation's DE list (by p-value), and direction agreement when given.
High-confidence direct relationships are expected to recover more strongly than indirect ones —
split by `confidence` in the output table to check this.

## Test 8 — `gene_sets_json`
A JSON mapping each perturbation gene to the genes of its expected pathway / regulon / signature:
```json
{
  "GATA1": ["HBB", "HBA1", "KLF1", "ALAS2", "SLC4A1"],
  "MYC":   ["CDKN1A", "CCND2", "NCL", "NPM1"]
}
```
Genes must match `var_names`. For each perturbation the test computes the AUROC of |LFC|
separating set members from non-members (>0.5 ⇒ the set is enriched among large-effect genes).
TF perturbations should enrich their regulons; signaling perturbations should recover pathway
response genes. Candidate sources: TF regulons (DoRothEA/CollecTRI), MSigDB/Reactome pathways,
Perturb-seq consensus signatures, curated literature sets.
