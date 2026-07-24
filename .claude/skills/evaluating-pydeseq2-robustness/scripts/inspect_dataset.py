#!/usr/bin/env python
"""Fingerprint an AnnData for PyDESeq2-backend robustness testing.

Reads the file backed (no full load) and reports what the robustness battery
needs: a perturbation column + control label, a replicate/sample column (PyDESeq2
pseudobulk unit), a raw-counts source (layer or .X), and optional sgRNA / target-gene
/ blocking columns. Emits a JSON config skeleton between BEGIN_JSON/END_JSON.

    uv run python .../inspect_dataset.py /path/to/data.h5ad
"""

from __future__ import annotations

import argparse
import json
import sys

import anndata as ad
import numpy as np


def _is_intish(x: np.ndarray) -> bool:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return False
    return bool(np.allclose(x, np.rint(x), atol=1e-6))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("h5ad")
    args = ap.parse_args()

    adata = ad.read_h5ad(args.h5ad, backed="r")
    obs = adata.obs
    print(f"# {args.h5ad}")
    print(f"shape: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"layers: {list(adata.layers.keys())}")
    print(f"obsm: {list(adata.obsm.keys())}")
    print("\n## obs columns")
    cand_pert, cand_rep, cand_block = [], [], []
    for c in obs.columns:
        col = obs[c]
        try:
            nun = int(col.nunique(dropna=True))
        except TypeError:
            continue
        dtype = str(col.dtype)
        print(f"  {c:30s} dtype={dtype:12s} n_unique={nun}")
        is_str = col.dtype == object or str(col.dtype) in ("category", "string")
        if is_str and 2 <= nun <= max(5, adata.n_obs // 20):
            cand_pert.append(c)
        if is_str and 2 <= nun <= 200:
            cand_rep.append(c)
        if is_str and 2 <= nun <= 50:
            cand_block.append(c)

    # raw-counts detection (sample up to 2000 cells)
    n = min(2000, adata.n_obs)
    Xsamp = adata[:n].to_memory().X
    Xarr = Xsamp.toarray() if hasattr(Xsamp, "toarray") else np.asarray(Xsamp)
    x_is_raw = _is_intish(Xarr.ravel()[:200000])
    raw_layers = []
    for lname in adata.layers.keys():
        L = adata[:n].to_memory().layers[lname]
        Larr = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
        if _is_intish(Larr.ravel()[:200000]):
            raw_layers.append(lname)

    print(f"\n## counts\n  .X looks like raw integer counts: {x_is_raw}")
    print(f"  raw-integer layers: {raw_layers}")

    # heuristic control-label guesses on the best pert candidate
    control_guess = None
    pert_guess = cand_pert[0] if cand_pert else (cand_rep[0] if cand_rep else None)
    if pert_guess:
        vals = obs[pert_guess].astype(str)
        for token in ("non-targeting", "non_targeting", "control", "ctrl", "NTC", "DMSO", "safe-harbor"):
            hit = [v for v in vals.unique() if token.lower() in str(v).lower()]
            if hit:
                control_guess = hit[0]
                break

    sgrna_guess = next((c for c in obs.columns if "guide" in c.lower() or "sgrna" in c.lower()), None)
    target_guess = next((c for c in obs.columns if "target" in c.lower() or "gene" in c.lower()), None)

    counts_layer = raw_layers[0] if raw_layers else None
    skeleton = {
        "adata_path": args.h5ad,
        "pert_col": pert_guess,
        "control_pert": control_guess,
        "replicate_col": cand_rep[0] if cand_rep else None,
        "counts_layer": counts_layer,
        "allow_discrete": counts_layer is None and x_is_raw,
        "target_gene_col": target_guess,
        "sgrna_col": sgrna_guess,
        "block_cols": [c for c in cand_block if c not in {pert_guess, sgrna_guess}][:3],
        "fdr_threshold": 0.05,
        "num_threads": 8,
        "n_repeats": 3,
        "downsample_grid": [25, 50, 100, 200, 500, 1000],
        "control_downsample_grid": [500, 1000, 2000, 5000],
        "stable_lfc_corr": 0.9,
        "min_pseudobulk_cells": 10,
        "outdir": "./pdrobust-out",
        "tests": ["all"],
        "_curated_targets_csv_optional": "path/to/curated.csv (test 7)",
        "_gene_sets_json_optional": "path/to/gene_sets.json (test 8)",
    }
    print("\n## candidate columns")
    print(f"  perturbation: {cand_pert}")
    print(f"  replicate   : {cand_rep}")
    print(f"  blocking    : {cand_block}")
    print("\nReview/edit the skeleton below, save as config.yaml/json, then run run_robustness.py.")
    print("\nBEGIN_JSON")
    print(json.dumps(skeleton, indent=2))
    print("END_JSON")

    warnings = []
    if not x_is_raw and not raw_layers:
        warnings.append("No raw-integer counts found in .X or layers — PyDESeq2 will reject this.")
    if not cand_rep:
        warnings.append("No replicate/sample column candidate — PyDESeq2 needs one for pseudobulk.")
    if control_guess is None:
        warnings.append("Could not guess a control label — set control_pert manually.")
    if warnings:
        print("\n## WARNINGS")
        for w in warnings:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
