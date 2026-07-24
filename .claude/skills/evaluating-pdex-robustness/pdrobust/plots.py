"""Diagnostic plots for the robustness battery (matplotlib, headless)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402


def genomic_inflation(pvals: np.ndarray) -> float:
    """Genomic-control lambda: median chi2(1) from p-values / 0.4549.

    ~1.0 for a well-calibrated null; >1 indicates inflation (anti-conservative),
    <1 deflation (conservative).
    """
    p = pvals[np.isfinite(pvals)]
    p = np.clip(p, 1e-300, 1.0)
    chi2 = stats.chi2.isf(p, df=1)
    return float(np.median(chi2) / stats.chi2.ppf(0.5, df=1))


def qq_uniform_plot(
    pvals: np.ndarray,
    out_path: str,
    *,
    title: str,
    subtitle: str | None = None,
    max_points: int = 50000,
) -> dict[str, float]:
    """QQ-plot of p-values against the Uniform[0,1] null, on -log10 axes.

    lambda_GC and n are computed on all finite p-values; for legibility/speed the
    plotted points are thinned to ``max_points`` (sorted-order-preserving stride).
    Saves a PNG to ``out_path`` and returns {lambda_gc, n, frac_p_eq_1}.
    """
    p = np.asarray(pvals, dtype=float)
    p = p[np.isfinite(p)]
    n = p.size
    frac_one = float(np.mean(p >= 1.0)) if n else float("nan")
    lam = genomic_inflation(p) if n else float("nan")

    p_sorted = np.sort(np.clip(p, 1e-300, 1.0))
    k = np.arange(1, n + 1)
    expected = (k - 0.5) / n
    # thin plotted points (keep extremes) while leaving lambda/n on the full set
    if n > max_points:
        idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    else:
        idx = np.arange(n)
    x = -np.log10(expected[idx])
    y = -np.log10(p_sorted[idx])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(x, y, s=5, alpha=0.35, color="#1a3c6e", rasterized=True, edgecolors="none")
    lim = float(max(x.max(), y.max())) if n else 1.0
    ax.plot([0, lim], [0, lim], color="#c0392b", lw=1.2, ls="--", label="null (y = x)")
    # 95% pointwise band under the uniform null (Beta order statistics)
    lo = stats.beta.ppf(0.025, k[idx], n - k[idx] + 1)
    hi = stats.beta.ppf(0.975, k[idx], n - k[idx] + 1)
    ax.fill_between(x, -np.log10(hi), -np.log10(lo), color="#999999", alpha=0.20,
                    label="95% null band")
    ax.set_xlabel(r"Expected $-\log_{10}(p)$  (Uniform)")
    ax.set_ylabel(r"Observed $-\log_{10}(p)$")
    full_title = title + (f"\n{subtitle}" if subtitle else "")
    ax.set_title(full_title, fontsize=10)
    ax.text(0.04, 0.93, rf"$\lambda_{{GC}}$ = {lam:.3f}    n = {n:,}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.8))
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"lambda_gc": lam, "n": float(n), "frac_p_eq_1": frac_one}
