"""Sparse-aware helpers shared between the cpu and gpu edistance paths.

These deliberately have no top-level cupy / rsc dependency — cupy is imported
lazily inside :func:`_to_host` so that cpu-only installs can import this
module freely.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def to_host(X):
    """Host view of X: numpy ndarray or scipy.sparse, accepting cupy / cupyx.sparse too."""
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cp_sparse
    except ImportError:
        cp = None
        cp_sparse = None
    if cp_sparse is not None and cp_sparse.issparse(X):
        return X.get()
    if cp is not None and isinstance(X, cp.ndarray):
        return cp.asnumpy(X)
    return X


def stack_real_pred_sparse(real_X, pred_X) -> tuple[object, int]:
    """Concatenate real / pred matrices on host, preserving sparsity where possible.

    Accepts any combination of numpy ndarray, scipy.sparse, cupy ndarray, or
    cupyx.scipy.sparse. Returns ``(merged, n_real)``: ``merged`` is a scipy CSR
    matrix if any input was sparse, otherwise a numpy ndarray.
    """
    real_h = to_host(real_X)
    pred_h = to_host(pred_X)
    n_real = real_h.shape[0]
    any_sparse = sp.issparse(real_h) or sp.issparse(pred_h)
    if any_sparse:
        a = real_h if sp.issparse(real_h) else sp.csr_matrix(real_h)
        b = pred_h if sp.issparse(pred_h) else sp.csr_matrix(pred_h)
        return sp.vstack([a, b], format="csr"), n_real
    return np.vstack([real_h, pred_h]), n_real
