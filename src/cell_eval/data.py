import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

PERT_COL = "perturbation"
CELLTYPE_COL = "celltype"
CONTROL_VAR = "control"

N_CELLS = 1000
N_GENES = 100
N_PERTS = 10
N_CELLTYPES = 3
MAX_UMI = 1e6
NORM_TOTAL = 1e4

RANDOM_SEED = 42

OUTDIR = "TEST_OUTPUT_DIRECTORY"


def build_random_anndata(
    n_cells: int = N_CELLS,
    n_genes: int = N_GENES,
    n_perts: int = N_PERTS,
    n_celltypes: int = N_CELLTYPES,
    pert_col: str = PERT_COL,
    celltype_col: str = CELLTYPE_COL,
    control_var: str = CONTROL_VAR,
    random_state: int = RANDOM_SEED,
    as_sparse: bool = False,
    normlog: bool = True,
    normtotal: int | float = NORM_TOTAL,
) -> ad.AnnData:
    """Generate a random AnnData object for testing and tutorials."""
    if random_state is not None:
        np.random.seed(random_state)

    # Randomly sample a matrix
    matrix = np.random.randint(0, int(MAX_UMI), size=(n_cells, n_genes))

    # Normalize and log transform if required
    if normlog:
        matrix = matrix / matrix.sum(axis=1, keepdims=True) * normtotal
        matrix = np.log1p(matrix)

    # Convert to sparse if required
    if as_sparse:
        matrix = csr_matrix(matrix)

    return ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {
                pert_col: np.random.choice(
                    [f"pert_{i}" for i in range(n_perts)] + [control_var],
                    size=n_cells,
                    replace=True,
                ),
                celltype_col: np.random.choice(
                    [f"celltype_{i}" for i in range(n_celltypes)],
                    size=n_cells,
                    replace=True,
                ),
            },
            index=np.arange(n_cells).astype(str),
        ),
    )


def downsample_cells(
    adata: ad.AnnData,
    fraction: float = 0.5,
) -> ad.AnnData:
    """Downsample cells in an AnnData object.

    Copies the output to avoid memory overlaps.
    """
    assert 0 <= fraction <= 1, "Fraction must be between 0 and 1"
    mask = np.random.rand(adata.shape[0]) < fraction
    return adata[mask, :].copy()
