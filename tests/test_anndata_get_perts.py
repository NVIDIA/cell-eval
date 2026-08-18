import numpy as np

from cell_eval import PerturbationAnndataPair
from cell_eval.data import build_random_anndata


def test_get_perts_include_control():
    """get_perts(include_control=True) must include the control perturbation.

    Before the fix, `include_control=True` returned `self.perts`, which had
    already been filtered to exclude the control in `__post_init__`, so the
    flag was a no-op.
    """
    adata = build_random_anndata(random_state=42)
    pair = PerturbationAnndataPair(
        real=adata,
        pred=adata,
        pert_col="perturbation",
        control_pert="control",
    )

    perts_excluding_control = pair.get_perts(include_control=False)
    perts_including_control = pair.get_perts(include_control=True)

    assert "control" not in perts_excluding_control
    assert len(perts_excluding_control) == 10

    assert "control" in perts_including_control
    assert len(perts_including_control) == 11
    assert np.array_equal(perts_including_control, np.sort(perts_including_control))
