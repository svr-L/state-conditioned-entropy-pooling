import numpy as np
import pandas as pd

from state_conditioned_entropy_pooling import weighted_var_es, weighted_min_cvar


def test_weighted_var_es_tail_mass():
    r = np.array([-0.10, -0.05, 0.00, 0.02])
    p = np.full(4, 0.25)
    out = weighted_var_es(r, p, alpha=0.25)
    assert np.isclose(out["tail_mass"], 0.25)
    assert np.isclose(out["VaR_75_loss"], 0.10)
    assert np.isclose(out["ES_75_loss"], 0.10)


def test_min_cvar_weights_are_feasible():
    R = pd.DataFrame({
        "a": [-0.10, 0.02, 0.02, 0.02],
        "b": [-0.02, -0.01, 0.01, 0.01],
    })
    p = np.full(len(R), 0.25)
    w, _, _ = weighted_min_cvar(R, p, alpha=0.25, max_weight=0.8)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-10).all()
    assert (w <= 0.8 + 1e-10).all()
