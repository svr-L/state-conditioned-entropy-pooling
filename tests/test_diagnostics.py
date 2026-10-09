import numpy as np

from state_conditioned_entropy_pooling import view_set_diagnostics


def test_effective_rank_diagnostic_full_rank():
    features = np.array([
        [1, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 1],
    ], dtype=float)
    p = np.full(features.shape[1], 1 / features.shape[1])
    out = view_set_diagnostics(features, p, threshold_fraction=0.5)
    assert out["algebraic_rank"] == 3
    assert out["effective_rank"] > 1.5
