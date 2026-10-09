import numpy as np

from state_conditioned_entropy_pooling import entropy_pool, relative_entropy


def test_entropy_pool_event_view_increases_event_probability():
    q = np.full(5, 0.2)
    event = np.array([0, 0, 0, 1, 1], dtype=float)
    target = 0.60
    res = entropy_pool(q, z_ineq=-event[None, :], mu_ineq=np.array([-target]))
    assert res.success
    assert float(event @ res.probabilities) >= target - 1e-7
    assert np.isclose(res.probabilities.sum(), 1.0)
    assert relative_entropy(res.probabilities, q) > 0


def test_no_view_returns_prior():
    q = np.array([0.1, 0.2, 0.7])
    res = entropy_pool(q)
    assert res.success
    assert np.allclose(res.probabilities, q)
