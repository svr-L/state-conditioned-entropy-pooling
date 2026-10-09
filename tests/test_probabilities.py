import numpy as np

from state_conditioned_entropy_pooling import effective_scenarios, state_conditioned_prior


def test_effective_scenarios_uniform():
    p = np.full(10, 0.1)
    assert np.isclose(effective_scenarios(p), 10.0)


def test_state_conditioned_prior_concentrates_near_query():
    states = np.arange(10, dtype=float)[:, None]
    p = state_conditioned_prior(states, np.array([9.0]), min_count=2, fraction=0.2, bandwidth_floor=0.1)
    assert np.argmax(p) == 9
    assert np.isclose(p.sum(), 1.0)
