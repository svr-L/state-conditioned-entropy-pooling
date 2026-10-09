"""Offline quick-start using a synthetic scenario support.

For the live project, replace the synthetic arrays with scenario and state outputs
from the separate multi-asset-scenario-engine repository.
"""
import numpy as np
import pandas as pd

from state_conditioned_entropy_pooling import (
    EventView,
    ScenarioSupport,
    posterior_summary,
    solve_event_views,
    state_conditioned_prior,
    weighted_var_es,
)

rng = np.random.default_rng(42)
n = 5_000
state_history = rng.normal(size=(n, 3))
query_state = np.array([1.25, 1.0, 0.5])
prior = state_conditioned_prior(state_history, query_state, fraction=0.10, min_count=100)

common = rng.normal(size=n)
returns = pd.DataFrame({
    "equity": 0.012 * common + 0.012 * rng.normal(size=n),
    "credit": 0.004 * common + 0.006 * rng.normal(size=n),
    "rates": -0.002 * common + 0.004 * rng.normal(size=n),
})
features = pd.DataFrame({
    "equity_loss_event": (returns["equity"] < -0.025).astype(float),
    "credit_widen_event": (returns["credit"] < -0.012).astype(float),
})
support = ScenarioSupport(returns, features, prior=prior)

views = [
    EventView("credit_widen_event", features["credit_widen_event"].to_numpy(), 0.12),
    EventView("equity_loss_event", features["equity_loss_event"].to_numpy(), 0.10),
]
posterior = solve_event_views(support.prior, views).probabilities

portfolio = 0.6 * returns["equity"].to_numpy() + 0.4 * returns["credit"].to_numpy()
print("Prior risk:", weighted_var_es(portfolio, support.prior))
print("Posterior risk:", weighted_var_es(portfolio, posterior))
print("Posterior diagnostics:", posterior_summary(support.prior, posterior))
