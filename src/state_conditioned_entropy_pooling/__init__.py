"""State-conditioned Flexible Probabilities / Entropy Pooling research layer."""

from .adapters import ScenarioSupport
from .diagnostics import effective_rank, participation_ratio, view_set_diagnostics, posterior_summary
from .entropy import EntropyPoolResult, entropy_pool
from .lockbox import design_hash, open_lockbox
from .probabilities import (
    normalize_probabilities,
    effective_scenarios,
    relative_entropy,
    kernel_analogue_weights,
    state_conditioned_prior,
)
from .risk import weighted_tail_weights, weighted_var_es, weighted_min_cvar
from .scoring import (
    fz0_loss,
    exact_binomial_pvalue,
    kupiec_test,
    christoffersen_independence,
    es_residual_test,
    diebold_mariano,
)
from .views import EventView, event_uplift, uplift_target, solve_event_views, build_state_prior

__all__ = [
    "ScenarioSupport",
    "EntropyPoolResult",
    "entropy_pool",
    "normalize_probabilities",
    "effective_scenarios",
    "relative_entropy",
    "kernel_analogue_weights",
    "state_conditioned_prior",
    "weighted_tail_weights",
    "weighted_var_es",
    "weighted_min_cvar",
    "effective_rank",
    "participation_ratio",
    "view_set_diagnostics",
    "posterior_summary",
    "EventView",
    "event_uplift",
    "uplift_target",
    "solve_event_views",
    "build_state_prior",
    "fz0_loss",
    "exact_binomial_pvalue",
    "kupiec_test",
    "christoffersen_independence",
    "es_residual_test",
    "diebold_mariano",
    "design_hash",
    "open_lockbox",
]
