# Integration with the Multi-Asset Scenario Engine

The repositories should remain loosely coupled.

## Ownership boundary

**Multi-Asset Scenario Engine** owns:

- market data and risk-driver construction;
- physical cross-asset scenario generation;
- conditional volatility / level dynamics;
- horizon market-state reconstruction;
- Treasury, corporate-bond and FX-forward repricing;
- instrument-level scenario returns.

**State-Conditioned Entropy Pooling** owns:

- Flexible-Probabilities priors;
- state / signal analogue weighting;
- probabilistic views;
- minimum-relative-entropy reweighting;
- posterior diagnostics;
- posterior-weighted risk and portfolio optimisation;
- predictive factorial scoring and research audit controls.

## Minimal hand-off

The layer needs two row-aligned objects:

```python
instrument_returns   # DataFrame: scenarios x instruments
scenario_features    # DataFrame: scenarios x conditioning/view variables
```

Optionally pass an initial physical prior. Uniform probabilities are used otherwise.

```python
from state_conditioned_entropy_pooling import ScenarioSupport

support = ScenarioSupport(
    instrument_returns=instrument_returns,
    scenario_features=scenario_features,
    prior=physical_prior,
)
```

Examples of scenario features include:

- equity horizon return;
- equity-loss indicator;
- credit spread change;
- credit-widening indicator;
- rate sell-off indicator;
- joint event indicators.

## State-conditioned prior

The historical state panel must correspond to the state **known immediately before each historical horizon window**.

```python
from state_conditioned_entropy_pooling import state_conditioned_prior

ffp_prior = state_conditioned_prior(
    historical_states,
    current_state,
    fraction=0.10,
    min_count=100,
    bandwidth_floor=0.35,
)
```

When the support is the current Monte-Carlo / bootstrap scenario set rather than historical windows, the state-conditioned historical distribution can be used to estimate view targets or to construct a prior over a reduced-form historical support. Do not silently map weights across unrelated support points.

## Posterior views

For a probabilistic credit-widening view:

```python
from state_conditioned_entropy_pooling import EventView, solve_event_views

view = EventView(
    "credit_widen_event",
    support.scenario_features["credit_widen_event"].to_numpy(),
    target_probability=0.12,
)

result = solve_event_views(support.prior, [view])
posterior = result.probabilities
```

The scenario generator and repricer are untouched; only scenario probabilities change.

## Posterior risk

```python
from state_conditioned_entropy_pooling import weighted_var_es

portfolio_return = support.instrument_returns @ portfolio_weights
risk = weighted_var_es(portfolio_return, posterior, alpha=0.05)
```

## Research caution

Keep three objects conceptually separate:

1. the **physical scenario support**;
2. the **prior probability measure** on that support;
3. the **posterior probability measure** after views.

This separation is the point of using the layer as a separate repository rather than embedding every conditioning choice inside the scenario generator.
