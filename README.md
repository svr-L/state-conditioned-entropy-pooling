# State-Conditioned Entropy Pooling

A Python research framework for **conditioning physical scenario distributions on market-state information** using **Flexible Probabilities**, **minimum-relative-entropy (Entropy Pooling)**, and **out-of-sample VaR/Expected-Shortfall evaluation**.

This repository is intentionally separate from two related projects:

- **multi-asset-scenario-engine** — owns the physical cross-asset scenario generator, horizon repricing stack and VaR/ES validation;
- **entropy-pooling-stress-testing** — owns standalone stress testing under explicit user-specified probabilistic and moment views;
- **state-conditioned-entropy-pooling** — asks a different question: *where should predictive state information enter the distribution, and does it add value out of sample?*

The project emerged as a research layer above the cross-asset engine and was subsequently redesigned around controlled experiments, channel-specific views, state-conditioned priors and preregistered validation.

## Research question

The core decomposition is

```text
physical scenario support
        ↓
state-conditioned prior probabilities (Flexible Probabilities)
        ↓
channel-specific probabilistic views
        ↓
minimum-relative-entropy posterior
        ↓
weighted VaR / ES / tail attribution / min-CVaR
        ↓
walk-forward factorial validation
```

The central empirical question is not whether Entropy Pooling can create a stressed posterior — it can by construction — but whether **state information belongs in the prior, in posterior views, or in both**, and whether the resulting conditioning improves tail calibration out of sample.

## What is implemented

### 1. Scenario-support interface

The conditioning layer treats the upstream scenario engine as a **support provider**. It does not alter the simulated states or repricer. It changes only scenario probabilities.

`ScenarioSupport` accepts:

- instrument-level scenario returns;
- scenario-level features / event indicators;
- an optional prior probability vector.

The package is therefore source-agnostic: the support can come from the companion cross-asset engine or any other physical scenario model.

### 2. State-conditioned Flexible-Probabilities prior

Historical horizon windows can be reweighted according to the distance between the **state preceding each historical window** and today's state. A Gaussian kernel with a k-nearest-neighbour bandwidth produces the prior:

```text
historical states + current state
        ↓
state distances
        ↓
Gaussian kernel weights
        ↓
Flexible-Probabilities prior
```

This generalizes time-decay weighting from chronological distance to **economic-state distance**.

### 3. Channel-specific views

The research branch originally used a generic multi-view funnel. Signal evaluation later showed that many signals collapsed into essentially the same posterior-tilt direction, so the layer was redesigned around **channel-specific event views**.

The reusable package exposes:

- kernel-analogue conditional event probabilities;
- conditional/unconditional probability uplifts;
- activation deadbands;
- lower-bound event-probability views;
- source-agnostic Entropy Pooling over the supplied scenario support.

The research notebook used equity-loss and credit-widening channels as the main examples; the public library keeps the machinery generic.

### 4. Minimum-relative-entropy solver

`entropy_pool` solves

```text
minimize    D_KL(p || q)
subject to  Z_ineq p <= mu_ineq
            Z_eq   p  = mu_eq
            p >= 0, 1'p = 1
```

through the dual problem with internally rescaled constraint rows. The result includes:

- posterior probabilities;
- dual multipliers;
- equality residuals;
- inequality violations;
- solver status and iterations.

### 5. Weighted portfolio risk

Posterior probabilities feed:

- exact weighted VaR;
- exact weighted Expected Shortfall with partial mass at the VaR atom;
- entropy effective scenario count;
- KL divergence from the prior;
- exact long-only weighted minimum-CVaR optimisation via linear programming.

### 6. View-set diagnostics

The package includes the ARPM-style non-redundancy sanity check used in the research branch:

- algebraic rank of the prior-measure view covariance;
- Roy-Vetterli effective rank;
- participation ratio as a cross-check;
- threshold-based pass/fail diagnostic.

This answers whether a view set is *capable* of expressing distinct directions under the prior. It is deliberately separate from posterior-tilt diagnostics, which ask whether different signals actually produce different posterior tilts.

### 7. Tail-forecast scoring

For out-of-sample comparison of conditioning designs, the package includes:

- exact binomial coverage p-value;
- Kupiec unconditional coverage;
- Christoffersen independence;
- Expected-Shortfall exceedance-residual test;
- FZ0 joint VaR-ES scoring loss;
- Diebold-Mariano comparison with HAC long-run variance.

These utilities support factorial experiments such as

```text
{uniform prior, state-conditioned FFP prior}
×
{views off, views on}
```

without hard-wiring the layer to one specific scenario engine.

### 8. Pre-registration / lockbox control

The research workflow includes a small audit utility that hashes a frozen design and records a single lockbox opening. Re-opening the same results under a different design hash is refused unless explicitly forced and recorded.

This is not presented as a substitute for research discipline; it is an audit trail designed to remove ambiguity about whether the evaluation design was changed after observing the held-out results.

## Research lineage

The package is the cleaned reusable layer extracted from a longer research sequence:

- **v20** — predictive-state Entropy Pooling overlay over a physical cross-asset support;
- **v21** — powered, purged/embargoed point-in-time signal evaluation and multiple-testing control;
- **v22** — 2x2 conditional-calibration factorial: FHS conditioning × state overlay;
- **v23** — data-directed redesign into state-conditioned FFP priors and channel-specific views;
- **v24** — production-style gates, full repricer factorial and preregistration / lockbox;
- **v25** — closed-form mechanism diagnosis of a conditional-correlation discrepancy, verified against Monte Carlo and a state-matched historical benchmark;
- **v26** — effective-rank sanity check for the view set.

The repository does **not** claim that every predictive hypothesis survived out of sample. One registered cross-portfolio ranking prediction did not hold in the production factorial. That result is retained as part of the research record rather than optimized away.

See [`docs/research_lineage.md`](docs/research_lineage.md) for the design evolution and what was learned at each stage.

## Repository layout

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── notebooks/
│   └── state_conditioned_entropy_pooling.ipynb
├── examples/
│   └── quickstart.py
├── docs/
│   ├── integration.md
│   └── research_lineage.md
├── src/
│   └── state_conditioned_entropy_pooling/
│       ├── __init__.py
│       ├── adapters.py
│       ├── probabilities.py
│       ├── entropy.py
│       ├── views.py
│       ├── diagnostics.py
│       ├── risk.py
│       ├── scoring.py
│       └── lockbox.py
└── tests/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

For the notebook / tests:

```bash
pip install -e ".[research,test]"
```

## Quick start

```python
import numpy as np
from state_conditioned_entropy_pooling import (
    EventView,
    entropy_pool,
    solve_event_views,
    weighted_var_es,
)

# q: prior probabilities over a fixed scenario support
q = np.full(5, 0.20)

# Event indicator over those five scenarios
credit_widen = np.array([0, 0, 0, 1, 1], dtype=float)

# Ask the posterior to assign at least 60% probability to the event
view = EventView("credit_widen_event", credit_widen, 0.60)
result = solve_event_views(q, [view])
p = result.probabilities

scenario_returns = np.array([0.02, 0.01, 0.00, -0.03, -0.06])
print(weighted_var_es(scenario_returns, p, alpha=0.20))
```

For a complete offline example:

```bash
python examples/quickstart.py
```

## Connecting to the Multi-Asset Scenario Engine

The recommended integration is deliberately thin:

```python
from state_conditioned_entropy_pooling import ScenarioSupport

support = ScenarioSupport(
    instrument_returns=instrument_returns,   # rows = physical scenarios
    scenario_features=scenario_features,     # event / moment variables on same rows
    prior=prior_probabilities,               # uniform, FFP, or another physical prior
)
```

The upstream engine remains responsible for scenario generation and repricing. This repository owns only **probability conditioning, diagnostics, scoring and experimental design**.

See [`docs/integration.md`](docs/integration.md).

## Why this is not the old Entropy-Pooling Stress-Testing repo

The standalone stress-testing project answers:

> *Given explicit stress views, how does the posterior distribution and portfolio tail risk change?*

This project answers:

> *Given point-in-time market-state information, how should the prior and/or views be conditioned, and does that conditioning improve out-of-sample tail calibration?*

The common numerical primitive is Entropy Pooling; the research objective is different enough to justify a separate repository.

## Important limitations

- The package is a research layer, not a production forecasting or risk platform.
- State-conditioned priors and analogue views are only as good as the state variables, event definitions and historical support supplied to them.
- Entropy Pooling preserves the scenario support; it does not create outcomes absent from the upstream generator.
- View feasibility can fail when the supplied support contains too little probability mass in the requested event region.
- Multiple-testing and signal-selection bias remain first-order concerns in predictive applications.
- A view set can have high algebraic/effective rank yet still generate nearly collinear posterior tilts once actual signals are mapped into it.
- The research results motivate the architecture but do not establish universal predictive superiority of state conditioning.

## Author

Saverio Lauriola
