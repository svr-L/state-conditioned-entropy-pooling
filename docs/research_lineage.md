# Research lineage: v20 → v26

This note records the reasoning that turned the original predictive-state overlay into the cleaned `state-conditioned-entropy-pooling` package.

## v20 — predictive-state entropy overlay

The physical scenario engine was frozen as a support provider. A separate layer built a point-in-time cross-asset state library, mapped market-turbulence / growth-credit / inflation-term-premium states through kernel-weighted historical analogues into probabilistic and moment views, and solved for minimum-relative-entropy posterior probabilities.

The first version also included:

- signal-, view- and transmission-space redundancy diagnostics;
- a 3×3×3 primitive stress grid;
- entropy-cost versus ES-impact frontiers;
- leave-one-view-out attribution;
- posterior min-CVaR and tail contributions.

Key design decision: the generator and repricer were left unchanged. The overlay changed probabilities only.

## v21 — powered signal evaluation

The project moved from a fixed intuitive state triplet to a point-in-time signal-evaluation stream:

- broad price-derived candidate register;
- purged / embargoed walk-forward design;
- in-window event thresholds;
- kernel-analogue forecasts;
- HAC Diebold-Mariano inference;
- family-wise multiple-testing control;
- streaming candidate selection with hysteresis;
- tilt-space redundancy diagnostics.

This was the first step from "interesting stress states" toward falsifiable predictive conditioning.

## v22 — where should state information live?

A 2×2 reduced-form factorial explicitly separated:

- FHS volatility conditioning in the prior;
- state information in the Entropy-Pooling overlay.

Each arm was scored with VaR coverage, ES residual tests and FZ0 joint VaR-ES loss. The exercise exposed a structural issue: stress-only views can repair under-coverage but are mechanically ill-suited to already conservative forecasts.

The absolute-transfer variant was rejected by the validation harness; the uplift-ratio design replaced it.

## v23 — data-directed redesign

Three conclusions from the earlier experiments were hard-coded into the redesign:

1. **Remove the second-moment view** from the posterior layer: conditional volatility belongs to the conditioned prior / FHS channel.
2. **Replace the generic multi-view funnel with channel-specific views** because many signals collapsed into approximately one posterior-tilt direction.
3. **Add a state-conditioned Flexible-Probabilities prior** using Gaussian-kernel weights on the distance between the state preceding each historical horizon and today's state.

The main channel map became:

- cyclical-underperformance signal → credit-widening event;
- equity-breadth-stress signal → equity-loss event.

The first had survived the multiple-testing screen; the second remained explicitly suggestive.

## v24 — gates, full repricer and preregistration

The research design was hardened before stronger out-of-sample claims:

- conditional-dependence gate with arm-specific historical references;
- convexity stress on the HY sensitivity bridge;
- production-style factorial using the full scenario engine / repricer at monthly anchors;
- reference portfolios fixed before scoring;
- FZ0 / DM scoring;
- registered cross-portfolio prediction;
- hashed design and one-opening lockbox control.

Synthetic harnesses were used to verify that the test machinery could recover planted effects before using the real data.

## v25 — mechanism diagnosis rather than parameter tweaking

A conditional-correlation discrepancy was decomposed into channels rather than "fixed" by tuning.

The analysis refuted two candidate explanations and showed that the mismatch lived in the filtering channel. A closed-form decomposition reproduced the observed correlation collapse and was cross-checked against a 60k-path Monte Carlo experiment. The historical benchmark was then rebuilt on the **same state variables used by the engine**, making the comparison like-for-like.

This version is important because it changed the project from feature accumulation into model-mechanism diagnosis.

## v26 — view-set non-redundancy

The final research version added the ARPM-style sanity check on the view set:

- full algebraic rank of the prior-measure view covariance;
- Roy-Vetterli effective rank;
- participation ratio;
- leave-one-view-out effective-rank attribution.

The distinction is important:

- **view-set rank** asks whether the design can represent independent directions under the prior;
- **posterior-tilt rank** asks whether the selected signals actually generate distinct posterior tilts.

The research view set passed the prior-measure rank sanity check even though earlier signal-to-view mappings had shown severe tilt-space collapse. Both statements can be true.

## Negative result retained

The production factorial included a registered prediction that the credit-channel view would improve calibration most for the `credit_tilt` portfolio and least for `equity_only`.

That ranking did **not** hold in the observed sample: `equal_weight` ranked highest in that run. The result is intentionally retained. The package therefore exposes the experimental machinery but does not advertise a universal predictive improvement.

## What the cleaned package keeps

The reusable repository keeps the pieces that are methodologically portable:

- state-conditioned Flexible-Probabilities priors;
- kernel-analogue event uplifts;
- channel-specific event views;
- dual minimum-relative-entropy solver;
- effective-rank / participation-ratio diagnostics;
- weighted VaR / ES / min-CVaR;
- FZ0, coverage and DM scoring utilities;
- preregistration / lockbox audit control;
- a thin scenario-support adapter.

The full cross-asset generator/repricer stays in the separate `multi-asset-scenario-engine` repository, while the old explicit-view stress-testing engine remains a separate project as well.
