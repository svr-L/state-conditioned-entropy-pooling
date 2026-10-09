from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np

from .entropy import entropy_pool, EntropyPoolResult
from .probabilities import normalize_probabilities, state_conditioned_prior


@dataclass(frozen=True)
class EventView:
    """Lower-bound probability view on a scenario event."""
    name: str
    indicator: np.ndarray
    target_probability: float


def event_uplift(
    signal_history,
    current_signal: float,
    event_history,
    *,
    fraction: float = 0.10,
    min_count: int = 100,
    bandwidth_floor: float = 0.35,
) -> float:
    """Conditional/unconditional event-probability ratio from 1-D kernel analogues."""
    z = np.asarray(signal_history, dtype=float).reshape(-1)
    y = np.asarray(event_history, dtype=float).reshape(-1)
    if len(z) != len(y):
        raise ValueError("signal_history and event_history must have the same length")
    ok = np.isfinite(z) & np.isfinite(y)
    if ok.sum() < min_count or not np.isfinite(current_signal):
        return 1.0
    zl, yl = z[ok], y[ok]
    d = np.abs(zl - float(current_signal))
    k = int(min(len(zl), max(min_count, np.ceil(fraction * len(zl)))))
    bw = max(float(np.partition(d, k - 1)[k - 1]), float(bandwidth_floor))
    w = np.exp(-0.5 * (d / bw) ** 2)
    w = w / w.sum()
    cond = float(w @ yl)
    uncond = float(np.mean(yl))
    return cond / max(uncond, 1e-12)


def uplift_target(prior_probability: float, uplift: float, *, deadband: float = 0.05) -> float:
    """Translate a multiplicative uplift into a stress-only event-probability target."""
    prior_probability = float(prior_probability)
    uplift = float(uplift)
    if uplift <= 1.0 + deadband:
        return prior_probability
    return float(np.clip(prior_probability * uplift, prior_probability, 1.0))


def solve_event_views(prior, views: list[EventView], *, solver_tolerance: float = 5e-7) -> EntropyPoolResult:
    q = normalize_probabilities(prior)
    rows, lows = [], []
    for view in views:
        row = np.asarray(view.indicator, dtype=float).reshape(-1)
        if len(row) != len(q):
            raise ValueError(f"View {view.name} does not match the scenario support")
        prior_value = float(q @ row)
        target = float(view.target_probability)
        has_support = np.nanmin(row) < np.nanmax(row)
        if has_support and target > prior_value + 1e-10:
            rows.append(row)
            lows.append(target)
    if not rows:
        return entropy_pool(q, solver_tolerance=solver_tolerance)
    # entropy_pool uses upper-bound inequalities, so -E[p*I] <= -target
    return entropy_pool(
        q,
        z_ineq=-np.vstack(rows),
        mu_ineq=-np.asarray(lows),
        solver_tolerance=solver_tolerance,
    )


def build_state_prior(
    state_history,
    current_state,
    *,
    fraction: float = 0.10,
    min_count: int = 100,
    bandwidth_floor: float = 0.35,
):
    """Named wrapper for the state-conditioned Flexible-Probabilities prior."""
    return state_conditioned_prior(
        state_history,
        current_state,
        fraction=fraction,
        min_count=min_count,
        bandwidth_floor=bandwidth_floor,
    )
