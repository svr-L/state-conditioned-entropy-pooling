from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from .probabilities import normalize_probabilities


def weighted_tail_weights(returns, probabilities, alpha: float = 0.05):
    """Exact alpha-tail masses, allowing a partial probability atom at VaR."""
    r = np.asarray(returns, dtype=float).reshape(-1)
    p = normalize_probabilities(probabilities)
    if len(r) != len(p):
        raise ValueError("returns and probabilities must have the same length")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    order = np.argsort(r)
    r_s, p_s = r[order], p[order]
    take_s = np.zeros_like(p_s)
    remaining = float(alpha)
    for i, mass in enumerate(p_s):
        if remaining <= 1e-15:
            break
        use = min(float(mass), remaining)
        take_s[i] = use
        remaining -= use
    if remaining > 1e-8:
        raise RuntimeError("Unable to allocate the requested tail probability mass")
    take = np.zeros_like(take_s)
    take[order] = take_s
    crossing = int(np.searchsorted(np.cumsum(p_s), alpha, side="left"))
    var_return = float(r_s[min(crossing, len(r_s) - 1)])
    return take, var_return


def weighted_var_es(returns, probabilities, alpha: float = 0.05) -> dict:
    r = np.asarray(returns, dtype=float).reshape(-1)
    p = normalize_probabilities(probabilities)
    tail_mass, var_return = weighted_tail_weights(r, p, alpha=alpha)
    es_return = float(np.sum(tail_mass * r) / alpha)
    mean = float(np.sum(p * r))
    var = float(np.sum(p * (r - mean) ** 2))
    return {
        "mean_return": mean,
        "volatility": float(np.sqrt(max(var, 0.0))),
        f"VaR_{int((1-alpha)*100)}_loss": -var_return,
        f"ES_{int((1-alpha)*100)}_loss": -es_return,
        "tail_mass": float(tail_mass.sum()),
    }


def weighted_min_cvar(
    instrument_returns,
    probabilities,
    *,
    alpha: float = 0.05,
    max_weight: float = 0.50,
):
    """Rockafellar-Uryasev long-only weighted minimum-CVaR linear program."""
    if isinstance(instrument_returns, pd.DataFrame):
        df = instrument_returns.copy()
        names = df.columns
        R = df.to_numpy(dtype=float)
    else:
        R = np.asarray(instrument_returns, dtype=float)
        names = [f"asset_{i}" for i in range(R.shape[1])]
        df = pd.DataFrame(R, columns=names)
    p = normalize_probabilities(probabilities)
    n_scen, n_assets = R.shape
    if len(p) != n_scen:
        raise ValueError("Scenario probabilities do not match instrument-return rows")

    c = np.concatenate([np.zeros(n_assets), [1.0], p / alpha])
    A_loss = sparse.hstack(
        [
            sparse.csr_matrix(-R),
            sparse.csr_matrix(-np.ones((n_scen, 1))),
            -sparse.eye(n_scen, format="csr"),
        ],
        format="csr",
    )
    b_loss = np.zeros(n_scen)
    A_eq = sparse.csr_matrix(np.concatenate([np.ones(n_assets), np.zeros(1 + n_scen)])[None, :])
    b_eq = np.array([1.0])
    bounds = [(0.0, max_weight)] * n_assets + [(None, None)] + [(0.0, None)] * n_scen
    res = linprog(c, A_ub=A_loss, b_ub=b_loss, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Weighted min-CVaR LP failed: {res.message}")

    weights = pd.Series(res.x[:n_assets], index=names, name="weight")
    port = R @ weights.to_numpy()
    stats = weighted_var_es(port, p, alpha=alpha)
    info = pd.Series({
        "LP objective": float(res.fun),
        "zeta": float(res.x[n_assets]),
        "max weight": float(weights.max()),
        "sum weights": float(weights.sum()),
        **stats,
    })
    return weights, pd.Series(port, index=df.index), info
