from __future__ import annotations

import numpy as np
import pandas as pd

from .probabilities import normalize_probabilities, relative_entropy, effective_scenarios


def effective_rank(cov) -> float:
    """Roy-Vetterli effective rank: exp(entropy of normalized eigenvalues)."""
    lam = np.clip(np.linalg.eigvalsh(np.asarray(cov, dtype=float)), 0.0, None)
    total = float(lam.sum())
    if total <= 0:
        return np.nan
    w = lam / total
    w = w[w > 1e-15]
    return float(np.exp(-np.sum(w * np.log(w))))


def participation_ratio(cov) -> float:
    lam = np.clip(np.linalg.eigvalsh(np.asarray(cov, dtype=float)), 0.0, None)
    s1, s2 = float(lam.sum()), float((lam ** 2).sum())
    return float(s1 * s1 / s2) if s2 > 0 else np.nan


def weighted_covariance_rows(feature_rows, probabilities):
    G = np.atleast_2d(np.asarray(feature_rows, dtype=float))
    p = normalize_probabilities(probabilities)
    if G.shape[1] != len(p):
        raise ValueError("feature_rows must have one value per scenario")
    mu = G @ p
    X = G - mu[:, None]
    return (X * p[None, :]) @ X.T


def view_set_diagnostics(feature_rows, probabilities, *, threshold_fraction: float = 0.5) -> dict:
    """ARPM-style non-redundancy diagnostic for a set of view variables."""
    G = np.atleast_2d(np.asarray(feature_rows, dtype=float))
    p = normalize_probabilities(probabilities)
    cov = weighted_covariance_rows(G, p)
    k = int(np.linalg.matrix_rank(cov, tol=1e-10))
    er = effective_rank(cov)
    pr = participation_ratio(cov)
    threshold = threshold_fraction * G.shape[0]
    return {
        "n_views": int(G.shape[0]),
        "algebraic_rank": k,
        "effective_rank": er,
        "participation_ratio": pr,
        "threshold": float(threshold),
        "passes": bool(k == G.shape[0] and er >= threshold),
        "covariance": cov,
    }


def posterior_summary(prior, posterior) -> dict:
    q = normalize_probabilities(prior)
    p = normalize_probabilities(posterior)
    return {
        "relative_entropy": relative_entropy(p, q),
        "prior_effective_n": effective_scenarios(q),
        "posterior_effective_n": effective_scenarios(p),
        "posterior_effective_ratio": effective_scenarios(p) / len(p),
        "max_probability": float(p.max()),
        "max_prior_ratio": float(np.max(p / np.clip(q, 1e-300, None))),
    }


def leave_one_view_out(full_metric: float, loo_metrics: dict[str, float]) -> pd.Series:
    """Generic attribution: positive values mean the omitted view reduced the metric."""
    return pd.Series({name: full_metric - value for name, value in loo_metrics.items()}, name="marginal_contribution")
