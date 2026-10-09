from __future__ import annotations

import numpy as np


def normalize_probabilities(p) -> np.ndarray:
    """Return a finite, non-negative probability vector summing to one."""
    p = np.asarray(p, dtype=float).reshape(-1)
    if np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError("Probabilities must be finite and non-negative.")
    total = float(p.sum())
    if total <= 0:
        raise ValueError("Probabilities must have positive mass.")
    return p / total


def effective_scenarios(p) -> float:
    """Entropy effective sample size exp(-sum p log p)."""
    p = normalize_probabilities(p)
    return float(np.exp(-np.sum(np.where(p > 0, p * np.log(p), 0.0))))


def relative_entropy(p, q) -> float:
    """KL divergence D_KL(p || q)."""
    p = normalize_probabilities(p)
    q = normalize_probabilities(q)
    if len(p) != len(q):
        raise ValueError("p and q must have the same length.")
    mask = p > 0
    return float(
        np.sum(p[mask] * (np.log(p[mask]) - np.log(np.clip(q[mask], 1e-300, None))))
    )


def gaussian_kernel_weights(distances, bandwidth: float) -> np.ndarray:
    distances = np.asarray(distances, dtype=float).reshape(-1)
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    w = np.exp(-0.5 * (distances / bandwidth) ** 2)
    return normalize_probabilities(w)


def kernel_analogue_weights(
    state_history,
    query_state,
    *,
    fraction: float = 0.10,
    min_count: int = 100,
    bandwidth_floor: float = 0.35,
):
    """Gaussian-kernel weights over historical states, using a k-NN bandwidth.

    Parameters
    ----------
    state_history : array-like, shape (n_history, n_state)
        Point-in-time state vectors attached to historical horizon windows.
    query_state : array-like, shape (n_state,)
        Current state.
    fraction, min_count :
        The bandwidth is the distance to k=max(min_count, ceil(fraction*n)) analogues,
        capped at n.
    bandwidth_floor :
        Lower bound on the state-space bandwidth.
    """
    H = np.asarray(state_history, dtype=float)
    q = np.asarray(query_state, dtype=float).reshape(-1)
    if H.ndim != 2 or H.shape[1] != len(q):
        raise ValueError("state_history must be 2-D and match query_state dimension")
    ok = np.all(np.isfinite(H), axis=1)
    H_ok = H[ok]
    if len(H_ok) == 0 or not np.all(np.isfinite(q)):
        raise ValueError("No finite historical states or non-finite query state")
    dist = np.sqrt(np.sum((H_ok - q[None, :]) ** 2, axis=1))
    k = int(min(len(H_ok), max(min_count, np.ceil(fraction * len(H_ok)))))
    kth = float(np.partition(dist, max(k - 1, 0))[max(k - 1, 0)])
    bw = max(kth, float(bandwidth_floor))
    weights_ok = gaussian_kernel_weights(dist, bw)
    weights = np.zeros(H.shape[0], dtype=float)
    weights[np.flatnonzero(ok)] = weights_ok
    return weights, dist, bw


def state_conditioned_prior(
    state_history,
    query_state,
    *,
    fraction: float = 0.10,
    min_count: int = 100,
    bandwidth_floor: float = 0.35,
) -> np.ndarray:
    """Flexible-probabilities prior obtained by kernel weighting historical states."""
    weights, _, _ = kernel_analogue_weights(
        state_history,
        query_state,
        fraction=fraction,
        min_count=min_count,
        bandwidth_floor=bandwidth_floor,
    )
    return normalize_probabilities(weights)
