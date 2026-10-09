from __future__ import annotations

import numpy as np
from scipy import stats


def long_run_variance(x, lag: int = 1) -> float:
    """Simple Bartlett/Newey-West long-run variance estimate."""
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    n = len(x)
    if n <= 1:
        return np.nan
    y = x - x.mean()
    gamma0 = float(y @ y / n)
    out = gamma0
    L = int(max(0, min(lag, n - 1)))
    for k in range(1, L + 1):
        gamma = float(y[k:] @ y[:-k] / n)
        weight = 1.0 - k / (L + 1.0)
        out += 2.0 * weight * gamma
    return max(out, 0.0)


def fz0_loss(realized, var_loss, es_loss, alpha: float):
    """FZ0 joint VaR-ES scoring loss; lower is better."""
    r = np.asarray(realized, dtype=float)
    v = -np.asarray(var_loss, dtype=float)
    e = np.minimum(-np.asarray(es_loss, dtype=float), v - 1e-12)
    hit = (r <= v).astype(float)
    return (-hit * (v - r) / (alpha * e)) + v / e + np.log(-e) - 1.0


def exact_binomial_pvalue(k: int, n: int, alpha: float) -> float:
    lo = stats.binom.cdf(k, n, alpha)
    hi = stats.binom.sf(k - 1, n, alpha)
    return float(min(1.0, 2.0 * min(lo, hi)))


def kupiec_test(k: int, n: int, alpha: float):
    if k in (0, n):
        lr = -2.0 * (n * np.log(1 - alpha) if k == 0 else n * np.log(alpha))
    else:
        ph = k / n
        lr = -2.0 * ((n - k) * np.log((1 - alpha) / (1 - ph)) + k * np.log(alpha / ph))
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_independence(hits):
    h = np.asarray(hits, dtype=int)
    if h.sum() in (0, len(h)) or len(h) < 2:
        return np.nan, np.nan
    n00 = int(np.sum((h[:-1] == 0) & (h[1:] == 0)))
    n01 = int(np.sum((h[:-1] == 0) & (h[1:] == 1)))
    n10 = int(np.sum((h[:-1] == 1) & (h[1:] == 0)))
    n11 = int(np.sum((h[:-1] == 1) & (h[1:] == 1)))
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p1 = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(p, a, b):
        return a * np.log(max(1 - p, 1e-300)) + b * np.log(max(p, 1e-300))

    lr = -2.0 * (ll(p1, n00 + n10, n01 + n11) - (ll(p01, n00, n01) + ll(p11, n10, n11)))
    return float(lr), float(stats.chi2.sf(lr, 1))


def es_residual_test(realized, var_loss, es_loss, *, hac_lag: int = 1):
    """HAC t-test that realized losses beyond VaR average to the ES forecast."""
    loss = -np.asarray(realized, dtype=float)
    hit = loss > np.asarray(var_loss, dtype=float)
    if hit.sum() < 8:
        return np.nan, np.nan, int(hit.sum())
    resid = (loss - np.asarray(es_loss, dtype=float))[hit]
    v = long_run_variance(resid, lag=hac_lag)
    if not np.isfinite(v) or v <= 0:
        v = float(np.var(resid, ddof=1))
    t = float(np.mean(resid) / np.sqrt(v / len(resid)))
    p = float(2.0 * stats.t.sf(abs(t), df=len(resid) - 1))
    return t, p, int(hit.sum())


def diebold_mariano(loss_a, loss_b, *, hac_lag: int = 1):
    """Two-sided DM test on loss_a-loss_b; negative mean favors model A."""
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    d = a - b
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return np.nan, np.nan, np.nan
    v = long_run_variance(d, lag=hac_lag)
    if not np.isfinite(v) or v <= 0:
        return np.nan, np.nan, float(d.mean())
    stat = float(d.mean() / np.sqrt(v / len(d)))
    p = float(2.0 * stats.norm.sf(abs(stat)))
    return stat, p, float(d.mean())
