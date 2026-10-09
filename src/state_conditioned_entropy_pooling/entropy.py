from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .probabilities import normalize_probabilities


@dataclass(frozen=True)
class EntropyPoolResult:
    probabilities: np.ndarray
    theta_scaled: np.ndarray
    theta_original: np.ndarray
    success: bool
    message: str
    eq_max_abs_residual: float
    ineq_max_violation: float
    nit: int


def _scale_constraint_rows(matrix, targets, prior):
    if matrix is None or len(matrix) == 0:
        return None, None, np.array([], dtype=float)
    M = np.asarray(matrix, dtype=float)
    m = np.asarray(targets, dtype=float).reshape(-1)
    q = normalize_probabilities(prior)
    if M.shape[1] != len(q):
        raise ValueError("Constraint rows must have one entry per scenario.")
    if M.shape[0] != len(m):
        raise ValueError("Constraint target length does not match number of rows.")
    means = M @ q
    var = ((M - means[:, None]) ** 2) @ q
    scales = np.sqrt(np.maximum(var, 0.0))
    fallback = np.maximum(np.max(np.abs(M), axis=1), np.abs(m))
    scales = np.where(scales > 1e-10, scales, np.where(fallback > 1e-10, fallback, 1.0))
    return M / scales[:, None], m / scales, scales


def entropy_pool(
    prior,
    *,
    z_ineq=None,
    mu_ineq=None,
    z_eq=None,
    mu_eq=None,
    solver_tolerance: float = 5e-7,
    optimizer_tolerance: float = 1e-10,
    maxiter: int = 3000,
) -> EntropyPoolResult:
    """Minimum-relative-entropy posterior under linear equality/inequality views.

    Solves
        min_p D_KL(p || q)
    subject to
        z_ineq @ p <= mu_ineq
        z_eq   @ p == mu_eq
        p >= 0, sum(p) = 1.

    The implementation is the cleaned generic form of the dual solver used in the
    v20-v26 research notebooks. Constraint rows are internally rescaled to improve
    numerical conditioning.
    """
    q = normalize_probabilities(prior)
    Zi = None if z_ineq is None else np.atleast_2d(np.asarray(z_ineq, dtype=float))
    mi = None if mu_ineq is None else np.asarray(mu_ineq, dtype=float).reshape(-1)
    Ze = None if z_eq is None else np.atleast_2d(np.asarray(z_eq, dtype=float))
    me = None if mu_eq is None else np.asarray(mu_eq, dtype=float).reshape(-1)

    if (Zi is None) != (mi is None):
        raise ValueError("z_ineq and mu_ineq must be supplied together")
    if (Ze is None) != (me is None):
        raise ValueError("z_eq and mu_eq must be supplied together")

    Zi_s, mi_s, si = _scale_constraint_rows(Zi, mi, q) if Zi is not None else (None, None, np.array([]))
    Ze_s, me_s, se = _scale_constraint_rows(Ze, me, q) if Ze is not None else (None, None, np.array([]))

    blocks, targets, bounds = [], [], []
    if Ze_s is not None:
        blocks.append(Ze_s)
        targets.append(me_s)
        bounds.extend([(None, None)] * Ze_s.shape[0])
    if Zi_s is not None:
        blocks.append(Zi_s)
        targets.append(mi_s)
        bounds.extend([(0.0, None)] * Zi_s.shape[0])

    if not blocks:
        return EntropyPoolResult(
            probabilities=q.copy(),
            theta_scaled=np.array([]),
            theta_original=np.array([]),
            success=True,
            message="No active views",
            eq_max_abs_residual=0.0,
            ineq_max_violation=0.0,
            nit=0,
        )

    M = np.vstack(blocks)
    m = np.concatenate(targets)
    log_q = np.log(np.clip(q, 1e-300, None))

    def probabilities(theta):
        log_u = log_q - M.T @ theta
        return np.exp(log_u - logsumexp(log_u))

    def objective(theta):
        return float(logsumexp(log_q - M.T @ theta) + theta @ m)

    def gradient(theta):
        p = probabilities(theta)
        return m - M @ p

    res = minimize(
        objective,
        np.zeros(M.shape[0]),
        jac=gradient,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": optimizer_tolerance, "gtol": 1e-10, "maxiter": maxiter, "maxls": 60},
    )
    p = normalize_probabilities(probabilities(res.x))

    eq_resid = 0.0 if Ze is None else float(np.max(np.abs(Ze @ p - me)))
    ineq_viol = 0.0 if Zi is None else float(max(0.0, np.max(Zi @ p - mi)))
    success = bool(res.success and eq_resid <= solver_tolerance and ineq_viol <= solver_tolerance)

    scales_all = np.concatenate([se, si])
    theta_original = np.asarray(res.x, dtype=float) / scales_all
    return EntropyPoolResult(
        probabilities=p,
        theta_scaled=np.asarray(res.x, dtype=float),
        theta_original=theta_original,
        success=success,
        message=str(res.message),
        eq_max_abs_residual=eq_resid,
        ineq_max_violation=ineq_viol,
        nit=int(getattr(res, "nit", -1)),
    )
