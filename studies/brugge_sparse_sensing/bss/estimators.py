"""State estimators from point measurements. All operate on all test snapshots at once.

B_S : (m, r) measured rows of the basis; Y : (m, T) measurements (normalised scale).
Return coefficients X : (r, T); the state is B @ X.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla


def least_squares(BS: np.ndarray, Y: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Minimum-norm least squares (gappy POD / DEIM-type reconstruction)."""
    X, *_ = np.linalg.lstsq(BS, Y, rcond=rcond)
    return X


def l1_admm(BS: np.ndarray, Y: np.ndarray, epsilon=1e-2, delta=1.0, relax=0.95,
            max_iter=5000, tol=1e-7) -> tuple[np.ndarray, int]:
    """min_x  0.5 ||B_S x - y||^2 + epsilon ||x||_1  (TBMD compressive sensing objective),
    solved by over-relaxed ADMM (Boyd et al., 2011) with a fixed penalty delta; vectorised over
    snapshots. The penalty schedule affects convergence speed only, not the minimiser."""
    r = BS.shape[1]
    T = Y.shape[1]
    L = sla.cho_factor(BS.T @ BS + delta * np.eye(r))
    AtY = BS.T @ Y
    x = np.zeros((r, T)); d = np.zeros((r, T)); p = np.zeros((r, T))
    kappa = epsilon / delta
    it = 0
    for it in range(1, max_iter + 1):
        x = sla.cho_solve(L, AtY + delta * (d - p))
        xh = relax * x + (1 - relax) * d
        d_old = d
        z = xh + p
        d = np.sign(z) * np.maximum(np.abs(z) - kappa, 0.0)
        p = p + xh - d
        if it % 10 == 0:
            prim = np.linalg.norm(x - d, axis=0).max()
            dual = delta * np.linalg.norm(d - d_old, axis=0).max()
            if max(prim, dual) < tol:
                break
    return d, it


def lasso_objective(BS, Y, X, epsilon):
    R = BS @ X - Y
    return 0.5 * np.sum(R**2, axis=0) + epsilon * np.sum(np.abs(X), axis=0)


def idw_residual(prior: np.ndarray, rows: np.ndarray, Y: np.ndarray, xy: np.ndarray,
                 n_active: int, power: float = 2.0) -> np.ndarray:
    """prior (2n, T) normalised; rows are stacked indices of measured channels; Y (m, T).
    The residual y - prior is interpolated per property by inverse-distance weighting in the
    grid-index plane; properties without measurements keep the prior."""
    out = prior.copy()
    for k in (0, 1):
        sel = (rows >= k * n_active) & (rows < (k + 1) * n_active)
        if not sel.any():
            continue
        cells = rows[sel] - k * n_active
        res = Y[sel] - prior[rows[sel]]
        dist = np.linalg.norm(xy[:, None, :] - xy[None, cells, :], axis=2)  # (n, m)
        w = 1.0 / np.maximum(dist, 1e-9) ** power
        exact = dist < 1e-9
        w[exact.any(axis=1)] = exact[exact.any(axis=1)].astype(float)
        w /= w.sum(axis=1, keepdims=True)
        out[k * n_active:(k + 1) * n_active] += w @ res
    return out
