"""Error metrics on active cells after inverse scaling, in supplied export units."""
from __future__ import annotations

import numpy as np


def evaluate(est_norm: np.ndarray, fold, n_active: int) -> dict:
    """est_norm (2n, T) normalised estimate. Returns per-snapshot arrays."""
    est = fold.scaler.inv(est_norm).reshape(2, n_active, -1)
    ref = fold.test_phys
    err = est - ref
    rmse_p = np.sqrt(np.mean(err[0] ** 2, axis=0))
    rmse_s = np.sqrt(np.mean(err[1] ** 2, axis=0))
    rel = np.linalg.norm(est_norm - fold.test, axis=0) / np.linalg.norm(fold.test, axis=0)
    maxabs_p = np.max(np.abs(err[0]), axis=0)
    return {"rmse_p": rmse_p, "rmse_so": rmse_s, "relerr_norm": rel, "maxabs_p": maxabs_p}
