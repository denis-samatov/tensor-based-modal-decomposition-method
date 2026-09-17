"""Reduced bases: POD (matrix SVD) and TBMD (fourth-order Tucker time-insensitive modes).

Both return a matrix B of shape (2 n_active, r) acting on the stacked, normalised state
(pressure block first). Inactive cells are zero in the gridded tensor and are excluded from B.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker


@dataclass
class Basis:
    name: str
    B: np.ndarray            # (2n, r)
    sv: np.ndarray | None    # training singular values (POD) or None
    seconds: float
    extra: dict


def energy_rank(sv: np.ndarray, threshold: float) -> int:
    e = np.cumsum(sv**2) / np.sum(sv**2)
    return int(np.searchsorted(e, threshold) + 1)


def pod(train: np.ndarray, r: int | None = None, threshold: float | None = None) -> Basis:
    t0 = time.perf_counter()
    U, s, _ = np.linalg.svd(train, full_matrices=False)
    if r is None:
        r = energy_rank(s, threshold)
    return Basis("POD", U[:, :r].copy(), s, time.perf_counter() - t0, {"rank": r})


def pod_energy(train: np.ndarray, r: int | None = None, threshold: float | None = None) -> Basis:
    """Energy-weighted POD atoms U_r diag(s_r). Algebraically these are the TBMD time-insensitive
    modes obtained with full (untruncated) spatial Tucker ranks; used as an ablation."""
    base = pod(train, r=r, threshold=threshold)
    r = base.extra["rank"]
    t0 = time.perf_counter()
    B = base.B * base.sv[:r]
    return Basis("POD-E", B, base.sv, base.seconds + time.perf_counter() - t0, {"rank": r})


def to_grid(stacked: np.ndarray, ij: np.ndarray, shape=(139, 48)) -> np.ndarray:
    """(2n, T) -> (I, J, 2, T) with zeros at inactive cells."""
    n = ij.shape[0]
    T = stacked.shape[1]
    g = np.zeros(shape + (2, T))
    g[ij[:, 0], ij[:, 1], 0, :] = stacked[:n]
    g[ij[:, 0], ij[:, 1], 1, :] = stacked[n:]
    return g


def from_grid(grid_ijk_w: np.ndarray, ij: np.ndarray) -> np.ndarray:
    """(I, J, 2, W) -> (2n, W)."""
    p = grid_ijk_w[ij[:, 0], ij[:, 1], 0, :]
    s = grid_ijk_w[ij[:, 0], ij[:, 1], 1, :]
    return np.concatenate([p, s], axis=0)


def tbmd(train: np.ndarray, ij: np.ndarray, r: int, spatial_ranks=(48, 48, 2),
         init="svd", n_iter_max=100, tol=1e-8, seed=0) -> Basis:
    """Tucker (HOOI, SVD initialisation) of the gridded (I, J, K, T) training tensor and the
    time-insensitive modes M_n = G[:, :, :, n] x1 U1 x2 U2 x3 U3 (Zhong et al., 2024)."""
    tl.set_backend("numpy")
    t0 = time.perf_counter()
    X = to_grid(train, ij)
    ranks = [min(spatial_ranks[0], X.shape[0]), min(spatial_ranks[1], X.shape[1]),
             min(spatial_ranks[2], X.shape[2]), min(r, X.shape[3])]
    core, factors = tucker(X, rank=ranks, init=init, n_iter_max=n_iter_max, tol=tol,
                           random_state=seed)
    modes = tl.tenalg.multi_mode_dot(core, factors[:3], modes=[0, 1, 2])  # (I, J, K, r)
    B = from_grid(modes, ij)
    Xh = tl.tucker_to_tensor((core, factors))
    fit = float(np.linalg.norm(X - Xh) / np.linalg.norm(X))
    return Basis("TBMD", B, None, time.perf_counter() - t0,
                 {"ranks": ranks, "train_rel_error": fit,
                  "storage_floats": int(core.size + sum(f.size for f in factors))})
