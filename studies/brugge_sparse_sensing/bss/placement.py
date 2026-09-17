"""Sensor placement strategies. Each returns an ordered array of selected units; budgets are
nested prefixes of that order.

* qr_dg    -- column-pivoted QR on B[C]^T for the first r points (Q-DEIM; equivalent to the
              TBMD tube-fibre pivoted QR on the mode-4 unfolding), then determinant-based greedy
              oversampling (Saito et al., 2021) beyond r.
* block_dg -- greedy log-det selection of *wells* (blocks of rows), ridge-regularised.
* random   -- uniform random order.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla


def qr_dg(B: np.ndarray, n_max: int, candidates: np.ndarray | None = None) -> np.ndarray:
    M, r = B.shape
    cand = np.arange(M) if candidates is None else np.asarray(candidates)
    Bc = B[cand]
    k = min(n_max, r, len(cand))
    _, _, piv = sla.qr(Bc.T, mode="economic", pivoting=True)
    order = list(piv[:k])
    if n_max > k:
        S = Bc[order]
        C = np.linalg.inv(S.T @ S)
        mask = np.ones(len(cand), bool)
        mask[order] = False
        for _ in range(min(n_max, len(cand)) - k):
            idx = np.flatnonzero(mask)
            V = Bc[idx]
            score = np.einsum("ij,jk,ik->i", V, C, V)
            j = idx[int(np.argmax(score))]
            b = Bc[j]
            Cb = C @ b
            C = C - np.outer(Cb, Cb) / (1.0 + b @ Cb)
            order.append(j)
            mask[j] = False
    return cand[np.asarray(order, dtype=int)]


def block_dg(B: np.ndarray, blocks: list[np.ndarray], n_max: int, ridge: float | None = None) -> np.ndarray:
    """Greedy maximisation of log det(B_S^T B_S + ridge I) over blocks (e.g. wells)."""
    r = B.shape[1]
    if ridge is None:
        ridge = 1e-6 * np.trace(B.T @ B) / B.shape[0]
    C = np.eye(r) / ridge
    left = list(range(len(blocks)))
    order = []
    for _ in range(min(n_max, len(blocks))):
        best, bestgain = None, -np.inf
        for w in left:
            Bw = B[blocks[w]]
            G = np.eye(Bw.shape[0]) + Bw @ C @ Bw.T
            gain = np.linalg.slogdet(G)[1]
            if gain > bestgain:
                best, bestgain = w, gain
        Bw = B[blocks[best]]
        K = np.linalg.inv(np.eye(Bw.shape[0]) + Bw @ C @ Bw.T)
        C = C - C @ Bw.T @ K @ Bw @ C
        order.append(best)
        left.remove(best)
    return np.asarray(order)


def random_order(n_units: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n_units)
