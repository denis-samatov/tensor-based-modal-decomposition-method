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


def block_dg(
    B: np.ndarray, blocks: list[np.ndarray], n_max: int, ridge: float | None = None
) -> np.ndarray:
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


def condition_aware_blocks(
    B: np.ndarray,
    blocks: list[np.ndarray],
    n_max: int,
    condition_weight: float = 0.1,
    ridge: float | None = None,
    candidates: np.ndarray | None = None,
) -> np.ndarray:
    """Greedily balance regularised D-optimality and sampled-basis conditioning."""
    if condition_weight < 0:
        raise ValueError("condition_weight must be non-negative")
    r = B.shape[1]
    if ridge is None:
        ridge = max(np.finfo(float).eps, 1e-8 * np.trace(B.T @ B) / max(B.shape[0], 1))
    remaining = list(range(len(blocks))) if candidates is None else [int(x) for x in candidates]
    chosen: list[int] = []
    gram_selected = np.zeros((r, r))
    for _ in range(min(n_max, len(remaining))):
        best = remaining[0]
        best_score = -np.inf
        for candidate in remaining:
            block = B[np.asarray(blocks[candidate], dtype=int)]
            gram = gram_selected + block.T @ block + ridge * np.eye(r)
            sign, logdet = np.linalg.slogdet(gram)
            score = (
                -np.inf if sign <= 0 else logdet - condition_weight * np.log(np.linalg.cond(gram))
            )
            if score > best_score:
                best = candidate
                best_score = score
        chosen.append(best)
        selected = B[np.asarray(blocks[best], dtype=int)]
        gram_selected += selected.T @ selected
        remaining.remove(best)
    return np.asarray(chosen, dtype=int)


def cluster_constrained_order(base_order: np.ndarray, labels: np.ndarray, n_max: int) -> np.ndarray:
    """Round-robin cluster coverage while preserving within-cluster base priority."""
    base = [int(x) for x in np.asarray(base_order)]
    labels = np.asarray(labels)
    queues = {
        int(label): [candidate for candidate in base if int(labels[candidate]) == int(label)]
        for label in sorted(np.unique(labels))
    }
    order: list[int] = []
    while len(order) < min(n_max, len(base)):
        progressed = False
        for label in sorted(queues):
            if queues[label] and len(order) < min(n_max, len(base)):
                order.append(queues[label].pop(0))
                progressed = True
        if not progressed:
            break
    return np.asarray(order, dtype=int)
