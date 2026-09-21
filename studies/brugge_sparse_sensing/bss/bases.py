"""Reduced bases: POD (matrix SVD) and TBMD (fourth-order Tucker time-insensitive modes).

Both return a matrix B of shape (2 n_active, r) acting on the stacked, normalised state
(pressure block first). Inactive cells are zero in the gridded tensor and are excluded from B.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import tensorly as tl
from sklearn.utils.extmath import randomized_svd
from tensorly.decomposition import tucker


@dataclass
class Basis:
    name: str
    B: np.ndarray  # (2n, r)
    sv: np.ndarray | None  # training singular values (POD) or None
    seconds: float
    extra: dict


@dataclass(frozen=True)
class StructuredTBMDConfig:
    """Configuration for nested, efficiently truncatable TBMD variants."""

    architecture: str = "joint"
    spatial_ranks: tuple[int, int] | None = (48, 48)
    spatial_energy: float | None = None
    max_spatial_ranks: tuple[int, int] = (139, 48)
    property_rank: int = 2
    modal_rank: int | None = 16
    modal_energy: float | None = None
    max_modal_rank: int = 64
    property_modal_ranks: tuple[int, int] | None = None
    property_weights: tuple[float, float] = (1.0, 1.0)
    residual_rank: int = 0
    exact_svd: bool = False
    svd_iterations: int = 6
    seed: int = 0


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


def tbmd(
    train: np.ndarray,
    ij: np.ndarray,
    r: int,
    spatial_ranks=(48, 48, 2),
    init="svd",
    n_iter_max=100,
    tol=1e-8,
    seed=0,
) -> Basis:
    """Tucker (HOOI, SVD initialisation) of the gridded (I, J, K, T) training tensor and the
    time-insensitive modes M_n = G[:, :, :, n] x1 U1 x2 U2 x3 U3 (Zhong et al., 2024)."""
    tl.set_backend("numpy")
    t0 = time.perf_counter()
    X = to_grid(train, ij)
    ranks = [
        min(spatial_ranks[0], X.shape[0]),
        min(spatial_ranks[1], X.shape[1]),
        min(spatial_ranks[2], X.shape[2]),
        min(r, X.shape[3]),
    ]
    core, factors = tucker(
        X, rank=ranks, init=init, n_iter_max=n_iter_max, tol=tol, random_state=seed
    )
    modes = tl.tenalg.multi_mode_dot(core, factors[:3], modes=[0, 1, 2])  # (I, J, K, r)
    B = from_grid(modes, ij)
    Xh = tl.tucker_to_tensor((core, factors))
    fit = float(np.linalg.norm(X - Xh) / np.linalg.norm(X))
    return Basis(
        "TBMD",
        B,
        None,
        time.perf_counter() - t0,
        {
            "ranks": ranks,
            "train_rel_error": fit,
            "storage_floats": int(core.size + sum(f.size for f in factors)),
        },
    )


def _canonical_eigensystem(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    for column in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1
    return values, vectors


def _energy_rank_from_values(values: np.ndarray, threshold: float, cap: int) -> int:
    if not 0 < threshold <= 1:
        raise ValueError("energy threshold must be in (0, 1]")
    total = float(np.sum(values))
    if total <= 0:
        return 1
    rank = int(np.searchsorted(np.cumsum(values) / total, threshold) + 1)
    return max(1, min(rank, cap))


def _leading_svd(
    matrix: np.ndarray, rank: int | None, cfg: StructuredTBMDConfig
) -> tuple[np.ndarray, np.ndarray]:
    limit = min(matrix.shape)
    requested = min(rank, limit) if rank is not None else min(cfg.max_modal_rank, limit)
    if cfg.exact_svd or requested == limit:
        left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
        left, singular = left[:, :requested], singular[:requested]
    else:
        left, singular, _ = randomized_svd(
            matrix,
            n_components=requested,
            n_iter=cfg.svd_iterations,
            random_state=cfg.seed,
        )
    if rank is None:
        if cfg.modal_energy is None:
            raise ValueError("modal_energy is required when modal_rank is None")
        total_energy = float(np.linalg.norm(matrix) ** 2)
        captured = (
            np.cumsum(singular**2) / total_energy if total_energy > 0 else np.ones_like(singular)
        )
        selected = int(np.searchsorted(captured, cfg.modal_energy) + 1)
        selected = min(max(selected, 1), requested)
        left, singular = left[:, :selected], singular[:selected]
    return left, singular


def _resolve_spatial_rank(values: np.ndarray, axis: int, cfg: StructuredTBMDConfig) -> int:
    cap = min(cfg.max_spatial_ranks[axis], len(values))
    if cfg.spatial_ranks is not None:
        return min(cfg.spatial_ranks[axis], cap)
    if cfg.spatial_energy is None:
        raise ValueError("spatial_energy is required when spatial_ranks is None")
    return _energy_rank_from_values(values, cfg.spatial_energy, cap)


def _joint_spatial_factors(grid: np.ndarray, cfg: StructuredTBMDConfig):
    i, j, k, _ = grid.shape
    r3 = min(cfg.property_rank, k)
    unfold1 = grid.reshape(i, -1)
    unfold2 = grid.transpose(1, 0, 2, 3).reshape(j, -1)
    unfold3 = grid.transpose(2, 0, 1, 3).reshape(k, -1)
    values1, vectors1 = _canonical_eigensystem(unfold1 @ unfold1.T)
    values2, vectors2 = _canonical_eigensystem(unfold2 @ unfold2.T)
    _, vectors3 = _canonical_eigensystem(unfold3 @ unfold3.T)
    u1 = vectors1[:, : _resolve_spatial_rank(values1, 0, cfg)]
    u2 = vectors2[:, : _resolve_spatial_rank(values2, 1, cfg)]
    u3 = vectors3[:, :r3]
    return u1, u2, u3


def _single_spatial_factors(grid: np.ndarray, cfg: StructuredTBMDConfig):
    i, j, _ = grid.shape
    unfold1 = grid.reshape(i, -1)
    unfold2 = grid.transpose(1, 0, 2).reshape(j, -1)
    values1, vectors1 = _canonical_eigensystem(unfold1 @ unfold1.T)
    values2, vectors2 = _canonical_eigensystem(unfold2 @ unfold2.T)
    u1 = vectors1[:, : _resolve_spatial_rank(values1, 0, cfg)]
    u2 = vectors2[:, : _resolve_spatial_rank(values2, 1, cfg)]
    return u1, u2


def _joint_structured(
    train: np.ndarray, ij: np.ndarray, cfg: StructuredTBMDConfig, shape: tuple[int, int]
) -> tuple[np.ndarray, dict]:
    weights = np.asarray(cfg.property_weights, dtype=float)
    if weights.shape != (2,) or np.any(weights <= 0):
        raise ValueError("property_weights must contain two positive values")
    grid = to_grid(train, ij, shape=shape)
    weighted = grid * weights.reshape(1, 1, 2, 1)
    u1, u2, u3 = _joint_spatial_factors(weighted, cfg)
    compressed = np.einsum("ia,jb,kc,ijkt->abct", u1, u2, u3, weighted, optimize=True)
    features = compressed.reshape(-1, compressed.shape[-1])
    left, singular = _leading_svd(features, cfg.modal_rank, cfg)
    r = left.shape[1]
    core_modes = (left * singular).reshape(u1.shape[1], u2.shape[1], u3.shape[1], r)
    modes = np.einsum("ia,jb,kc,abcr->ijkr", u1, u2, u3, core_modes, optimize=True)
    modes /= weights.reshape(1, 1, 2, 1)
    basis = from_grid(modes, ij)
    storage = u1.size + u2.size + u3.size + core_modes.size
    extra = {
        "architecture": "joint",
        "realized_ranks": [u1.shape[1], u2.shape[1], u3.shape[1], r],
        "storage_floats": int(storage),
        "property_weights": weights.tolist(),
        "singular_values": singular.tolist(),
        "rank_strategy": "energy"
        if cfg.modal_rank is None or cfg.spatial_ranks is None
        else "fixed",
    }
    return basis, extra


def _independent_structured(
    train: np.ndarray,
    ij: np.ndarray,
    cfg: StructuredTBMDConfig,
    shape: tuple[int, int],
    *,
    shared: bool,
) -> tuple[np.ndarray, dict]:
    n = len(ij)
    weights = np.asarray(cfg.property_weights, dtype=float)
    ranks = cfg.property_modal_ranks
    if ranks is None:
        ranks = (
            (None, None)
            if cfg.modal_rank is None
            else (
                cfg.modal_rank // 2,
                cfg.modal_rank - cfg.modal_rank // 2,
            )
        )
    if any(rank is not None and rank < 1 for rank in ranks):
        raise ValueError("property modal ranks must be positive")
    grid = to_grid(train, ij, shape=shape)
    weighted = grid * weights.reshape(1, 1, 2, 1)
    shared_factors = None
    storage = 0
    if shared:
        u1, u2, _ = _joint_spatial_factors(weighted, replace(cfg, property_rank=2))
        shared_factors = (u1, u2)
        storage += u1.size + u2.size
    blocks = []
    realized = []
    singular_values = []
    for prop, rank in enumerate(ranks):
        if shared_factors is None:
            u1, u2 = _single_spatial_factors(weighted[:, :, prop, :], cfg)
            storage += u1.size + u2.size
        else:
            u1, u2 = shared_factors
        compressed = np.einsum("ia,jb,ijt->abt", u1, u2, weighted[:, :, prop, :], optimize=True)
        left, singular = _leading_svd(compressed.reshape(-1, compressed.shape[-1]), rank, cfg)
        r = left.shape[1]
        core_modes = (left * singular).reshape(u1.shape[1], u2.shape[1], r)
        modes = np.einsum("ia,jb,abr->ijr", u1, u2, core_modes, optimize=True)
        active_modes = modes[ij[:, 0], ij[:, 1]] / weights[prop]
        block = np.zeros((2 * n, r))
        block[prop * n : (prop + 1) * n] = active_modes
        blocks.append(block)
        storage += core_modes.size
        realized.append([u1.shape[1], u2.shape[1], r])
        singular_values.append(singular.tolist())
    architecture = "shared-independent" if shared else "independent"
    return np.concatenate(blocks, axis=1), {
        "architecture": architecture,
        "realized_ranks": realized,
        "storage_floats": int(storage),
        "property_weights": weights.tolist(),
        "singular_values": singular_values,
        "rank_strategy": "energy"
        if cfg.modal_rank is None or cfg.spatial_ranks is None
        else "fixed",
    }


def structured_tbmd(
    train: np.ndarray, ij: np.ndarray, cfg: StructuredTBMDConfig, shape: tuple[int, int] = (139, 48)
) -> Basis:
    """Fit a deterministic HOSVD-style TBMD basis with nested spatial factors.

    The spatial factors are learned from mode covariances; a (possibly
    randomized) SVD of the spatially compressed tensor supplies energy-weighted
    time-insensitive modes.  This makes rank sweeps inexpensive while retaining
    the tensor product representation.
    """

    start = time.perf_counter()
    if cfg.architecture in {"joint", "joint-residual"}:
        basis, extra = _joint_structured(train, ij, cfg, shape)
    elif cfg.architecture == "shared-independent":
        basis, extra = _independent_structured(train, ij, cfg, shape, shared=True)
    elif cfg.architecture == "independent":
        basis, extra = _independent_structured(train, ij, cfg, shape, shared=False)
    else:
        raise ValueError(f"unknown TBMD architecture: {cfg.architecture}")

    if cfg.architecture == "joint-residual":
        if cfg.residual_rank < 1:
            raise ValueError("joint-residual requires residual_rank >= 1")
        q, _ = np.linalg.qr(basis)
        residual = train - q @ (q.T @ train)
        residual_cfg = replace(cfg, modal_rank=cfg.residual_rank)
        left, singular = _leading_svd(residual, cfg.residual_rank, residual_cfg)
        residual_modes = left * singular
        basis = np.concatenate([basis, residual_modes], axis=1)
        extra["architecture"] = "joint-residual"
        extra["residual_rank"] = residual_modes.shape[1]
        extra["storage_floats"] += int(residual_modes.size)

    coefficients = np.linalg.lstsq(basis, train, rcond=None)[0]
    extra["train_rel_error"] = float(
        np.linalg.norm(train - basis @ coefficients) / np.linalg.norm(train)
    )
    return Basis(
        name=f"TBMD-{extra['architecture']}",
        B=basis,
        sv=None,
        seconds=time.perf_counter() - start,
        extra=extra,
    )
