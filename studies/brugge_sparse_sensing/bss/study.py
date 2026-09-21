"""Reusable components for the nested TBMD optimization benchmark."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from . import bases, estimators, placement
from .optimization import PropertyScaler


def _spec_id(spec: dict[str, Any]) -> str:
    fields = {key: value for key, value in spec.items() if key != "id"}
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def build_representation_specs(optimization: dict) -> list[dict]:
    """Build the bounded Stage-A representation search declared in ``config.json``."""
    modal_ranks = sorted(set(int(rank) for rank in optimization["modal_ranks"]))
    spatial = [tuple(int(x) for x in ranks) for ranks in optimization["joint_spatial_ranks"]]
    independent_spatial = [
        tuple(int(x) for x in ranks) for ranks in optimization["independent_spatial_ranks"]
    ]
    property_ranks = [
        tuple(int(x) for x in ranks) for ranks in optimization["property_modal_ranks"]
    ]
    default_modal = 16 if 16 in modal_ranks else modal_ranks[len(modal_ranks) // 2]
    specs: list[dict] = []

    for family in ("POD", "POD-E"):
        for rank in modal_ranks:
            specs.append(
                {
                    "family": family,
                    "architecture": "matrix",
                    "scaling": "minmax",
                    "rank_strategy": "fixed",
                    "modal_rank": rank,
                }
            )
        for threshold in optimization["energy_candidates"]:
            specs.append(
                {
                    "family": family,
                    "architecture": "matrix",
                    "scaling": "minmax",
                    "rank_strategy": "energy",
                    "modal_energy": float(threshold),
                }
            )

    for ranks in spatial:
        specs.append(
            {
                "family": "TBMD",
                "architecture": "joint",
                "scaling": "minmax",
                "rank_strategy": "fixed",
                "spatial_ranks": list(ranks),
                "property_rank": 2,
                "modal_rank": default_modal,
                "property_weights": [1.0, 1.0],
            }
        )
    anchors = [ranks for ranks in spatial if ranks[0] in (64, 80) and ranks[1] == 48]
    for ranks in anchors:
        for rank in modal_ranks:
            specs.append(
                {
                    "family": "TBMD",
                    "architecture": "joint",
                    "scaling": "minmax",
                    "rank_strategy": "fixed",
                    "spatial_ranks": list(ranks),
                    "property_rank": 2,
                    "modal_rank": rank,
                    "property_weights": [1.0, 1.0],
                }
            )
    if anchors:
        specs.append(
            {
                "family": "TBMD",
                "architecture": "joint",
                "scaling": "minmax",
                "rank_strategy": "fixed",
                "spatial_ranks": list(anchors[0]),
                "property_rank": 1,
                "modal_rank": default_modal,
                "property_weights": [1.0, 1.0],
            }
        )

    for architecture in ("shared-independent", "independent"):
        ranks_to_use = (
            independent_spatial
            if architecture == "shared-independent"
            else independent_spatial[-2:]
        )
        for ranks in ranks_to_use:
            for modal_by_property in property_ranks:
                specs.append(
                    {
                        "family": "TBMD",
                        "architecture": architecture,
                        "scaling": "minmax",
                        "rank_strategy": "fixed",
                        "spatial_ranks": list(ranks),
                        "property_rank": 2,
                        "property_modal_ranks": list(modal_by_property),
                        "modal_rank": sum(modal_by_property),
                        "property_weights": [1.0, 1.0],
                    }
                )

    residual_anchors = spatial[2:4] if len(spatial) >= 4 else spatial[-1:]
    for ranks in residual_anchors:
        for residual_rank in optimization["residual_ranks"]:
            specs.append(
                {
                    "family": "TBMD",
                    "architecture": "joint-residual",
                    "scaling": "minmax",
                    "rank_strategy": "fixed",
                    "spatial_ranks": list(ranks),
                    "property_rank": 2,
                    "modal_rank": default_modal,
                    "residual_rank": int(residual_rank),
                    "property_weights": [1.0, 1.0],
                }
            )

    for threshold in optimization["energy_candidates"]:
        specs.append(
            {
                "family": "TBMD",
                "architecture": "joint",
                "scaling": "minmax",
                "rank_strategy": "energy",
                "spatial_energy": float(threshold),
                "modal_energy": float(threshold),
                "max_spatial_ranks": [139, 48],
                "max_modal_rank": int(optimization["max_modal_rank"]),
                "property_rank": 2,
                "property_weights": [1.0, 1.0],
            }
        )

    unique: dict[str, dict] = {}
    for spec in specs:
        identifier = _spec_id(spec)
        unique[identifier] = {"id": identifier, **spec}
    return list(unique.values())


def solver_id(spec: dict) -> str:
    """Stable human-readable solver identifier."""
    name = str(spec["name"])
    parameters = [f"{key}={spec[key]:g}" for key in sorted(spec) if key != "name"]
    return name if not parameters else name + "(" + ",".join(parameters) + ")"


def solve_coefficients(
    design: np.ndarray, targets: np.ndarray, spec: dict
) -> tuple[np.ndarray, dict]:
    """Dispatch a configured sparse-recovery solver with scale-relative L2 penalties."""
    name = spec["name"]
    identifier = solver_id(spec)
    relative_scale = float(np.trace(design.T @ design) / max(design.shape[1], 1))
    numerical_floor = float(np.finfo(float).eps * max(relative_scale, 1.0))
    metadata: dict[str, Any] = {"solver": identifier, "iterations": 0}
    if name == "least_squares":
        coefficients = estimators.least_squares(design, targets, float(spec.get("rcond", 1e-10)))
    elif name == "ridge":
        alpha = max(float(spec["alpha_relative"]) * relative_scale, numerical_floor)
        coefficients = estimators.ridge(design, targets, alpha)
        metadata["alpha_absolute"] = alpha
    elif name == "lasso":
        coefficients, iterations = estimators.l1_admm(
            design,
            targets,
            epsilon=float(spec["l1"]),
            delta=float(spec.get("delta", 1.0)),
            relax=float(spec.get("relax", 0.95)),
            max_iter=int(spec.get("max_iter", 5000)),
            tol=float(spec.get("tol", 1e-7)),
        )
        metadata["iterations"] = iterations
    elif name == "elastic_net":
        l2 = max(float(spec["l2_relative"]) * relative_scale, numerical_floor)
        coefficients, iterations = estimators.elastic_net_admm(
            design,
            targets,
            l1=float(spec["l1"]),
            l2=l2,
            delta=float(spec.get("delta", 1.0)),
            relax=float(spec.get("relax", 0.95)),
            max_iter=int(spec.get("max_iter", 5000)),
            tol=float(spec.get("tol", 1e-7)),
        )
        metadata["iterations"] = iterations
        metadata["l2_absolute"] = l2
    else:
        raise ValueError(f"unknown solver: {name}")
    return coefficients, metadata


def physical_error_summary(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Mean snapshot-wise RMSE and MAE for pressure and saturation."""
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if estimate.shape != truth.shape or estimate.shape[0] != 2:
        raise ValueError("estimate and truth must have matching (2, n_active, time) shapes")
    error = estimate - truth
    return {
        "rmse_p": float(np.sqrt(np.mean(error[0] ** 2, axis=0)).mean()),
        "rmse_so": float(np.sqrt(np.mean(error[1] ** 2, axis=0)).mean()),
        "mae_p": float(np.mean(np.abs(error[0]), axis=0).mean()),
        "mae_so": float(np.mean(np.abs(error[1]), axis=0).mean()),
    }


def transform_runs(raw: np.ndarray, scaler: PropertyScaler) -> np.ndarray:
    """Scale and concatenate scenario snapshots in property-major row order."""
    raw = np.asarray(raw, dtype=float)
    if raw.ndim == 3:
        raw = raw[None, ...]
    if raw.ndim != 4 or raw.shape[1] != 2:
        raise ValueError("raw data must have shape (runs, 2, n_active, time)")
    scaled = scaler.fwd(raw)
    return np.concatenate([run.reshape(-1, run.shape[-1]) for run in scaled], axis=1)


def fit_representation(
    raw_fit: np.ndarray,
    ij: np.ndarray,
    spec: dict,
    *,
    shape: tuple[int, int] = (139, 48),
    exact_svd: bool = False,
):
    """Fit one declared representation on training scenarios only."""
    scaler = PropertyScaler.fit(raw_fit, method=spec["scaling"])
    train = transform_runs(raw_fit, scaler)
    if spec["family"] == "POD":
        basis = bases.pod(
            train,
            r=spec.get("modal_rank"),
            threshold=spec.get("modal_energy"),
        )
    elif spec["family"] == "POD-E":
        basis = bases.pod_energy(
            train,
            r=spec.get("modal_rank"),
            threshold=spec.get("modal_energy"),
        )
    elif spec["family"] == "TBMD":
        rank_strategy = spec.get("rank_strategy", "fixed")
        config = bases.StructuredTBMDConfig(
            architecture=spec["architecture"],
            spatial_ranks=None if rank_strategy == "energy" else tuple(spec["spatial_ranks"]),
            spatial_energy=spec.get("spatial_energy"),
            max_spatial_ranks=tuple(spec.get("max_spatial_ranks", shape)),
            property_rank=int(spec.get("property_rank", 2)),
            modal_rank=None if rank_strategy == "energy" else int(spec["modal_rank"]),
            modal_energy=spec.get("modal_energy"),
            max_modal_rank=int(spec.get("max_modal_rank", 64)),
            property_modal_ranks=tuple(spec["property_modal_ranks"])
            if spec.get("property_modal_ranks") is not None
            else None,
            property_weights=tuple(spec.get("property_weights", (1.0, 1.0))),
            residual_rank=int(spec.get("residual_rank", 0)),
            exact_svd=exact_svd,
            svd_iterations=int(spec.get("svd_iterations", 6)),
            seed=int(spec.get("seed", 0)),
        )
        basis = bases.structured_tbmd(train, ij, config, shape=shape)
    else:
        raise ValueError(f"unknown representation family: {spec['family']}")
    basis.extra["spec_id"] = spec["id"]
    basis.extra["scaling"] = spec["scaling"]
    basis.extra.setdefault("storage_floats", int(basis.B.size))
    return scaler, basis, train


def placement_orders(
    B: np.ndarray,
    well_rows: np.ndarray,
    ij: np.ndarray,
    *,
    max_wells: int,
    max_grid: int,
    condition_weight: float,
    cluster_labels: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Create deterministic TBMD/POD-specific well and grid placement orders."""
    n = B.shape[0] // 2
    well_rows = np.asarray(well_rows, dtype=int)
    joint_blocks = [np.array([row, row + n]) for row in well_rows]
    pressure_blocks = [np.array([row]) for row in well_rows]
    well = {
        "configured": np.arange(len(well_rows), dtype=int)[:max_wells],
        "dg_joint": placement.block_dg(B, joint_blocks, max_wells),
        "condition_joint": placement.condition_aware_blocks(
            B, joint_blocks, max_wells, condition_weight=condition_weight
        ),
        "dg_pressure": placement.block_dg(B[:n], pressure_blocks, max_wells),
        "condition_pressure": placement.condition_aware_blocks(
            B[:n], pressure_blocks, max_wells, condition_weight=condition_weight
        ),
    }

    grid_blocks = [np.array([row]) for row in range(2 * n)]
    qr = placement.qr_dg(B, max_grid)
    norms = np.linalg.norm(B, axis=0)
    whitened = placement.qr_dg(B / np.maximum(norms, np.finfo(float).eps), max_grid)
    condition_depth = min(max_grid, max(2 * B.shape[1], 64))
    pool_size = min(2 * n, max(2 * condition_depth, condition_depth))
    qr_pool = placement.qr_dg(B, pool_size)
    leverage_pool = np.argsort(np.linalg.norm(B, axis=1))[::-1][:pool_size]
    pool = np.asarray(
        list(dict.fromkeys(np.concatenate([qr_pool, leverage_pool]).tolist())), dtype=int
    )
    condition = placement.condition_aware_blocks(
        B,
        grid_blocks,
        condition_depth,
        condition_weight=condition_weight,
        candidates=pool,
    )
    if len(condition) < max_grid:
        condition_set = set(condition.tolist())
        condition = np.concatenate(
            [condition, np.asarray([row for row in qr if row not in condition_set], dtype=int)]
        )[:max_grid]
    channel_clusters = np.concatenate([cluster_labels, cluster_labels])
    clustered = placement.cluster_constrained_order(qr, channel_clusters, max_grid)

    coordinates = np.asarray(ij, dtype=float)
    well_coordinates = coordinates[well_rows]
    distance = np.min(
        np.linalg.norm(coordinates[:, None, :] - well_coordinates[None, :, :], axis=2), axis=1
    )
    region_cells = np.flatnonzero(distance <= 3.0)
    region_channels = np.concatenate([region_cells, region_cells + n])
    if len(region_channels) < max_grid:
        region_channels = np.arange(2 * n)
    well_region = placement.qr_dg(B, max_grid, candidates=region_channels)
    grid = {
        "qr_joint": qr,
        "qr_whitened": whitened,
        "condition_joint": condition,
        "cluster_constrained": clustered,
        "well_region": well_region,
    }
    return {"well": well, "grid": grid}
