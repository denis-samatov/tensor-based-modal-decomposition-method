"""Predeclared matched-capacity components for experiment E9."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker

from . import bases


@dataclass(frozen=True)
class E9Transform:
    """Train-fitted preprocessing with an explicit physical-space baseline."""

    mode: str
    baseline: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, training_fields: np.ndarray, mode: str) -> "E9Transform":
        fields = np.asarray(training_fields, dtype=float)
        if fields.ndim != 4 or fields.shape[1] != 2:
            raise ValueError("training_fields must have shape (runs, 2, active_cells, time)")
        values = np.moveaxis(fields, 1, 0).reshape(2, -1)
        if mode == "raw_minmax":
            baseline = values.min(axis=1).reshape(2, 1, 1)
            scale = np.ptp(values, axis=1).reshape(2, 1, 1)
        elif mode == "mean_centered":
            baseline = values.mean(axis=1).reshape(2, 1, 1)
            scale = np.ones((2, 1, 1))
        elif mode in {"ensemble_anomaly", "standardized_anomaly"}:
            baseline = fields.mean(axis=0)
            anomalies = fields - baseline[None, ...]
            if mode == "standardized_anomaly":
                scale = anomalies.std(axis=(0, 2, 3)).reshape(2, 1, 1)
            else:
                scale = np.ones((2, 1, 1))
        else:
            raise ValueError(f"unknown E9 preprocessing mode: {mode}")
        if not np.isfinite(scale).all() or np.any(scale <= np.finfo(float).eps):
            raise ValueError(f"invalid property scale for E9 preprocessing mode {mode}")
        return cls(mode=mode, baseline=baseline.copy(), scale=scale.copy())

    def fwd(self, fields: np.ndarray) -> np.ndarray:
        values = np.asarray(fields, dtype=float)
        return (values - self.baseline) / self.scale

    def inv(self, fields: np.ndarray) -> np.ndarray:
        values = np.asarray(fields, dtype=float)
        return values * self.scale + self.baseline


def _identifier(spec: dict) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return f"{spec['variant']}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def build_e9_candidate_specs(
    *,
    spatial_ranks: tuple[tuple[int, int], ...] = ((64, 48), (80, 48)),
    preprocessing: tuple[str, ...] = (
        "raw_minmax",
        "mean_centered",
        "ensemble_anomaly",
        "standardized_anomaly",
    ),
) -> list[dict]:
    """Build the bounded search frozen before any E9 outer evaluation."""

    specs: list[dict] = []
    for capacity_regime, budget in (("equal_total", 32), ("equal_per_property", 64)):
        property_rank = budget // 2
        for ranks in spatial_ranks:
            for transform in preprocessing:
                base = {
                    "capacity_regime": capacity_regime,
                    "coefficient_budget": budget,
                    "spatial_ranks": list(ranks),
                    "preprocessing": transform,
                    "property_weights": [1.0, 1.0],
                }
                variants = [
                    {
                        **base,
                        "order": "3D",
                        "variant": "3D-independent",
                        "architecture": "independent",
                        "property_modal_ranks": [property_rank, property_rank],
                    },
                    {
                        **base,
                        "order": "4D",
                        "variant": "4D-A",
                        "architecture": "joint",
                        "property_rank": 2,
                        "modal_rank": budget,
                    },
                    {
                        **base,
                        "order": "4D",
                        "variant": "4D-B",
                        "architecture": "shared-independent",
                        "property_modal_ranks": [property_rank, property_rank],
                    },
                ]
                if transform == "standardized_anomaly":
                    variants.extend(
                        [
                            {
                                **base,
                                "order": "4D",
                                "variant": "4D-A",
                                "architecture": "joint",
                                "property_rank": 1,
                                "modal_rank": budget,
                            },
                            {
                                **base,
                                "order": "4D",
                                "variant": "4D-D",
                                "architecture": "partial-spatial",
                                "shared_property_ranks": [
                                    3 * property_rank // 4,
                                    3 * property_rank // 4,
                                ],
                                "property_residual_ranks": [
                                    property_rank // 4,
                                    property_rank // 4,
                                ],
                            },
                            {
                                **base,
                                "order": "4D",
                                "variant": "4D-E",
                                "architecture": "joint-property-residual",
                                "modal_rank": budget // 2,
                                "property_residual_ranks": [budget // 4, budget // 4],
                                "property_rank": 2,
                            },
                        ]
                    )
                    for saturation_weight in (0.5, 2.0):
                        variants.append(
                            {
                                **base,
                                "order": "4D",
                                "variant": "4D-C",
                                "architecture": "joint",
                                "property_rank": 2,
                                "modal_rank": budget,
                                "property_weights": [1.0, saturation_weight],
                            }
                        )
                specs.extend(variants)
    unique: dict[str, dict] = {}
    for spec in specs:
        identifier = _identifier(spec)
        unique[identifier] = {"id": identifier, **spec}
    for capacity_regime, budget in (("equal_total", 32), ("equal_per_property", 64)):
        property_depth = budget // 2
        current_specs = [
            {
                "capacity_regime": capacity_regime,
                "coefficient_budget": budget,
                "spatial_ranks": [48, 48],
                "preprocessing": "raw_minmax",
                "property_weights": [1.0, 1.0],
                "order": "3D",
                "variant": "3D-current",
                "architecture": "legacy-independent",
                "property_modal_ranks": [property_depth, property_depth],
            },
            {
                "capacity_regime": capacity_regime,
                "coefficient_budget": budget,
                "spatial_ranks": [48, 48],
                "preprocessing": "raw_minmax",
                "property_weights": [1.0, 1.0],
                "order": "4D",
                "variant": "4D-A-current",
                "architecture": "legacy-joint",
                "property_rank": 2,
                "modal_rank": budget,
            },
        ]
        for spec in current_specs:
            identifier = _identifier(spec)
            unique[identifier] = {"id": identifier, **spec}
    return list(unique.values())


def _matrix(fields: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [run.reshape(-1, run.shape[-1]) for run in np.asarray(fields, dtype=float)], axis=1
    )


def _config(
    spec: dict,
    architecture: str,
    *,
    modal_rank: int | None = None,
    property_modal_ranks: tuple[int, int] | None = None,
    exact_svd: bool,
) -> bases.StructuredTBMDConfig:
    return bases.StructuredTBMDConfig(
        architecture=architecture,
        spatial_ranks=tuple(spec["spatial_ranks"]),
        property_rank=int(spec.get("property_rank", 2)),
        modal_rank=modal_rank,
        property_modal_ranks=property_modal_ranks,
        property_weights=tuple(spec.get("property_weights", (1.0, 1.0))),
        max_modal_rank=max(int(spec["coefficient_budget"]), 64),
        exact_svd=exact_svd,
        seed=20260921,
    )


def _project_residual(matrix: np.ndarray, basis: np.ndarray) -> np.ndarray:
    coefficients = np.linalg.lstsq(basis, matrix, rcond=None)[0]
    return matrix - basis @ coefficients


def _legacy_independent(
    matrix: np.ndarray,
    ij: np.ndarray,
    spec: dict,
    shape: tuple[int, int],
) -> bases.Basis:
    tl.set_backend("numpy")
    start = time.perf_counter()
    n = len(ij)
    blocks = []
    storage = 0
    for prop, depth in enumerate(spec["property_modal_ranks"]):
        grid = np.zeros(shape + (matrix.shape[1],))
        grid[ij[:, 0], ij[:, 1]] = matrix[prop * n : (prop + 1) * n]
        ranks = [min(48, shape[0]), min(48, shape[1]), int(depth)]
        core, factors = tucker(
            grid,
            rank=ranks,
            init="svd",
            n_iter_max=100,
            tol=1e-8,
            random_state=20260921,
        )
        modes = tl.tenalg.multi_mode_dot(core, factors[:2], modes=[0, 1])
        active_modes = modes[ij[:, 0], ij[:, 1]]
        block = np.zeros((2 * n, int(depth)))
        block[prop * n : (prop + 1) * n] = active_modes
        blocks.append(block)
        storage += core.size + sum(factor.size for factor in factors)
    return bases.Basis(
        "TBMD-legacy-independent",
        np.concatenate(blocks, axis=1),
        None,
        time.perf_counter() - start,
        {"architecture": "legacy-independent", "storage_floats": int(storage)},
    )


def fit_e9_model(
    raw_fit: np.ndarray,
    ij: np.ndarray,
    spec: dict,
    *,
    shape: tuple[int, int] = (139, 48),
    exact_svd: bool = False,
) -> tuple[E9Transform, bases.Basis, np.ndarray]:
    """Fit one frozen E9 representation on training scenarios only."""

    transform = E9Transform.fit(raw_fit, spec["preprocessing"])
    transformed = transform.fwd(raw_fit)
    matrix = _matrix(transformed)
    architecture = spec["architecture"]
    if architecture == "legacy-joint":
        model = bases.tbmd(
            matrix,
            ij,
            int(spec["modal_rank"]),
            spatial_ranks=(48, 48, 2),
            seed=20260921,
        )
        model.extra["architecture"] = "legacy-joint"
    elif architecture == "legacy-independent":
        model = _legacy_independent(matrix, ij, spec, shape)
    elif architecture in {"joint", "shared-independent", "independent"}:
        config = _config(
            spec,
            architecture,
            modal_rank=int(spec["modal_rank"]) if architecture == "joint" else None,
            property_modal_ranks=tuple(spec["property_modal_ranks"])
            if architecture != "joint"
            else None,
            exact_svd=exact_svd,
        )
        model = bases.structured_tbmd(matrix, ij, config, shape=shape)
    elif architecture == "partial-spatial":
        shared = bases.structured_tbmd(
            matrix,
            ij,
            _config(
                spec,
                "shared-independent",
                property_modal_ranks=tuple(spec["shared_property_ranks"]),
                exact_svd=exact_svd,
            ),
            shape=shape,
        )
        residual = _project_residual(matrix, shared.B)
        specific = bases.structured_tbmd(
            residual,
            ij,
            _config(
                spec,
                "independent",
                property_modal_ranks=tuple(spec["property_residual_ranks"]),
                exact_svd=exact_svd,
            ),
            shape=shape,
        )
        combined = np.concatenate([shared.B, specific.B], axis=1)
        model = bases.Basis(
            "TBMD-partial-spatial",
            combined,
            None,
            shared.seconds + specific.seconds,
            {
                "architecture": architecture,
                "storage_floats": shared.extra["storage_floats"]
                + specific.extra["storage_floats"],
                "components": [shared.extra, specific.extra],
            },
        )
    elif architecture == "joint-property-residual":
        joint = bases.structured_tbmd(
            matrix,
            ij,
            _config(
                spec,
                "joint",
                modal_rank=int(spec["modal_rank"]),
                exact_svd=exact_svd,
            ),
            shape=shape,
        )
        residual = _project_residual(matrix, joint.B)
        specific = bases.structured_tbmd(
            residual,
            ij,
            _config(
                spec,
                "independent",
                property_modal_ranks=tuple(spec["property_residual_ranks"]),
                exact_svd=exact_svd,
            ),
            shape=shape,
        )
        combined = np.concatenate([joint.B, specific.B], axis=1)
        model = bases.Basis(
            "TBMD-joint-property-residual",
            combined,
            None,
            joint.seconds + specific.seconds,
            {
                "architecture": architecture,
                "storage_floats": joint.extra["storage_floats"]
                + specific.extra["storage_floats"],
                "components": [joint.extra, specific.extra],
            },
        )
    else:
        raise ValueError(f"unknown E9 architecture: {architecture}")
    if model.B.shape[1] != int(spec["coefficient_budget"]):
        raise ValueError(
            f"realized coefficient count {model.B.shape[1]} does not match "
            f"the declared budget {spec['coefficient_budget']}"
        )
    model.extra["spec_id"] = spec["id"]
    model.extra["variant"] = spec["variant"]
    model.extra["preprocessing"] = spec["preprocessing"]
    return transform, model, matrix


def headline_selection_loss(scores: dict[str, float]) -> float:
    """Equal weight to error/structure groups and pressure/saturation."""

    error = np.mean(
        [
            scores["anomaly_relative_frobenius_p"],
            scores["anomaly_relative_frobenius_so"],
        ]
    )
    structural_dissimilarity = np.mean(
        [
            (1.0 - scores["anomaly_ssim_p"]) / 2.0,
            (1.0 - scores["anomaly_ssim_so"]) / 2.0,
        ]
    )
    return float(0.5 * error + 0.5 * structural_dissimilarity)
