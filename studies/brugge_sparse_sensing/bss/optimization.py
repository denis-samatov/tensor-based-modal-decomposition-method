"""Leakage-safe helpers for the nested TBMD optimization study."""

from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import combinations

import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PropertyScaler:
    """Train-fitted affine property transform.

    Arrays use the study convention with the property axis at ``-3``.  The
    transform is intentionally immutable so validation/test calls cannot alter
    train-fitted statistics.
    """

    method: str
    offset: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, fields: np.ndarray, method: str) -> "PropertyScaler":
        values = np.moveaxis(np.asarray(fields, dtype=float), -3, 0).reshape(2, -1)
        if method == "none":
            offset = np.zeros(2)
            scale = np.ones(2)
        elif method == "minmax":
            offset = values.min(axis=1)
            scale = values.max(axis=1) - offset
        elif method == "standard":
            offset = values.mean(axis=1)
            scale = values.std(axis=1)
        elif method == "rms":
            offset = np.zeros(2)
            scale = np.sqrt(np.mean(values**2, axis=1))
        else:
            raise ValueError(f"unknown property scaling method: {method}")
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError(f"non-positive or non-finite property scale for method {method}")
        return cls(method=method, offset=offset, scale=scale)

    def fwd(self, fields: np.ndarray) -> np.ndarray:
        return self._apply(fields, inverse=False)

    def inv(self, fields: np.ndarray) -> np.ndarray:
        return self._apply(fields, inverse=True)

    def _apply(self, fields: np.ndarray, *, inverse: bool) -> np.ndarray:
        values = np.asarray(fields, dtype=float)
        if values.ndim >= 3 and values.shape[-3] == 2:
            shape = [1] * values.ndim
            shape[-3] = 2
            offset = self.offset.reshape(shape)
            scale = self.scale.reshape(shape)
        else:
            n = values.shape[-2] // 2
            shape = [1] * values.ndim
            shape[-2] = 2 * n
            offset = np.repeat(self.offset, n).reshape(shape)
            scale = np.repeat(self.scale, n).reshape(shape)
        return values * scale + offset if inverse else (values - offset) / scale


def inner_scenario_splits(
    train_runs: list[int], n_splits: int = 3
) -> list[tuple[list[int], list[int]]]:
    """Deterministic grouped inner CV over outer-training scenario identifiers."""

    runs = sorted(int(run) for run in train_runs)
    if not 2 <= n_splits <= len(runs):
        raise ValueError("n_splits must be between 2 and the number of training runs")
    validation_groups = [group.tolist() for group in np.array_split(np.asarray(runs), n_splits)]
    splits: list[tuple[list[int], list[int]]] = []
    for validation in validation_groups:
        validation_set = set(validation)
        fit = [run for run in runs if run not in validation_set]
        splits.append((fit, validation))
    return splits


def normalized_joint_score(
    rmse_pressure: float, rmse_saturation: float, *, prior_pressure: float, prior_saturation: float
) -> float:
    """Equal-property validation loss relative to the no-measurement prior."""
    if prior_pressure <= 0 or prior_saturation <= 0:
        raise ValueError("prior errors must be positive")
    return 0.5 * (rmse_pressure / prior_pressure + rmse_saturation / prior_saturation)


def compare_reproduction_values(
    expected: np.ndarray,
    observed: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float | bool]:
    """Compare deterministic outputs without misclassifying equal infinities.

    A relative term is essential for diagnostics such as condition numbers,
    whose meaningful reproducibility error is scale-relative rather than an
    absolute difference at magnitudes above ``1e20``.
    """

    expected = np.asarray(expected, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if expected.shape != observed.shape:
        return {
            "matches": False,
            "max_absolute_difference": float("inf"),
            "max_relative_difference": float("inf"),
        }
    finite = np.isfinite(expected) & np.isfinite(observed)
    nonfinite_equal = (
        (np.isnan(expected) & np.isnan(observed))
        | (np.isposinf(expected) & np.isposinf(observed))
        | (np.isneginf(expected) & np.isneginf(observed))
    )
    absolute = np.abs(expected[finite] - observed[finite])
    scale = np.maximum(np.abs(expected[finite]), np.abs(observed[finite]))
    relative = absolute / np.maximum(scale, np.finfo(float).tiny)
    finite_matches = absolute <= atol + rtol * scale
    matches = bool(np.all(finite | nonfinite_equal) and np.all(finite_matches))
    return {
        "matches": matches,
        "max_absolute_difference": float(absolute.max(initial=0.0)),
        "max_relative_difference": float(relative.max(initial=0.0)),
    }


def select_candidate(records: list[dict], tolerance: float = 0.0) -> dict:
    """Select by mean inner score, using storage only within an accuracy tolerance."""
    if not records:
        raise ValueError("candidate records are empty")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record["candidate"]), []).append(record)
    summary = []
    for candidate, rows in grouped.items():
        summary.append(
            {
                "candidate": candidate,
                "mean_score": float(np.mean([float(row["score"]) for row in rows])),
                "score_std": float(np.std([float(row["score"]) for row in rows], ddof=1))
                if len(rows) > 1
                else 0.0,
                "storage_floats": int(round(np.mean([int(row["storage_floats"]) for row in rows]))),
                "n_inner": len(rows),
            }
        )
    best_score = min(row["mean_score"] for row in summary)
    limit = best_score * (1.0 + tolerance)
    eligible = [row for row in summary if row["mean_score"] <= limit + 1e-15]
    return min(
        eligible, key=lambda row: (row["storage_floats"], row["mean_score"], row["candidate"])
    )


def fit_active_clusters(
    train: np.ndarray,
    ij: np.ndarray,
    *,
    k_values: tuple[int, ...] = (2, 3, 4, 5, 6),
    seeds: tuple[int, ...] = (0, 1, 2),
) -> dict:
    """Select deterministic active-cell clusters from train-only spatial/dynamic features."""
    train = np.asarray(train, dtype=float)
    ij = np.asarray(ij, dtype=float)
    n = len(ij)
    if train.shape[0] != 2 * n:
        raise ValueError("train must contain pressure and saturation blocks for every active cell")
    dynamics = np.column_stack(
        [
            train[:n].mean(axis=1),
            train[:n].std(axis=1),
            train[n:].mean(axis=1),
            train[n:].std(axis=1),
        ]
    )
    spatial = StandardScaler().fit_transform(ij)
    dynamic = StandardScaler().fit_transform(dynamics)
    features = np.column_stack([2.0 * spatial, dynamic])
    diagnostics = []
    label_sets: dict[int, list[np.ndarray]] = {}
    for k in sorted(set(int(value) for value in k_values)):
        if not 2 <= k < n:
            continue
        labels_for_k = []
        silhouettes = []
        for seed in seeds:
            labels = KMeans(n_clusters=k, n_init=10, random_state=int(seed)).fit_predict(features)
            labels_for_k.append(labels)
            silhouettes.append(
                float(
                    silhouette_score(
                        features,
                        labels,
                        sample_size=min(1000, n),
                        random_state=int(seed),
                    )
                )
            )
        stability = [adjusted_rand_score(a, b) for a, b in combinations(labels_for_k, 2)]
        mean_stability = float(np.mean(stability)) if stability else 1.0
        diagnostics.append(
            {
                "k": k,
                "silhouette": float(np.mean(silhouettes)),
                "stability": mean_stability,
                "selection_score": float(np.mean(silhouettes) + 0.1 * mean_stability),
            }
        )
        label_sets[k] = labels_for_k
    if not diagnostics:
        raise ValueError("no admissible cluster counts")
    selected = max(diagnostics, key=lambda row: (row["selection_score"], -row["k"]))
    selected_k = int(selected["k"])
    return {
        "selected_k": selected_k,
        "labels": label_sets[selected_k][0],
        "diagnostics": diagnostics,
        "feature_names": [
            "grid_i",
            "grid_j",
            "pressure_mean",
            "pressure_std",
            "saturation_mean",
            "saturation_std",
        ],
    }
