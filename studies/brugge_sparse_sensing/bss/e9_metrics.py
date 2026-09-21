"""Leakage-safe reconstruction and structural metrics for experiment E9.

The historical Brugge workflow evaluated rectangular arrays with inactive
background cells. This module instead computes local SSIM moments with
mask-normalised Gaussian weights and averages only windows whose centre is
active and whose in-domain support contains enough active cells. Dynamic
ranges and anomaly references are fitted from the current training partition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve


@dataclass(frozen=True)
class MetricReference:
    """Train-only reference quantities used by all E9 metrics."""

    ensemble_mean: np.ndarray
    raw_range: np.ndarray
    anomaly_range: np.ndarray

    @classmethod
    def fit(cls, training_fields: np.ndarray) -> "MetricReference":
        fields = np.asarray(training_fields, dtype=float)
        if fields.ndim != 4 or fields.shape[1] != 2:
            raise ValueError("training_fields must have shape (runs, 2, active_cells, time)")
        if not np.isfinite(fields).all():
            raise ValueError("training_fields contain non-finite values")
        ensemble_mean = fields.mean(axis=0)
        anomalies = fields - ensemble_mean[None, ...]
        raw_range = np.ptp(fields, axis=(0, 2, 3))
        anomaly_range = np.ptp(anomalies, axis=(0, 2, 3))
        floor = np.finfo(float).eps
        if np.any(raw_range <= floor):
            raise ValueError("training property range is zero")
        anomaly_range = np.where(anomaly_range > floor, anomaly_range, raw_range)
        return cls(
            ensemble_mean=ensemble_mean.copy(),
            raw_range=raw_range.copy(),
            anomaly_range=anomaly_range.copy(),
        )


def _gaussian_kernel(win_size: int, sigma: float) -> np.ndarray:
    if win_size < 3 or win_size % 2 == 0:
        raise ValueError("win_size must be an odd integer of at least 3")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = win_size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=float)
    one_dimensional = np.exp(-(coordinates**2) / (2.0 * sigma**2))
    one_dimensional /= one_dimensional.sum()
    return np.outer(one_dimensional, one_dimensional)


def masked_ssim(
    reference: np.ndarray,
    estimate: np.ndarray,
    active_mask: np.ndarray,
    *,
    data_range: float,
    win_size: int = 7,
    sigma: float = 1.5,
    min_active_fraction: float = 0.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Return deterministic local SSIM without inactive-background leakage.

    Local moments use a Gaussian window renormalised over active cells.
    Windows are retained only when the centre is active and active cells
    account for at least min_active_fraction of the in-domain window weight.
    Thus values outside active_mask cannot change the score.
    """

    truth = np.asarray(reference, dtype=float)
    prediction = np.asarray(estimate, dtype=float)
    mask = np.asarray(active_mask, dtype=bool)
    if truth.ndim != 2 or prediction.shape != truth.shape or mask.shape != truth.shape:
        raise ValueError("reference, estimate, and active_mask must share one 2-D shape")
    return masked_ssim_stack(
        truth[None, ...],
        prediction[None, ...],
        mask,
        data_range=data_range,
        win_size=win_size,
        sigma=sigma,
        min_active_fraction=min_active_fraction,
        k1=k1,
        k2=k2,
    )


def masked_ssim_stack(
    reference: np.ndarray,
    estimate: np.ndarray,
    active_mask: np.ndarray,
    *,
    data_range: float,
    win_size: int = 7,
    sigma: float = 1.5,
    min_active_fraction: float = 0.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Vectorised mean of the exact snapshot-wise mask-aware SSIM definition."""

    truth = np.asarray(reference, dtype=float)
    prediction = np.asarray(estimate, dtype=float)
    mask = np.asarray(active_mask, dtype=bool)
    if (
        truth.ndim != 3
        or prediction.shape != truth.shape
        or mask.shape != truth.shape[1:]
    ):
        raise ValueError(
            "reference and estimate must be (time, I, J), with a matching 2-D active_mask"
        )
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be finite and positive")
    if not 0 < min_active_fraction <= 1:
        raise ValueError("min_active_fraction must be in (0, 1]")
    if not mask.any():
        raise ValueError("active_mask is empty")
    if not np.isfinite(truth[:, mask]).all() or not np.isfinite(prediction[:, mask]).all():
        raise ValueError("active field values must be finite")

    kernel_2d = _gaussian_kernel(win_size, sigma)
    kernel = kernel_2d[None, ...]
    active = mask.astype(float)
    domain_weight = convolve(
        np.ones_like(active), kernel_2d, mode="constant", cval=0.0
    )
    active_weight = convolve(active, kernel_2d, mode="constant", cval=0.0)
    active_fraction = active_weight / np.maximum(domain_weight, np.finfo(float).tiny)
    valid = mask & (active_fraction >= min_active_fraction) & (active_weight > 0)
    if not valid.any():
        raise ValueError("no valid SSIM windows for the requested active-cell threshold")

    x = np.where(mask[None, ...], truth, 0.0)
    y = np.where(mask[None, ...], prediction, 0.0)
    normalizer = np.maximum(active_weight, np.finfo(float).tiny)
    mu_x = convolve(x, kernel, mode="constant", cval=0.0) / normalizer[None, ...]
    mu_y = convolve(y, kernel, mode="constant", cval=0.0) / normalizer[None, ...]
    second_x = (
        convolve(x * x, kernel, mode="constant", cval=0.0) / normalizer[None, ...]
    )
    second_y = (
        convolve(y * y, kernel, mode="constant", cval=0.0) / normalizer[None, ...]
    )
    cross = convolve(x * y, kernel, mode="constant", cval=0.0) / normalizer[None, ...]
    variance_x = np.maximum(second_x - mu_x * mu_x, 0.0)
    variance_y = np.maximum(second_y - mu_y * mu_y, 0.0)
    covariance = cross - mu_x * mu_y

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * covariance + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (variance_x + variance_y + c2)
    local = numerator / np.maximum(denominator, np.finfo(float).tiny)
    return float(np.mean(local[:, valid]))


def _active_to_grid(values: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    grid = np.zeros((values.shape[-1], *active_mask.shape), dtype=float)
    grid[:, active_mask] = values.T
    return grid


def _safe_relative(numerator: np.ndarray, denominator: np.ndarray) -> float:
    scale = float(np.linalg.norm(denominator))
    if scale <= np.finfo(float).eps:
        return 0.0 if np.linalg.norm(numerator) <= np.finfo(float).eps else float("inf")
    return float(np.linalg.norm(numerator) / scale)


def evaluate_reconstruction(
    estimate: np.ndarray,
    truth: np.ndarray,
    reference: MetricReference,
    active_mask: np.ndarray,
    *,
    win_size: int = 7,
    sigma: float = 1.5,
    min_active_fraction: float = 0.5,
) -> dict[str, float]:
    """Evaluate E9 headline and diagnostic metrics in physical units."""

    prediction = np.asarray(estimate, dtype=float)
    target = np.asarray(truth, dtype=float)
    mask = np.asarray(active_mask, dtype=bool)
    if prediction.shape != target.shape or target.ndim != 3 or target.shape[0] != 2:
        raise ValueError("estimate and truth must have matching shape (2, active_cells, time)")
    if target.shape[1] != int(mask.sum()):
        raise ValueError("active_mask count does not match the active-cell dimension")
    if reference.ensemble_mean.shape != target.shape:
        raise ValueError("metric reference and evaluation fields have incompatible shapes")

    errors = prediction - target
    anomalies_truth = target - reference.ensemble_mean
    anomalies_prediction = prediction - reference.ensemble_mean
    result: dict[str, float] = {}
    suffixes = ("p", "so")
    for prop, suffix in enumerate(suffixes):
        result[f"relative_frobenius_{suffix}"] = _safe_relative(errors[prop], target[prop])
        result[f"anomaly_relative_frobenius_{suffix}"] = _safe_relative(
            errors[prop], anomalies_truth[prop]
        )
        rmse = float(np.sqrt(np.mean(errors[prop] ** 2)))
        result[f"rmse_{suffix}"] = rmse
        result[f"train_range_nrmse_{suffix}"] = rmse / float(reference.raw_range[prop])

        raw_truth = _active_to_grid(target[prop], mask)
        raw_prediction = _active_to_grid(prediction[prop], mask)
        anomaly_truth = _active_to_grid(anomalies_truth[prop], mask)
        anomaly_prediction = _active_to_grid(anomalies_prediction[prop], mask)
        try:
            result[f"ssim_{suffix}"] = masked_ssim_stack(
                raw_truth,
                raw_prediction,
                mask,
                data_range=float(reference.raw_range[prop]),
                win_size=win_size,
                sigma=sigma,
                min_active_fraction=min_active_fraction,
            )
            result[f"anomaly_ssim_{suffix}"] = masked_ssim_stack(
                anomaly_truth,
                anomaly_prediction,
                mask,
                data_range=float(reference.anomaly_range[prop]),
                win_size=win_size,
                sigma=sigma,
                min_active_fraction=min_active_fraction,
            )
        except ValueError as error:
            if "valid SSIM windows" not in str(error):
                raise
            result[f"ssim_{suffix}"] = float("nan")
            result[f"anomaly_ssim_{suffix}"] = float("nan")
    return result
