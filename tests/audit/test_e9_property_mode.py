"""Scientific contracts for E9 third- versus fourth-order reconstruction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STUDY = Path(__file__).resolve().parents[2] / "studies" / "brugge_sparse_sensing"
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(STUDY / "scripts"))

from bss.e9_metrics import (  # noqa: E402
    MetricReference,
    evaluate_reconstruction,
    masked_ssim,
    masked_ssim_stack,
)
from bss.e9_study import (  # noqa: E402
    E9Transform,
    build_e9_candidate_specs,
    fit_e9_model,
    headline_selection_loss,
)
from e9_compare_reproduction import compare  # noqa: E402


def _irregular_mask() -> np.ndarray:
    mask = np.ones((13, 11), dtype=bool)
    mask[:3, :4] = False
    mask[9:, 8:] = False
    mask[6, 5] = False
    return mask


def test_metric_reference_is_fitted_from_training_only() -> None:
    train = np.zeros((3, 2, 5, 4))
    train[:, 0] = np.arange(60).reshape(3, 5, 4)
    train[:, 1] = 0.1 + np.arange(60).reshape(3, 5, 4) / 100
    reference = MetricReference.fit(train)
    before = (
        reference.ensemble_mean.copy(),
        reference.raw_range.copy(),
        reference.anomaly_range.copy(),
    )
    extreme_test = np.full((2, 5, 4), 1e9)
    evaluate_reconstruction(extreme_test, extreme_test, reference, np.ones((1, 5), bool))
    np.testing.assert_array_equal(reference.ensemble_mean, before[0])
    np.testing.assert_array_equal(reference.raw_range, before[1])
    np.testing.assert_array_equal(reference.anomaly_range, before[2])


def test_reconstruction_metrics_are_property_separate_and_match_formulas() -> None:
    train = np.array(
        [
            [
                [[10.0, 12.0], [14.0, 16.0]],
                [[0.1, 0.2], [0.3, 0.4]],
            ],
            [
                [[12.0, 14.0], [16.0, 18.0]],
                [[0.2, 0.3], [0.4, 0.5]],
            ],
        ]
    )
    truth = train[0]
    estimate = truth.copy()
    estimate[0] += 2.0
    estimate[1] -= 0.05
    reference = MetricReference.fit(train)
    scores = evaluate_reconstruction(estimate, truth, reference, np.ones((1, 2), bool))
    expected_p = np.linalg.norm(estimate[0] - truth[0]) / np.linalg.norm(truth[0])
    expected_s = np.linalg.norm(estimate[1] - truth[1]) / np.linalg.norm(truth[1])
    assert np.isclose(scores["relative_frobenius_p"], expected_p)
    assert np.isclose(scores["relative_frobenius_so"], expected_s)
    assert np.isclose(scores["rmse_p"], 2.0)
    assert np.isclose(scores["rmse_so"], 0.05)
    assert scores["anomaly_relative_frobenius_p"] != scores["relative_frobenius_p"]


def test_anomaly_relative_error_is_not_diluted_by_large_pressure_baseline() -> None:
    time = np.arange(7, dtype=float)
    train = np.stack(
        [
            np.stack(
                [
                    np.broadcast_to(1000.0 + time, (9, 7)),
                    np.broadcast_to(0.2 + 0.01 * time, (9, 7)),
                ]
            ),
            np.stack(
                [
                    np.broadcast_to(1002.0 + time, (9, 7)),
                    np.broadcast_to(0.22 + 0.01 * time, (9, 7)),
                ]
            ),
        ]
    )
    truth = train[0]
    estimate = truth.copy()
    estimate[0] += 1.0
    reference = MetricReference.fit(train)
    scores = evaluate_reconstruction(estimate, truth, reference, np.ones((3, 3), bool))
    assert scores["relative_frobenius_p"] < 0.002
    assert scores["anomaly_relative_frobenius_p"] > 0.9


def test_masked_ssim_identity_symmetry_and_determinism() -> None:
    mask = _irregular_mask()
    rng = np.random.default_rng(9)
    field = rng.normal(size=mask.shape)
    first = masked_ssim(field, field, mask, data_range=5.0)
    second = masked_ssim(field, field, mask, data_range=5.0)
    assert first == pytest.approx(1.0, abs=1e-12)
    assert second == first
    perturbed = field.copy()
    perturbed[mask] += 0.2 * rng.normal(size=mask.sum())
    forward = masked_ssim(field, perturbed, mask, data_range=5.0)
    reverse = masked_ssim(perturbed, field, mask, data_range=5.0)
    assert forward == pytest.approx(reverse, abs=1e-14)
    assert forward < first


def test_vectorized_ssim_matches_slice_wise_definition() -> None:
    mask = _irregular_mask()
    rng = np.random.default_rng(19)
    truth = rng.normal(size=(5, *mask.shape))
    estimate = truth + 0.1 * rng.normal(size=truth.shape)
    expected = np.mean(
        [masked_ssim(truth[index], estimate[index], mask, data_range=4.0) for index in range(5)]
    )
    assert masked_ssim_stack(truth, estimate, mask, data_range=4.0) == pytest.approx(
        expected, abs=1e-14
    )


def test_masked_ssim_ignores_inactive_background_values() -> None:
    mask = _irregular_mask()
    y = np.linspace(0.0, 1.0, mask.size).reshape(mask.shape)
    x = y.copy()
    x[mask] += 0.03
    score = masked_ssim(y, x, mask, data_range=1.0)
    y2, x2 = y.copy(), x.copy()
    y2[~mask] = -1e12
    x2[~mask] = 1e12
    assert masked_ssim(y2, x2, mask, data_range=1.0) == pytest.approx(score, abs=1e-14)


def test_masked_ssim_rejects_invalid_windows_and_ranges() -> None:
    field = np.ones((7, 7))
    sparse_mask = np.zeros((7, 7), dtype=bool)
    sparse_mask[3, 3] = True
    with pytest.raises(ValueError, match="valid SSIM windows"):
        masked_ssim(field, field, sparse_mask, data_range=1.0, min_active_fraction=0.8)
    with pytest.raises(ValueError, match="data_range"):
        masked_ssim(field, field, np.ones((7, 7), bool), data_range=0.0)


def _synthetic_runs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    mask = np.ones((4, 3), dtype=bool)
    ij = np.argwhere(mask)
    time = np.linspace(0.0, 1.0, 12)
    runs = []
    for run in range(4):
        pressure = np.stack(
            [100.0 + 2 * run + i * time + 0.2 * np.sin((j + 1) * time) for i, j in ij]
        )
        saturation = np.stack(
            [0.2 + 0.01 * run + 0.02 * j * time + 0.01 * np.cos((i + 1) * time) for i, j in ij]
        )
        runs.append(np.stack([pressure, saturation]))
    fields = np.stack(runs)
    fields += rng.normal(scale=np.array([1e-3, 1e-5])[None, :, None, None], size=fields.shape)
    return fields, ij


@pytest.mark.parametrize(
    "mode",
    ("raw_minmax", "mean_centered", "ensemble_anomaly", "standardized_anomaly"),
)
def test_e9_transform_is_train_only_and_round_trips(mode: str) -> None:
    fields, _ = _synthetic_runs()
    transform = E9Transform.fit(fields[:3], mode)
    transformed = transform.fwd(fields[3])
    np.testing.assert_allclose(transform.inv(transformed), fields[3], rtol=0, atol=1e-12)
    before = (transform.baseline.copy(), transform.scale.copy())
    transform.fwd(np.full_like(fields[3], 1e12))
    np.testing.assert_array_equal(transform.baseline, before[0])
    np.testing.assert_array_equal(transform.scale, before[1])


def test_e9_candidates_cover_matched_capacities_and_fourth_order_variants() -> None:
    specs = build_e9_candidate_specs()
    assert len({spec["id"] for spec in specs}) == len(specs)
    assert {"equal_total", "equal_per_property"} == {spec["capacity_regime"] for spec in specs}
    fourth_order = {spec["variant"] for spec in specs if spec["order"] == "4D"}
    assert {"4D-A", "4D-B", "4D-C", "4D-D", "4D-E"} <= fourth_order
    for spec in specs:
        expected = 32 if spec["capacity_regime"] == "equal_total" else 64
        assert spec["coefficient_budget"] == expected


def test_partial_and_joint_residual_models_preserve_coefficient_budget() -> None:
    fields, ij = _synthetic_runs()
    specs = [
        spec
        for spec in build_e9_candidate_specs(spatial_ranks=((4, 3),), preprocessing=("standardized_anomaly",))
        if spec["capacity_regime"] == "equal_total" and spec["variant"] in {"4D-D", "4D-E"}
    ]
    assert len(specs) == 2
    for spec in specs:
        _, model, transformed = fit_e9_model(fields, ij, spec, shape=(4, 3), exact_svd=True)
        assert model.B.shape == (2 * len(ij), 32)
        assert transformed.shape == (2 * len(ij), fields.shape[0] * fields.shape[-1])
        assert model.extra["storage_floats"] > 0


def test_headline_selection_loss_balances_error_and_structure() -> None:
    scores = {
        "anomaly_relative_frobenius_p": 0.4,
        "anomaly_relative_frobenius_so": 0.6,
        "anomaly_ssim_p": 0.8,
        "anomaly_ssim_so": 0.6,
    }
    # 0.5 * mean errors + 0.5 * mean DSSIM, DSSIM=(1-SSIM)/2.
    assert headline_selection_loss(scores) == pytest.approx(0.325)


def test_e9_reproduction_comparison_ignores_runtime_but_not_science() -> None:
    row = {
        "outer_fold": 1,
        "capacity_regime": "equal_total",
        "geometry": "wells",
        "budget": 5,
        "pair_4d_variant": "4D-D",
        "pair_role": "3D",
        "candidate": "candidate-3d",
        "fit_seconds": 1.0,
        "anomaly_ssim_p": 0.75,
    }
    canonical = pd.DataFrame([row])
    rerun = pd.DataFrame([{**row, "fit_seconds": 999.0}])
    assert compare(canonical, rerun)["status"] == "pass"
    rerun.loc[0, "anomaly_ssim_p"] += 1e-5
    assert compare(canonical, rerun)["status"] == "fail"
