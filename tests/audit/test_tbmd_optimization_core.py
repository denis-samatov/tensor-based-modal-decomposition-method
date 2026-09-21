"""Contracts for the leakage-safe TBMD optimization study."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[2] / "studies" / "brugge_sparse_sensing"
sys.path.insert(0, str(STUDY))

from bss import bases, estimators, placement  # noqa: E402
from bss.optimization import (  # noqa: E402
    PropertyScaler,
    compare_reproduction_values,
    fit_active_clusters,
    inner_scenario_splits,
    normalized_joint_score,
    select_candidate,
)
from bss.study import (  # noqa: E402
    build_representation_specs,
    fit_representation,
    physical_error_summary,
    placement_orders,
    solve_coefficients,
    solver_id,
    transform_runs,
)


def _synthetic_train() -> tuple[np.ndarray, np.ndarray]:
    """Return a fully active 4x3 two-property training matrix."""
    rng = np.random.default_rng(17)
    ij = np.argwhere(np.ones((4, 3), dtype=bool))
    time = np.linspace(0.0, 1.0, 20)
    p = np.stack([3.0 + (i + 1) * time + 0.2 * np.sin((j + 1) * np.pi * time) for i, j in ij])
    s = np.stack([0.2 + 0.02 * j + 0.04 * np.cos((i + 1) * np.pi * time) for i, j in ij])
    train = np.concatenate([p, s], axis=0)
    train += 1e-4 * rng.standard_normal(train.shape)
    return train, ij


def test_property_scalers_are_train_only_and_round_trip() -> None:
    train = np.array(
        [
            [[[1.0, 2.0], [3.0, 4.0]], [[0.1, 0.2], [0.3, 0.4]]],
            [[[2.0, 3.0], [4.0, 5.0]], [[0.2, 0.3], [0.4, 0.5]]],
        ]
    )
    extreme_validation = np.array([[[[1000.0]], [[-1000.0]]]])
    for method in ("none", "minmax", "standard", "rms"):
        scaler = PropertyScaler.fit(train, method=method)
        before = (scaler.offset.copy(), scaler.scale.copy())
        transformed = scaler.fwd(train)
        np.testing.assert_allclose(scaler.inv(transformed), train, rtol=0, atol=1e-12)
        scaler.fwd(extreme_validation)
        np.testing.assert_array_equal(scaler.offset, before[0])
        np.testing.assert_array_equal(scaler.scale, before[1])


def test_inner_scenario_splits_are_disjoint_and_cover_each_training_run_once() -> None:
    runs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    splits = inner_scenario_splits(runs, n_splits=3)
    assert len(splits) == 3
    validation = []
    for fit_runs, validation_runs in splits:
        assert set(fit_runs).isdisjoint(validation_runs)
        assert set(fit_runs) | set(validation_runs) == set(runs)
        validation.extend(validation_runs)
    assert sorted(validation) == runs


def test_untruncated_structured_tbmd_matches_pod_energy_subspace() -> None:
    train, ij = _synthetic_train()
    cfg = bases.StructuredTBMDConfig(
        architecture="joint",
        spatial_ranks=(4, 3),
        property_rank=2,
        modal_rank=5,
        property_weights=(1.0, 1.0),
        exact_svd=True,
    )
    tbmd = bases.structured_tbmd(train, ij, cfg, shape=(4, 3))
    pod_e = bases.pod_energy(train, r=5)
    q_tbmd = np.linalg.qr(tbmd.B)[0]
    q_pod = np.linalg.qr(pod_e.B)[0]
    distance = np.linalg.norm(q_tbmd @ q_tbmd.T - q_pod @ q_pod.T)
    assert distance < 1e-9
    assert tbmd.extra["storage_floats"] > 0
    assert tbmd.extra["realized_ranks"] == [4, 3, 2, 5]


def test_shared_independent_basis_has_property_specific_coefficients() -> None:
    train, ij = _synthetic_train()
    cfg = bases.StructuredTBMDConfig(
        architecture="shared-independent",
        spatial_ranks=(3, 3),
        property_modal_ranks=(3, 2),
        exact_svd=True,
    )
    model = bases.structured_tbmd(train, ij, cfg, shape=(4, 3))
    n = len(ij)
    assert model.B.shape == (2 * n, 5)
    np.testing.assert_array_equal(model.B[n:, :3], 0.0)
    np.testing.assert_array_equal(model.B[:n, 3:], 0.0)
    assert model.extra["architecture"] == "shared-independent"


def test_energy_selected_structured_ranks_respect_caps_and_thresholds() -> None:
    train, ij = _synthetic_train()
    cfg = bases.StructuredTBMDConfig(
        architecture="joint",
        spatial_ranks=None,
        spatial_energy=0.99,
        max_spatial_ranks=(4, 3),
        modal_rank=None,
        modal_energy=0.999,
        max_modal_rank=8,
        exact_svd=True,
    )
    model = bases.structured_tbmd(train, ij, cfg, shape=(4, 3))
    r1, r2, r3, r4 = model.extra["realized_ranks"]
    assert 1 <= r1 <= 4
    assert 1 <= r2 <= 3
    assert r3 == 2
    assert 1 <= r4 <= 8
    assert model.extra["rank_strategy"] == "energy"


def test_ridge_matches_closed_form_and_elastic_net_reduces_to_ridge() -> None:
    rng = np.random.default_rng(5)
    design = rng.normal(size=(7, 4))
    targets = rng.normal(size=(7, 3))
    alpha = 0.17
    expected = np.linalg.solve(design.T @ design + alpha * np.eye(4), design.T @ targets)
    np.testing.assert_allclose(estimators.ridge(design, targets, alpha), expected)
    actual, iterations = estimators.elastic_net_admm(
        design,
        targets,
        l1=0.0,
        l2=alpha,
        max_iter=20_000,
        tol=1e-10,
    )
    assert iterations < 20_000
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)


def test_condition_aware_block_order_is_deterministic_nested_and_unique() -> None:
    basis = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.1, 0.9],
            [0.0, 0.0, 1.0],
            [0.2, 0.6, 0.6],
        ]
    )
    blocks = [np.array([i]) for i in range(len(basis))]
    first = placement.condition_aware_blocks(basis, blocks, n_max=5, condition_weight=0.25)
    second = placement.condition_aware_blocks(basis, blocks, n_max=5, condition_weight=0.25)
    np.testing.assert_array_equal(first, second)
    assert len(first) == len(set(first.tolist())) == 5
    assert set(first[:3]).issubset(first[:5])


def test_cluster_constrained_order_is_deterministic_and_preserves_candidates() -> None:
    base = np.array([7, 3, 0, 5, 2, 6, 1, 4])
    labels = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    order = placement.cluster_constrained_order(base, labels, n_max=8)
    np.testing.assert_array_equal(order, placement.cluster_constrained_order(base, labels, n_max=8))
    assert set(order.tolist()) == set(base.tolist())
    assert len(set(labels[order[:3]].tolist())) == 3


def test_candidate_selection_uses_inner_scores_and_compression_tie_break_only() -> None:
    records = [
        {"candidate": "large", "inner_fold": 0, "score": 0.80, "storage_floats": 1000},
        {"candidate": "large", "inner_fold": 1, "score": 1.00, "storage_floats": 1000},
        {"candidate": "small", "inner_fold": 0, "score": 0.91, "storage_floats": 100},
        {"candidate": "small", "inner_fold": 1, "score": 0.91, "storage_floats": 100},
    ]
    assert select_candidate(records, tolerance=0.0)["candidate"] == "large"
    assert select_candidate(records, tolerance=0.02)["candidate"] == "small"


def test_normalized_joint_score_balances_properties() -> None:
    assert normalized_joint_score(2.0, 0.1, prior_pressure=4.0, prior_saturation=0.2) == 0.5
    assert np.isclose(
        normalized_joint_score(1.0, 0.3, prior_pressure=2.0, prior_saturation=0.1),
        1.75,
    )


def test_reproduction_comparison_uses_relative_tolerance_for_large_condition_numbers() -> None:
    expected = np.array([2.151980890980397e36, np.inf])
    reproduced = np.array([2.1519808909803966e36, np.inf])
    comparison = compare_reproduction_values(expected, reproduced, atol=0.0, rtol=1e-12)
    assert comparison["matches"]
    assert comparison["max_relative_difference"] < 1e-12
    mismatch = compare_reproduction_values(
        np.array([1.0]), np.array([1.0 + 1e-9]), atol=1e-12, rtol=1e-12
    )
    assert not mismatch["matches"]


def test_active_clustering_selects_k_on_training_features_deterministically() -> None:
    rng = np.random.default_rng(9)
    ij = np.vstack(
        [
            rng.normal(loc=(0, 0), scale=0.1, size=(12, 2)),
            rng.normal(loc=(4, 4), scale=0.1, size=(12, 2)),
            rng.normal(loc=(0, 5), scale=0.1, size=(12, 2)),
        ]
    )
    train = rng.normal(size=(2 * len(ij), 15))
    first = fit_active_clusters(train, ij, k_values=(2, 3, 4), seeds=(3, 7))
    second = fit_active_clusters(train, ij, k_values=(2, 3, 4), seeds=(3, 7))
    assert first["selected_k"] == 3
    np.testing.assert_array_equal(first["labels"], second["labels"])
    assert len(first["labels"]) == len(ij)


def test_representation_specs_are_unique_and_cover_required_architectures() -> None:
    optimization = {
        "joint_spatial_ranks": [[48, 48], [64, 48]],
        "modal_ranks": [8, 16],
        "property_ranks": [1, 2],
        "independent_spatial_ranks": [[48, 48]],
        "property_modal_ranks": [[4, 4], [8, 8]],
        "residual_ranks": [4],
        "energy_candidates": [0.99],
        "max_modal_rank": 32,
        "randomized_svd_iterations": 4,
    }
    specs = build_representation_specs(optimization)
    identifiers = [spec["id"] for spec in specs]
    assert len(identifiers) == len(set(identifiers))
    architectures = {spec["architecture"] for spec in specs if spec["family"] == "TBMD"}
    assert {"joint", "shared-independent", "independent", "joint-residual"} <= architectures
    assert {"POD", "POD-E"} <= {spec["family"] for spec in specs}
    assert any(spec["rank_strategy"] == "energy" for spec in specs)


def test_solver_dispatch_has_stable_ids_and_finite_coefficients() -> None:
    design = np.array([[1.0, 0.0], [1.0, 1e-5], [0.0, 1.0]])
    targets = np.array([[1.0], [1.0], [0.5]])
    solvers = [
        {"name": "least_squares", "rcond": 1e-10},
        {"name": "ridge", "alpha_relative": 1e-4},
        {"name": "lasso", "l1": 1e-3},
        {"name": "elastic_net", "l1": 1e-3, "l2_relative": 1e-4},
    ]
    assert len({solver_id(solver) for solver in solvers}) == len(solvers)
    for solver in solvers:
        coefficients, metadata = solve_coefficients(design, targets, solver)
        assert coefficients.shape == (2, 1)
        assert np.isfinite(coefficients).all()
        assert metadata["solver"] == solver_id(solver)


def test_relative_ridge_remains_solvable_for_zero_information_design() -> None:
    design = np.zeros((1, 4))
    targets = np.ones((1, 2))
    coefficients, metadata = solve_coefficients(
        design,
        targets,
        {"name": "ridge", "alpha_relative": 1e-8},
    )
    np.testing.assert_array_equal(coefficients, np.zeros((4, 2)))
    assert metadata["alpha_absolute"] > 0


def test_physical_error_summary_reports_rmse_and_mae_per_property() -> None:
    truth = np.array([[[1.0, 3.0], [2.0, 4.0]], [[0.1, 0.3], [0.2, 0.4]]])
    estimate = truth.copy()
    estimate[0] += 2.0
    estimate[1] -= 0.1
    summary = physical_error_summary(estimate, truth)
    assert summary["rmse_p"] == 2.0
    assert np.isclose(summary["rmse_so"], 0.1)
    assert summary["mae_p"] == 2.0
    assert np.isclose(summary["mae_so"], 0.1)


def test_fit_representation_and_transform_runs_preserve_declared_shapes() -> None:
    train, ij = _synthetic_train()
    n = len(ij)
    raw = train.reshape(2, n, 20)[None, ...]
    spec = {
        "id": "synthetic",
        "family": "TBMD",
        "architecture": "joint",
        "scaling": "standard",
        "rank_strategy": "fixed",
        "spatial_ranks": [4, 3],
        "property_rank": 2,
        "modal_rank": 4,
        "property_weights": [1.0, 1.0],
    }
    scaler, basis, stacked = fit_representation(raw, ij, spec, shape=(4, 3), exact_svd=True)
    assert stacked.shape == (2 * n, 20)
    assert basis.B.shape == (2 * n, 4)
    transformed = transform_runs(raw, scaler)
    np.testing.assert_allclose(transformed, stacked)


def test_placement_orders_cover_declared_well_and_grid_objectives() -> None:
    rng = np.random.default_rng(41)
    basis = rng.normal(size=(20, 4))
    well_rows = np.array([0, 2, 4, 6, 8])
    ij = np.column_stack([np.arange(10), np.zeros(10)])
    clusters = np.arange(10) % 2
    orders = placement_orders(
        basis,
        well_rows,
        ij,
        max_wells=5,
        max_grid=8,
        condition_weight=0.1,
        cluster_labels=clusters,
    )
    assert {
        "configured",
        "dg_joint",
        "condition_joint",
        "dg_pressure",
        "condition_pressure",
    } <= set(orders["well"])
    assert {
        "qr_joint",
        "qr_whitened",
        "condition_joint",
        "cluster_constrained",
        "well_region",
    } <= set(orders["grid"])
    for order in orders["well"].values():
        assert len(order) == len(set(order.tolist())) == 5
    for order in orders["grid"].values():
        assert len(order) == len(set(order.tolist())) == 8
