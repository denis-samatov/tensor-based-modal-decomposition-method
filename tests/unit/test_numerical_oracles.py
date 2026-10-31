"""CPU float64 analytic oracles, frozen before evaluation (2026-10-31).

These cases qualify numerical contracts, not Brugge accuracy or generalization.
No Tucker/ADMM iteration is replaced by an oracle implementation.
"""

import numpy as np
import pytest
import torch

from TBMD.core.decomposition import TuckerDecomposer
from TBMD.core.reconstruction.tensor_compressive_sensing import (
    CompressiveSensingConfig,
    ExtensionCompressiveSensingConfig,
    TensorCompressiveSensing,
)


def test_tucker_recovers_known_subspaces_and_independent_dense_einsum():
    rng = np.random.default_rng(20261031)
    factors = [np.linalg.qr(rng.normal(size=(n, r)))[0] for n, r in zip((6, 5, 4), (2, 3, 2))]
    core = rng.normal(size=(2, 3, 2))
    source = np.einsum("abc,ia,jb,kc->ijk", core, *factors)
    decomposer = TuckerDecomposer(
        source, ranks=[2, 3, 2], epsilon=1e-12, random_state=20261031, dtype=torch.float64
    )
    decomposer.decompose()
    decomposer.reconstruct()
    recovered = [factor.numpy() for factor in decomposer.factors]
    independent = np.einsum("abc,ia,jb,kc->ijk", decomposer.cores.numpy(), *recovered)
    np.testing.assert_allclose(independent, source, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        decomposer.reconstructed_tensors.numpy(), independent, rtol=1e-10, atol=1e-10
    )
    for actual, expected in zip(recovered, factors):
        np.testing.assert_allclose(actual @ actual.T, expected @ expected.T, rtol=1e-10, atol=1e-10)
    # The check must reject a changed factor; signs/permutations alone are not compared.
    corrupted = recovered.copy()
    corrupted[0] = corrupted[0] + 0.1
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            np.einsum("abc,ia,jb,kc->ijk", decomposer.cores.numpy(), *corrupted),
            source,
            rtol=1e-10,
            atol=1e-10,
        )


def test_rank_one_matrix_matches_eckart_young_residual():
    source = np.diag([5.0, 3.0, 1.0])
    decomposer = TuckerDecomposer(source, ranks=[1, 1], epsilon=1e-12, dtype=torch.float64)
    decomposer.decompose()
    decomposer.reconstruct()
    np.testing.assert_allclose(
        decomposer.reconstructed_tensors.numpy(), np.diag([5.0, 0.0, 0.0]), rtol=1e-10, atol=1e-10
    )
    assert decomposer.reconstruction_errors == pytest.approx(np.sqrt(10 / 35), abs=1e-10)


def test_zero_tensor_reconstructs_with_finite_zero_error():
    decomposer = TuckerDecomposer(np.zeros((3, 2, 2)), ranks=[3, 2, 2], dtype=torch.float64)
    decomposer.decompose()
    decomposer.reconstruct()
    assert decomposer.reconstruction_errors == 0
    assert torch.count_nonzero(decomposer.reconstructed_tensors) == 0


def diagonal_problem(solver, sentinel=1e6, zero=False):
    # Selected flattened spatial positions are non-contiguous; the last axis is modes.
    design = np.arange(18, dtype=float).reshape(6, 3) + 10
    selected = np.array([0, 2, 5])
    design[selected] = np.eye(3)
    mask = np.zeros(6, dtype=bool)
    mask[selected] = True
    observations = np.full(6, sentinel)
    observations[selected] = [0.0, 0.0, 0.0] if zero else [2.0, -0.5, 0.05]
    return TensorCompressiveSensing(
        design.reshape(2, 3, 3),
        mask.reshape(2, 3),
        observations.reshape(2, 3),
        core_cfg=CompressiveSensingConfig(
            max_iter=5000,
            tol=1e-10,
            epsilon_l1=0.1,
            delta_init=1.0,
            delta_max=1.0,
            relax_lambda=0.95,
            dtype=torch.float64,
        ),
        ext_cfg=ExtensionCompressiveSensingConfig(
            solver=solver, reg=0.0, delta_policy="cap_only", stop_policy="residual"
        ),
    )


def kkt_violation(x):
    # Independent stationarity of 0.5||x-y||² + .1||x||_1 for identity selected design.
    gradient = x - np.array([2.0, -0.5, 0.05])
    active = np.abs(x) > 2e-7
    violations = np.where(
        active, np.abs(gradient + 0.1 * np.sign(x)), np.maximum(np.abs(gradient) - 0.1, 0.0)
    )
    return float(violations.max())


@pytest.mark.parametrize("solver", ["cholesky", "direct", "svd"])
def test_masked_lasso_matches_soft_threshold_and_kkt_with_ignored_observations(solver):
    first = diagonal_problem(solver)
    solution, metrics = first.solve()
    expected = np.array([1.9, -0.4, 0.0])
    np.testing.assert_allclose(solution.numpy(), expected, atol=2e-7, rtol=0.0)
    assert metrics.converged
    assert kkt_violation(solution.numpy()) <= 2e-7
    other, other_metrics = diagonal_problem(solver, sentinel=-1e6).solve()
    assert other_metrics.converged
    torch.testing.assert_close(solution, other, rtol=0.0, atol=0.0)
    # Negative controls: unregularized solution and incorrect threshold signs are detectable.
    assert kkt_violation(np.array([2.0, -0.5, 0.05])) > 2e-7
    assert kkt_violation(np.array([2.1, -0.6, 0.15])) > 2e-7


def test_cs_zero_measurement_error_is_finite_and_nonzero_residual_is_undefined():
    problem = diagonal_problem("cholesky", zero=True)
    solution, metrics = problem.solve()
    assert metrics.converged
    assert problem.reconstruction_error(solution) == 0.0
    with pytest.raises(ValueError, match="zero"):
        problem.reconstruction_error(np.ones(3))


def test_nonzero_float32_error_keeps_torch_dtype_division():
    design = torch.eye(3, dtype=torch.float32)
    observations = torch.tensor([0.3, 0.7, 0.2], dtype=torch.float32)
    candidate = torch.tensor([0.11, 0.19, 0.43], dtype=torch.float32)
    problem = TensorCompressiveSensing(design, torch.ones(3, dtype=torch.bool), observations)
    expected = (torch.norm(design @ candidate - observations) / torch.norm(observations)).item()
    assert problem.reconstruction_error(candidate) == expected


@pytest.mark.parametrize(("dtype", "tiny"), [(torch.float32, 1e-30), (torch.float64, 1e-200)])
def test_nonzero_measurements_with_norm_underflow_keep_relative_error(dtype, tiny):
    observations = torch.tensor([tiny, 2 * tiny, -tiny], dtype=dtype)
    problem = TensorCompressiveSensing(
        torch.eye(3, dtype=dtype),
        torch.ones(3, dtype=torch.bool),
        observations,
        core_cfg=CompressiveSensingConfig(dtype=dtype),
    )
    assert torch.count_nonzero(observations) == 3
    assert torch.norm(observations).item() == 0.0  # reproduce the dtype's norm underflow
    assert problem.reconstruction_error(torch.zeros(3, dtype=dtype)) == pytest.approx(1.0)
    assert problem.reconstruction_error(observations) == 0.0


def test_zero_measurements_with_tiny_residual_norm_underflow_are_undefined():
    problem = TensorCompressiveSensing(
        torch.eye(3), torch.ones(3, dtype=torch.bool), torch.zeros(3)
    )
    candidate = torch.tensor([1e-30, 2e-30, -1e-30])
    assert torch.count_nonzero(candidate) == 3 and torch.norm(candidate).item() == 0.0
    with pytest.raises(ValueError, match="zero"):
        problem.reconstruction_error(candidate)


@pytest.mark.parametrize("coefficient", [1.0, 1e-30])
def test_norm_underflow_with_large_finite_relative_error_avoids_overflow(coefficient):
    import math

    observations = torch.tensor([1e-200, 2e-200, -1e-200], dtype=torch.float64)
    candidate = torch.tensor([coefficient, 0.0, 0.0], dtype=torch.float64)
    problem = TensorCompressiveSensing(
        torch.eye(3, dtype=torch.float64),
        torch.ones(3, dtype=torch.bool),
        observations,
        core_cfg=CompressiveSensingConfig(dtype=torch.float64),
    )
    # Python hypot is an independent scaled Euclidean norm, not the implementation formula.
    expected = math.hypot(*(candidate - observations).tolist()) / math.hypot(*observations.tolist())
    assert math.isfinite(expected)
    actual = problem.reconstruction_error(candidate)
    assert math.isfinite(actual)
    assert actual == pytest.approx(expected, rel=1e-14)


def test_norm_underflow_with_unrepresentable_relative_error_returns_infinity():
    import math

    observations = torch.tensor([1e-200, 2e-200, -1e-200], dtype=torch.float64)
    problem = TensorCompressiveSensing(
        torch.eye(3, dtype=torch.float64),
        torch.ones(3, dtype=torch.bool),
        observations,
        core_cfg=CompressiveSensingConfig(dtype=torch.float64),
    )
    assert math.isinf(
        problem.reconstruction_error(torch.tensor([1e200, 0.0, 0.0], dtype=torch.float64))
    )
