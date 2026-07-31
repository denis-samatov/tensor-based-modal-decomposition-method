#!/usr/bin/env python3
"""Run the public synthetic TBMD end-to-end pipeline.

The example exercises the current v2 API:

1. Tucker/HOSVD decomposition.
2. Time-insensitive modal-basis construction.
3. Tensor Tube QR sensor placement.
4. Sparse full-field reconstruction.

The synthetic data are intended as a deterministic software smoke test. They
do not reproduce the Brugge benchmark or the manuscript's reported metrics.
"""

from __future__ import annotations

import argparse
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from TBMD.config import CompressiveSensingConfig, DecompositionConfig, SensorPlacementConfig
from TBMD.core import TensorCompressiveSensing, TensorTubeQRDecomposition, TuckerDecomposer
from TBMD.core.modal_processor.modes import ModalProcessorConfig, ModalTensorProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TBMD synthetic complete pipeline")
    parser.add_argument(
        "--n-modes",
        type=int,
        default=20,
        help="Number of modes in the reconstruction dictionary",
    )
    parser.add_argument(
        "--n-sensors",
        type=int,
        default=12,
        help="Number of spatial-variable sensor locations",
    )
    parser.add_argument(
        "--solver",
        choices=("least_squares", "admm", "ista"),
        default="admm",
        help="Coefficient-recovery solver",
    )
    parser.add_argument("--spatial-points", type=int, default=200)
    parser.add_argument("--time-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", action="store_true", help="Save a diagnostic PNG")
    return parser.parse_args()


def create_synthetic_reservoir_data(
    I: int = 200,
    J: int = 3,
    T: int = 50,
    seed: int = 42,
) -> torch.Tensor:
    """Create deterministic reservoir-like fields for a software smoke test.

    The three variables are pressure, oil saturation, and temperature. The
    fields are analytic patterns with additive Gaussian noise; they are not
    derived from the Brugge benchmark.
    """
    if I < 2:
        raise ValueError("I must be at least 2")
    if J != 3:
        raise ValueError("This synthetic generator requires J=3 variables")
    if T < 2:
        raise ValueError("T must be at least 2")

    torch.manual_seed(seed)
    np.random.seed(seed)

    x = torch.linspace(0, 1, I)
    time = torch.linspace(0, 1, T)
    data = torch.zeros(I, J, T)

    pressure_base = 1.0 - 0.5 * x
    for ti, t in enumerate(time):
        decay = torch.exp(-0.5 * t)
        data[:, 0, ti] = pressure_base * decay + 0.1 * torch.sin(5 * x) * decay

    for ti, t in enumerate(time):
        wavefront = torch.sigmoid(10 * (x - 0.5 * t - 0.2))
        data[:, 1, ti] = wavefront + 0.05 * torch.sin(3 * x)

    temp_center = I // 2
    for ti, t in enumerate(time):
        spread = 0.1 + 0.3 * t
        data[:, 2, ti] = torch.exp(-((x - x[temp_center]) ** 2) / spread)

    data += 0.02 * torch.randn_like(data)

    for variable in range(J):
        variable_data = data[:, variable, :]
        data[:, variable, :] = (variable_data - variable_data.mean()) / (
            variable_data.std() + 1e-8
        )

    return data


def _validate_pipeline_parameters(
    data: torch.Tensor,
    n_modes: int,
    n_sensors: int,
    solver: str,
) -> None:
    if data.ndim != 3:
        raise ValueError(f"data must have shape (space, variables, time), got {data.shape}")
    if not torch.isfinite(data).all():
        raise ValueError("data contains NaN or infinite values")
    if n_modes < 1 or n_modes > min(data.shape[0], data.shape[-1]):
        raise ValueError(
            "n_modes must be between 1 and min(spatial_points, time_steps), "
            f"got {n_modes}"
        )
    if n_sensors < 1 or n_sensors > n_modes:
        raise ValueError(
            "n_sensors must be between 1 and n_modes because Tensor Tube QR "
            f"can select at most one sensor per modal tube; got {n_sensors}"
        )
    if solver not in {"least_squares", "admm", "ista"}:
        raise ValueError(f"Unsupported solver: {solver}")


def _recover_coefficients(
    modal_basis: torch.Tensor,
    sensor_mask: torch.Tensor,
    field: torch.Tensor,
    solver: str,
    max_iterations: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Recover modal coefficients from one sparse field observation."""
    sampled_basis = modal_basis[sensor_mask]
    sampled_values = field[sensor_mask]

    if solver == "least_squares":
        solution = torch.linalg.lstsq(sampled_basis, sampled_values.unsqueeze(1)).solution
        return solution.squeeze(1), {"iterations": 1, "converged": True}

    if solver == "ista":
        lipschitz = torch.linalg.matrix_norm(sampled_basis, ord=2).square().clamp_min(1e-8)
        step_size = 1.0 / lipschitz
        l1_weight = 1e-2
        coefficients = torch.zeros(modal_basis.shape[-1], dtype=modal_basis.dtype)

        for _ in range(max_iterations):
            gradient = sampled_basis.T @ (sampled_basis @ coefficients - sampled_values)
            update = coefficients - step_size * gradient
            threshold = step_size * l1_weight
            coefficients = torch.sign(update) * torch.clamp(torch.abs(update) - threshold, min=0)

        return coefficients, {"iterations": max_iterations, "converged": False}

    config = CompressiveSensingConfig(
        max_iter=max_iterations,
        tol=1e-4,
        epsilon_l1=1e-2,
        device="cpu",
        dtype=modal_basis.dtype,
    )
    reconstructor = TensorCompressiveSensing(
        modal_basis,
        sensor_mask,
        field,
        core_cfg=config,
    )
    coefficients, metrics = reconstructor.solve()
    return coefficients, {
        "iterations": metrics.iterations,
        "converged": metrics.converged,
        "primal_residual": metrics.primal_residual,
        "dual_residual": metrics.dual_residual,
    }


def run_tbmd_pipeline(
    data: torch.Tensor,
    n_modes: int,
    n_sensors: int,
    solver: str = "admm",
    *,
    max_iterations: int = 100,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run decomposition, modal processing, placement, and reconstruction."""
    _validate_pipeline_parameters(data, n_modes, n_sensors, solver)

    spatial_points, n_variables, time_steps = data.shape
    ranks = [n_modes, n_variables, n_modes]

    if verbose:
        print("\n" + "=" * 60)
        print("Step 1: Tucker decomposition")
        print("=" * 60)

    decomp_config = DecompositionConfig(
        ranks=ranks,
        backend="torch",
        random_state=42,
        verbose=verbose,
    )
    decomposer = TuckerDecomposer(data, config=decomp_config)
    decomposer.decompose()
    decomposer.reconstruct()

    core = decomposer.cores
    factors = decomposer.factors
    decomposition_error = float(decomposer.reconstruction_errors)
    retained_energy = max(0.0, 1.0 - decomposition_error**2)

    modal_processor = ModalTensorProcessor(
        ModalProcessorConfig(
            device="cpu",
            return_numpy=False,
            enable_progress_logging=False,
        )
    )
    modal_basis = modal_processor.process_single_subject(core, factors)

    compressed_size = core.numel() + sum(factor.numel() for factor in factors)
    decomposition_result = {
        "core": core,
        "factors": factors,
        "relative_error": decomposition_error,
        "retained_energy": retained_energy,
        "compressed_size": compressed_size,
    }

    if verbose:
        print(f"Core tensor: {tuple(core.shape)}")
        print(f"Factor matrices: {[tuple(factor.shape) for factor in factors]}")
        print(f"Modal basis: {tuple(modal_basis.shape)}")
        print(f"Relative decomposition error: {decomposition_error:.4f}")
        print(f"Retained energy estimate: {retained_energy:.2%}")

        print("\n" + "=" * 60)
        print("Step 2: Tensor Tube QR sensor placement")
        print("=" * 60)

    sensor_config = SensorPlacementConfig(
        n_sensors=n_sensors,
        random_state=42,
        backend="torch",
        verbose=verbose,
    )
    sensor_placer = TensorTubeQRDecomposition(modal_basis, config=sensor_config)
    sensor_mask_int, q_factor, r_factor = sensor_placer.factorize()
    sensor_mask = sensor_mask_int.bool()
    sensor_indices = torch.nonzero(sensor_mask, as_tuple=False)
    actual_sensors = int(sensor_mask.sum().item())
    factorization_valid, factorization_error, factorization_metrics = (
        sensor_placer.check_factorization(tol=1e-4)
    )

    if actual_sensors != n_sensors:
        raise RuntimeError(
            f"Tensor Tube QR placed {actual_sensors} of {n_sensors} requested sensors"
        )
    if not factorization_valid:
        raise RuntimeError(
            "Tensor Tube QR validation failed with relative factorization error "
            f"{factorization_error:.3e}"
        )

    sensor_result = {
        "mask": sensor_mask,
        "indices": sensor_indices,
        "q_factor": q_factor,
        "r_factor": r_factor,
        "requested": n_sensors,
        "actual": actual_sensors,
        "sampling_ratio": actual_sensors / (spatial_points * n_variables),
        "factorization_valid": factorization_valid,
        "factorization_error": factorization_error,
        "factorization_metrics": factorization_metrics,
    }

    if verbose:
        print(f"Placed sensors: {actual_sensors}")
        print(f"Sensor mask: {tuple(sensor_mask.shape)}")
        print(f"Sampling ratio: {sensor_result['sampling_ratio']:.2%}")
        print(f"QR relative factorization error: {factorization_error:.3e}")

        print("\n" + "=" * 60)
        print(f"Step 3: Sparse reconstruction ({solver})")
        print("=" * 60)

    reconstructed_data = torch.zeros_like(data)
    reconstruction_errors: list[float] = []
    solver_diagnostics: list[dict[str, Any]] = []

    for time_index in range(time_steps):
        field = data[:, :, time_index]
        coefficients, diagnostics = _recover_coefficients(
            modal_basis,
            sensor_mask,
            field,
            solver,
            max_iterations,
        )
        reconstructed = modal_basis @ coefficients
        reconstructed_data[:, :, time_index] = reconstructed
        relative_error = torch.linalg.vector_norm(field - reconstructed) / torch.linalg.vector_norm(
            field
        )
        reconstruction_errors.append(float(relative_error.item()))
        solver_diagnostics.append(diagnostics)

    reconstruction_result = {
        "data": reconstructed_data,
        "errors": reconstruction_errors,
        "mean_error": float(np.mean(reconstruction_errors)),
        "std_error": float(np.std(reconstruction_errors)),
        "solver": solver,
        "diagnostics": solver_diagnostics,
    }

    if verbose:
        print(f"Mean relative error: {reconstruction_result['mean_error']:.4f}")
        print(f"Standard deviation: {reconstruction_result['std_error']:.4f}")
        print(f"Minimum error: {min(reconstruction_errors):.4f}")
        print(f"Maximum error: {max(reconstruction_errors):.4f}")

        print("\n" + "=" * 60)
        print("Step 4: Size and sampling summary")
        print("=" * 60)
        original_size = data.numel()
        print(f"Original tensor: {original_size} elements")
        print(f"Tucker representation: {compressed_size} elements")
        print(f"Representation ratio: {original_size / compressed_size:.2f}x")
        print(
            f"Sensors: {actual_sensors} / {spatial_points * n_variables} "
            f"({sensor_result['sampling_ratio']:.2%})"
        )

    return {
        "decomposition": decomposition_result,
        "modal_basis": modal_basis,
        "sensor_mask": sensor_mask,
        "sensors": sensor_result,
        "reconstruction": reconstruction_result,
    }


def visualize_results(
    data: torch.Tensor,
    results: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    """Save a compact diagnostic plot and return its filename."""
    spatial_points, n_variables, time_steps = data.shape
    modal_basis = results["modal_basis"].detach().cpu()
    sensor_mask = results["sensor_mask"].detach().cpu()
    reconstruction = results["reconstruction"]
    reconstructed_data = reconstruction["data"].detach().cpu()

    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

    for mode_index in range(3):
        axis = fig.add_subplot(grid[0, mode_index])
        axis.plot(modal_basis[:, :, mode_index].mean(dim=1).numpy(), linewidth=2)
        axis.set_title(f"Modal basis vector {mode_index + 1}")
        axis.set_xlabel("Spatial point")
        axis.grid(True, alpha=0.3)

    axis = fig.add_subplot(grid[0, 3])
    axis.imshow(sensor_mask.numpy(), aspect="auto", cmap="Greys")
    axis.set_title(f"Sensor mask ({int(sensor_mask.sum())} locations)")
    axis.set_xlabel("Variable")
    axis.set_ylabel("Spatial point")

    sample_time = time_steps // 2
    for variable in range(n_variables):
        axis = fig.add_subplot(grid[1, variable])
        axis.plot(data[:, variable, sample_time].numpy(), label="Original", linewidth=2)
        axis.plot(
            reconstructed_data[:, variable, sample_time].numpy(),
            label="Reconstructed",
            linestyle="--",
            linewidth=2,
        )
        variable_sensors = torch.nonzero(sensor_mask[:, variable], as_tuple=False).flatten()
        axis.scatter(
            variable_sensors.numpy(),
            data[variable_sensors, variable, sample_time].numpy(),
            color="red",
            s=35,
            zorder=5,
        )
        axis.set_title(f"Variable {variable}, time {sample_time}")
        axis.legend()
        axis.grid(True, alpha=0.3)

    axis = fig.add_subplot(grid[1, 3])
    axis.plot(reconstruction["errors"], linewidth=2)
    axis.axhline(
        reconstruction["mean_error"],
        color="red",
        linestyle="--",
        label=f"Mean {reconstruction['mean_error']:.3f}",
    )
    axis.set_title("Relative reconstruction error")
    axis.set_xlabel("Time step")
    axis.legend()
    axis.grid(True, alpha=0.3)

    for column, (field, title) in enumerate(
        (
            (data[:, 0, :], "Original pressure-like field"),
            (reconstructed_data[:, 0, :], "Reconstructed pressure-like field"),
            (torch.abs(data[:, 0, :] - reconstructed_data[:, 0, :]), "Absolute error"),
        )
    ):
        axis = fig.add_subplot(grid[2, column])
        image = axis.imshow(field.numpy(), aspect="auto", cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("Time")
        axis.set_ylabel("Spatial point")
        plt.colorbar(image, ax=axis)

    axis = fig.add_subplot(grid[2, 3])
    axis.axis("off")
    axis.text(
        0.05,
        0.95,
        "Synthetic smoke test\n"
        f"Modes: {args.n_modes}\n"
        f"Sensors: {args.n_sensors}\n"
        f"Solver: {args.solver}\n"
        f"Mean error: {reconstruction['mean_error']:.4f}",
        va="top",
        fontsize=12,
    )

    fig.suptitle("TBMD synthetic end-to-end pipeline", fontsize=16)
    filename = (
        f"tbmd_complete_pipeline_{args.n_modes}modes_"
        f"{args.n_sensors}sensors_{args.solver}.png"
    )
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filename


def main() -> dict[str, Any]:
    args = parse_args()

    print("=" * 60)
    print("TBMD - Synthetic Complete Pipeline")
    print("=" * 60)
    print(f"Modes: {args.n_modes}")
    print(f"Sensors: {args.n_sensors}")
    print(f"Solver: {args.solver}")
    print(f"Seed: {args.seed}")

    print("\nGenerating synthetic reservoir-like data...")
    data = create_synthetic_reservoir_data(
        I=args.spatial_points,
        J=3,
        T=args.time_steps,
        seed=args.seed,
    )
    print(f"Data shape: {tuple(data.shape)}")

    results = run_tbmd_pipeline(
        data,
        args.n_modes,
        args.n_sensors,
        args.solver,
    )

    if args.visualize:
        filename = visualize_results(data, results, args)
        print(f"\nVisualization saved: {filename}")

    print("\n" + "=" * 60)
    print("TBMD synthetic complete pipeline completed successfully.")
    print("=" * 60)
    print("This smoke test does not reproduce manuscript benchmark metrics.")
    return results


if __name__ == "__main__":
    main()
