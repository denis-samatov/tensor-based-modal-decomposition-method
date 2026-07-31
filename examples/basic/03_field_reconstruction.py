#!/usr/bin/env python3
"""Reconstruct synthetic fields from Tensor Tube QR sensor measurements."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from TBMD.config import CompressiveSensingConfig, DecompositionConfig, SensorPlacementConfig
from TBMD.core import TensorCompressiveSensing, TensorTubeQRDecomposition, TuckerDecomposer
from TBMD.core.modal_processor.modes import ModalProcessorConfig, ModalTensorProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visualize", action="store_true", help="Save a diagnostic PNG")
    return parser.parse_args()


def create_dynamic_fields(
    spatial_points: int = 80,
    variables: int = 3,
    time_steps: int = 24,
    seed: int = 42,
) -> torch.Tensor:
    """Create deterministic propagating-wave fields."""
    torch.manual_seed(seed)
    x = torch.linspace(0, 2 * torch.pi, spatial_points)
    time = torch.linspace(0, 4 * torch.pi, time_steps)
    data = torch.zeros(spatial_points, variables, time_steps)

    for variable in range(variables):
        for time_index, phase in enumerate(time):
            data[:, variable, time_index] = torch.sin(
                (variable + 1) * x + 0.5 * phase
            ) + 0.25 * torch.cos(x - phase)

    return data + 0.04 * torch.randn_like(data)


def build_modal_basis(data: torch.Tensor, n_modes: int) -> torch.Tensor:
    """Create the time-insensitive modal dictionary used by QR and CS."""
    decomposer = TuckerDecomposer(
        data,
        config=DecompositionConfig(
            ranks=[n_modes, data.shape[1], n_modes],
            random_state=42,
            verbose=False,
        ),
    )
    decomposer.decompose()
    processor = ModalTensorProcessor(
        ModalProcessorConfig(return_numpy=False, enable_progress_logging=False)
    )
    return processor.process_single_subject(decomposer.cores, decomposer.factors)


def place_sensors(modal_basis: torch.Tensor, n_sensors: int) -> torch.Tensor:
    """Return a boolean sensor mask."""
    placer = TensorTubeQRDecomposition(
        modal_basis,
        config=SensorPlacementConfig(
            n_sensors=n_sensors,
            random_state=42,
            verbose=False,
        ),
    )
    placement, _, _ = placer.factorize()
    mask = placement.bool()
    if int(mask.sum()) != n_sensors:
        raise RuntimeError(f"Placed {int(mask.sum())} of {n_sensors} requested sensors")
    return mask


def recover_coefficients(
    modal_basis: torch.Tensor,
    sensor_mask: torch.Tensor,
    field: torch.Tensor,
    method: str,
) -> tuple[torch.Tensor, int]:
    """Recover modal coefficients with a selected current solver path."""
    sampled_basis = modal_basis[sensor_mask]
    sampled_values = field[sensor_mask]

    if method == "least_squares":
        coefficients = torch.linalg.lstsq(
            sampled_basis,
            sampled_values.unsqueeze(1),
        ).solution.squeeze(1)
        return coefficients, 1

    if method == "ista":
        lipschitz = torch.linalg.matrix_norm(sampled_basis, ord=2).square().clamp_min(1e-8)
        step = 1.0 / lipschitz
        coefficients = torch.zeros(modal_basis.shape[-1], dtype=modal_basis.dtype)
        for _ in range(100):
            gradient = sampled_basis.T @ (sampled_basis @ coefficients - sampled_values)
            update = coefficients - step * gradient
            coefficients = torch.sign(update) * torch.clamp(
                torch.abs(update) - step * 1e-2,
                min=0,
            )
        return coefficients, 100

    reconstructor = TensorCompressiveSensing(
        modal_basis,
        sensor_mask,
        field,
        core_cfg=CompressiveSensingConfig(
            max_iter=100,
            tol=1e-4,
            epsilon_l1=1e-2,
            device="cpu",
            dtype=modal_basis.dtype,
        ),
    )
    coefficients, metrics = reconstructor.solve()
    return coefficients, metrics.iterations


def reconstruct_field(
    modal_basis: torch.Tensor,
    sensor_mask: torch.Tensor,
    field: torch.Tensor,
    method: str,
) -> tuple[torch.Tensor, float, int]:
    coefficients, iterations = recover_coefficients(modal_basis, sensor_mask, field, method)
    reconstructed = modal_basis @ coefficients
    error = torch.linalg.vector_norm(field - reconstructed) / torch.linalg.vector_norm(field)
    return reconstructed, float(error.item()), iterations


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("TBMD - Sparse Field Reconstruction")
    print("=" * 60)

    data = create_dynamic_fields()
    modal_basis = build_modal_basis(data, n_modes=12)
    sensor_mask = place_sensors(modal_basis, n_sensors=10)
    test_field = data[:, :, data.shape[-1] // 2]

    method_results: dict[str, tuple[torch.Tensor, float, int]] = {}
    for method in ("least_squares", "admm", "ista"):
        reconstructed, error, iterations = reconstruct_field(
            modal_basis,
            sensor_mask,
            test_field,
            method,
        )
        method_results[method] = (reconstructed, error, iterations)
        print(f"{method}: relative error={error:.4f}, iterations={iterations}")

    sequence_errors: list[float] = []
    reconstructed_sequence = torch.zeros_like(data)
    for time_index in range(data.shape[-1]):
        reconstructed, error, _ = reconstruct_field(
            modal_basis,
            sensor_mask,
            data[:, :, time_index],
            "admm",
        )
        reconstructed_sequence[:, :, time_index] = reconstructed
        sequence_errors.append(error)

    print(
        "ADMM sequence error: "
        f"{np.mean(sequence_errors):.4f} +/- {np.std(sequence_errors):.4f}"
    )

    if args.visualize:
        figure, axes = plt.subplots(1, 4, figsize=(18, 4))
        axes[0].imshow(test_field.numpy(), aspect="auto", cmap="viridis")
        axes[0].set_title("Original")
        for axis, method in zip(axes[1:], ("least_squares", "admm", "ista")):
            reconstruction, error, _ = method_results[method]
            axis.imshow(reconstruction.numpy(), aspect="auto", cmap="viridis")
            axis.set_title(f"{method} ({error:.3f})")
        for axis in axes:
            axis.set_xlabel("Variable")
            axis.set_ylabel("Spatial point")
        figure.tight_layout()
        filename = "field_reconstruction_results.png"
        figure.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(f"Visualization saved: {filename}")

    print("Field reconstruction example completed successfully.")


if __name__ == "__main__":
    main()
