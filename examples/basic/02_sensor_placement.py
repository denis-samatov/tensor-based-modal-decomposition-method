#!/usr/bin/env python3
"""Demonstrate Tensor Tube QR placement with the current TBMD v2 API."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch

from TBMD.config import DecompositionConfig, SensorPlacementConfig
from TBMD.core import TensorTubeQRDecomposition, TuckerDecomposer
from TBMD.core.modal_processor.modes import ModalProcessorConfig, ModalTensorProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visualize", action="store_true", help="Save a diagnostic PNG")
    return parser.parse_args()


def create_synthetic_data(
    spatial_points: int = 80,
    variables: int = 3,
    time_steps: int = 24,
    seed: int = 42,
) -> torch.Tensor:
    """Create deterministic multi-variable wave fields."""
    torch.manual_seed(seed)
    x = torch.linspace(0, 2 * torch.pi, spatial_points)
    time = torch.linspace(0, 3 * torch.pi, time_steps)
    data = torch.zeros(spatial_points, variables, time_steps)

    for variable in range(variables):
        for time_index, phase in enumerate(time):
            data[:, variable, time_index] = torch.sin(
                (variable + 1) * x + 0.4 * phase
            ) + 0.35 * torch.cos(0.5 * x - phase)

    return data + 0.03 * torch.randn_like(data)


def build_modal_basis(data: torch.Tensor, n_modes: int) -> tuple[torch.Tensor, float]:
    """Decompose data and construct the time-insensitive modal tensor."""
    ranks = [n_modes, data.shape[1], n_modes]
    decomposer = TuckerDecomposer(
        data,
        config=DecompositionConfig(ranks=ranks, random_state=42, verbose=False),
    )
    decomposer.decompose()
    decomposer.reconstruct()

    processor = ModalTensorProcessor(
        ModalProcessorConfig(return_numpy=False, enable_progress_logging=False)
    )
    modal_basis = processor.process_single_subject(decomposer.cores, decomposer.factors)
    return modal_basis, float(decomposer.reconstruction_errors)


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("TBMD - Tensor Tube QR Sensor Placement")
    print("=" * 60)

    data = create_synthetic_data()
    n_modes = 12
    modal_basis, decomposition_error = build_modal_basis(data, n_modes)
    print(f"Data shape: {tuple(data.shape)}")
    print(f"Modal basis shape: {tuple(modal_basis.shape)}")
    print(f"Relative decomposition error: {decomposition_error:.4f}")

    placements: dict[int, torch.Tensor] = {}
    factorization_errors: dict[int, float] = {}
    for n_sensors in (4, 8, 12):
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
        _, factorization_error, _ = placer.check_factorization(tol=1e-4)
        placements[n_sensors] = mask
        factorization_errors[n_sensors] = factorization_error

        sampled_basis = modal_basis[mask]
        condition_number = float(torch.linalg.cond(sampled_basis).item())
        print(
            f"Sensors={int(mask.sum())}: sampling={mask.float().mean():.2%}, "
            f"condition={condition_number:.2e}, QR error={factorization_error:.2e}"
        )

    selected_count = 8
    selected_mask = placements[selected_count]
    test_field = data[:, :, data.shape[-1] // 2]
    sampled_basis = modal_basis[selected_mask]
    sampled_values = test_field[selected_mask]
    coefficients = torch.linalg.lstsq(
        sampled_basis,
        sampled_values.unsqueeze(1),
    ).solution.squeeze(1)
    reconstructed = modal_basis @ coefficients
    relative_error = torch.linalg.vector_norm(test_field - reconstructed) / torch.linalg.vector_norm(
        test_field
    )
    print(f"Least-squares reconstruction error: {relative_error.item():.4f}")

    if args.visualize:
        figure, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].imshow(selected_mask.numpy(), aspect="auto", cmap="Greys")
        axes[0].set_title(f"Sensor mask ({selected_count})")
        axes[1].imshow(test_field.numpy(), aspect="auto", cmap="viridis")
        axes[1].set_title("Original field")
        axes[2].imshow(reconstructed.numpy(), aspect="auto", cmap="viridis")
        axes[2].set_title(f"Reconstructed ({relative_error.item():.3f})")
        for axis in axes:
            axis.set_xlabel("Variable")
            axis.set_ylabel("Spatial point")
        figure.tight_layout()
        filename = "sensor_placement_results.png"
        figure.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(f"Visualization saved: {filename}")

    print("Sensor placement example completed successfully.")


if __name__ == "__main__":
    main()
