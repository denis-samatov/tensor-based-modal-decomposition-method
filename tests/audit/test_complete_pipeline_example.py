"""Regression coverage for the public synthetic end-to-end example."""

import importlib.util
from pathlib import Path

import torch


def _load_complete_pipeline_module():
    example_path = (
        Path(__file__).resolve().parents[2] / "examples" / "basic" / "04_complete_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location("tbmd_complete_pipeline_example", example_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_pipeline_runs_with_current_public_api():
    """Exercise decomposition, QR sensor placement, and sparse reconstruction."""
    example = _load_complete_pipeline_module()
    data = example.create_synthetic_reservoir_data(I=16, J=3, T=10, seed=7)

    results = example.run_tbmd_pipeline(
        data,
        n_modes=5,
        n_sensors=4,
        solver="least_squares",
        verbose=False,
    )

    assert results["modal_basis"].shape == (16, 3, 5)
    assert results["sensor_mask"].dtype == torch.bool
    assert int(results["sensor_mask"].sum().item()) == 4
    assert results["sensors"]["factorization_valid"] is True
    assert results["reconstruction"]["data"].shape == data.shape
    assert torch.isfinite(results["reconstruction"]["data"]).all()
    assert results["reconstruction"]["mean_error"] >= 0.0
