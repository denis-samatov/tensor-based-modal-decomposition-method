# Python API

The library exposes Python classes, not an HTTP service. The core API uses `decompose()`,
`reconstruct()`, `factorize()` and `solve()`. It does not provide a uniform scikit-learn-style
`fit/transform/inverse_transform` interface.

## Minimal decomposition and sparse recovery

Run after installing the package. The final reconstruction uses the same synthetic sequence that
built the dictionary; this is an API smoke example, not a held-out evaluation.

```python
import torch
from TBMD.config import CompressiveSensingConfig, DecompositionConfig, SensorPlacementConfig
from TBMD.core import TensorCompressiveSensing, TensorTubeQRDecomposition, TuckerDecomposer
from TBMD.core.modal_processor.modes import ModalProcessorConfig, ModalTensorProcessor

torch.manual_seed(42)
data = torch.randn(12, 2, 8)
decomposer = TuckerDecomposer(
    data,
    config=DecompositionConfig(ranks=[4, 2, 4], device="cpu", random_state=42, verbose=False),
)
decomposer.decompose()
decomposer.reconstruct()
assert decomposer.reconstructed_tensors.shape == data.shape

processor = ModalTensorProcessor(
    ModalProcessorConfig(return_numpy=False, enable_progress_logging=False)
)
basis = processor.process_single_subject(decomposer.cores, decomposer.factors)
placement, _, _ = TensorTubeQRDecomposition(
    basis,
    config=SensorPlacementConfig(n_sensors=4, device="cpu", random_state=42, verbose=False),
).factorize()
mask = placement.bool()
assert int(mask.sum()) == 4

field = data[:, :, 0]
solver = TensorCompressiveSensing(
    basis, mask, field,
    core_cfg=CompressiveSensingConfig(max_iter=100, device="cpu", dtype=torch.float32),
)
coefficients, metrics = solver.solve()
recovered = basis @ coefficients
assert recovered.shape == field.shape
assert torch.isfinite(recovered).all()
print(metrics.converged, metrics.iterations, metrics.primal_residual, metrics.dual_residual)
```

`decompose()` and `reconstruct()` store results on the decomposer and return `None`.
Accessing results before the required operation raises a state error. `solve()` returns a CPU
coefficient vector and metrics; callers using other devices must handle device alignment explicitly.

## Configuration and examples

General dataclasses are in [TBMD.config](../../src/TBMD/config/__init__.py).
Geometry-aware constructors have additional configuration contracts; see
[components](../architecture/components.md) and [geometry examples](../../examples/geometry_aware/README.md).
The [complete synthetic example](../../examples/basic/04_complete_pipeline.py) shows the composed workflow.
Forecasting-specific APIs belong to
[tbmd-forecasting](https://github.com/denis-samatov/tbmd-forecasting).

## Validation

```bash
MPLBACKEND=Agg python -m pytest tests/unit/test_decomposition.py tests/unit/test_sensor_placement.py tests/unit/test_reconstruction.py -q
```
