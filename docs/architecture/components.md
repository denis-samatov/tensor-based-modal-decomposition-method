# Components

Use the public classes exposed by [TBMD.core](../../src/TBMD/core/__init__.py).

| Component | Interface | Result |
|---|---|---|
| `TuckerDecomposer` | `decompose()`, then `reconstruct()` | `cores`, `factors`, `reconstructed_tensors`, `reconstruction_errors` |
| `ModalTensorProcessor` | `process_single_subject(core, factors)` | Time-insensitive modal dictionary; final axis indexes coefficients |
| `BatchModalProcessor` | `process_multiple_subjects(cores, factors)` | Modal dictionaries for a collection of decompositions |
| `ModalTensorStacker` | `stack_modal_tensors(...)` | Combined modal dictionaries; inspect configuration and source before stacking |
| `TensorTubeQRDecomposition` | `factorize()` | Binary placement tensor `P` and factorization diagnostics `Q`, `R` |
| `TensorCompressiveSensing` | `solve()` | Coefficient vector and `CompressiveSensingMetrics` |

The modal-processing classes are in [core/modal_processor/modes.py](../../src/TBMD/core/modal_processor/modes.py).
The [Python API](../interfaces/python-api.md) provides a complete minimal example.

## Geometry extensions

`GeometryAwareTuckerDecomposer`, `GeometryAwareTensorQR` and `GeometryAwareTensorCS` use mesh/graph
information for their respective stages. Mesh-node ordering must match the spatial tensor mode.
Geometry-aware decomposition takes `tensor`, `mesh`, `geo_config` and `ranks`; its
`GeometryAwareConfig` is defined in [geometry_aware.py](../../src/TBMD/core/decomposition/geometry_aware.py),
not the identically named general configuration namespace.

See [geometry examples](../../examples/geometry_aware/README.md) for the actual constructors.
Geometry regularization does not establish physical fidelity by itself.

## Validation

```bash
MPLBACKEND=Agg python -m pytest tests/unit/test_decomposition.py tests/unit/test_geometry.py -q
```

[Data flow](data-flow.md) · [Python API](../interfaces/python-api.md)
