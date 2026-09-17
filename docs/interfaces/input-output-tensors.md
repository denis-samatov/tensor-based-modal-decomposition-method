# Tensor contracts

Axis order is experiment-specific. The synthetic complete example uses `(space, variable, time)`;
the Brugge study defines its reservoir representation separately. Do not silently transpose a
physical dataset to match an assumed universal layout.

| Stage | Input | Output |
|---|---|---|
| Tucker decomposition | Tensor with one rank per mode, or a supported tensor collection | Core and factor matrices; single-input factor `i` has rows matching input mode `i` |
| Dense Tucker reconstruction | Stored core and factors | Input-shaped tensor in `reconstructed_tensors` |
| Modal processing | Compatible core and factor matrices | Dictionary `A`, with coefficients on its final axis |
| Tensor Tube QR | Modal dictionary `A` | Binary `P` of shape `A.shape[:-1]`, plus `Q`, `R` |
| Sparse coefficient solve | `A`, placement mask `P`, field-shaped `Y` | CPU coefficient vector of length `A.shape[-1]`, plus metrics |
| Field expansion | Dictionary and compatible coefficient vector | `A @ coefficients`, shape `A.shape[:-1]` |

The solver validates matching `P`/`Y` shapes and rejects an empty sensor mask. A sensor-only vector
is not the `Y` constructor contract. Physical units, missing-data policy and preprocessing must be
recorded by the caller; the library does not infer them from tensor dimensions.

## Validation

Execute the complete [Python API example](python-api.md) and run:

```bash
MPLBACKEND=Agg python -m pytest tests/unit/test_decomposition.py tests/unit/test_reconstruction.py -q
```

[Data flow](../architecture/data-flow.md) · [Brugge study](../../studies/brugge_sparse_sensing/README.md)
