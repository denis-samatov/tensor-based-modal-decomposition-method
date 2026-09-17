# Data flow

The synthetic complete example uses `(space, variable, time)`. Other experiments must explicitly
record their axis order; the library does not infer physical units or dataset meaning.

1. Construct a tensor and define ranks for its modes.
2. Call `TuckerDecomposer.decompose()` to obtain the core and factors.
3. Call `ModalTensorProcessor.process_single_subject()` to build a modal dictionary `A`.
4. Call `TensorTubeQRDecomposition.factorize()` and convert returned `P` to a boolean mask.
5. Provide `A`, the mask and a field-shaped measurement array `Y` to `TensorCompressiveSensing`.
6. Call `solve()` to recover coefficients `x`; compute the field as `A @ x` on a compatible device.

`P` and `Y` must have shape `A.shape[:-1]`. The solver samples only entries selected by `P`.
This differs from passing a vector containing only sensor values.
`TuckerDecomposer.reconstruct()` reconstructs the decomposed input; it is separate from sparse
measurement recovery.

## Validation

```bash
MPLBACKEND=Agg python examples/basic/04_complete_pipeline.py --spatial-points 40 --time-steps 12 --n-modes 8 --n-sensors 6 --solver admm
```

This is an in-sequence synthetic software demonstration, not held-out Brugge or URANS qualification.

[Tensor contracts](../interfaces/input-output-tensors.md) · [Python API](../interfaces/python-api.md)
