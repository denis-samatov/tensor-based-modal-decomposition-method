# Sparse reconstruction workflow

The [synthetic complete example](../../examples/basic/04_complete_pipeline.py) composes:

1. `TuckerDecomposer`: factor the tensor and retain a reduced representation.
2. `ModalTensorProcessor`: form the time-insensitive dictionary.
3. `TensorTubeQRDecomposition`: select entries of the spatial-variable field.
4. `TensorCompressiveSensing`: recover coefficients from measurements at the selected entries.
5. Dictionary expansion: compute `basis @ coefficients` to recover the field.

The public solver consumes a field-shaped `Y` plus a mask, not only a sensor-value vector.
The caller supplies the representation and measurement semantics. Any held-out protocol must fit
its representation and preprocessing on training data only and freeze the sensor design as required
by the experiment. The synthetic example itself does not establish that held-out protocol.

The [Brugge study](../../studies/brugge_sparse_sensing/README.md) defines its own benchmark, splits,
noise treatment and matrix/tensor baseline comparison. Do not substitute the synthetic ADMM example
for that numerical evaluation.

## Validation

```bash
MPLBACKEND=Agg python -m pytest tests/unit/test_reconstruction.py -q
```

[Python API](../interfaces/python-api.md) · [Experiment orchestration](experiment-orchestration.md)
