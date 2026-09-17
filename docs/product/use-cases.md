# Use cases

| Task | Starting point |
|---|---|
| Factor a tensor into a core and mode factors | [Tucker example](../../examples/basic/01_tucker_decomposition.py) |
| Select sparse spatial-variable measurements | [Sensor-placement example](../../examples/basic/02_sensor_placement.py) |
| Recover a field from selected measurements | [Reconstruction example](../../examples/basic/03_field_reconstruction.py) |
| Compose the complete synthetic workflow | [Complete example](../../examples/basic/04_complete_pipeline.py) |
| Reproduce revised reservoir benchmark claims | [Brugge study](../../studies/brugge_sparse_sensing/README.md) |
| Explore mesh-aware regularization | [Geometry examples](../../examples/geometry_aware/README.md) |

## Validation

From the installed checkout root:

```bash
MPLBACKEND=Agg python examples/basic/04_complete_pipeline.py --spatial-points 40 --time-steps 12 --n-modes 8 --n-sensors 6 --solver admm
```

This checks a generated-data demonstration. To adapt it to physical data, define the tensor axes,
units, normalization, split, sensor constraints and evaluation baseline explicitly.

[Overview](overview.md) · [Python API](../interfaces/python-api.md) · [Setup](../setup/local-development.md)
