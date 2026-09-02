# TBMD Examples

This directory contains runnable examples for the TBMD package. Run commands from the repository root after installing the package.

## Directory Overview

| Directory | Contents |
| --- | --- |
| `basic/` | Minimal decomposition, sensor placement, reconstruction, and complete-pipeline examples. |

| `geometry_aware/` | Examples for graph and mesh-aware workflows. |
| `advanced/` | Advanced and legacy workflows. |
| `applications/` | Dataset-specific scripts. |
| `experiments/` | Experimental visualization and validation scripts. |

## Basic Examples

```bash
python examples/basic/01_tucker_decomposition.py
python examples/basic/02_sensor_placement.py
python examples/basic/03_field_reconstruction.py
python examples/basic/04_complete_pipeline.py
```

The complete-pipeline example uses deterministic generated data and the current
public v2 API. It verifies decomposition, modal processing, Tensor Tube QR
placement, and sparse reconstruction without downloading a dataset. See
[`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) for the boundary between this
software check and the non-public Brugge experiment artifacts.

## Geometry-Aware Examples

```bash
python examples/geometry_aware/01_graph_based_tbmd.py
python examples/geometry_aware/02_geometry_aware_cs.py
python examples/geometry_aware/03_geometry_aware_decomposition.py
python examples/geometry_aware/04_geometry_utils.py
python examples/geometry_aware/05_test_components.py
python examples/geometry_aware/06_geometry_aware_run.py
```

## Dataset-Specific Examples

Dataset-specific forecasting experiments are maintained in the separate
`tbmd-forecasting` repository. This repository keeps only reusable TBMD examples
and deterministic synthetic smoke tests. Keep local datasets and generated
outputs out of version control.

## Additional Documentation

- [Quick start](../README.md)
- [Python API](../docs/interfaces/python-api.md)
- [Input and output tensors](../docs/interfaces/input-output-tensors.md)
