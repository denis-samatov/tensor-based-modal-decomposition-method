# Tensor-Based Modal Decomposition Method

[![CI](https://github.com/denis-samatov/tensor-based-modal-decomposition-method/actions/workflows/ci.yml/badge.svg)](https://github.com/denis-samatov/tensor-based-modal-decomposition-method/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2607.09687-b31b1b.svg)](https://arxiv.org/abs/2607.09687)

A Python research library for reduced-order modeling of spatiotemporal tensor data.

## Reproduce the public example

The repository includes a synthetic end-to-end example and software tests. The Brugge study of the revised manuscript is in [`studies/brugge_sparse_sensing`](studies/brugge_sparse_sensing/README.md): its tables, statistics and number macros can be verified from the committed result files, but figures and a full re-run needs the Brugge simulation outputs, which are not redistributed here (TNO data-use agreement). The arXiv:2607.09687 results of the earlier version are **not** reproducible and are superseded by that study. See the [reproducibility matrix](REPRODUCIBILITY.md#public-reproducibility-scope).

After installation, run:

```bash
python examples/basic/04_complete_pipeline.py --spatial-points 40 --time-steps 12 --n-modes 8 --n-sensors 6 --solver admm --visualize
```

This generates analytic fields with seed 42, selects six spatial-variable sensor locations, reconstructs the fields, and writes `tbmd_complete_pipeline_8modes_6sensors_admm.png`. Expected completion: `TBMD synthetic complete pipeline completed successfully.` This is a software demonstration on generated data, not a held-out Brugge evaluation.

![Synthetic TBMD: modal basis, selected sensors, reconstructed fields, and errors](docs/examples/synthetic-pipeline.png)

[Captured run and environment](docs/examples/synthetic-run.md). The modal basis and reconstruction in this smoke test use the same generated sequence; the displayed error is not a held-out generalization metric.

## What this project does
Tensor-Based Modal Decomposition Method (TBMD) compresses high-dimensional spatiotemporal data (such as computational fluid dynamics or reservoir-modeling datasets) into a compact modal representation. It uses these representations to select sensor placements with a QR-based method and reconstruct full fields from sparse measurements.

## Who this is for
- **ML/AI Engineers & Data Scientists**: For building and orchestrating modal decomposition pipelines.
- **Scientific Computing Researchers**: For experimenting with tensor decompositions (Tucker/HOSVD) and geometry-aware representations.
- **Developers**: For extending and integrating the core mathematical components into larger simulation workflows.

## Core capabilities
- **Tucker/HOSVD decomposition** for spatiotemporal tensor data.
- **Modal tensor processing** utilities for building reduced bases.
- **Tensor QR-based sensor placement** to find the most informative measurement locations.
- **Compressive sensing reconstruction** with ADMM-based solvers.
- **Geometry-aware variants** for decomposition, reconstruction, and sensor placement on irregular grids.

## Architecture at a glance
The library is composed of modular components built primarily on PyTorch.
The synthetic example uses a `(space, variable, time)` tensor. Tucker decomposition produces factors and a core; modal processing builds a basis; Tensor Tube QR selects spatial-variable locations; a coefficient solver reconstructs the field from those measurements. Manuscript inputs use a separate four-dimensional reservoir representation.

For more details, see the [Architecture Overview](docs/architecture/overview.md).

Dataset-specific one-step forecasting, including the local OpenFOAM URANS workflow, is maintained
in the separate [`tbmd-forecasting`](https://github.com/denis-samatov/tbmd-forecasting)
repository. This repository remains the reusable TBMD library and synthetic-example source.

## Quick start
1. Clone the repository:
```bash
git clone https://github.com/denis-samatov/tensor-based-modal-decomposition-method.git
cd tensor-based-modal-decomposition-method
```
2. Install as an editable package with development dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```
3. Run a basic decomposition script:
```bash
python examples/basic/01_tucker_decomposition.py
```

## Configuration
Configuration is managed strictly through Python dataclasses located in `TBMD.config`, rather than environment variables or external files. See the [Configuration Guide](docs/setup/configuration.md) for details.

## Testing
To verify the installation and run unit tests:
```bash
pytest
```
To run targeted repository hygiene and architecture checks:
```bash
pytest tests/audit -q
```
For more information, see the [Testing Guide](docs/development/testing.md).

## Benchmarks

Measured stage costs for the Brugge study (single thread, one leave-one-scenario-out fold) are in
[`studies/brugge_sparse_sensing/outputs/e6_cost.json`](studies/brugge_sparse_sensing/outputs/e6_cost.json)
and Table 2 of the revised manuscript. Earlier timings produced by `measure_brugge_runtime.py` timed
QR placement on a random matrix and are not used.

### Map of documentation

- **Product & Concepts**: [`docs/product/overview.md`](docs/product/overview.md)
- **Architecture**: [`docs/architecture/overview.md`](docs/architecture/overview.md)
- **Mathematical & Research Pipeline**: [`docs/research-system/reconstruction-pipeline.md`](docs/research-system/reconstruction-pipeline.md)
- **Interfaces & Python Usage**: [`docs/interfaces/python-api.md`](docs/interfaces/python-api.md)
- **Installation & Setup**: [`docs/setup/local-development.md`](docs/setup/local-development.md)
- **Running Experiments**: [`docs/operations/runbook.md`](docs/operations/runbook.md)
- **Contributing & Code Style**: [`docs/development/contribution-guide.md`](docs/development/contribution-guide.md)
- **Operations & Runbooks**: [`docs/operations/runbook.md`](docs/operations/runbook.md)


## Known limitations
This project is an experimental research codebase. Claims regarding accuracy, performance, or "production-readiness" require explicit verification. Local datasets and generated artifacts must not be tracked in version control. See [Limitations](docs/product/limitations.md).

## Contributing
We welcome improvements! Please review the [Contribution Guidelines](CONTRIBUTING.md) before opening a Pull Request.

## License / ownership
MIT License. See `LICENSE`.

## Citation

This repository implements the method described in:

> D. Samatov, B. Merzlikin, and G. Shishaev, "Tensor-Based Modal Decomposition and
> Sparse Sensor Placement for the Brugge Field Simulation Model," arXiv:2607.09687, 2026.
> https://arxiv.org/abs/2607.09687

```bibtex
@article{samatov2026tbmd,
  title   = {Tensor-Based Modal Decomposition and Sparse Sensor Placement for the Brugge Field Simulation Model},
  author  = {Samatov, D. and Merzlikin, B. and Shishaev, G.},
  journal = {arXiv preprint arXiv:2607.09687},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.09687}
}
```

The numerical results of that preprint could not be reproduced (see the Brugge study below); the revised
manuscript *"Sparse-sensor reconstruction of reservoir states with tensor-based and matrix modal bases: a
reproducible benchmark on the Brugge model"* replaces them.

See [`CITATION.cff`](CITATION.cff) for citing this software directly.

## Reproducing the Brugge sparse-sensing study (revised manuscript)

All numbers, tables and figures of the revised manuscript *"Sparse-sensor reconstruction of
reservoir states with tensor-based and matrix modal bases: a reproducible benchmark on the Brugge
model"* are produced by the study package in
[`studies/brugge_sparse_sensing`](studies/brugge_sparse_sensing/README.md):

```bash
python -m pip install -r studies/brugge_sparse_sensing/requirements-lock.txt
python -m pip install -e .
cd studies/brugge_sparse_sensing
TBMD_DATA_DIR=/path/to/data ./run_all.sh 8
```

The two input files (`data_exp_4_.h5`, `all_wells_exp_4.json`) are identified by SHA-256 checksums in
the study README. They are simulations of the TNO Brugge benchmark model, whose data-use agreement does
not permit redistribution, so they are not in this repository; `./verify_outputs.sh` checks all tables,
statistics and number macros without them. The study also contains an audit
(`scripts/e0_*`) showing that the numerical results of the earlier Computers & Geosciences submission
(CAGEO-D-26-01439, arXiv:2607.09687) could not be reproduced with this library.

**Synthetic end-to-end smoke test (no external data):**
```bash
python examples/basic/04_complete_pipeline.py \
  --spatial-points 40 \
  --time-steps 12 \
  --n-modes 8 \
  --n-sensors 6 \
  --solver admm
```

This command generates its data in memory and exercises Tucker decomposition, modal processing,
Tensor Tube QR sensor placement, and sparse reconstruction.
