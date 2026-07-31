# Reproducibility guide

This guide separates the software checks available in a clean public clone from
the Brugge numerical experiments reported in the manuscript. The distinction is
intentional: passing the synthetic checks verifies the public TBMD
implementation, but it does not reproduce the manuscript's Brugge metrics or
figures.

## Public reproducibility scope

| Item | Public status | Verification |
| --- | --- | --- |
| Package installation and imports | Reproducible from a clean clone | Install the package and run `pytest tests/audit -q`. |
| Tucker decomposition | Reproducible with generated data | Run `examples/basic/01_tucker_decomposition.py`. |
| Decomposition → modal basis → Tensor Tube QR → sparse reconstruction | Reproducible with generated data | Run `examples/basic/04_complete_pipeline.py` as shown below. |
| Manuscript Brugge tables and field figures | Not reproducible from this repository alone | The processed tensors, simulator outputs, exact experiment orchestration, and complete run metadata are not distributed here. |

The repository is currently distributed through
[GitHub](https://github.com/denis-samatov/tensor_based_modal_decomposition_method)
under the MIT License. This guide does not claim a Zenodo deposit or DOI.

## Environment

The package supports Python 3.10–3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Record the exact source revision and installed environment with:

```bash
git rev-parse HEAD
python --version
python -m pip freeze
```

The public smoke tests run on CPU and do not require a GPU.

## Deterministic software checks

### 1. Basic Tucker decomposition

```bash
python examples/basic/01_tucker_decomposition.py
```

Expected result: the script prints approximation errors for several Tucker
ranks and finishes without an exception.

### 2. Synthetic end-to-end TBMD–QR–TBCS pipeline

Run a compact deterministic check:

```bash
python examples/basic/04_complete_pipeline.py \
  --spatial-points 40 \
  --time-steps 12 \
  --n-modes 8 \
  --n-sensors 6 \
  --solver admm
```

Expected result: the script prints the Tucker core and factor shapes, a
`(40, 3, 8)` modal basis, six selected sensor locations, reconstruction-error
statistics, and the final message:

```text
TBMD synthetic complete pipeline completed successfully.
```

The example uses a fixed random seed and analytic reservoir-like fields. It
does not load external data and does not write output unless `--visualize` is
provided.

Alternative coefficient recovery paths can be checked with:

```bash
python examples/basic/04_complete_pipeline.py \
  --spatial-points 40 \
  --time-steps 12 \
  --n-modes 8 \
  --n-sensors 6 \
  --solver least_squares
```

For Tensor Tube QR, `n_sensors` must not exceed `n_modes`.

### 3. Repository tests

```bash
python -m compileall src tests examples scripts
pytest tests/audit -q
pytest tests/unit -q
```

The audit suite includes a regression test that executes the synthetic
decomposition, modal-processing, QR-placement, and reconstruction path against
the current public API.

## Brugge data and manuscript-result boundary

The manuscript uses third-party Brugge benchmark inputs and locally generated
reservoir-simulator outputs and processed tensors. Those locally generated
artifacts are excluded from Git by the repository's data-isolation policy and
are not available at a verified public archive referenced by this guide.

There is therefore no valid GitHub or Zenodo download instruction for the exact
manuscript data package. Local authorized copies, when available to the
research team, belong under ignored paths such as `data/brugge/`; the presence
of local `.h5` files is not evidence that they are publicly distributed.

The public repository also does not contain the exact experiment orchestration
and run metadata required to map commands to the manuscript's Brugge tables and
figures. Accordingly, this guide makes no claims about output filenames,
runtimes, or one-command regeneration of those manuscript artifacts.

## Troubleshooting

- **Import error after installation**: confirm that the active interpreter is
  the virtual environment's Python and rerun `python -m pip install -e ".[dev]"`.
- **Matplotlib cache warning on a read-only or headless system**: set
  `MPLBACKEND=Agg` and set `MPLCONFIGDIR` to a writable temporary directory.
- **Unexpected QR sensor count**: use `1 <= n_sensors <= n_modes`; the modal
  tube dimension bounds the number of effective QR pivots.
- **Oversubscribed numerical libraries in CI**: set `OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, and `VECLIB_MAXIMUM_THREADS=1`.

## What the synthetic check establishes

The deterministic example establishes API compatibility and dimensional
continuity across Tucker decomposition, modal processing, Tensor Tube QR
placement, and sparse reconstruction. It does not validate the manuscript's
Brugge split, well geometry, reported metrics, uncertainty summaries, or
archived figures.
