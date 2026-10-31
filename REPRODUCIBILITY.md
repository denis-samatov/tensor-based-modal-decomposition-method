# Reproducibility guide

This guide separates the software checks available in a clean public clone from
the Brugge numerical experiments reported in the manuscript. The distinction is
intentional: passing the synthetic checks verifies the public TBMD
implementation, but it does not reproduce the manuscript's Brugge metrics or
figures.

See [the 2026-10-31 numerical qualification](docs/numerical_qualification.md) for
independent analytic checks and a failed exact table replay in the maintained
software profile. Passing software tests does not qualify scientific replay; use
the historical study profile for its separate acceptance gate.

## Public reproducibility scope

| Item | Public status | Verification |
| --- | --- | --- |
| Package installation and imports | Reproducible from a clean clone | Install the package and run `pytest tests/audit -q`. |
| Tucker decomposition | Reproducible with generated data | Run `examples/basic/01_tucker_decomposition.py`. |
| Decomposition → modal basis → Tensor Tube QR → sparse reconstruction | Reproducible with generated data | Run `examples/basic/04_complete_pipeline.py` as shown below. |
| Twelve core Brugge tables, statistics and number macros | Statistics-only replay from committed rows | Use the isolated Python 3.13 profile below; byte comparison of a fixed twelve-file set. |
| Broader revised-manuscript stored-result gate, including E8/E9 | Requires the historical full environment | Run `verify_environment.py`, then `studies/brugge_sparse_sensing/verify_outputs.sh`. |
| Revised-manuscript Brugge experiments from simulation outputs | Requires the two input files, which are not redistributed | Obtain the files (study README, section 2), then run `studies/brugge_sparse_sensing/run_all.sh`. |
| Brugge results of the earlier version (arXiv:2607.09687) | Not reproducible | Audited in `studies/brugge_sparse_sensing/scripts/e0_*` and Supplementary Section S2 of the revised manuscript. |

The repository is currently distributed through
[GitHub](https://github.com/denis-samatov/tensor-based-modal-decomposition-method)
under the MIT License. Release v2.1.0, which contains the Brugge study package, is archived on
Zenodo: https://doi.org/10.5281/zenodo.22814377. The archive contains code and result files only,
not the Brugge simulation inputs.

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
python -m compileall src tests examples studies/brugge_sparse_sensing
pytest tests/audit -q
pytest tests/unit -q
```

The audit suite includes a regression test that executes the synthetic
decomposition, modal-processing, QR-placement, and reconstruction path against
the current public API.

## Statistics-only replay: twelve core artifacts

This separate profile uses Python 3.13, NumPy 2.5.2, pandas 3.0.5 and SciPy 1.18.1.
The generated hash lock also fixes their transitive dependencies. It reruns the
unchanged `make_tables.py`, `make_numbers_tex.py` and `make_latex_tables.py` over
fifteen explicitly selected committed input files. It never copies a canonical
expected output into the scratch directory. All twelve required outputs must
exist and match byte for byte; missing files fail the gate. The committed
`SHA256SUMS` is checked before generation, and source hashes before/after must agree.

Run from the repository root, using new environment/output paths outside the
checkout (the output path must not already exist):

```bash
python3.13 -m venv /tmp/tbmd-statistics-env
/tmp/tbmd-statistics-env/bin/python -m pip install --require-hashes \
  -r studies/brugge_sparse_sensing/requirements-statistics.txt
/tmp/tbmd-statistics-env/bin/python studies/brugge_sparse_sensing/scripts/verify_statistics.py \
  --output-dir /tmp/tbmd-statistics-replay
```

The runner records interpreter/platform/package versions, source/output hashes,
generator exit codes and exact comparisons in `receipt.json`; `logs/` and scratch
outputs remain available on failure. A successful result prints `12/12 exact
artifacts`. `tests/audit/test_statistics_replay.py` exercises missing/mutated
outputs, runtime drift, subprocess failures and source/output isolation. The
independent `statistics-only-replay` CI job runs this same hash-locked profile on
Ubuntu 24.04; consult its actual workflow run before claiming hosted acceptance.

The lock is generated, not edited manually:

```bash
uv pip compile studies/brugge_sparse_sensing/requirements-statistics.in \
  --python-version 3.13 --universal --generate-hashes \
  --output-file studies/brugge_sparse_sensing/requirements-statistics.txt
```

This is **statistics-only stored-row rendering**, not the historical Python 3.12
study environment or Python 3.13 library-support qualification. It excludes E8,
optimization/E9, figures, manuscript claim checks and raw-input experiments.
Committed rows, their original timings and provenance are reused; no model is
refitted, parser/tolerance/statistic altered or scientific claim newly validated.
The isolated older numerical pins are for trusted committed data, not a general
deployment recommendation. The failed modern-profile 6/12 replay remains recorded
in [numerical qualification](docs/numerical_qualification.md).

## Brugge data and manuscript-result boundary

The revised manuscript's Brugge study, its orchestration (`run_all.sh`), configuration, seeds, run
logs, result files and checksums are in
[`studies/brugge_sparse_sensing`](studies/brugge_sparse_sensing/README.md). Its inputs are supplied,
transformed t-Navigator state exports for the Brugge benchmark. The archive does not establish the
t-Navigator version, exporter, vertical reduction or physical pressure unit. TNO's data-use notice
reserves all rights and permits non-commercial use; unrestricted public redistribution of the
transformed simulation exports has not been established, so the inputs are excluded from Git by the
repository's data-isolation policy and there is no GitHub or Zenodo download for them. The
corresponding author can provide the transformed HDF5/JSON inputs upon reasonable request, subject
to applicable data-use and access conditions. Local authorised copies belong under ignored paths
such as `data/brugge/`; the study verifies them by SHA-256 before running.

Without the inputs, `studies/brugge_sparse_sensing/verify_outputs.sh` rebuilds every table,
statistic and number macro from the committed result files and checks them byte for byte; the
figures also plot simulated fields and need the inputs.

The third- versus fourth-order property-mode study has its own deterministic entry point:
`studies/brugge_sparse_sensing/run_e9_property_mode.sh --reproduce`. It validates the environment
and inputs, independently refits the ten frozen outer-fold configurations into a separate output
directory, compares every scientific value with the canonical 2,080-row E9 registry, and runs the
manuscript claim checker. `--full` additionally rebuilds the nested train-only search, cross-property
diagnostics, single-pass outer results, frozen-final LASSO sensitivity, Figs. 15--19, tables and E9
macros. Protocol and preregistration SHA-256 values are stored in every fold selection file.

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

## Canonical manuscript

The current [paper source and build](docs/paper/README.md) reside in this checkout. TeX directly
references canonical study results; no external author-machine file is needed to verify stored
results or build the manuscript. Restricted simulation inputs are still needed for a full rerun.
The simulator/export placeholder remains an author-only submission blocker. Historical notebooks
are preserved in [scientific provenance](docs/paper/audit/README.md), not the active pipeline.
