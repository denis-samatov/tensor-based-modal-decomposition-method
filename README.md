# Tensor-Based Modal Decomposition Method

A Python research library for Tucker/HOSVD decomposition, tensor QR sensor placement and
sparse reconstruction of spatiotemporal fields. The current reservoir benchmark lives in
[`studies/brugge_sparse_sensing`](studies/brugge_sparse_sensing/README.md).
Physical URANS forecasting is maintained in the separate
[tbmd-forecasting repository](https://github.com/denis-samatov/tbmd-forecasting).

## Paper

*Nested optimization of tensor-based modal decomposition for sparse reservoir-state reconstruction: accuracy–compression regimes on Brugge* — D. Samatov, B. Merzlikin, G. Shishaev.

The [canonical manuscript and supplement](docs/paper/README.md) are in this checkout.
The revised manuscript supersedes the unreproduced quantitative results of
[arXiv:2607.09687](https://arxiv.org/abs/2607.09687). Submission remains pending archival publication of the reviewed revision, three transferred-portal checks; see the paper README.

## Installation

The library supports Python 3.10–3.12. The recorded study environment uses Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r studies/brugge_sparse_sensing/requirements-lock.txt
python -m pip install --no-deps -e .
```

For library development, use `python -m pip install -e ".[dev]"` instead of the study lock.

## Reproducibility

From the checkout root, verify committed results without restricted raw data:

```bash
studies/brugge_sparse_sensing/verify_outputs.sh
MPLBACKEND=Agg python -m pytest
```

The verifier checks checksums, rebuilds all tables/statistics/number macros in a disposable
scratch directory and checks the canonical manuscript. It also regenerates the figures when
the input files are available. Software tests and stored-result verification do not replace
a full independent scientific rerun; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

A compact software demonstration needs no external data:

```bash
python examples/basic/04_complete_pipeline.py --spatial-points 40 --time-steps 12 --n-modes 8 --n-sensors 6 --solver admm
```

## Data and full study

The study requires the exact author-generated `data_exp_4_.h5` and `all_wells_exp_4.json`
under `$TBMD_DATA_DIR/brugge/`; their hashes and acquisition/provenance limitations are in
[the study data guide](studies/brugge_sparse_sensing/README.md#2-data).
The [TNO Brugge model](https://github.com/TNO/Brugge) and the derived simulation inputs are not
redistributed here. The corresponding author may provide the transformed HDF5/JSON inputs upon
reasonable request, subject to applicable data-use and access conditions. The simulator is confirmed
as t-Navigator; its version, HDF5 exporter, vertical reduction and physical pressure unit are not
recoverable from the current archive. Obtaining the base model alone does not recreate the exact
simulation exports.

Run into temporary output directories to preserve the paper's canonical results:

```bash
cd studies/brugge_sparse_sensing
TBMD_DATA_DIR=/path/to/data BSS_OUT=/tmp/brugge-rerun/outputs BSS_FIG=/tmp/brugge-rerun/figures BSS_LOG=/tmp/brugge-rerun/logs ./run_all.sh 8
python scripts/compare_rerun.py /tmp/brugge-rerun/outputs
```

E1–E5/E7 results are deterministic for the recorded environment; E6 timings and original-library
E0 re-executions have separate variability. Remove temporary rerun outputs after reviewing the comparison.

## Structure and outputs

```text
src/TBMD/                     reusable library, public configuration and compatibility imports
studies/brugge_sparse_sensing/ config, bss library, stage scripts and verified inputs manifest
  outputs/                    canonical experiment results, tables and number macros
  figures/                    canonical publication PDFs
  run_logs/                   recorded independent-run evidence
  scripts/                    experiments, generators and verification
examples/                     basic and geometry-aware software demonstrations
tests/                        unit and repository/paper integration checks
docs/                         library guides; paper/manuscript, submission sources and audit
```

[Documentation index](docs/index.md) · Repository audit (local audit material, not distributed) ·
Cleanup and validation report (local audit material, not distributed).

## Build the paper

Install `tectonic` and `pdftotext` separately, then run:

```bash
python docs/paper/build_submission_ready.py --arxiv
```

This generates ignored `docs/paper/submission_ready/`, `arxiv_v2/` and `arxiv_v2.tar.gz`.
Build success verifies the local package but does not perform or authorise an upload.

## Citation and licence

Cite this software using [CITATION.cff](CITATION.cff). The recorded software release v2.1.0
is identified by [Zenodo DOI 10.5281/zenodo.22814377](https://doi.org/10.5281/zenodo.22814377).
The [v2.2.0 GitHub release](https://github.com/denis-samatov/tensor-based-modal-decomposition-method/releases/tag/v2.2.0) includes nested optimization and E9. Its version-specific archival DOI is pending; the old DOI does not identify this revision. The current paper title is above;
the earlier arXiv entry retains its original title until the authors replace it.
Code is MIT licensed ([LICENSE](LICENSE)); TNO data rights are separate.
[Contribution guide](CONTRIBUTING.md).
