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


## Zero-reference reconstruction and file loading

An exactly reconstructed zero tensor has relative error 0 in single, CPU collection and GPU
collection paths. A nonzero residual against a zero reference norm raises `ValidationError`;
its relative error is mathematically undefined. This does not change nonzero-reference errors.

`BaseDataLoader` loads `.pt`/`.pth` files with `weights_only=True`. The payload must be a tensor,
or a dictionary with an explicit tensor key. Arbitrary Python objects are not accepted.
This is a deserialization contract, not a memory quota for untrusted large files.

Validation: `MPLBACKEND=Agg pytest tests/unit/test_audit_boundaries.py -q`.

## Separate execution profile (2026-10-29)

`requirements-execution.in` is the maintained input; `requirements-execution.txt` is generated
with `uv pip compile requirements-execution.in --python-version 3.13 --generate-hashes --output-file requirements-execution.txt`.
It is a new Python 3.13 software-validation environment, not the historical scientific environment.
Create a local venv, then install with `python -m pip install --require-hashes -r requirements-execution.txt`.
Do not edit the generated lock by hand. OSV exact-version scanning covers the resolved public
packages; it does not establish exploit reachability or guarantee that undisclosed issues are absent.
Historical locks, predictions and numerical results remain unchanged. A new experimental result
requires its own inputs, source revision, environment and numerical qualification.

This profile was resolved on macOS arm64 for Python 3.13. It is not a universal lock for
all operating systems, Python versions or accelerators; regenerate and qualify another target.
