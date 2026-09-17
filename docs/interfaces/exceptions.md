# Exceptions and errors

The decomposition module defines `TensorDecompositionError`, `InvalidRankError`, `StateError` and
`ValidationError`. Source: [hosvd.py](../../src/TBMD/core/decomposition/hosvd.py).

| Failure | Check |
|---|---|
| Result accessed before decomposition/reconstruction | Call `decompose()` or `reconstruct()` before reading the corresponding properties |
| Invalid ranks or dimensions | Record input shape and verify ranks against every mode |
| Sparse solver shape mismatch | Verify `P.shape == Y.shape == A.shape[:-1]` |
| Empty sensor mask | Confirm QR selected the requested count; do not infer success from method completion |
| Non-finite values or linear algebra failure | Inspect input validity, sampled-dictionary rank/conditioning, dtype and regularization |
| Memory exhaustion | Reduce problem size or use a documented data strategy; modal batching alone does not make dense SVD out-of-core |

Do not suppress failures by changing the physical data meaning or silently switching to synthetic
inputs. Exact exception text can depend on the component and numerical-library version.

## Validation

```bash
MPLBACKEND=Agg python -m pytest tests/unit/test_decomposition.py tests/unit/test_TensorValidator.py tests/unit/test_reconstruction.py -q
```

[Troubleshooting](../operations/troubleshooting.md) · [Tensor contracts](input-output-tensors.md)
