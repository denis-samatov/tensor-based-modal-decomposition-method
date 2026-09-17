# Troubleshooting

| Symptom | Investigation |
|---|---|
| Tensor shape failure | Check the exact component contract and recorded axis order; there is no universal four-dimensional input layout |
| Empty/incomplete placement | Inspect returned `P`, available/rejected locations and modal rank; assert the actual sensor count |
| High field error | Separate basis truncation error, sampled-dictionary conditioning, measurement error and solver behavior |
| Solver reaches iteration limit | Inspect residuals, stopping policy, dtype and penalty settings; do not claim convergence |
| SVD memory exhaustion | Estimate unfolded tensor/matrix size and temporary storage; reduce or restructure the experiment deliberately |
| Plotting fails in batch tests | Set `MPLBACKEND=Agg` before starting Python; choose a writable `MPLCONFIGDIR` if needed |

`BatchModalProcessor` batches modal processing; it is not a guarantee of streaming/out-of-core
Tucker SVD. Changing axis semantics or inserting synthetic data is not a valid recovery strategy.

## Validation

```bash
MPLBACKEND=Agg python -m pytest tests/unit/test_decomposition.py tests/unit/test_reconstruction.py -q
```

[Errors](../interfaces/exceptions.md) · [Tensor contracts](../interfaces/input-output-tensors.md) ·
[Diagnostics](monitoring.md)
