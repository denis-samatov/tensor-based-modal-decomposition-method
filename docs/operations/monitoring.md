# Experiment diagnostics

Monitoring means observing numerical execution, not service uptime.
`TensorCompressiveSensing.solve()` returns `CompressiveSensingMetrics` containing convergence,
iteration count, final primal/dual residuals, objective, penalty, optional history and elapsed time.

- Inspect `converged` together with the stopping policy and iteration limit.
- Compare measurement fit and field error separately from residual convergence.
- Use the optional `hook` to record per-iteration primal/dual residuals and objective when needed.
- Residuals are not guaranteed to decrease strictly at every iteration.
- Record sampled-dictionary conditioning and sensor count for sparse recovery diagnostics.
- Rank/energy diagnostics describe a representation; they do not establish held-out prediction skill.

Save diagnostics with the exact configuration, input provenance and source revision.
The [Python API example](../interfaces/python-api.md) prints actual solver metrics.

## Validation

```bash
MPLBACKEND=Agg python -m pytest tests/unit/test_reconstruction.py -q
```

[Runbook](runbook.md) · [Troubleshooting](troubleshooting.md)
