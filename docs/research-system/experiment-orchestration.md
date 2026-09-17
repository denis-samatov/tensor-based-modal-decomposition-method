# Experiment orchestration

Reusable components are composed by the synthetic examples, `TBMD.experiments` helpers and the
separate Brugge study scripts. There is no documented universal `FullPipelineConfig` that
normalizes arbitrary reservoir data automatically.

Each experiment owns:

- Data loading, axis order, units, missing-data and normalization rules.
- Training/validation/test split and protection against target leakage.
- Ranks, dictionary construction, sensor constraints and solver configuration.
- Metrics, baselines, provenance and artifact destinations.

Follow [examples/basic/04_complete_pipeline.py](../../examples/basic/04_complete_pipeline.py) for
synthetic API composition, and the [Brugge study](../../studies/brugge_sparse_sensing/README.md) for
its dataset-specific benchmark. Forecasting is a separate project.

## Validation

```bash
MPLBACKEND=Agg python -m pytest tests/unit/test_experiment_plotting.py -q
```

This tests experiment plotting behavior, not the entire scientific benchmark.
[Architecture decisions](../architecture/decisions.md) · [Runbook](../operations/runbook.md)
