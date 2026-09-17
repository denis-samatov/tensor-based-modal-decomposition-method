# Experiment runbook

## Synthetic software example

From the installed checkout root:

```bash
MPLBACKEND=Agg python examples/basic/04_complete_pipeline.py --spatial-points 40 --time-steps 12 --n-modes 8 --n-sensors 6 --solver admm
```

Add `--visualize` to write the diagnostic PNG named by the example. Use an ignored artifact
location when adapting examples into repeated experiments. Preserve exact arguments, versions,
seed and source revision alongside results.

## Brugge benchmark

Use the [study README](../../studies/brugge_sparse_sensing/README.md), its pinned environment and
input checksums. The study generators produce the revised manuscript numbers and figures.
Restricted inputs are not redistributed. Existing curated result files support the documented
subset of verification; they do not imply that a clean clone includes all simulation data.

## Forecasting

Physical URANS and `t+1` experiment commands belong to
[tbmd-forecasting](https://github.com/denis-samatov/tbmd-forecasting) and its output/provenance contract.

## Validation

```bash
MPLBACKEND=Agg python -m pytest tests/audit tests/unit -q
```

[Diagnostics](monitoring.md) · [Reconstruction workflow](../research-system/reconstruction-pipeline.md)
