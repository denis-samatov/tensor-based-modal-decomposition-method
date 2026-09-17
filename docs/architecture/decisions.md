# Architecture decisions

## Reusable library and experiments

Reusable algorithms live in `src/TBMD/core/`. Synthetic examples and the Brugge study compose these
algorithms without treating a dataset-specific workflow as a universal API. Physical URANS and
`t+1` forecasting live in the separate
[tbmd-forecasting repository](https://github.com/denis-samatov/tbmd-forecasting).

## Explicit tensor representation

Tucker decomposition produces a core and mode factors. Tensor Tube QR works on a modal dictionary,
and sparse reconstruction recovers its coefficient vector. Axis order and flattening decisions
belong in each experiment's contract. A tensor representation does not guarantee a more accurate
result than a matrix baseline; comparisons require the same split and sensor budget.

## ADMM reconstruction

`TensorCompressiveSensing` implements an iterative coefficient solver with configurable penalties,
linear-solver and stopping policies. Its metrics describe the executed optimization; convergence,
measurement fit and held-out field accuracy are separate checks.

## Data and artifacts

Local datasets and exploratory run artifacts are ignored. Curated synthetic documentation assets
and Brugge study results/generators are intentionally tracked for reproducibility. The policy is
not a blanket prohibition on every generated file. Inspect `.gitignore`,
[REPRODUCIBILITY.md](../../REPRODUCIBILITY.md) and the study README before adding artifacts.

## Compatibility

Deprecated `TBMD.modules` and `TBMD.utils` paths remain compatibility wrappers. Current examples
use the core API; explicitly named legacy examples demonstrate compatibility. Removing wrappers
requires an API migration, not a filesystem cleanup.

## Validation

```bash
MPLBACKEND=Agg python -m pytest tests/audit -q
```

[Architecture overview](overview.md) · [Brugge study](../../studies/brugge_sparse_sensing/README.md)
