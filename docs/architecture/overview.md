# Architecture overview

TBMD is a Python research library for decomposition, modal-basis construction, sensor placement and
sparse-field reconstruction. Dataset-specific forecasting belongs to the separate
[tbmd-forecasting repository](https://github.com/denis-samatov/tbmd-forecasting).

| Layer | Source | Responsibility |
|---|---|---|
| Configuration | `src/TBMD/config/` | Explicit dataclasses for computation, decomposition, sensors and reconstruction |
| Core algorithms | `src/TBMD/core/` | Decomposition, modal processing, geometry, placement, reconstruction, data and metrics |
| Experiment helpers | `src/TBMD/experiments/`, `src/TBMD/visualization/` | Experiment composition and diagnostic plotting |
| Examples | `examples/` | Synthetic demonstrations and explicit compatibility/geometry examples |
| Brugge study | `studies/brugge_sparse_sensing/` | Dataset-specific benchmark, curated results and manuscript generators |
| Compatibility | `src/TBMD/modules/`, `src/TBMD/utils/` | Deprecated import paths retained for existing callers |

Algorithm instances retain computed results. Configuration seeding can change random-library and
PyTorch deterministic settings; this is not a stateless service. Dataset provenance, split policy
and held-out evaluation are responsibilities of the experiment, not automatic library guarantees.

## Validation

From the installed checkout root:

```bash
MPLBACKEND=Agg python -m pytest tests/audit -q
```

[Data flow](data-flow.md) · [Components](components.md) · [Decisions](decisions.md) ·
[Documentation index](../index.md)
