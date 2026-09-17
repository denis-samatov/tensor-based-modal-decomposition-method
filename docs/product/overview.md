# TBMD overview

TBMD is a Python research library for reduced-order representation, sparse sensor placement and
field reconstruction. The reusable flow is decomposition → modal dictionary → sensor selection →
coefficient recovery → reconstructed field.

The synthetic examples demonstrate library execution. The separate Brugge study provides the
revised manuscript benchmark with its own data representation, split and baseline comparisons.
Neither a low decomposition error nor successful tests establish production suitability.

The library does not run a reservoir simulator and does not own physical URANS/`t+1` forecasting.
Algorithm suitability depends on the data, rank, measurements and held-out evaluation.

## Validation

```bash
MPLBACKEND=Agg python -m pytest tests/unit/test_decomposition.py -q
```

[Use cases](use-cases.md) · [Limitations](limitations.md) · [Architecture](../architecture/overview.md) ·
[Brugge study](../../studies/brugge_sparse_sensing/README.md)
