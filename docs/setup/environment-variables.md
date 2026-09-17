# Environment variables

The core library requires no credentials and does not parse an `.env` file. Runtime settings are
passed through Python configuration objects.

| Variable | Scope |
|---|---|
| `MPLBACKEND=Agg` | Headless Matplotlib in tests and batch examples |
| `TBMD_DATA_DIR` | Brugge study data location, documented by the study runner |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` | Numerical-library thread controls; use explicit settings when comparing timing results |
| `MPLCONFIGDIR` | Optional writable Matplotlib configuration/cache directory |

These variables belong to tools and experiment execution, not an implicit core configuration
service. Physical URANS configuration belongs to
[tbmd-forecasting](https://github.com/denis-samatov/tbmd-forecasting).

## Validation

```bash
python -c "import TBMD; print(TBMD.__version__)"
```

Never commit credentials or personal `.env` files.
[Configuration](configuration.md) · [Brugge study](../../studies/brugge_sparse_sensing/README.md)
