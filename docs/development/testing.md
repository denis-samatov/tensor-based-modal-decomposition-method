# Testing

Run commands from the installed checkout root. Headless plotting avoids depending on an interactive
Matplotlib backend.

```bash
MPLBACKEND=Agg python -m pytest tests -q
```

## Focused checks

```bash
MPLBACKEND=Agg python -m pytest tests/audit -q
MPLBACKEND=Agg python -m pytest tests/unit -q
MPLBACKEND=Agg python -m pytest tests/unit/test_decomposition.py -q
python -m compileall src tests examples
python -m ruff check src tests examples
python -m mypy src/TBMD/core
```

`pyproject.toml` records Ruff settings and existing mypy type-debt exclusions. A passing mypy command
covers its configured scope; it does not prove that excluded modules have no type errors.
Audit tests cover repository hygiene, local links, installation instructions and public boundaries.
Compileall checks syntax rather than numerical behavior.

## Scientific checks

The software suite uses synthetic fixtures. Run Brugge-specific verification using the
[study instructions](../../studies/brugge_sparse_sensing/README.md).
Dataset-specific URANS/forecasting tests belong to
[tbmd-forecasting](https://github.com/denis-samatov/tbmd-forecasting).
Software PASS does not establish physical forecast skill, field qualification or article readiness.

[Contribution guide](contribution-guide.md) · [Code style](code-style.md) ·
[Release process](release-process.md)
