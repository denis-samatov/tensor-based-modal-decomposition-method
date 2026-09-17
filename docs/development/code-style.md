# Code style

Follow the surrounding implementation and the actual configuration in `pyproject.toml`.
Ruff targets Python 3.10 with a 100-character line length and the configured error/import rules.
Public docstrings should describe arguments, return values, tensor axes and failure conditions.
Preserve established interfaces and avoid unrelated formatting or algorithm refactors.

Type-checking uses explicit existing debt exclusions. Add accurate annotations without weakening
the configured checks. Use Google-style docstrings for new public interfaces.

## Validation

```bash
python -m ruff check src tests examples
python -m mypy src/TBMD/core
python -m compileall src tests examples
```

[Contribution guide](contribution-guide.md) · [Testing](testing.md)
