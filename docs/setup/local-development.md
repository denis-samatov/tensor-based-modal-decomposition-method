# Local development

`pyproject.toml` defines Python and dependency requirements; `.python-version` records the local
Python choice. Python 3.10+ is supported. Use a virtual environment from the checkout root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Core synthetic examples require no external credentials or private data. Brugge figure generation
and a full study rerun have separate data/environment requirements in the
[study README](../../studies/brugge_sparse_sensing/README.md).

## Validate the environment

```bash
python -c "import TBMD; print(TBMD.__version__)"
MPLBACKEND=Agg python -m pytest tests/audit -q
python -m compileall src tests examples
```

`MPLBACKEND=Agg` selects headless plotting for tests and batch examples. It does not validate a
native GUI. Compilation checks syntax; audit tests check repository contracts.

[Configuration](configuration.md) · [Testing](../development/testing.md) ·
[Documentation index](../index.md)
