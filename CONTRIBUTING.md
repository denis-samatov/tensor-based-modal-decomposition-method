# Contributing

## Purpose
To outline how developers can safely and effectively contribute code, documentation, and tests to the Tensor-Based Modal Decomposition Method (TBMD) repository.

## Audience
External contributors and internal developers.

## Summary
Contributors should use a local virtual environment with an editable installation, avoid committing generated artifacts or local datasets, and run validation checks before opening a pull request.

## Details

### Development Setup
To set up your local environment for active development:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Contribution Guidelines
- **Scope**: Keep algorithmic changes separate from documentation or cleanup changes.
- **Hygiene**: Do not commit local datasets, virtual environments, caches, `.env` files, generated plots, generated metrics, or model artifacts.
- **Documentation**: Document new scripts with required inputs, outputs, and whether they need local data.
- **Testing**: Add tests for public API behavior, shape contracts, and bug fixes.
- **Claims**: Avoid unsupported performance or accuracy claims unless the repository includes a reproducible command and dataset description.
- **Links**: Keep documentation links case-correct so they work on GitHub and Linux file systems.

### Code Style
Ruff and mypy settings are declared in `pyproject.toml`. Follow the surrounding style, document tensor contracts, preserve configured type-debt exclusions and avoid unrelated refactors.

## Validation
Before opening a Pull Request, run the relevant checks:
```bash
# Run the full test suite
MPLBACKEND=Agg python -m pytest tests -q

# Ensure syntax is correct
python -m compileall src tests examples scripts
python -m ruff check src tests examples
python -m mypy src/TBMD/core
```

For documentation or repository-structure changes, also run:
```bash
MPLBACKEND=Agg python -m pytest tests/audit -q
```
Record actual results and any warnings. A successful software suite is not scientific or production qualification.

## Related docs
- [Testing Guide](docs/development/testing.md)
- [Code Style Guide](docs/development/code-style.md)
