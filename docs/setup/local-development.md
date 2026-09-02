# Local Development Setup

## Purpose
To provide step-by-step instructions for establishing a reproducible local development environment for TBMD.

## Audience
Developers and researchers preparing to run, test, or contribute to the TBMD codebase.

## Summary
The project requires Python 3.10+ (3.12.7 recommended) and standard data science packages (PyTorch). `pyproject.toml` is the single dependency source of truth.

## Details

### Requirements
- Python 3.10 or newer (The repository records Python 3.12.7 in `.python-version`).
- A local virtual environment (`venv`).
- Sufficient disk space for optional datasets and generated experiment outputs.
- No external service credentials are required for core examples.

### Editable Development Install
For contributors who plan to edit the source code, an editable install is recommended:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```
This installs the package from `src/` along with the development dependencies declared in `pyproject.toml`.

## Examples
*(See above for command-line installation examples).*

## Validation
To verify that the installation succeeded and the core code compiles, run:
```bash
python -c "import TBMD; print(TBMD.__version__)"
pytest tests/audit -q
python -m compileall src tests examples
```
Expected result: The version should print, the audit tests should pass with no errors, and the compilation check should complete silently (meaning no syntax errors exist in the codebase).

## Related docs
- [Configuration](configuration.md)
- [Environment Variables](environment-variables.md)
