# Contribution guide

[CONTRIBUTING.md](../../CONTRIBUTING.md) defines the development workflow. Keep changes focused,
preserve scientific provenance and add tests when public behavior changes.

Document required inputs, outputs, axis contracts and whether a command needs private data.
Current user instructions belong in operational documentation; dated run reports retain their
original results and scope.

## Validation

```bash
MPLBACKEND=Agg python -m pytest tests/audit -q
python -m ruff check src tests examples
```

For code changes, also run the affected unit tests and the broader checks justified by the change.
[Testing](testing.md) · [Code style](code-style.md) · [Release process](release-process.md)
