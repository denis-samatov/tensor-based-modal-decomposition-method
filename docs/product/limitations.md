# Limitations and data handling

TBMD is research software. Accuracy, scalability and production qualification require evidence
for the specific data and deployment. An owner decision alone cannot replace that evidence.

## Numerical limits

- Dense tensor decomposition can allocate large unfolded matrices and SVD intermediates.
- Truncation ranks trade representation size against approximation error; no universal rank is optimal.
- Sparse recovery depends on sampled-dictionary conditioning and the relationship between sensor
  count and modal degrees of freedom.
- Solver convergence is separate from physical reconstruction accuracy and held-out prediction skill.
- Mesh regularization depends on correct node ordering and geometry. It does not automatically
  improve every dataset.

No maximum supported physical grid size is asserted here without a corresponding measured run.

## Artifacts and privacy

Put local datasets in ignored `data/` and exploratory results in ignored artifact directories.
Curated documentation assets and Brugge study outputs are intentional tracked exceptions.
Record input checksums, units, axis order, preprocessing, source revision and split policy.
Do not commit credentials, personal `.env` files or restricted simulator exports.

## Validation

```bash
git status --ignored --short
MPLBACKEND=Agg python -m pytest tests/audit -q
```

Inspect tracked/staged files too: an ignored directory alone does not prove that all data are
untracked. See [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md) for the public scientific scope.

[Overview](overview.md) · [Testing](../development/testing.md)
