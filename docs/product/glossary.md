# Glossary

| Term | Meaning in this project |
|---|---|
| Tensor | Multidimensional numerical array with an explicitly documented axis order |
| Tucker decomposition | Core tensor and mode-factor representation |
| Modal dictionary | Array used to expand a coefficient vector into a field; coefficients occupy its final axis |
| Sensor mask | Binary/boolean array selecting entries of a field-shaped measurement array |
| Reconstruction | Recovering coefficients from measurements and expanding them into the field |
| Decomposition reconstruction error | Error when representing the decomposed input; not a held-out prediction metric |
| ADMM convergence | Satisfaction of configured numerical stopping criteria |
| Brugge study | Reservoir-state benchmark with restricted inputs and curated reproducibility outputs |
| Synthetic example | Generated-data software demonstration |
| Qualification | Evidence that a specified scientific/physical/deployment contract is satisfied |

Use these distinctions in new documentation. Release tags, successful tests and directories named
`production` are not qualification by themselves.

## Validation

Compare terminology with the [tensor contracts](../interfaces/input-output-tensors.md) and
[study definitions](../../studies/brugge_sparse_sensing/README.md).

[Overview](overview.md) · [Architecture](../architecture/overview.md)
