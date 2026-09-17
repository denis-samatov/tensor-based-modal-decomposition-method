# TBMD documentation

Start with the [repository README](../README.md) for installation and the complete synthetic example.
This index covers the library. Forecasting is maintained in
[tbmd-forecasting](https://github.com/denis-samatov/tbmd-forecasting).

## Directory responsibilities

| Directory | Purpose |
|---|---|
| `src/TBMD/core/` | Reusable numerical components |
| `src/TBMD/config/` | Public configuration dataclasses |
| `src/TBMD/experiments/`, `src/TBMD/visualization/` | Experiment and plotting helpers |
| `src/TBMD/modules/`, `src/TBMD/utils/` | Compatibility import paths |
| `examples/`, `notebooks/` | Synthetic examples, geometry demonstrations and historical notebooks |
| `studies/brugge_sparse_sensing/` | Revised manuscript benchmark and curated outputs |
| `docs/` | Current guides and a separate captured synthetic-run record |
| `data/`, `outputs/`, `results/`, `dist/` | Local datasets and artifact/build directories |

## Guides

### Scope and concepts

- [Glossary](product/glossary.md)
- [Limitations and data handling](product/limitations.md)
- [TBMD overview](product/overview.md)
- [Use cases](product/use-cases.md)

### Setup

- [Configuration](setup/configuration.md)
- [Environment variables](setup/environment-variables.md)
- [Local development](setup/local-development.md)

### Architecture

- [Components](architecture/components.md)
- [Data flow](architecture/data-flow.md)
- [Architecture decisions](architecture/decisions.md)
- [Architecture overview](architecture/overview.md)

### Python and tensor contracts

- [Exceptions and errors](interfaces/exceptions.md)
- [Tensor contracts](interfaces/input-output-tensors.md)
- [Python API](interfaces/python-api.md)

### Research workflows

- [Experiment orchestration](research-system/experiment-orchestration.md)
- [Sparse reconstruction workflow](research-system/reconstruction-pipeline.md)

### Running and diagnosing experiments

- [Experiment diagnostics](operations/monitoring.md)
- [Experiment runbook](operations/runbook.md)
- [Troubleshooting](operations/troubleshooting.md)

### Development

- [Code style](development/code-style.md)
- [Contribution guide](development/contribution-guide.md)
- [Release Process](development/release-process.md)
- [Testing](development/testing.md)

## Examples and evidence

- [Example catalogue](../examples/README.md)
- [Notebook status](../notebooks/README.md)
- [Captured synthetic run](examples/synthetic-run.md) — dated execution evidence, not a new benchmark
- [Brugge study](../studies/brugge_sparse_sensing/README.md)
- [Reproducibility matrix](../REPRODUCIBILITY.md)

## Repository guidance

- [Contribution policy](../CONTRIBUTING.md)
- [Security policy](../SECURITY.md)
- [Changelog](../CHANGELOG.md)
- [Codex instructions](../AGENTS.md)

Optional local Claude and Gemini instructions are not distributed with the repository.

## Validation

```bash
MPLBACKEND=Agg python -m pytest tests/audit -q
```

Guides describe current interfaces. Dated evidence records retain the executed revision and scope;
software success does not establish physical or production qualification.
