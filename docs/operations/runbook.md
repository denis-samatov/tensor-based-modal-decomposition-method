# Experiment Runbook

## Purpose
Provides a standard procedure for running and logging numerical experiments reproducibly.

## Audience
Researchers running end-to-end benchmarks.

## Summary
Experiments should be run via standardized scripts, logging configuration metadata alongside the results, rather than relying on interactive Jupyter notebooks.

## Details
### 1. Preparation
- Install the project from `pyproject.toml`.
- Verify that `TBMD.config` parameters are correct for the experiment.

### 2. Execution
Run the experiment script:
```bash
python examples/basic/04_complete_pipeline.py
```

### 3. Artifact Logging
- When adapting the example, write generated artifacts to an ignored directory such as `results/`.
- Save exact configuration parameters alongside numerical outputs. RANS/URANS forecasting
  runs use the standalone `tbmd-forecasting` project and its structured output contract.
