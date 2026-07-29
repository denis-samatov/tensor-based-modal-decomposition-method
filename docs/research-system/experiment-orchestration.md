# Experiment Orchestration

## Purpose
Describes how the pipeline orchestrates data flow between the core mathematical components.

## Audience
Developers and ML Engineers setting up new experiments or benchmarks.

## Summary
The orchestration layer manages the lifecycle of offline model training and state reconstruction, allowing researchers to run full end-to-end benchmarks on datasets like Brugge.

## Details
### Orchestration Flow
1. **Configuration**: The pipeline receives a `FullPipelineConfig` that groups `DecompositionConfig`, `SensorPlacementConfig`, and `ReconstructionConfig`.
2. **Data Ingestion**: The pipeline accepts raw tensors (e.g., `[features, x, y, time]`) and handles normalization.
3. **Offline Phase**: Calls the decomposer to extract the spatial basis and temporal modes.
4. **Reconstruction**: Integrates sparse measurements to project the state back into the original high-dimensional space.

### Interfaces
The orchestration scripts expose methods to standardize evaluation.

## Related docs
- [Current Architecture Decisions](../architecture/decisions.md)
