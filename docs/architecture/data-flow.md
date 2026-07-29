# Data Flow and Lifecycle

## Purpose
To map the lifecycle of data as it passes through the TBMD pipeline, from raw tensors to sensor placement and field reconstruction.

## Audience
Developers and ML/AI engineers writing custom integration code or troubleshooting pipeline bottlenecks.

## Summary
Data flows through the system in a linear sequence: historical tensor data is decomposed, transformed into a modal basis, and used to determine sensor placements. Measurements at these locations are then used to reconstruct full fields.

## Details

### 1. Historical Data Ingestion
Input data is loaded into memory as a PyTorch tensor.
Typical shapes include `(x, y, time)`, `(x, y, z, time)`, or `(active_cells, time)`.
If Geometry-Aware components are used, a graph or mesh adjacency matrix is also provided.

### 2. Tensor Decomposition
The input tensor is passed to a decomposer (e.g., `TuckerDecomposer`).
**Output**: A reduced Core Tensor and a set of Factor Matrices representing the dominant modes of the spatial and temporal dimensions.

### 3. Modal Processing
The core tensor and factor matrices are passed to `BatchModalProcessor` and `ModalTensorStacker`.
**Output**: A Modal Basis (`A_tensor`) that compactly represents the spatial dynamics.

### 4. Sensor Placement
The Modal Basis is passed to `TensorTubeQRDecomposition`.
**Output**: Permutation matrices (`P`), orthogonal matrices (`Q`), and upper triangular matrices (`R`). The `P` matrix acts as the mask defining optimal sensor locations.

### 5. Operational Phase (Reconstruction)
In operational use, sparse sensor measurements are taken at locations defined by `P`.
`TensorCompressiveSensing` reconstructs the full modal coefficients from these sparse readings.
The estimated modal coefficients are expanded back into the full spatial domain using the Factor Matrices.
## Validation
To test the end-to-end data flow pipeline computationally:
```bash
python examples/basic/04_complete_pipeline.py
```
Expected result: The script will trace data from initialization to final reconstruction and print the intermediate tensor shapes, proving that dimensions align across the pipeline.

## Related docs
- [Architecture Overview](overview.md)
- [Components](components.md)
