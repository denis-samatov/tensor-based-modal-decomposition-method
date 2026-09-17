# Configuration

General configuration dataclasses are exported by [TBMD.config](../../src/TBMD/config/__init__.py).
Pass the relevant object to the component; there is no single `FullPipelineConfig` contract.

| Class | Responsibility |
|---|---|
| `BaseConfig` | TensorLy backend, dtype, device, seed, deterministic settings and logging |
| `DecompositionConfig` | Ranks, energy thresholds and decomposition/numerical options |
| `SensorPlacementConfig` | Sensor count, reproducibility and placement options |
| `CompressiveSensingConfig` | Core ADMM coefficient-solver settings |
| `ExtensionCompressiveSensingConfig` | Linear-solver/stopping/history extension options |
| `ReconstructionConfig` | Higher-level reconstruction settings; `to_core_config()` adapts the core solver configuration |
| `ExperimentConfig`, `ModalProcessorConfig` | Experiment and modal-processing settings |

For `TensorCompressiveSensing`, use `core_cfg=CompressiveSensingConfig(...)` and optional `ext_cfg`.
Some geometry modules define their own configuration dataclasses; import the class documented by
that module rather than substituting a similarly named class.

```python
from TBMD.config import DecompositionConfig
config = DecompositionConfig(
    ranks=[4, 2, 4], backend="pytorch", device="cpu", dtype="float32",
    seed=42, random_state=42, verbose=False,
)
```

Creating a base-derived configuration can seed numerical libraries and enable deterministic
PyTorch operations. Seeds do not guarantee bitwise equality across platforms and library versions.
Consult constructor/source behavior rather than assuming every inherited option changes every
algorithm path.

## Validation

```bash
python -m pytest tests/unit/test_config.py -q
```

[Python API](../interfaces/python-api.md) · [Environment variables](environment-variables.md)
