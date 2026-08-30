"""Configuration objects for the public TBMD modules."""

from .base import BaseConfig
from .decomposition import DecompositionConfig, GeometryAwareDecompositionConfig
from .experiments import ExperimentConfig
from .modal_processor import ModalProcessorConfig
from .reconstruction import (
    CompressiveSensingConfig,
    ExtensionCompressiveSensingConfig,
    GeometryAwareReconstructionConfig,
    ReconstructionConfig,
)
from .sensor_placement import GeometricSensorConfig, SensorPlacementConfig

__all__ = [
    "BaseConfig",
    "DecompositionConfig",
    "GeometryAwareDecompositionConfig",
    "SensorPlacementConfig",
    "GeometricSensorConfig",
    "ReconstructionConfig",
    "CompressiveSensingConfig",
    "ExtensionCompressiveSensingConfig",
    "GeometryAwareReconstructionConfig",
    "ExperimentConfig",
    "ModalProcessorConfig",
    "SEED",
    "SET_BACKEND",
]

# Instantiate default configs for quick access
_decomposition_config = DecompositionConfig()
_sensor_placement_config = SensorPlacementConfig()
_reconstruction_config = ReconstructionConfig()
_experiment_config = ExperimentConfig()

# Default seed and TensorLy backend, derived from BaseConfig so they never
# drift from the values every *Config class already uses.
SEED = BaseConfig.__dataclass_fields__["seed"].default
SET_BACKEND = BaseConfig.__dataclass_fields__["backend"].default
