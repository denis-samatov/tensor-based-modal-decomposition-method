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
]

# Instantiate default configs for quick access
_decomposition_config = DecompositionConfig()
_sensor_placement_config = SensorPlacementConfig()
_reconstruction_config = ReconstructionConfig()
_experiment_config = ExperimentConfig()
