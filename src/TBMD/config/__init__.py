"""
Configuration objects for TBMD module.
"""

from .base import BaseConfig
from .decomposition import DecompositionConfig, GeometryAwareDecompositionConfig
from .sensor_placement import SensorPlacementConfig, GeometricSensorConfig
from .reconstruction import (
    ReconstructionConfig, 
    CompressiveSensingConfig, 
    ExtensionCompressiveSensingConfig, 
    GeometryAwareReconstructionConfig
)

from .experiments import ExperimentConfig
from .modal_processor import ModalProcessorConfig
from .forecaster import (
    LatentModalForecasterConfig,
    LinearForecasterConfig,
    MLPForecasterConfig,
    LSTMForecasterConfig,
    MultiResolutionTBMDConfig,
)

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
    "LatentModalForecasterConfig",
    "LinearForecasterConfig",
    "MLPForecasterConfig",
    "LSTMForecasterConfig",
    "MultiResolutionTBMDConfig",
]

# Instantiate default configs for quick access
_decomposition_config = DecompositionConfig()
_sensor_placement_config = SensorPlacementConfig()
_reconstruction_config = ReconstructionConfig()
_experiment_config = ExperimentConfig()
