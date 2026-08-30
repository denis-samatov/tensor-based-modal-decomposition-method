"""
TBMD Package - Tensor-Based Modal Decomposition Method

Main package for tensor-based modal decomposition algorithms with geometry awareness.
"""

# Re-export geometry modules for backward compatibility
# This allows "from TBMD.geometry import ..." to work
from .core import geometry
from .core.decomposition.geometry_aware import GeometryAwareTuckerDecomposer
from .core.decomposition.hosvd import TuckerDecomposerInterface
from .core.modal_processor.modes import ModalTensorProcessor
from .core.reconstruction.geometry_aware import GeometryAwareTensorCS
from .core.reconstruction.tensor_compressive_sensing import TensorCompressiveSensing
from .core.sensor_placement.geometry_aware import GeometryAwareTensorQR
from .core.sensor_placement.tensor_qr_factorization import TensorTubeQRDecomposition

__all__ = [
    "geometry",
    "TuckerDecomposerInterface",
    "GeometryAwareTuckerDecomposer",
    "TensorTubeQRDecomposition",
    "GeometryAwareTensorQR",
    "TensorCompressiveSensing",
    "GeometryAwareTensorCS",
    "ModalTensorProcessor",
]

__version__ = "2.0.0"
