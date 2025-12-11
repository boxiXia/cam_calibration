"""Multi-camera calibration for mobile legged robots."""

from .data_structures import (
    CameraConfig,
    RobotConfig,
    Capture,
    Detection,
    ValidationResult,
)
from .calibrator import Calibrator
from .simulation import SimulationFramework

__version__ = "0.1.0"

__all__ = [
    "CameraConfig",
    "RobotConfig",
    "Capture",
    "Detection",
    "ValidationResult",
    "Calibrator",
    "SimulationFramework",
]
