"""Data structures for multi-camera calibration."""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class CameraConfig:
    """Camera with parent link and initial transform estimate."""
    camera_id: str
    parent_link: str
    init_link_T_cam: np.ndarray  # 4x4 SE3: link to camera


@dataclass
class RobotConfig:
    """Multi-camera system configuration."""
    cameras: List[CameraConfig]

    def __post_init__(self):
        assert len(self.cameras) >= 2, "Need ≥2 cameras"
        assert len({c.camera_id for c in self.cameras}) == len(self.cameras), "Duplicate camera IDs"

    def get_camera_by_id(self, camera_id: str) -> CameraConfig:
        """Get camera by ID."""
        cam = next((c for c in self.cameras if c.camera_id == camera_id), None)
        if not cam:
            raise ValueError(f"Camera {camera_id} not found")
        return cam


@dataclass
class Detection:
    """Single detection: camera observing calibration object."""
    camera_id: str
    object_id: int
    cam_T_object: np.ndarray  # 4x4 SE3: camera to object
    num_tags: int  # AprilTag count (1-6)

    @property
    def weight(self) -> float:
        """Detection quality weight based on tag count."""
        return 1.0 - 1.0 / self.num_tags


@dataclass
class Capture:
    """Synchronized multi-camera snapshot at one instant."""
    capture_id: int
    link_poses: Dict[str, np.ndarray]  # link_name → base_T_link (4x4 SE3)
    detections: List[Detection]


@dataclass
class ValidationResult:
    """Calibration validation results."""
    mean_translation_error_mm: float
    mean_rotation_error_deg: float
    max_translation_error_mm: float
    max_rotation_error_deg: float
    num_constraints: int
    passed_sanity_check: bool
    correction_magnitudes: Dict[str, tuple]  # camera_id → (trans_mm, rot_deg)

    def __str__(self) -> str:
        lines = [f"Mean: {self.mean_translation_error_mm:.2f}mm, {self.mean_rotation_error_deg:.2f}°",
                f"Max: {self.max_translation_error_mm:.2f}mm, {self.max_rotation_error_deg:.2f}°",
                f"Constraints: {self.num_constraints}, Sanity: {'PASS' if self.passed_sanity_check else 'FAIL'}"]
        lines.extend(f"  {k}: {v[0]:.2f}mm, {v[1]:.2f}°" for k, v in self.correction_magnitudes.items())
        return "\n".join(lines)
