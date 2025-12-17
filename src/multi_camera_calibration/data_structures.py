"""Data structures for multi-camera calibration."""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class CameraConfig:
    """Configuration for a single camera.

    Attributes:
        camera_id: Unique identifier for the camera
        parent_link: Name of the link the camera is mounted on
        init_link_T_cam: Initial estimate of transform from parent link to camera optical frame (4x4 SE3 matrix)
    """
    camera_id: str
    parent_link: str
    init_link_T_cam: np.ndarray  # 4x4 SE3 matrix

    def __post_init__(self):
        """Validate the camera configuration."""
        assert self.init_link_T_cam.shape == (4, 4), "init_link_T_cam must be 4x4 matrix"
        # Verify it's approximately SE3
        assert np.allclose(self.init_link_T_cam[3, :], [0, 0, 0, 1]), "Bottom row must be [0,0,0,1]"


@dataclass
class RobotConfig:
    """Configuration for the robot's camera system.

    Attributes:
        cameras: List of camera configurations
    """
    cameras: List[CameraConfig]

    def __post_init__(self):
        """Validate the robot configuration."""
        camera_ids = [cam.camera_id for cam in self.cameras]
        assert len(camera_ids) == len(set(camera_ids)), "Camera IDs must be unique"
        assert len(self.cameras) >= 2, "Need at least 2 cameras for calibration"

    def get_camera_by_id(self, camera_id: str) -> CameraConfig:
        """Get camera configuration by ID."""
        cam = next((c for c in self.cameras if c.camera_id == camera_id), None)
        if not cam:
            raise ValueError(f"Camera {camera_id} not found")
        return cam


@dataclass
class Detection:
    """A single detection of a calibration object by one camera.

    Attributes:
        camera_id: ID of the camera that made the detection
        object_id: ID of the detected object
        cam_T_object: Transform from camera optical frame to object frame (4x4 SE3 matrix)
        num_tags: Number of AprilTags detected (1-6), used for weighting
    """
    camera_id: str
    object_id: int
    cam_T_object: np.ndarray  # 4x4 SE3 matrix
    num_tags: int

    def __post_init__(self):
        """Validate the detection."""
        assert self.cam_T_object.shape == (4, 4), "cam_T_object must be 4x4 matrix"
        assert np.allclose(self.cam_T_object[3, :], [0, 0, 0, 1]), "Bottom row must be [0,0,0,1]"
        assert 1 <= self.num_tags <= 6, "num_tags must be between 1 and 6"

    @property
    def weight(self) -> float:
        """Compute detection weight based on number of tags.

        Returns:
            Weight in [0, 1], where single-tag detections get 0 weight
        """
        return 1.0 - 1.0 / self.num_tags


@dataclass
class Capture:
    """A synchronized snapshot from all cameras at one instant.

    Attributes:
        capture_id: Unique identifier for this capture
        link_poses: Dictionary mapping link names to base_T_link transforms (4x4 SE3 matrices)
        detections: List of all detections in this capture
    """
    capture_id: int
    link_poses: Dict[str, np.ndarray]  # link_name -> base_T_link (4x4)
    detections: List[Detection]

    def __post_init__(self):
        """Validate the capture."""
        for link_name, pose in self.link_poses.items():
            assert pose.shape == (4, 4), f"Pose for link {link_name} must be 4x4 matrix"
            assert np.allclose(pose[3, :], [0, 0, 0, 1]), f"Pose for link {link_name} must have bottom row [0,0,0,1]"


@dataclass
class ValidationResult:
    """Results from validation of calibration.

    Attributes:
        mean_translation_error_mm: Mean translation error in millimeters
        mean_rotation_error_deg: Mean rotation error in degrees
        max_translation_error_mm: Maximum translation error in millimeters
        max_rotation_error_deg: Maximum rotation error in degrees
        num_constraints: Number of pairwise constraints evaluated
        passed_sanity_check: Whether corrections are within reasonable bounds
        correction_magnitudes: Dictionary mapping camera_id to (trans_mm, rot_deg) correction magnitudes
    """
    mean_translation_error_mm: float
    mean_rotation_error_deg: float
    max_translation_error_mm: float
    max_rotation_error_deg: float
    num_constraints: int
    passed_sanity_check: bool
    correction_magnitudes: Dict[str, tuple]  # camera_id -> (trans_mm, rot_deg)

    def __str__(self) -> str:
        """Human-readable summary of validation results."""
        lines = [
            "Validation Results:",
            f"  Mean error: {self.mean_translation_error_mm:.2f}mm, {self.mean_rotation_error_deg:.2f}°",
            f"  Max error: {self.max_translation_error_mm:.2f}mm, {self.max_rotation_error_deg:.2f}°",
            f"  Constraints: {self.num_constraints}",
            f"  Sanity check: {'PASS' if self.passed_sanity_check else 'FAIL'}",
            "  Correction magnitudes:",
        ]
        for cam_id, (trans_mm, rot_deg) in self.correction_magnitudes.items():
            lines.append(f"    {cam_id}: {trans_mm:.2f}mm, {rot_deg:.2f}°")
        return "\n".join(lines)
