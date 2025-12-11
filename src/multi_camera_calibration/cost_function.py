"""Cost function and residual computation for calibration optimization."""

import numpy as np
from typing import List, Tuple, Dict
from .data_structures import Capture, Detection, RobotConfig, CameraConfig
from .transforms import (
    apply_correction,
    compose_transforms,
    rotation_matrix_to_axis_angle,
    get_translation,
    get_rotation,
)


class PairwiseConstraint:
    """A pairwise constraint from two cameras viewing the same object.

    Attributes:
        camera_a_id: ID of first camera
        camera_b_id: ID of second camera
        object_id: ID of the object both cameras see
        base_T_link_a: Transform from base to camera A's parent link
        base_T_link_b: Transform from base to camera B's parent link
        cam_a_T_object: Transform from camera A to object
        cam_b_T_object: Transform from camera B to object
        weight: Combined weight based on detection quality
    """

    def __init__(
        self,
        camera_a_id: str,
        camera_b_id: str,
        object_id: int,
        base_T_link_a: np.ndarray,
        base_T_link_b: np.ndarray,
        cam_a_T_object: np.ndarray,
        cam_b_T_object: np.ndarray,
        weight: float,
    ):
        self.camera_a_id = camera_a_id
        self.camera_b_id = camera_b_id
        self.object_id = object_id
        self.base_T_link_a = base_T_link_a
        self.base_T_link_b = base_T_link_b
        self.cam_a_T_object = cam_a_T_object
        self.cam_b_T_object = cam_b_T_object
        self.weight = weight


def generate_pairwise_constraints(
    captures: List[Capture],
    robot_config: RobotConfig,
    min_tags: int = 2,
) -> List[PairwiseConstraint]:
    """Generate pairwise constraints from capture data.

    For each capture, for each object seen by multiple cameras,
    create pairwise constraints between all camera pairs.

    Args:
        captures: List of capture data
        robot_config: Robot configuration
        min_tags: Minimum number of tags required for valid detection

    Returns:
        List of pairwise constraints
    """
    constraints = []

    for capture in captures:
        # Group detections by object
        objects_to_detections = {}
        for det in capture.detections:
            if det.num_tags >= min_tags:
                if det.object_id not in objects_to_detections:
                    objects_to_detections[det.object_id] = []
                objects_to_detections[det.object_id].append(det)

        # For each object seen by multiple cameras, create pairwise constraints
        for object_id, detections in objects_to_detections.items():
            if len(detections) < 2:
                continue

            # Create constraints for all pairs
            for i in range(len(detections)):
                for j in range(i + 1, len(detections)):
                    det_a = detections[i]
                    det_b = detections[j]

                    camera_a = robot_config.get_camera_by_id(det_a.camera_id)
                    camera_b = robot_config.get_camera_by_id(det_b.camera_id)

                    # Get link poses from capture
                    base_T_link_a = capture.link_poses[camera_a.parent_link]
                    base_T_link_b = capture.link_poses[camera_b.parent_link]

                    # Compute combined weight
                    weight = det_a.weight * det_b.weight

                    constraint = PairwiseConstraint(
                        camera_a_id=det_a.camera_id,
                        camera_b_id=det_b.camera_id,
                        object_id=object_id,
                        base_T_link_a=base_T_link_a,
                        base_T_link_b=base_T_link_b,
                        cam_a_T_object=det_a.cam_T_object,
                        cam_b_T_object=det_b.cam_T_object,
                        weight=weight,
                    )

                    constraints.append(constraint)

    return constraints


def compute_pairwise_residual(
    constraint: PairwiseConstraint,
    link_T_cam_a: np.ndarray,
    link_T_cam_b: np.ndarray,
    w_rot: float,
) -> np.ndarray:
    """Compute residual for a single pairwise constraint.

    The constraint enforces that both cameras should agree on the object's
    pose in the base frame.

    Args:
        constraint: The pairwise constraint
        link_T_cam_a: Transform from camera A's parent link to camera A
        link_T_cam_b: Transform from camera B's parent link to camera B
        w_rot: Weight for rotation error

    Returns:
        6-element residual vector: [weighted_trans_error (3), weighted_rot_error (3)]
    """
    # Compute base_T_object from camera A's view
    base_T_object_a = compose_transforms(
        constraint.base_T_link_a,
        link_T_cam_a,
        constraint.cam_a_T_object,
    )

    # Compute base_T_object from camera B's view
    base_T_object_b = compose_transforms(
        constraint.base_T_link_b,
        link_T_cam_b,
        constraint.cam_b_T_object,
    )

    # Compute translation error
    trans_error = get_translation(base_T_object_a) - get_translation(base_T_object_b)

    # Compute rotation error (as axis-angle)
    R_a = get_rotation(base_T_object_a)
    R_b = get_rotation(base_T_object_b)
    R_diff = R_a.T @ R_b
    rot_error = rotation_matrix_to_axis_angle(R_diff)

    # Apply weights
    weighted_trans = np.sqrt(constraint.weight) * trans_error
    weighted_rot = np.sqrt(constraint.weight * w_rot) * rot_error

    return np.concatenate([weighted_trans, weighted_rot])


def compute_residuals(
    corrections: np.ndarray,
    constraints: List[PairwiseConstraint],
    robot_config: RobotConfig,
    w_rot: float,
    lambda_reg: float,
) -> np.ndarray:
    """Compute all residuals for optimization.

    Args:
        corrections: Flattened array of all camera corrections (6 * num_cameras)
        constraints: List of pairwise constraints
        robot_config: Robot configuration
        w_rot: Weight for rotation error
        lambda_reg: Regularization weight

    Returns:
        Residual vector containing:
        - 6 residuals per pairwise constraint
        - 6 residuals per camera for regularization
    """
    num_cameras = len(robot_config.cameras)
    corrections_dict = {}

    # Reshape corrections into per-camera corrections
    for i, camera in enumerate(robot_config.cameras):
        correction = corrections[i * 6 : (i + 1) * 6]
        link_T_cam = apply_correction(camera.init_link_T_cam, correction)
        corrections_dict[camera.camera_id] = {
            "link_T_cam": link_T_cam,
            "correction": correction,
        }

    residuals = []

    # Compute pairwise residuals
    for constraint in constraints:
        link_T_cam_a = corrections_dict[constraint.camera_a_id]["link_T_cam"]
        link_T_cam_b = corrections_dict[constraint.camera_b_id]["link_T_cam"]

        residual = compute_pairwise_residual(
            constraint, link_T_cam_a, link_T_cam_b, w_rot
        )
        residuals.append(residual)

    # Compute regularization residuals
    reg_weight = np.sqrt(lambda_reg)
    for camera in robot_config.cameras:
        correction = corrections_dict[camera.camera_id]["correction"]
        # Weight translation and rotation components
        reg_residual = np.concatenate([
            reg_weight * correction[:3],  # rotation
            reg_weight * correction[3:],  # translation
        ])
        # Apply rotation weight to rotation component
        reg_residual[:3] *= np.sqrt(w_rot)
        residuals.append(reg_residual)

    return np.concatenate(residuals)


def compute_pairwise_errors(
    constraints: List[PairwiseConstraint],
    robot_config: RobotConfig,
    corrections: np.ndarray,
) -> Tuple[List[float], List[float]]:
    """Compute pairwise translation and rotation errors.

    Args:
        constraints: List of pairwise constraints
        robot_config: Robot configuration
        corrections: Flattened array of all camera corrections

    Returns:
        Tuple of (translation_errors, rotation_errors) in meters and radians
    """
    corrections_dict = {}
    for i, camera in enumerate(robot_config.cameras):
        correction = corrections[i * 6 : (i + 1) * 6]
        link_T_cam = apply_correction(camera.init_link_T_cam, correction)
        corrections_dict[camera.camera_id] = link_T_cam

    trans_errors = []
    rot_errors = []

    for constraint in constraints:
        link_T_cam_a = corrections_dict[constraint.camera_a_id]
        link_T_cam_b = corrections_dict[constraint.camera_b_id]

        # Compute base_T_object from both cameras
        base_T_object_a = compose_transforms(
            constraint.base_T_link_a,
            link_T_cam_a,
            constraint.cam_a_T_object,
        )

        base_T_object_b = compose_transforms(
            constraint.base_T_link_b,
            link_T_cam_b,
            constraint.cam_b_T_object,
        )

        # Translation error
        trans_error = np.linalg.norm(
            get_translation(base_T_object_a) - get_translation(base_T_object_b)
        )
        trans_errors.append(trans_error)

        # Rotation error
        R_a = get_rotation(base_T_object_a)
        R_b = get_rotation(base_T_object_b)
        R_diff = R_a.T @ R_b
        axis_angle = rotation_matrix_to_axis_angle(R_diff)
        rot_error = np.linalg.norm(axis_angle)
        rot_errors.append(rot_error)

    return trans_errors, rot_errors
