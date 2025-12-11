"""Cost function and residual computation for calibration optimization."""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from .data_structures import Capture, RobotConfig
from .transforms import apply_correction, compose_transforms, rotation_matrix_to_axis_angle


@dataclass
class PairwiseConstraint:
    """Pairwise constraint: two cameras viewing same object must agree on its base-frame pose."""
    camera_a_id: str
    camera_b_id: str
    object_id: int
    base_T_link_a: np.ndarray  # Base to camera A's parent link
    base_T_link_b: np.ndarray  # Base to camera B's parent link
    cam_a_T_object: np.ndarray  # Camera A to object
    cam_b_T_object: np.ndarray  # Camera B to object
    weight: float  # Combined detection weight


def generate_pairwise_constraints(
    captures: List[Capture],
    robot_config: RobotConfig,
    min_tags: int = 2,
) -> List[PairwiseConstraint]:
    """Generate pairwise constraints: for each object seen by 2+ cameras, enforce pose agreement."""
    constraints = []

    for capture in captures:
        # Group valid detections (≥ min_tags) by object
        objects_to_detections = {}
        for det in capture.detections:
            if det.num_tags >= min_tags:
                objects_to_detections.setdefault(det.object_id, []).append(det)

        # Create pairwise constraints for multi-view objects
        for detections in objects_to_detections.values():
            if len(detections) < 2:
                continue

            # All pairs of detections
            for i in range(len(detections)):
                for j in range(i + 1, len(detections)):
                    det_a, det_b = detections[i], detections[j]
                    cam_a = robot_config.get_camera_by_id(det_a.camera_id)
                    cam_b = robot_config.get_camera_by_id(det_b.camera_id)

                    constraints.append(PairwiseConstraint(
                        camera_a_id=det_a.camera_id,
                        camera_b_id=det_b.camera_id,
                        object_id=det_a.object_id,
                        base_T_link_a=capture.link_poses[cam_a.parent_link],
                        base_T_link_b=capture.link_poses[cam_b.parent_link],
                        cam_a_T_object=det_a.cam_T_object,
                        cam_b_T_object=det_b.cam_T_object,
                        weight=det_a.weight * det_b.weight,  # Combined weight
                    ))

    return constraints


def compute_pairwise_residual(
    constraint: PairwiseConstraint,
    link_T_cam_a: np.ndarray,
    link_T_cam_b: np.ndarray,
    w_rot: float,
) -> np.ndarray:
    """Compute 6-DOF residual enforcing base_T_object agreement between two cameras."""
    # Both cameras should compute same base_T_object
    base_T_object_a = compose_transforms(constraint.base_T_link_a, link_T_cam_a, constraint.cam_a_T_object)
    base_T_object_b = compose_transforms(constraint.base_T_link_b, link_T_cam_b, constraint.cam_b_T_object)

    # Translation error (3D)
    trans_error = base_T_object_a[:3, 3] - base_T_object_b[:3, 3]

    # Rotation error as axis-angle (3D)
    R_diff = base_T_object_a[:3, :3].T @ base_T_object_b[:3, :3]
    rot_error = rotation_matrix_to_axis_angle(R_diff)

    # Apply weights: sqrt(weight) for residuals (squared in cost)
    return np.concatenate([
        np.sqrt(constraint.weight) * trans_error,
        np.sqrt(constraint.weight * w_rot) * rot_error,
    ])


def compute_residuals(
    corrections: np.ndarray,
    constraints: List[PairwiseConstraint],
    robot_config: RobotConfig,
    w_rot: float,
    lambda_reg: float,
) -> np.ndarray:
    """Compute residuals: pairwise errors + regularization penalties."""
    # Build camera transforms from corrections
    cameras_T = {}
    for i, camera in enumerate(robot_config.cameras):
        correction = corrections[i * 6 : (i + 1) * 6]
        cameras_T[camera.camera_id] = {
            'T': apply_correction(camera.init_link_T_cam, correction),
            'corr': correction,
        }

    residuals = []

    # Pairwise constraint residuals (6 per constraint)
    for c in constraints:
        residuals.append(compute_pairwise_residual(
            c, cameras_T[c.camera_a_id]['T'], cameras_T[c.camera_b_id]['T'], w_rot
        ))

    # Regularization residuals (6 per camera): penalize large corrections
    reg_sqrt = np.sqrt(lambda_reg)
    for camera in robot_config.cameras:
        corr = cameras_T[camera.camera_id]['corr']
        # Weight rotation and translation separately
        residuals.append(np.concatenate([
            reg_sqrt * np.sqrt(w_rot) * corr[:3],  # Rotation
            reg_sqrt * corr[3:],  # Translation
        ]))

    return np.concatenate(residuals)


def compute_pairwise_errors(
    constraints: List[PairwiseConstraint],
    robot_config: RobotConfig,
    corrections: np.ndarray,
) -> Tuple[List[float], List[float]]:
    """Compute unweighted translation (m) and rotation (rad) errors for validation."""
    # Build camera transforms
    cameras_T = {}
    for i, camera in enumerate(robot_config.cameras):
        correction = corrections[i * 6 : (i + 1) * 6]
        cameras_T[camera.camera_id] = apply_correction(camera.init_link_T_cam, correction)

    trans_errors, rot_errors = [], []

    for c in constraints:
        # Compute base_T_object from both cameras
        T_a = compose_transforms(c.base_T_link_a, cameras_T[c.camera_a_id], c.cam_a_T_object)
        T_b = compose_transforms(c.base_T_link_b, cameras_T[c.camera_b_id], c.cam_b_T_object)

        # Translation error
        trans_errors.append(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))

        # Rotation error
        rot_errors.append(np.linalg.norm(rotation_matrix_to_axis_angle(T_a[:3, :3].T @ T_b[:3, :3])))

    return trans_errors, rot_errors
