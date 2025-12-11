"""SE3 transform operations and axis-angle conversions for camera calibration."""

import numpy as np


def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle to rotation matrix via Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²"""
    theta = np.linalg.norm(axis_angle)
    if theta < 1e-10:
        return np.eye(3)

    k = axis_angle / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])  # Skew-symmetric
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to axis-angle representation."""
    if np.allclose(R, np.eye(3)):
        return np.zeros(3)

    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-10:
        return np.zeros(3)

    # Special case: near 180° rotation
    if np.abs(theta - np.pi) < 1e-6:
        k_idx = np.argmax(np.diag(R))  # Largest diagonal element
        axis = np.zeros(3)
        axis[k_idx] = np.sqrt((R[k_idx, k_idx] + 1) / 2)
        # Fill remaining elements
        for i in range(3):
            if i != k_idx:
                axis[i] = R[k_idx, i] / (2 * axis[k_idx])
        return theta * axis

    # General case: extract axis from skew-symmetric part
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
    return theta * axis


def correction_to_transform(correction: np.ndarray) -> np.ndarray:
    """Convert 6-DOF [rx,ry,rz,tx,ty,tz] to 4x4 SE3 matrix."""
    T = np.eye(4)
    T[:3, :3] = axis_angle_to_rotation_matrix(correction[:3])
    T[:3, 3] = correction[3:]
    return T


def transform_to_correction(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 SE3 matrix to 6-DOF [rx,ry,rz,tx,ty,tz]."""
    return np.concatenate([rotation_matrix_to_axis_angle(T[:3, :3]), T[:3, 3]])


def compose_transforms(*transforms: np.ndarray) -> np.ndarray:
    """Compose SE3 transforms left-to-right: T1 @ T2 @ T3 @ ..."""
    result = np.eye(4)
    for T in transforms:
        result = result @ T
    return result


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert SE3 transform: T^-1 = [R' | -R't; 0 | 1]"""
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T  # R' (transpose)
    T_inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]  # -R't
    return T_inv


def apply_correction(init_transform: np.ndarray, correction: np.ndarray) -> np.ndarray:
    """Apply correction: T_new = T_init @ correction_matrix"""
    return init_transform @ correction_to_transform(correction)


def translation_error(T1: np.ndarray, T2: np.ndarray) -> float:
    """Euclidean distance between translations: ||t1 - t2||"""
    return np.linalg.norm(T1[:3, 3] - T2[:3, 3])


def rotation_error(T1: np.ndarray, T2: np.ndarray) -> float:
    """Angular difference between rotations in radians."""
    return np.linalg.norm(rotation_matrix_to_axis_angle(T1[:3, :3].T @ T2[:3, :3]))
