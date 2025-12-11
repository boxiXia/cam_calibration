"""Transform utilities for SE3 operations and axis-angle conversions."""

import numpy as np
from typing import Tuple


def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle representation to rotation matrix using Rodrigues' formula.

    Args:
        axis_angle: 3-element array [rx, ry, rz] representing rotation

    Returns:
        3x3 rotation matrix
    """
    theta = np.linalg.norm(axis_angle)

    if theta < 1e-10:
        # Small angle approximation
        return np.eye(3)

    k = axis_angle / theta
    K = skew_symmetric(k)

    # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

    return R


def rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to axis-angle representation.

    Args:
        R: 3x3 rotation matrix

    Returns:
        3-element array [rx, ry, rz] representing rotation
    """
    # Handle identity matrix
    if np.allclose(R, np.eye(3)):
        return np.zeros(3)

    # Compute angle
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

    if theta < 1e-10:
        return np.zeros(3)

    # Handle angle close to π (180 degrees)
    if np.abs(theta - np.pi) < 1e-6:
        # Find the column with largest diagonal element
        diag = np.diag(R)
        k_idx = np.argmax(diag)
        k = np.sqrt((R[k_idx, k_idx] + 1) / 2)
        axis = np.zeros(3)
        axis[k_idx] = k

        # Fill in other elements
        for i in range(3):
            if i != k_idx:
                axis[i] = R[k_idx, i] / (2 * k)

        return theta * axis

    # General case
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2 * np.sin(theta))

    return theta * axis


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from 3D vector.

    Args:
        v: 3-element vector

    Returns:
        3x3 skew-symmetric matrix
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def correction_to_transform(correction: np.ndarray) -> np.ndarray:
    """Convert 6-DOF correction vector to SE3 transform.

    Args:
        correction: 6-element array [rx, ry, rz, tx, ty, tz]

    Returns:
        4x4 SE3 transform matrix
    """
    axis_angle = correction[:3]
    translation = correction[3:]

    T = np.eye(4)
    T[:3, :3] = axis_angle_to_rotation_matrix(axis_angle)
    T[:3, 3] = translation

    return T


def transform_to_correction(T: np.ndarray) -> np.ndarray:
    """Convert SE3 transform to 6-DOF correction vector.

    Args:
        T: 4x4 SE3 transform matrix

    Returns:
        6-element array [rx, ry, rz, tx, ty, tz]
    """
    axis_angle = rotation_matrix_to_axis_angle(T[:3, :3])
    translation = T[:3, 3]

    return np.concatenate([axis_angle, translation])


def compose_transforms(*transforms: np.ndarray) -> np.ndarray:
    """Compose multiple SE3 transforms (left to right).

    Args:
        transforms: Variable number of 4x4 SE3 matrices

    Returns:
        4x4 SE3 matrix representing the composition
    """
    result = np.eye(4)
    for T in transforms:
        result = result @ T
    return result


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert an SE3 transform.

    Args:
        T: 4x4 SE3 transform matrix

    Returns:
        4x4 SE3 inverse transform
    """
    T_inv = np.eye(4)
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv


def translation_error(T1: np.ndarray, T2: np.ndarray) -> float:
    """Compute Euclidean distance between translations of two transforms.

    Args:
        T1: 4x4 SE3 transform matrix
        T2: 4x4 SE3 transform matrix

    Returns:
        Euclidean distance between translations
    """
    return np.linalg.norm(T1[:3, 3] - T2[:3, 3])


def rotation_error(T1: np.ndarray, T2: np.ndarray) -> float:
    """Compute angular difference between rotations of two transforms.

    Args:
        T1: 4x4 SE3 transform matrix
        T2: 4x4 SE3 transform matrix

    Returns:
        Angular difference in radians
    """
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]

    R_diff = R1.T @ R2
    axis_angle = rotation_matrix_to_axis_angle(R_diff)

    return np.linalg.norm(axis_angle)


def apply_correction(init_transform: np.ndarray, correction: np.ndarray) -> np.ndarray:
    """Apply a 6-DOF correction to an initial transform.

    The correction is applied as: T_new = T_init · correction

    Args:
        init_transform: Initial 4x4 SE3 transform
        correction: 6-element correction [rx, ry, rz, tx, ty, tz]

    Returns:
        Corrected 4x4 SE3 transform
    """
    correction_T = correction_to_transform(correction)
    return compose_transforms(init_transform, correction_T)


def get_translation(T: np.ndarray) -> np.ndarray:
    """Extract translation vector from SE3 transform.

    Args:
        T: 4x4 SE3 transform matrix

    Returns:
        3-element translation vector
    """
    return T[:3, 3]


def get_rotation(T: np.ndarray) -> np.ndarray:
    """Extract rotation matrix from SE3 transform.

    Args:
        T: 4x4 SE3 transform matrix

    Returns:
        3x3 rotation matrix
    """
    return T[:3, :3]


def rotation_angle_between(R1: np.ndarray, R2: np.ndarray) -> float:
    """Compute the rotation angle between two rotation matrices.

    Args:
        R1: 3x3 rotation matrix
        R2: 3x3 rotation matrix

    Returns:
        Angle in radians
    """
    R_diff = R1.T @ R2
    # Ensure trace is in valid range for arccos
    trace = np.trace(R_diff)
    theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    return theta
