"""Tests for transform utilities."""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multi_camera_calibration.transforms import (
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_axis_angle,
    correction_to_transform,
    transform_to_correction,
    compose_transforms,
    invert_transform,
    translation_error,
    rotation_error,
)


def test_axis_angle_round_trip():
    """Test axis-angle to rotation matrix and back."""
    print("Testing axis-angle conversions...")

    # Test various rotations
    test_cases = [
        np.array([0, 0, 0]),  # Identity
        np.array([0.1, 0, 0]),  # Small rotation around X
        np.array([0, 0.5, 0]),  # Medium rotation around Y
        np.array([0, 0, 1.0]),  # Large rotation around Z
        np.array([0.3, 0.4, 0.5]),  # General rotation
        np.array([np.pi - 0.1, 0, 0]),  # Near 180 degrees
    ]

    for axis_angle in test_cases:
        R = axis_angle_to_rotation_matrix(axis_angle)

        # Check it's a valid rotation matrix
        assert np.allclose(R @ R.T, np.eye(3)), "R should be orthogonal"
        assert np.allclose(np.linalg.det(R), 1.0), "det(R) should be 1"

        # Round trip
        axis_angle_recovered = rotation_matrix_to_axis_angle(R)

        # Should recover same rotation (may differ by sign for 180° rotations)
        R_recovered = axis_angle_to_rotation_matrix(axis_angle_recovered)
        assert np.allclose(R, R_recovered), f"Round trip failed for {axis_angle}"

    print("  ✓ Axis-angle conversions passed")


def test_correction_round_trip():
    """Test 6-DOF correction to transform and back."""
    print("Testing correction conversions...")

    test_cases = [
        np.array([0, 0, 0, 0, 0, 0]),  # Identity
        np.array([0.1, 0, 0, 0.01, 0, 0]),  # Small rotation + translation
        np.array([0.3, 0.4, 0.5, 0.1, 0.2, 0.3]),  # General case
    ]

    for correction in test_cases:
        T = correction_to_transform(correction)

        # Check it's a valid SE3 matrix
        assert T.shape == (4, 4)
        assert np.allclose(T[3, :], [0, 0, 0, 1])

        # Round trip
        correction_recovered = transform_to_correction(T)
        assert np.allclose(correction, correction_recovered, atol=1e-6)

    print("  ✓ Correction conversions passed")


def test_compose_and_invert():
    """Test transform composition and inversion."""
    print("Testing composition and inversion...")

    # Create two random transforms
    T1 = correction_to_transform(np.array([0.1, 0.2, 0.3, 0.5, 0.6, 0.7]))
    T2 = correction_to_transform(np.array([0.4, 0.5, 0.6, 0.1, 0.2, 0.3]))

    # Compose
    T12 = compose_transforms(T1, T2)

    # Verify associativity
    T3 = correction_to_transform(np.array([0.2, 0.1, 0.4, 0.3, 0.4, 0.5]))
    T123_a = compose_transforms(compose_transforms(T1, T2), T3)
    T123_b = compose_transforms(T1, compose_transforms(T2, T3))
    assert np.allclose(T123_a, T123_b), "Composition should be associative"

    # Verify inversion
    T1_inv = invert_transform(T1)
    identity = compose_transforms(T1, T1_inv)
    assert np.allclose(identity, np.eye(4), atol=1e-10), "T · T^-1 should be identity"

    print("  ✓ Composition and inversion passed")


def test_error_metrics():
    """Test translation and rotation error functions."""
    print("Testing error metrics...")

    T1 = np.eye(4)
    T1[:3, 3] = [1, 2, 3]

    T2 = np.eye(4)
    T2[:3, 3] = [1.1, 2.0, 3.0]

    # Translation error should be 0.1
    trans_err = translation_error(T1, T2)
    assert np.allclose(trans_err, 0.1), f"Expected 0.1, got {trans_err}"

    # Rotation error between identical rotations should be 0
    rot_err = rotation_error(T1, T2)
    assert np.allclose(rot_err, 0, atol=1e-10), f"Expected 0, got {rot_err}"

    # Test rotation error
    T3 = correction_to_transform(np.array([0.1, 0, 0, 0, 0, 0]))  # 0.1 rad around X
    rot_err = rotation_error(np.eye(4), T3)
    assert np.allclose(rot_err, 0.1, atol=1e-6), f"Expected 0.1, got {rot_err}"

    print("  ✓ Error metrics passed")


def run_all_tests():
    """Run all transform tests."""
    print("\n" + "=" * 60)
    print("Running Transform Tests")
    print("=" * 60)

    test_axis_angle_round_trip()
    test_correction_round_trip()
    test_compose_and_invert()
    test_error_metrics()

    print("\n✓ All transform tests passed!\n")


if __name__ == "__main__":
    run_all_tests()
