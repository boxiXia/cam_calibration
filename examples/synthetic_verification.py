"""Synthetic verification of calibration algorithm.

This script generates synthetic data with known ground truth and verifies
that the calibration algorithm can recover the true camera transforms.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multi_camera_calibration import (
    Calibrator,
    SimulationFramework,
)


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_no_noise():
    """Test with perfect (no noise) data - should recover exact transforms."""
    print_section("Test 1: No Noise (Perfect Data)")

    # Create simulation with 4 cameras
    sim, robot_config, gt_transforms = SimulationFramework.create_from_scratch(
        camera_ids=["cam1", "cam2", "cam3", "cam4"],
        parent_links=["link1", "link2", "link3", "link4"],
        perturbation_translation_m=0.020,  # 20mm
        perturbation_rotation_deg=3.0,  # 3 degrees
        seed=42,
    )

    # Zero noise
    sim.noise_translation_m = 0.0
    sim.noise_rotation_rad = 0.0

    print(f"Cameras: {len(robot_config.cameras)}")

    # Generate captures
    captures = sim.generate_dataset(num_captures=15, num_objects_per_capture=3)
    print(f"Generated {len(captures)} captures")

    # Run calibration with fine-tuning for perfect data
    calibrator = Calibrator(robot_config, w_rot=0.25, lambda_reg=1.0)
    for capture in captures:
        calibrator.add_capture(capture.link_poses, capture.detections)

    # Use fine-tuning to achieve near-perfect recovery with noiseless data
    optimized = calibrator.optimize(verbose=False, fine_tune=True, fine_tune_lambda=0.001)

    # Validate
    validation = calibrator.validate()
    print(f"\nValidation (training data):")
    print(f"  Mean error: {validation.mean_translation_error_mm:.4f}mm, "
          f"{validation.mean_rotation_error_deg:.4f}°")
    print(f"  Max error: {validation.max_translation_error_mm:.4f}mm, "
          f"{validation.max_rotation_error_deg:.4f}°")

    # Compare to ground truth
    gt_errors = sim.compute_ground_truth_errors(optimized)
    print(f"\nGround truth errors:")
    for cam_id, (trans_err, rot_err) in gt_errors.items():
        print(f"  {cam_id}: {trans_err:.4f}mm, {rot_err:.4f}°")

    # Check success
    max_trans_err = max(trans_err for trans_err, _ in gt_errors.values())
    max_rot_err = max(rot_err for _, rot_err in gt_errors.values())

    # With fine-tuning and no noise, should achieve near-perfect recovery
    success = max_trans_err < 0.1 and max_rot_err < 0.01  # < 0.1mm and 0.01°
    print(f"\n{'✓ PASS' if success else '✗ FAIL'}: Max error {max_trans_err:.4f}mm, "
          f"{max_rot_err:.4f}° (with fine-tuning)")

    return success


def test_realistic_noise():
    """Test with realistic noise levels."""
    print_section("Test 2: Realistic Noise (2mm, 1°)")

    sim, robot_config, gt_transforms = SimulationFramework.create_from_scratch(
        camera_ids=["cam1", "cam2", "cam3", "cam4"],
        parent_links=["link1", "link2", "link3", "link4"],
        perturbation_translation_m=0.020,
        perturbation_rotation_deg=3.0,
        seed=43,
    )

    # Realistic noise: 2mm translation, 1° rotation
    sim.noise_translation_m = 0.002
    sim.noise_rotation_rad = np.deg2rad(1.0)

    print(f"Cameras: {len(robot_config.cameras)}")
    print(f"Noise: {sim.noise_translation_m*1000:.1f}mm, "
          f"{np.rad2deg(sim.noise_rotation_rad):.1f}°")

    # Generate more captures to handle noise
    captures = sim.generate_dataset(num_captures=30, num_objects_per_capture=3)
    print(f"Generated {len(captures)} captures")

    # Run calibration
    calibrator = Calibrator(robot_config, w_rot=0.25, lambda_reg=1.0)
    for capture in captures:
        calibrator.add_capture(capture.link_poses, capture.detections)

    optimized = calibrator.optimize(verbose=False)

    # Validate
    validation = calibrator.validate()
    print(f"\nValidation (training data):")
    print(f"  Mean error: {validation.mean_translation_error_mm:.2f}mm, "
          f"{validation.mean_rotation_error_deg:.2f}°")
    print(f"  Max error: {validation.max_translation_error_mm:.2f}mm, "
          f"{validation.max_rotation_error_deg:.2f}°")
    print(f"  Sanity check: {'PASS' if validation.passed_sanity_check else 'FAIL'}")

    # Compare to ground truth
    gt_errors = sim.compute_ground_truth_errors(optimized)
    print(f"\nGround truth errors:")
    for cam_id, (trans_err, rot_err) in gt_errors.items():
        print(f"  {cam_id}: {trans_err:.2f}mm, {rot_err:.2f}°")

    # Check success (should be within noise level)
    max_trans_err = max(trans_err for trans_err, _ in gt_errors.values())
    max_rot_err = max(rot_err for _, rot_err in gt_errors.values())

    success = max_trans_err < 5.0 and max_rot_err < 2.0  # < 5mm, 2°
    print(f"\n{'✓ PASS' if success else '✗ FAIL'}: Max error {max_trans_err:.2f}mm, "
          f"{max_rot_err:.2f}°")

    return success


def test_generalization():
    """Test generalization to held-out test data."""
    print_section("Test 3: Generalization (Train/Test Split)")

    sim, robot_config, gt_transforms = SimulationFramework.create_from_scratch(
        camera_ids=["cam1", "cam2", "cam3", "cam4"],
        parent_links=["link1", "link2", "link3", "link4"],
        perturbation_translation_m=0.020,
        perturbation_rotation_deg=3.0,
        seed=44,
    )

    sim.noise_translation_m = 0.002
    sim.noise_rotation_rad = np.deg2rad(1.0)

    # Generate train and test splits
    train_captures = sim.generate_dataset(num_captures=25, num_objects_per_capture=3)
    test_captures = sim.generate_dataset(num_captures=10, num_objects_per_capture=3)

    print(f"Training captures: {len(train_captures)}")
    print(f"Test captures: {len(test_captures)}")

    # Run calibration on training data
    calibrator = Calibrator(robot_config, w_rot=0.25, lambda_reg=1.0)
    for capture in train_captures:
        calibrator.add_capture(capture.link_poses, capture.detections)

    optimized = calibrator.optimize(verbose=False)

    # Validate on training data
    train_validation = calibrator.validate()
    print(f"\nTraining validation:")
    print(f"  Mean error: {train_validation.mean_translation_error_mm:.2f}mm, "
          f"{train_validation.mean_rotation_error_deg:.2f}°")

    # Validate on test data
    test_validation = calibrator.validate(test_captures=test_captures)
    print(f"\nTest validation:")
    print(f"  Mean error: {test_validation.mean_translation_error_mm:.2f}mm, "
          f"{test_validation.mean_rotation_error_deg:.2f}°")

    # Ground truth
    gt_errors = sim.compute_ground_truth_errors(optimized)
    max_trans_err = max(trans_err for trans_err, _ in gt_errors.values())
    max_rot_err = max(rot_err for _, rot_err in gt_errors.values())

    print(f"\nGround truth max error: {max_trans_err:.2f}mm, {max_rot_err:.2f}°")

    # Success if test error is similar to training error
    error_diff = abs(test_validation.mean_translation_error_mm -
                     train_validation.mean_translation_error_mm)
    success = error_diff < 2.0  # Test and train errors within 2mm

    print(f"\n{'✓ PASS' if success else '✗ FAIL'}: Train/test error difference "
          f"{error_diff:.2f}mm")

    return success


def test_large_initial_error():
    """Test convergence with large initial errors."""
    print_section("Test 4: Large Initial Error (50mm, 10°)")

    sim, robot_config, gt_transforms = SimulationFramework.create_from_scratch(
        camera_ids=["cam1", "cam2", "cam3", "cam4"],
        parent_links=["link1", "link2", "link3", "link4"],
        perturbation_translation_m=0.050,  # 50mm - large!
        perturbation_rotation_deg=10.0,  # 10° - large!
        seed=45,
    )

    sim.noise_translation_m = 0.002
    sim.noise_rotation_rad = np.deg2rad(1.0)

    print(f"Initial perturbation: 50mm, 10°")

    # Generate captures
    captures = sim.generate_dataset(num_captures=40, num_objects_per_capture=3)
    print(f"Generated {len(captures)} captures")

    # Check initial errors
    initial_errors = sim.compute_ground_truth_errors(
        {cam.camera_id: cam.init_link_T_cam for cam in robot_config.cameras}
    )
    print(f"\nInitial errors (before calibration):")
    for cam_id, (trans_err, rot_err) in initial_errors.items():
        print(f"  {cam_id}: {trans_err:.2f}mm, {rot_err:.2f}°")

    # Run calibration
    calibrator = Calibrator(robot_config, w_rot=0.25, lambda_reg=1.0)
    for capture in captures:
        calibrator.add_capture(capture.link_poses, capture.detections)

    optimized = calibrator.optimize(verbose=False)

    # Check final errors
    final_errors = sim.compute_ground_truth_errors(optimized)
    print(f"\nFinal errors (after calibration):")
    for cam_id, (trans_err, rot_err) in final_errors.items():
        print(f"  {cam_id}: {trans_err:.2f}mm, {rot_err:.2f}°")

    # Compute improvement
    initial_trans = np.mean([t for t, _ in initial_errors.values()])
    final_trans = np.mean([t for t, _ in final_errors.values()])
    improvement = initial_trans - final_trans

    print(f"\nImprovement: {improvement:.2f}mm ({improvement/initial_trans*100:.1f}%)")

    success = final_trans < 5.0  # Final error < 5mm
    print(f"\n{'✓ PASS' if success else '✗ FAIL'}: Final error {final_trans:.2f}mm")

    return success


def main():
    """Run all verification tests."""
    print("\n" + "█" * 70)
    print("  MULTI-CAMERA CALIBRATION - SYNTHETIC VERIFICATION")
    print("█" * 70)

    results = []

    # Run all tests
    results.append(("No Noise", test_no_noise()))
    results.append(("Realistic Noise", test_realistic_noise()))
    results.append(("Generalization", test_generalization()))
    results.append(("Large Initial Error", test_large_initial_error()))

    # Summary
    print_section("SUMMARY")
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    all_passed = all(passed for _, passed in results)
    print(f"\n{'=' * 70}")
    if all_passed:
        print("  ✓ ALL TESTS PASSED")
    else:
        print("  ✗ SOME TESTS FAILED")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
