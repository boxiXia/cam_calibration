"""Unitree G1 calibration example using URDF-based simulation.

This example shows how to use the actual robot URDF model to create
realistic ground truth camera transforms for verification.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multi_camera_calibration import Calibrator, SimulationFramework


def unitree_g1_camera_specs():
    """Define Unitree G1 camera specifications.

    These are realistic camera positions based on the actual G1 URDF:
    - Chest camera: Forward-facing on torso_link
    - Head camera: Forward-facing on head_link
    - Left hand camera: Downward-facing on left_wrist_yaw_link
    - Right hand camera: Downward-facing on right_wrist_yaw_link
    """
    return [
        {
            "camera_id": "chest",
            "parent_link": "torso_link",
            "xyz": [0.15, 0.0, 0.20],  # 15cm forward, 20cm up from torso center
            "rpy": [0, 0, 0],  # Looking forward
        },
        {
            "camera_id": "head",
            "parent_link": "head_link",
            "xyz": [0.08, 0.0, 0.03],  # 8cm forward, 3cm up from head center
            "rpy": [0, 0, 0],  # Looking forward
        },
        {
            "camera_id": "hand_L",
            "parent_link": "left_wrist_yaw_link",
            "xyz": [0.05, 0.0, -0.02],  # 5cm forward, 2cm down from wrist
            "rpy": [0, np.deg2rad(45), 0],  # Tilted down 45°
        },
        {
            "camera_id": "hand_R",
            "parent_link": "right_wrist_yaw_link",
            "xyz": [0.05, 0.0, -0.02],  # 5cm forward, 2cm down from wrist
            "rpy": [0, np.deg2rad(45), 0],  # Tilted down 45°
        },
    ]


def run_urdf_based_verification(urdf_path: str):
    """Run calibration verification using URDF-based ground truth.

    Args:
        urdf_path: Path to G1 URDF file (required)
    """
    print("=" * 70)
    print("  Unitree G1 URDF-Based Calibration Verification")
    print("=" * 70)

    # Camera specifications
    camera_specs = unitree_g1_camera_specs()

    print(f"\nCamera configuration:")
    for spec in camera_specs:
        print(f"  {spec['camera_id']:10s} on {spec['parent_link']:20s}")
        xyz_str = f"xyz=[{spec['xyz'][0]:.2f}, {spec['xyz'][1]:.2f}, {spec['xyz'][2]:.2f}]"
        rpy_deg = np.rad2deg(spec['rpy'])
        rpy_str = f"rpy=[{rpy_deg[0]:.1f}°, {rpy_deg[1]:.1f}°, {rpy_deg[2]:.1f}°]"
        print(f"              {xyz_str}, {rpy_str}")

    # Create simulation using actual URDF
    print(f"\nUsing URDF: {urdf_path}")
    print("Creating simulation framework with forward kinematics...")

    sim, robot_config, gt_transforms = SimulationFramework.create_from_urdf(
        urdf_path=urdf_path,
        camera_specs=camera_specs,
        perturbation_translation_m=0.020,  # 20mm
        perturbation_rotation_deg=3.0,  # 3°
        seed=42,
    )

    print(f"Created {len(robot_config.cameras)} cameras")
    print(f"URDF contains {len(sim.urdf_model.link_map)} links, {len(sim.urdf_model.joint_map)} joints")

    # Generate synthetic data using URDF forward kinematics
    print("\nGenerating synthetic calibration data using URDF FK...")
    print("  Each capture uses random joint configuration from URDF")
    sim.noise_translation_m = 0.002  # 2mm
    sim.noise_rotation_rad = np.deg2rad(1.0)  # 1°

    captures = sim.generate_dataset(num_captures=30, num_objects_per_capture=3)
    print(f"Generated {len(captures)} captures with realistic robot poses")

    # Run calibration
    print("\nRunning calibration...")
    calibrator = Calibrator(robot_config, w_rot=0.25, lambda_reg=1.0)

    for capture in captures:
        calibrator.add_capture(capture.link_poses, capture.detections)

    # Optimize with fine-tuning
    results = calibrator.optimize(verbose=False, fine_tune=True, fine_tune_lambda=0.001)

    # Validate
    print("\nValidation results:")
    validation = calibrator.validate()
    print(f"  Mean error: {validation.mean_translation_error_mm:.2f}mm, "
          f"{validation.mean_rotation_error_deg:.2f}°")
    print(f"  Max error: {validation.max_translation_error_mm:.2f}mm, "
          f"{validation.max_rotation_error_deg:.2f}°")
    print(f"  Sanity check: {'PASS' if validation.passed_sanity_check else 'FAIL'}")

    # Compare to ground truth
    print("\nGround truth comparison:")
    gt_errors = sim.compute_ground_truth_errors(results)

    max_trans_err = 0
    max_rot_err = 0

    for cam_id, (trans_err, rot_err) in gt_errors.items():
        print(f"  {cam_id:10s}: {trans_err:.3f}mm, {rot_err:.3f}°")
        max_trans_err = max(max_trans_err, trans_err)
        max_rot_err = max(max_rot_err, rot_err)

    # Display ground truth transforms
    print("\nGround truth camera transforms:")
    for cam_id in ["chest", "head", "hand_L", "hand_R"]:
        T = gt_transforms[cam_id]
        trans = T[:3, 3]
        print(f"\n  {cam_id}:")
        print(f"    Translation: [{trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f}]")
        print(f"    Rotation matrix:")
        for row in T[:3, :3]:
            print(f"      [{row[0]:7.4f}, {row[1]:7.4f}, {row[2]:7.4f}]")

    # Display optimized transforms
    print("\nOptimized camera transforms:")
    for cam_id in ["chest", "head", "hand_L", "hand_R"]:
        T = results[cam_id]
        trans = T[:3, 3]
        print(f"\n  {cam_id}:")
        print(f"    Translation: [{trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f}]")

    # Summary
    print("\n" + "=" * 70)
    print(f"  Maximum error: {max_trans_err:.3f}mm, {max_rot_err:.3f}°")
    print(f"  URDF-based FK: Using actual robot kinematics for link poses")

    # More realistic threshold for URDF-based simulation
    success = max_trans_err < 2.0 and max_rot_err < 0.3
    print(f"  Status: {'PASS' if success else 'FAIL'}")
    print("=" * 70)

    return success


def main():
    """Run Unitree G1 URDF-based verification example."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, '..', 'urdf', 'g1_dual_arm.urdf')

    if not os.path.exists(urdf_path):
        print(f"ERROR: URDF file not found at {urdf_path}")
        print("\nDownload with:")
        print("  mkdir -p urdf")
        print("  curl -sL https://raw.githubusercontent.com/unitreerobotics/"
              "unitree_ros/master/robots/g1_description/g1_dual_arm.urdf "
              "-o urdf/g1_dual_arm.urdf")
        return False

    return run_urdf_based_verification(urdf_path)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
