"""Example calibration setup for Unitree G1 humanoid robot.

This demonstrates how to configure the calibration system for a real robot
with multiple cameras (chest, head, and two hand cameras).
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multi_camera_calibration import (
    CameraConfig,
    RobotConfig,
    Calibrator,
    Detection,
)


def create_unitree_g1_config():
    """Create robot configuration for Unitree G1.

    The G1 has 4 cameras:
    - Chest camera (RealSense D435i) on torso_link
    - Head camera (RealSense D435i) on head_link
    - Left hand camera (RealSense D405) on left_wrist_link
    - Right hand camera (RealSense D405) on right_wrist_link
    """
    # Initial transforms from URDF (example values - replace with actual URDF)
    # These are rough estimates that will be refined by calibration

    # Chest camera - forward-facing on torso
    chest_T_cam = np.array([
        [0, 0, 1, 0.15],   # Looking forward (X), 15cm in front
        [-1, 0, 0, 0.0],   # Y to the left
        [0, -1, 0, 0.30],  # Z up, 30cm above torso origin
        [0, 0, 0, 1],
    ])

    # Head camera - forward-facing on head
    head_T_cam = np.array([
        [0, 0, 1, 0.10],   # Looking forward, 10cm in front
        [-1, 0, 0, 0.0],   # Y to the left
        [0, -1, 0, 0.05],  # Z up, 5cm above head origin
        [0, 0, 0, 1],
    ])

    # Left hand camera - looking down and forward
    left_hand_T_cam = np.array([
        [0.707, 0, 0.707, 0.05],   # Forward and down
        [0, 1, 0, 0.02],            # Y to the left
        [-0.707, 0, 0.707, 0.0],   # Mix of forward and down
        [0, 0, 0, 1],
    ])

    # Right hand camera - looking down and forward
    right_hand_T_cam = np.array([
        [0.707, 0, 0.707, 0.05],   # Forward and down
        [0, 1, 0, -0.02],           # Y to the right (negative)
        [-0.707, 0, 0.707, 0.0],   # Mix of forward and down
        [0, 0, 0, 1],
    ])

    cameras = [
        CameraConfig(
            camera_id="chest",
            parent_link="torso_link",
            init_link_T_cam=chest_T_cam,
        ),
        CameraConfig(
            camera_id="head",
            parent_link="head_link",
            init_link_T_cam=head_T_cam,
        ),
        CameraConfig(
            camera_id="hand_L",
            parent_link="left_wrist_link",
            init_link_T_cam=left_hand_T_cam,
        ),
        CameraConfig(
            camera_id="hand_R",
            parent_link="right_wrist_link",
            init_link_T_cam=right_hand_T_cam,
        ),
    ]

    return RobotConfig(cameras=cameras)


def example_calibration_workflow():
    """Demonstrate the calibration workflow for Unitree G1."""
    print("=" * 70)
    print("  Unitree G1 Multi-Camera Calibration Example")
    print("=" * 70)

    # 1. Create robot configuration
    print("\n1. Creating robot configuration...")
    robot_config = create_unitree_g1_config()
    print(f"   Configured {len(robot_config.cameras)} cameras:")
    for cam in robot_config.cameras:
        print(f"     - {cam.camera_id} on {cam.parent_link}")

    # 2. Create calibrator
    print("\n2. Creating calibrator...")
    # w_rot = 0.25 assumes ~0.5m working distance
    # Adjust based on actual calibration setup
    calibrator = Calibrator(
        robot_config,
        w_rot=0.25,  # d² where d=0.5m
        lambda_reg=1.0,
        min_tags=2,
    )
    print("   Calibrator initialized")

    # 3. Data collection workflow (pseudocode - requires real hardware)
    print("\n3. Data collection workflow:")
    print("   ┌─────────────────────────────────────────────────────┐")
    print("   │ For each calibration pose:                          │")
    print("   │                                                      │")
    print("   │  a) Position robot so 2+ cameras see calibration    │")
    print("   │     object (AprilTag board with 6 tags)             │")
    print("   │                                                      │")
    print("   │  b) Get link poses from robot SDK:                  │")
    print("   │     link_poses = {                                  │")
    print("   │         'torso_link': base_T_torso,                 │")
    print("   │         'head_link': base_T_head,                   │")
    print("   │         'left_wrist_link': base_T_left_wrist,       │")
    print("   │         'right_wrist_link': base_T_right_wrist,     │")
    print("   │     }                                                │")
    print("   │                                                      │")
    print("   │  c) Trigger all cameras (synchronized)              │")
    print("   │                                                      │")
    print("   │  d) Run AprilTag detection on each image            │")
    print("   │                                                      │")
    print("   │  e) Create Detection objects:                       │")
    print("   │     detections = [                                  │")
    print("   │         Detection(                                  │")
    print("   │             camera_id='chest',                      │")
    print("   │             object_id=0,                            │")
    print("   │             cam_T_object=computed_transform,        │")
    print("   │             num_tags=6                              │")
    print("   │         ),                                          │")
    print("   │         ...                                         │")
    print("   │     ]                                                │")
    print("   │                                                      │")
    print("   │  f) Add capture:                                    │")
    print("   │     calibrator.add_capture(link_poses, detections)  │")
    print("   │                                                      │")
    print("   │ Repeat for 20-30 different robot configurations     │")
    print("   └─────────────────────────────────────────────────────┘")

    # 4. Run optimization (would run after data collection)
    print("\n4. Run optimization:")
    print("   results = calibrator.optimize()")
    print("   # Returns dict: camera_id -> optimized link_T_cam")

    # 5. Validation
    print("\n5. Validation:")
    print("   validation = calibrator.validate()")
    print("   print(validation)")
    print("   # Check: mean error < 5mm, 1°")
    print("   # Check: passed_sanity_check = True")

    # 6. Save results
    print("\n6. Save calibrated transforms:")
    print("   np.savez('g1_camera_calibration.npz', **results)")
    print("   # Load later: data = np.load('g1_camera_calibration.npz')")

    print("\n" + "=" * 70)
    print("  Recommended Data Collection Strategy")
    print("=" * 70)

    print("""
For Unitree G1, collect data in these poses:

1. Standing poses (5-10 captures):
   - Torso upright, head looking straight
   - Head looking left/right/up/down
   - Chest and head should see calibration object

2. Arms extended poses (5-10 captures):
   - Arms extended forward at different heights
   - One arm extended, one at side
   - Hand cameras and chest/head should see object

3. Manipulation poses (5-10 captures):
   - Both hands near calibration object
   - Hands at different distances (7-50cm for D405)
   - All cameras should see object if possible

4. Dynamic poses (5-10 captures):
   - Various combinations of head and arm positions
   - Ensure constraint graph stays connected

Total: 20-40 captures recommended for robust calibration

Calibration object:
   - AprilTag board with 6 tags (e.g., tag36h11 family)
   - Board size: ~30x30cm for visibility
   - Place at 0.5-1.0m from robot
    """)

    print("=" * 70 + "\n")


def example_integration_with_sdk():
    """Example of how to integrate with robot SDK."""
    print("=" * 70)
    print("  Integration with Robot SDK - Pseudocode")
    print("=" * 70)

    code_example = '''
# Pseudocode for integration with Unitree SDK

import unitree_sdk  # Hypothetical SDK
from multi_camera_calibration import Calibrator, Detection
import apriltag_detector  # Hypothetical detector

# 1. Initialize robot and cameras
robot = unitree_sdk.Robot()
cameras = robot.get_cameras()
calibrator = Calibrator(robot_config)

# 2. Data collection loop
for i in range(30):  # 30 captures
    print(f"Capture {i+1}/30")
    print("Position robot and press Enter...")
    input()

    # Get current link poses from SDK
    link_poses = {
        'torso_link': robot.get_link_pose('torso_link'),
        'head_link': robot.get_link_pose('head_link'),
        'left_wrist_link': robot.get_link_pose('left_wrist_link'),
        'right_wrist_link': robot.get_link_pose('right_wrist_link'),
    }

    # Trigger all cameras synchronously
    images = {
        'chest': cameras['chest'].capture(),
        'head': cameras['head'].capture(),
        'hand_L': cameras['hand_L'].capture(),
        'hand_R': cameras['hand_R'].capture(),
    }

    # Detect AprilTags in each image
    detections = []
    for camera_id, image in images.items():
        tags = apriltag_detector.detect(image)

        if tags:
            # Compute pose of tag board (6 tags -> 1 object)
            cam_T_object = apriltag_detector.compute_board_pose(tags)
            num_tags = len(tags)

            detection = Detection(
                camera_id=camera_id,
                object_id=0,  # Single calibration board
                cam_T_object=cam_T_object,
                num_tags=min(num_tags, 6),  # Cap at 6
            )
            detections.append(detection)

    # Add to calibrator
    calibrator.add_capture(link_poses, detections)
    print(f"  Added {len(detections)} detections")

# 3. Run calibration
print("\\nRunning calibration...")
results = calibrator.optimize()

# 4. Validate
validation = calibrator.validate()
print(validation)

# 5. Save results
import numpy as np
np.savez('g1_calibration.npz', **results)
print("Calibration saved to g1_calibration.npz")
    '''

    print(code_example)
    print("=" * 70 + "\n")


def main():
    """Run all examples."""
    example_calibration_workflow()
    example_integration_with_sdk()


if __name__ == "__main__":
    main()
