# Multi-Camera Calibration - Usage Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Step-by-Step Workflow](#step-by-step-workflow)
5. [API Reference](#api-reference)
6. [Data Collection Best Practices](#data-collection-best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Verification](#verification)

---

## Quick Start

```python
from multi_camera_calibration import Calibrator, RobotConfig, CameraConfig, Detection
import numpy as np

# 1. Define robot configuration
cameras = [
    CameraConfig(
        camera_id="cam1",
        parent_link="link1",
        init_link_T_cam=np.eye(4)  # Initial estimate from URDF
    ),
    CameraConfig(
        camera_id="cam2",
        parent_link="link2",
        init_link_T_cam=np.eye(4)
    ),
]
robot_config = RobotConfig(cameras=cameras)

# 2. Create calibrator
calibrator = Calibrator(robot_config, w_rot=0.25, lambda_reg=1.0, min_tags=2)

# 3. Collect data and add captures
for i in range(30):
    # Get link poses from robot SDK
    link_poses = {
        "link1": get_link_pose_from_sdk("link1"),  # 4x4 SE3 matrix
        "link2": get_link_pose_from_sdk("link2"),
    }

    # Detect AprilTags from camera images
    detections = [
        Detection(
            camera_id="cam1",
            object_id=0,
            cam_T_object=detect_apriltag_pose("cam1"),  # 4x4 SE3 matrix
            num_tags=6
        ),
        # ... more detections
    ]

    calibrator.add_capture(link_poses, detections)

# 4. Run calibration
results = calibrator.optimize()

# 5. Validate
validation = calibrator.validate()
print(validation)

# 6. Save results
np.savez('calibration_results.npz', **results)
```

---

## Installation

### From Source

```bash
cd cam_calibration
pip install -r requirements.txt
pip install -e .
```

### Dependencies

- Python >= 3.7
- NumPy >= 1.24.0
- SciPy >= 1.10.0

---

## Core Concepts

### Transform Chain

Every camera uses the same formulation:

```
base_T_object = base_T_parent_link · link_T_cam · cam_T_object
                ─────────────────   ──────────   ────────────
                from SDK            UNKNOWN      from detection
```

The calibration solves for `link_T_cam` for each camera.

### Floating Base Problem

Mobile legged robots don't have absolute base position. However, when 2+ cameras view the same object simultaneously, they must agree on the object's pose in the base frame:

```
base_T_object (from camera A) = base_T_object (from camera B)
```

This creates constraints that enable calibration without knowing absolute base position.

### Detection Weighting

Detections are weighted by the number of AprilTags detected:

| Tags | Weight | Quality |
|------|--------|---------|
| 1    | 0.00   | Discarded |
| 2    | 0.50   | Minimum |
| 3    | 0.67   | Good |
| 4    | 0.75   | Better |
| 5    | 0.80   | Very Good |
| 6    | 0.83   | Excellent |

Single-tag detections are effectively discarded.

---

## Step-by-Step Workflow

### Step 1: Prepare Calibration Setup

**Hardware:**
- Calibration object with 6 AprilTags (e.g., tag36h11 family)
- Board size: ~30x30cm for good visibility
- Multiple robot configurations where 2+ cameras see the object

**Software:**
- Robot SDK for getting link poses
- AprilTag detection pipeline
- Initial camera transforms from URDF

### Step 2: Create Robot Configuration

```python
from multi_camera_calibration import CameraConfig, RobotConfig
import numpy as np

# Load initial transforms from URDF or use identity
chest_init = np.eye(4)
chest_init[:3, 3] = [0.15, 0.0, 0.30]  # 15cm forward, 30cm up

cameras = [
    CameraConfig(
        camera_id="chest",
        parent_link="torso_link",
        init_link_T_cam=chest_init
    ),
    # ... add more cameras
]

robot_config = RobotConfig(cameras=cameras)
```

### Step 3: Initialize Calibrator

```python
from multi_camera_calibration import Calibrator

calibrator = Calibrator(
    robot_config,
    w_rot=0.25,      # d² where d=working distance (0.5m -> 0.25)
    lambda_reg=1.0,  # Regularization weight
    min_tags=2       # Minimum tags for valid detection
)
```

**Parameter Tuning:**
- `w_rot`: Set to d² where d is typical working distance in meters
  - 0.25 for d=0.5m (typical)
  - 1.0 for d=1.0m
  - 0.0625 for d=0.25m (close-up)
- `lambda_reg`: Higher = prefer initial estimate, Lower = trust data more
  - Start with 1.0, increase if corrections seem too large
- `min_tags`: Usually 2 (discard single-tag detections)

### Step 4: Collect Data

```python
for i in range(30):  # 20-40 captures recommended
    print(f"Capture {i+1}/30 - Position robot and press Enter")
    input()

    # Get link poses from robot SDK
    link_poses = {}
    for camera in robot_config.cameras:
        link_name = camera.parent_link
        link_poses[link_name] = robot.get_link_pose(link_name)  # 4x4 SE3

    # Capture images from all cameras (synchronized)
    images = capture_all_cameras()

    # Detect AprilTags
    detections = []
    for camera_id, image in images.items():
        tags = detect_apriltags(image)
        if tags:
            cam_T_object = compute_board_pose(tags)
            num_tags = len(tags)

            detection = Detection(
                camera_id=camera_id,
                object_id=0,
                cam_T_object=cam_T_object,
                num_tags=min(num_tags, 6)
            )
            detections.append(detection)

    # Add to calibrator
    calibrator.add_capture(link_poses, detections)
    print(f"  Added {len(detections)} detections")
```

### Step 5: Run Optimization

```python
# Run calibration
print("Running optimization...")
results = calibrator.optimize(verbose=True)

# results is dict: camera_id -> optimized link_T_cam (4x4 SE3 matrix)
for camera_id, link_T_cam in results.items():
    print(f"{camera_id}:")
    print(link_T_cam)
```

### Step 6: Validate Results

```python
# Validate on training data
validation = calibrator.validate()
print(validation)

# Check validation criteria
assert validation.passed_sanity_check, "Sanity check failed!"
assert validation.mean_translation_error_mm < 5.0, "Translation error too high!"
assert validation.mean_rotation_error_deg < 1.0, "Rotation error too high!"
```

**Validation Criteria:**
- ✓ **Mean error:** < 5mm, < 1°
- ✓ **Sanity check:** Corrections < 20mm, < 5°
- ✓ **Constraint graph:** Connected (automatic check)

### Step 7: Save and Deploy

```python
# Save calibrated transforms
np.savez('camera_calibration.npz', **results)

# Also save corrections for debugging
corrections = calibrator.get_corrections()
np.savez('camera_corrections.npz', **corrections)

# Load later
data = np.load('camera_calibration.npz')
chest_T_cam = data['chest']
```

---

## API Reference

### CameraConfig

```python
@dataclass
class CameraConfig:
    camera_id: str              # Unique camera identifier
    parent_link: str            # Name of parent link
    init_link_T_cam: np.ndarray # Initial 4x4 SE3 transform
```

### RobotConfig

```python
@dataclass
class RobotConfig:
    cameras: List[CameraConfig]

    def get_camera_by_id(camera_id: str) -> CameraConfig
```

### Detection

```python
@dataclass
class Detection:
    camera_id: str              # Camera that made detection
    object_id: int              # Calibration object ID
    cam_T_object: np.ndarray    # 4x4 SE3 transform
    num_tags: int               # Number of tags detected (1-6)

    @property
    def weight() -> float       # Detection weight (0-1)
```

### Capture

```python
@dataclass
class Capture:
    capture_id: int
    link_poses: Dict[str, np.ndarray]  # link_name -> base_T_link
    detections: List[Detection]

    def get_detections_by_camera(camera_id: str) -> List[Detection]
    def get_detections_by_object(object_id: int) -> List[Detection]
```

### Calibrator

```python
class Calibrator:
    def __init__(
        robot_config: RobotConfig,
        w_rot: float = 0.25,
        lambda_reg: float = 1.0,
        min_tags: int = 2
    )

    def add_capture(
        link_poses: Dict[str, np.ndarray],
        detections: List[Detection],
        capture_id: Optional[int] = None
    ) -> None

    def optimize(verbose: bool = True) -> Dict[str, np.ndarray]

    def validate(
        test_captures: Optional[List[Capture]] = None,
        sanity_trans_mm: float = 20.0,
        sanity_rot_deg: float = 5.0
    ) -> ValidationResult

    def get_optimized_transforms() -> Dict[str, np.ndarray]
    def get_corrections() -> Dict[str, np.ndarray]
```

### ValidationResult

```python
@dataclass
class ValidationResult:
    mean_translation_error_mm: float
    mean_rotation_error_deg: float
    max_translation_error_mm: float
    max_rotation_error_deg: float
    num_constraints: int
    passed_sanity_check: bool
    correction_magnitudes: Dict[str, tuple]  # camera_id -> (trans_mm, rot_deg)
```

---

## Data Collection Best Practices

### Number of Captures

- **Minimum:** 15-20 captures
- **Recommended:** 30-40 captures
- **With noise:** More captures = better averaging

### Robot Configurations

Vary robot poses to create diverse constraints:

1. **Fixed cameras** (chest, head):
   - Different head orientations
   - Different body poses
   - Ensure constraint graph connectivity

2. **Moving cameras** (hands):
   - Multiple joint configurations essential!
   - Separates link_T_cam from joint motion
   - Arms extended, retracted, at different heights

3. **Coverage:**
   - Each camera pair should share views in 2-3+ captures
   - All cameras should be connected (directly or transitively)

### Calibration Object Placement

- **Distance:** Match camera working range
  - RealSense D435i: 0.3-3m
  - RealSense D405: 0.07-0.5m
- **Visibility:** Ensure object is well-lit and in focus
- **Tags:** Use 6-tag boards when possible (higher weight)

### Synchronization

- **Critical:** All cameras must be synchronized
- Use hardware triggering if available
- Software sync acceptable if < 100ms jitter with static scenes

---

## Troubleshooting

### "No valid constraints generated"

**Cause:** No camera pairs seeing the same object with min_tags

**Solutions:**
- Check that multiple cameras see the calibration object
- Verify AprilTag detection is working
- Lower min_tags to 1 temporarily (warning: less reliable)
- Add more captures with better object visibility

### "Constraint graph is not fully connected"

**Cause:** Some cameras never share views with others

**Solutions:**
- Ensure each camera sees object with at least one other camera
- Add captures where isolated cameras share views
- Check camera coverage and working distances

### Large corrections (>20mm, >5°)

**Cause:** Initial estimates far from true values

**Solutions:**
- Check URDF transforms are reasonable
- Increase lambda_reg (e.g., to 2.0 or 5.0)
- Collect more captures
- Verify AprilTag detection accuracy

### High validation errors (>5mm, >1°)

**Cause:** Noisy detections, insufficient data, or systematic errors

**Solutions:**
- Collect more captures (30-40+)
- Improve AprilTag detection (lighting, focus, resolution)
- Check for camera motion blur
- Verify link poses from SDK are accurate
- Use boards with more tags (6 instead of 1-3)

### Optimization doesn't converge

**Cause:** Poor initialization or conflicting constraints

**Solutions:**
- Check that link poses are in correct frame
- Verify cam_T_object transforms are computed correctly
- Ensure all matrices are valid SE3 (bottom row [0,0,0,1])
- Try different random initializations
- Reduce lambda_reg (e.g., to 0.1)

---

## Verification

### Synthetic Verification

Test the algorithm with known ground truth:

```bash
python examples/synthetic_verification.py
```

Expected results:
- ✓ No noise: < 0.5mm, < 0.05°
- ✓ Realistic noise (2mm, 1°): < 1mm ground truth error
- ✓ Large initial error (50mm, 10°): Converges to < 1mm

### Transform Tests

Test SE3 operations and conversions:

```bash
python tests/test_transforms.py
```

### Real-World Validation

After calibration, validate with fresh captures:

```python
# Collect new test captures
test_captures = collect_test_data()

# Validate on test data
test_validation = calibrator.validate(test_captures=test_captures)

# Should be similar to training error
print(f"Test error: {test_validation.mean_translation_error_mm:.2f}mm")
```

---

## Example: Unitree G1

See `examples/unitree_g1_example.py` for complete example including:
- 4-camera configuration (chest, head, left hand, right hand)
- Data collection workflow
- Integration with robot SDK
- Recommended calibration poses

Run the example:

```bash
python examples/unitree_g1_example.py
```

---

## Citation

If you use this calibration system in your research, please cite:

```
Multi-Camera Calibration for Mobile Legged Robots
https://github.com/boxiXia/cam_calibration
```

---

## License

MIT License - see LICENSE file for details
