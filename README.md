# Multi-Camera Calibration for Mobile Legged Robots

A calibration system for determining camera-to-link transforms on floating-base robots (humanoids, quadrupeds) using multi-camera constraints.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Core Concepts](#core-concepts)
5. [Usage Guide](#usage-guide)
6. [API Reference](#api-reference)
7. [Examples](#examples)
8. [Verification](#verification)
9. [Troubleshooting](#troubleshooting)
10. [Implementation](#implementation)
11. [Performance](#performance)

---

## Overview

This package addresses camera calibration for floating-base robots where absolute base position is unknown. When 2+ cameras view the same calibration object simultaneously, they must agree on the object's pose in the base frame, creating constraints that enable calibration without absolute base position.

### Features

- Floating-base compatible: No absolute base position required
- Unified formulation: Treats fixed and articulated cameras identically
- Detection weighting: Weights detections by AprilTag count (1-6 tags)
- Levenberg-Marquardt optimization with regularization
- Optional fine-tuning pass with reduced regularization
- Synthetic test coverage with known ground truth

### Test Results

| Test Scenario | Result |
|---------------|--------|
| Perfect data (no noise) | 0.0004mm, 0.0000° with fine-tuning |
| Realistic noise (2mm, 1°) | 0.73mm, 0.04° ground truth error |
| Large initial error (50mm, 10°) | 99% improvement → 0.43mm |

---

## Quick Start

```python
from multi_camera_calibration import Calibrator, RobotConfig, CameraConfig, Detection
import numpy as np

# 1. Configure cameras
cameras = [
    CameraConfig("chest", "torso_link", init_link_T_cam=np.eye(4)),
    CameraConfig("head", "head_link", init_link_T_cam=np.eye(4)),
]
calibrator = Calibrator(RobotConfig(cameras))

# 2. Collect data (30+ synchronized captures)
for i in range(30):
    link_poses = get_link_poses_from_sdk()  # Dict: link_name -> base_T_link (4x4)
    detections = detect_apriltags()  # List of Detection objects
    calibrator.add_capture(link_poses, detections)

# 3. Optimize
results = calibrator.optimize(fine_tune=True)

# 4. Validate
validation = calibrator.validate()
print(validation)

# 5. Save
np.savez('calibration.npz', **results)
```

---

## Installation

```bash
cd cam_calibration
pip install -r requirements.txt
pip install -e .
```

### Dependencies

- Python ≥ 3.7
- NumPy ≥ 1.24.0
- SciPy ≥ 1.10.0

---

## Core Concepts

### Transform Chain

Every camera uses the same formulation:

```
base_T_object = base_T_parent_link · link_T_cam · cam_T_object
                ─────────────────   ──────────   ────────────
                from SDK            OPTIMIZE     from detection
```

Goal: Solve for `link_T_cam` for each camera.

### Floating Base Solution

When cameras A and B both see the same object:

```
base_T_object (from A) = base_T_object (from B)
```

This creates pairwise constraints that cancel the unknown base position.

### Detection Weighting

Weight based on number of AprilTags detected:

| Tags | Weight | Quality |
|------|--------|---------|
| 1    | 0.00   | Discarded |
| 2    | 0.50   | Minimum |
| 3    | 0.67   | Good |
| 4    | 0.75   | Better |
| 5    | 0.80   | Very Good |
| 6    | 0.83   | Excellent |

Formula: `w = 1 - 1/num_tags`

### Cost Function

```
cost = Σ w_pair · (||Δtrans||² + w_rot · ||Δrot||²) + λ · Σ ||correction||²
       ─────────────────────────────────────────   ─────────────────────
       pairwise errors                              regularization
```

- Pairwise errors: Cameras must agree on object pose
- Regularization: Penalizes large deviations from initial estimate
- w_rot: Rotation weight = d² (working distance²)
- λ: Regularization weight

---

## Usage Guide

### Step 1: Create Robot Configuration

```python
from multi_camera_calibration import CameraConfig, RobotConfig
import numpy as np

# Initial transforms from URDF
chest_T_cam = np.array([
    [0, 0, 1, 0.15],   # Looking forward
    [-1, 0, 0, 0.0],
    [0, -1, 0, 0.30],  # 30cm up
    [0, 0, 0, 1],
])

cameras = [
    CameraConfig(
        camera_id="chest",
        parent_link="torso_link",
        init_link_T_cam=chest_T_cam
    ),
    CameraConfig(
        camera_id="head",
        parent_link="head_link",
        init_link_T_cam=head_T_cam
    ),
]

robot_config = RobotConfig(cameras=cameras)
```

### Step 2: Initialize Calibrator

```python
from multi_camera_calibration import Calibrator

calibrator = Calibrator(
    robot_config,
    w_rot=0.25,      # For 0.5m working distance (d²)
    lambda_reg=1.0,  # Regularization weight
    min_tags=2       # Minimum tags for valid detection
)
```

**Parameter Selection:**

- **w_rot**: Set to d² where d = working distance (meters)
  - 0.0625 for d=0.25m
  - 0.25 for d=0.5m
  - 1.0 for d=1.0m
- **lambda_reg**: Higher values trust initial estimate more (default: 1.0)
- **min_tags**: Minimum tags per detection (default: 2)

### Step 3: Collect Data

```python
for i in range(30):  # 20-40 captures recommended
    input(f"Capture {i+1}/30 - Position robot and press Enter")

    # Get link poses from robot SDK
    link_poses = {
        'torso_link': robot.get_link_pose('torso_link'),  # 4x4 SE3
        'head_link': robot.get_link_pose('head_link'),
    }

    # Detect AprilTags from camera images
    detections = []
    for camera_id, image in capture_images().items():
        tags = detect_apriltags(image)
        if tags:
            cam_T_object = compute_board_pose(tags)
            detections.append(Detection(
                camera_id=camera_id,
                object_id=0,
                cam_T_object=cam_T_object,
                num_tags=len(tags)
            ))

    calibrator.add_capture(link_poses, detections)
```

**Data Collection Guidelines:**

- Minimum: 15-20 captures
- Recommended: 30-40 captures
- Vary robot configurations significantly
- Each camera pair should share views 2-3+ times
- Moving cameras require multiple joint configurations

### Step 4: Optimize

```python
# Standard optimization
results = calibrator.optimize(verbose=True)

# With fine-tuning
results = calibrator.optimize(
    verbose=True,
    fine_tune=True,
    fine_tune_lambda=0.001
)
```

### Step 5: Validate

```python
# Validate on training data
validation = calibrator.validate()
print(validation)

# Validate on test data
test_validation = calibrator.validate(test_captures=test_captures)

# Check criteria
assert validation.passed_sanity_check
assert validation.mean_translation_error_mm < 5.0
assert validation.mean_rotation_error_deg < 1.0
```

**Validation Criteria:**

- Mean error: < 5mm, < 1°
- Sanity check: Corrections < 20mm, < 5°
- Constraint graph: Connected
- Test error similar to training error

### Step 6: Save Results

```python
# Save calibrated transforms
np.savez('calibration.npz', **results)

# Save corrections
corrections = calibrator.get_corrections()
np.savez('corrections.npz', **corrections)

# Load
data = np.load('calibration.npz')
chest_T_cam = data['chest']
```

---

## API Reference

### Data Structures

#### CameraConfig

```python
@dataclass
class CameraConfig:
    camera_id: str              # Unique camera identifier
    parent_link: str            # Name of parent link
    init_link_T_cam: np.ndarray # Initial 4x4 SE3 transform
```

#### RobotConfig

```python
@dataclass
class RobotConfig:
    cameras: List[CameraConfig]

    def get_camera_by_id(camera_id: str) -> CameraConfig
```

#### Detection

```python
@dataclass
class Detection:
    camera_id: str              # Camera that made detection
    object_id: int              # Calibration object ID
    cam_T_object: np.ndarray    # 4x4 SE3 transform
    num_tags: int               # Number of tags (1-6)

    @property
    def weight() -> float       # Detection weight (0-1)
```

#### Capture

```python
@dataclass
class Capture:
    capture_id: int
    link_poses: Dict[str, np.ndarray]  # link_name -> base_T_link (4x4)
    detections: List[Detection]
```

#### ValidationResult

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

### Calibrator Class

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

    def optimize(
        verbose: bool = True,
        fine_tune: bool = False,
        fine_tune_lambda: float = 0.01
    ) -> Dict[str, np.ndarray]

    def validate(
        test_captures: Optional[List[Capture]] = None,
        sanity_trans_mm: float = 20.0,
        sanity_rot_deg: float = 5.0
    ) -> ValidationResult

    def get_optimized_transforms() -> Dict[str, np.ndarray]
    def get_corrections() -> Dict[str, np.ndarray]
```

### Transform Utilities

```python
# Conversions
def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray
def rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray
def correction_to_transform(correction: np.ndarray) -> np.ndarray
def transform_to_correction(T: np.ndarray) -> np.ndarray

# Operations
def compose_transforms(*transforms: np.ndarray) -> np.ndarray
def invert_transform(T: np.ndarray) -> np.ndarray
def apply_correction(init_transform: np.ndarray, correction: np.ndarray) -> np.ndarray

# Metrics
def translation_error(T1: np.ndarray, T2: np.ndarray) -> float
def rotation_error(T1: np.ndarray, T2: np.ndarray) -> float
```

---

## Examples

### Unitree G1 Humanoid Robot

```python
from multi_camera_calibration import CameraConfig, RobotConfig, Calibrator
import numpy as np

# 4-camera configuration
cameras = [
    CameraConfig("chest", "torso_link", chest_T_cam),
    CameraConfig("head", "head_link", head_T_cam),
    CameraConfig("hand_L", "left_wrist_link", left_hand_T_cam),
    CameraConfig("hand_R", "right_wrist_link", right_hand_T_cam),
]

calibrator = Calibrator(RobotConfig(cameras), w_rot=0.25)

# Run: python examples/unitree_g1_example.py
```

### Synthetic Verification

```python
from multi_camera_calibration import SimulationFramework

# Create simulation with known ground truth
sim, robot_config, gt_transforms = SimulationFramework.create_from_scratch(
    camera_ids=["cam1", "cam2", "cam3", "cam4"],
    parent_links=["link1", "link2", "link3", "link4"],
    seed=42
)

# Generate synthetic data
captures = sim.generate_dataset(num_captures=30)

# Calibrate
calibrator = Calibrator(robot_config)
for capture in captures:
    calibrator.add_capture(capture.link_poses, capture.detections)

results = calibrator.optimize(fine_tune=True)

# Compare to ground truth
errors = sim.compute_ground_truth_errors(results)

# Run: python examples/synthetic_verification.py
```

---

## Verification

### Test Suite

```bash
# Transform utilities
python tests/test_transforms.py

# Synthetic verification
python examples/synthetic_verification.py
```

### Synthetic Test Results

| Test | Setup | Result |
|------|-------|--------|
| No Noise | 15 captures, 0mm noise | 0.0004mm, 0.0000° |
| Realistic Noise | 30 captures, 2mm/1° noise | 0.73mm, 0.04° GT error |
| Generalization | 25 train / 10 test split | Test error ≈ training |
| Large Initial Error | 50mm, 10° initial error | 99% improvement |

### Real-World Validation

```python
# Collect test captures
test_captures = collect_new_data()

# Validate
test_validation = calibrator.validate(test_captures=test_captures)
print(f"Test error: {test_validation.mean_translation_error_mm:.2f}mm")
```

---

## Troubleshooting

### "No valid constraints generated"

**Cause**: No camera pairs seeing same object with ≥ min_tags

**Solutions:**
- Verify multiple cameras see calibration object
- Check AprilTag detection pipeline
- Lower `min_tags` temporarily
- Improve object visibility and lighting

### "Constraint graph disconnected"

**Cause**: Some cameras never share views with others

**Solutions:**
- Ensure each camera shares views with ≥1 other camera
- Add captures with better camera coverage
- Check working distances match camera specs

### Large corrections (>20mm, >5°)

**Cause**: Initial estimates far from true values

**Solutions:**
- Verify URDF transforms are reasonable
- Increase `lambda_reg` (e.g., to 2.0 or 5.0)
- Collect more captures (40+)
- Check AprilTag detection accuracy

### High validation errors (>5mm, >1°)

**Cause**: Noisy detections or insufficient data

**Solutions:**
- Collect more captures (30-40+)
- Improve AprilTag detection (lighting, focus, resolution)
- Use boards with more tags (6 instead of 1-3)
- Verify link poses from SDK are accurate
- Check camera synchronization

### Optimization doesn't converge

**Cause**: Poor initialization or conflicting constraints

**Solutions:**
- Verify link poses are in correct frame
- Check `cam_T_object` transforms are computed correctly
- Ensure all matrices are valid SE3 (bottom row [0,0,0,1])
- Reduce `lambda_reg` (e.g., to 0.1)
- Check for systematic errors in detection

---

## Implementation

### Architecture

```
src/multi_camera_calibration/
├── data_structures.py    # Data classes
├── transforms.py         # SE3 operations (84 lines)
├── cost_function.py      # Residuals & constraints (153 lines)
├── calibrator.py         # Main engine (221 lines)
└── simulation.py         # Synthetic data generation
```

### Transform Chain

```python
# For each camera-object pair:
base_T_object = compose_transforms(
    base_T_parent_link,  # From robot SDK
    link_T_cam,          # Optimized (unknown)
    cam_T_object         # From AprilTag detection
)
```

### Optimization Variables

- Parameterization: 6-DOF corrections per camera `[rx, ry, rz, tx, ty, tz]`
- Total unknowns: 6 × num_cameras
- Initial value: All zeros (identity correction)
- Update rule: `link_T_cam = init_link_T_cam @ correction_matrix`

### Residuals

For each pairwise constraint:
- 3 residuals: weighted translation error
- 3 residuals: weighted rotation error (axis-angle)

For each camera:
- 6 residuals: regularization penalty

Total: `6 × num_constraints + 6 × num_cameras`

### Algorithm

1. Generate pairwise constraints from captures
2. Check constraint graph connectivity (BFS)
3. Levenberg-Marquardt optimization:
   - Minimize: `||residuals||²`
   - Method: `scipy.optimize.least_squares(method='lm')`
4. Optional fine-tuning with reduced regularization
5. Validation on training or test set

---

## Performance

### Code Metrics

| File | Lines |
|------|-------|
| `transforms.py` | 84 |
| `cost_function.py` | 153 |
| `calibrator.py` | 221 |
| **Total core** | **458** |

### Runtime

- Typical: < 1 minute for 30 captures, 4 cameras
- Complexity: O(n_constraints × n_cameras) per iteration
- Iterations: 10-50 (Levenberg-Marquardt)
- Memory: O(captures × detections)

### Accuracy

| Scenario | Result |
|----------|--------|
| Perfect data + fine-tuning | 0.0004mm |
| Realistic noise (2mm/1°) | 0.7mm ground truth |
| Large initial error | 99% correction |

---

## Quick Reference

### Minimal Example

```python
from multi_camera_calibration import Calibrator, RobotConfig, CameraConfig

# Setup
calibrator = Calibrator(RobotConfig(cameras=[...]))

# Collect (30+ times)
calibrator.add_capture(link_poses, detections)

# Optimize
results = calibrator.optimize(fine_tune=True)

# Validate
validation = calibrator.validate()

# Save
np.savez('calibration.npz', **results)
```

### Detection Weights

```python
num_tags:  1    2     3     4     5     6
weight:    0   0.5   0.67  0.75  0.8   0.83
```

### Parameters

```python
w_rot = 0.25        # For d=0.5m working distance
lambda_reg = 1.0    # Regularization weight
min_tags = 2        # Minimum tags per detection
```

### SE3 Format

```
┌           ┐
│ R₃ₓ₃  t₃ₓ₁│  R = rotation (orthonormal)
│ 0₁ₓ₃   1  │  t = translation
└           ┘
```

Bottom row must be `[0, 0, 0, 1]`

---

## Citation

```bibtex
@software{multi_camera_calibration,
  title = {Multi-Camera Calibration for Mobile Legged Robots},
  year = {2025},
  url = {https://github.com/boxiXia/cam_calibration}
}
```

---

## License

MIT License
