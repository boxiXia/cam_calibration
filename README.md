# Multi-Camera Calibration for Mobile Legged Robots

A calibration system for determining camera-to-link transforms on floating-base robots (e.g., humanoids, quadrupeds) using multi-camera constraints.

## Overview

This package solves the problem of calibrating multiple cameras on a mobile legged robot where absolute base position is unknown (floating base). When 2+ cameras view the same calibration object simultaneously, they must agree on the object's pose in the base frame, creating constraints that enable calibration without knowing absolute base position.

## Key Features

- **Floating-base compatible**: No need for absolute base position
- **Unified formulation**: Treats fixed and articulated cameras identically
- **Detection weighting**: Automatically weights by AprilTag count (1-6 tags)
- **Robust optimization**: Levenberg-Marquardt with regularization
- **Simulation framework**: Verify correctness with synthetic data
- **Comprehensive validation**: Multiple validation checks

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from multi_camera_calibration import Calibrator, RobotConfig, CameraConfig, Capture, Detection
import numpy as np

# Define robot configuration
cameras = [
    CameraConfig(camera_id="chest", parent_link="torso_link", init_link_T_cam=np.eye(4)),
    CameraConfig(camera_id="head", parent_link="head_link", init_link_T_cam=np.eye(4)),
]
robot_config = RobotConfig(cameras=cameras)

# Create calibrator
calibrator = Calibrator(robot_config)

# Add captures (synchronized snapshots)
for capture_data in capture_dataset:
    calibrator.add_capture(
        link_poses=capture_data['link_poses'],  # dict: link_name -> base_T_link
        detections=capture_data['detections']   # list of Detection objects
    )

# Optimize
results = calibrator.optimize()

# Validate
validation = calibrator.validate(test_captures)
print(f"Validation error: {validation.mean_translation_error_mm:.2f}mm, {validation.mean_rotation_error_deg:.2f}°")
```

## Architecture

### Transform Chain

Every camera uses the same formulation:

```
base_T_object = base_T_parent_link · link_T_cam · cam_T_object
                ─────────────────   ──────────   ────────────
                from SDK            unknown      measured
```

### Data Flow

1. **Data Collection**: Robot SDK provides `base_T_link`, detection pipeline provides `cam_T_object`
2. **Constraint Generation**: Pairs of cameras viewing same object create pairwise constraints
3. **Optimization**: Levenberg-Marquardt finds `link_T_cam` corrections minimizing pairwise disagreement
4. **Validation**: Check residuals, sanity bounds, generalization

## Core Concepts

### Detection Confidence

Weight based on number of AprilTags detected:
- 1 tag: weight = 0 (discarded)
- 2 tags: weight = 0.5
- 3 tags: weight = 0.67
- 6 tags: weight = 0.83

### Cost Function

- **Pairwise error**: Weighted disagreement between camera pairs
- **Regularization**: Penalize large corrections from initial estimate
- **Total cost**: `Σ(pairwise errors) + λ·reg`

### Parameters

- `w_rot`: Rotation weight = d² (working distance²), default 0.25 for d=0.5m
- `λ`: Regularization weight, default 1.0
- `min_tags`: Minimum tags for valid detection, default 2

## Verification

Run synthetic tests to verify implementation:

```bash
python examples/synthetic_verification.py
```

Expected results with no noise: < 0.001mm error
Expected results with realistic noise: error < noise level

## Directory Structure

```
cam_calibration/
├── src/multi_camera_calibration/
│   ├── data_structures.py    # CameraConfig, Capture, Detection, etc.
│   ├── transforms.py          # SE3 operations, axis-angle conversions
│   ├── calibrator.py          # Main calibration engine
│   ├── cost_function.py       # Cost and residual computation
│   ├── validation.py          # Validation metrics
│   └── simulation.py          # Synthetic data generation
├── tests/                     # Unit tests
└── examples/                  # Usage examples
```

## License

MIT
