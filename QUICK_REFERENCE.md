# Multi-Camera Calibration - Quick Reference

## Installation
```bash
pip install -r requirements.txt
pip install -e .
```

## Basic Usage
```python
from multi_camera_calibration import Calibrator, RobotConfig, CameraConfig, Detection
import numpy as np

# 1. Setup
cameras = [CameraConfig("cam1", "link1", np.eye(4)), ...]
calibrator = Calibrator(RobotConfig(cameras))

# 2. Collect (repeat 30+ times)
calibrator.add_capture(link_poses, detections)

# 3. Optimize
results = calibrator.optimize()

# 4. Validate
validation = calibrator.validate()
print(validation)

# 5. Save
np.savez('calibration.npz', **results)
```

## Data Structures

### CameraConfig
```python
CameraConfig(
    camera_id="chest",           # str: unique ID
    parent_link="torso_link",    # str: parent link name
    init_link_T_cam=np.eye(4)    # 4x4 SE3 from URDF
)
```

### Detection
```python
Detection(
    camera_id="chest",           # str: camera ID
    object_id=0,                 # int: object ID
    cam_T_object=np.eye(4),      # 4x4 SE3 transform
    num_tags=6                   # int: 1-6 tags
)
```

### Capture
```python
link_poses = {
    "torso_link": base_T_torso,      # 4x4 SE3
    "head_link": base_T_head,        # 4x4 SE3
    ...
}

detections = [Detection(...), ...]

calibrator.add_capture(link_poses, detections)
```

## Parameters

### Calibrator.__init__
- `w_rot`: Rotation weight = d² (working distance²)
  - 0.0625 for 0.25m
  - 0.25 for 0.5m (default)
  - 1.0 for 1.0m
- `lambda_reg`: Regularization weight (default: 1.0)
  - Higher = trust initial estimate more
- `min_tags`: Minimum tags for valid detection (default: 2)

## Detection Weights

| Tags | Weight | Use Case |
|------|--------|----------|
| 1    | 0.00   | Discarded |
| 2    | 0.50   | Minimum acceptable |
| 3    | 0.67   | Good |
| 4    | 0.75   | Better |
| 5    | 0.80   | Very good |
| 6    | 0.83   | Excellent |

## Validation Criteria

✓ Mean error < 5mm, 1°
✓ Corrections < 20mm, 5° (sanity check)
✓ Constraint graph connected
✓ Test error ≈ Training error

## Common Issues

### No constraints
- Ensure 2+ cameras see same object
- Check min_tags setting
- Verify AprilTag detection

### Disconnected graph
- Each camera needs shared views with another
- Add more diverse captures

### Large corrections
- Check URDF initial estimates
- Increase lambda_reg
- Verify detection accuracy

### High validation error
- Collect more captures (30-40+)
- Improve tag detection quality
- Check camera synchronization

## Data Collection

### Minimum Requirements
- 20-30 captures
- 2+ cameras per object per capture
- 2+ tags per detection
- Connected constraint graph

### Best Practices
- Vary robot configurations
- Multiple poses for moving cameras
- Good lighting, sharp focus
- Synchronized capture
- 6-tag calibration boards

## Verification

```bash
# Run synthetic tests
python examples/synthetic_verification.py

# Run unit tests
python tests/test_transforms.py

# View examples
python examples/unitree_g1_example.py
```

## File Format

### Save Results
```python
np.savez('calibration.npz', **results)
```

### Load Results
```python
data = np.load('calibration.npz')
chest_T_cam = data['chest']  # 4x4 SE3 matrix
```

## SE3 Transform Format

All transforms are 4×4 homogeneous matrices:
```
┌           ┐
│ R₃ₓ₃  t₃ₓ₁│  R = rotation matrix (orthonormal)
│ 0₁ₓ₃   1  │  t = translation vector
└           ┘
```

Bottom row must be `[0, 0, 0, 1]`

## Transform Chain

```
base_T_object = base_T_link · link_T_cam · cam_T_object
                ───────────   ──────────   ────────────
                from SDK      OPTIMIZE     from detect
```

## Quick Checks

```python
# Check validation passed
assert validation.passed_sanity_check
assert validation.mean_translation_error_mm < 5.0
assert validation.mean_rotation_error_deg < 1.0

# Check convergence
assert result.success

# Check corrections
corrections = calibrator.get_corrections()
for cam_id, corr in corrections.items():
    trans_mm = np.linalg.norm(corr[3:]) * 1000
    rot_deg = np.rad2deg(np.linalg.norm(corr[:3]))
    print(f"{cam_id}: {trans_mm:.1f}mm, {rot_deg:.1f}°")
```

## More Info

- Full guide: `USAGE_GUIDE.md`
- Implementation details: `IMPLEMENTATION_SUMMARY.md`
- API docs: Inline docstrings in source code
