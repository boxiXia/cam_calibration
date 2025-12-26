"""Levenberg-Marquardt calibration engine for multi-camera systems."""

import numpy as np
from scipy.optimize import least_squares
from typing import Dict, List, Optional, Literal
from collections import deque
import warnings

from .data_structures import RobotConfig, Capture, Detection, ValidationResult
from .cost_function import generate_pairwise_constraints, compute_residuals, compute_pairwise_errors, PairwiseConstraint
from .transforms import apply_correction


class Calibrator:
    """Multi-camera calibration via pairwise constraints and Levenberg-Marquardt.

    Usage: add_capture() → optimize() → validate()

    Args:
        w_rot: Rotation weight = d² where d=working distance (default 0.25 for d=0.5m)
        lambda_reg: Regularization weight (default 1.0)
        min_tags: Minimum AprilTags per detection (default 2)
    """

    def __init__(self, robot_config: RobotConfig, w_rot: float = 0.25, lambda_reg: float = 1.0, min_tags: int = 2):
        self.robot_config = robot_config
        self.captures: List[Capture] = []  # Synchronized multi-camera snapshots
        self.w_rot = w_rot  # Rotation weight (squared working distance)
        self.lambda_reg = lambda_reg  # Regularization to prevent large corrections
        self.min_tags = min_tags  # Quality filter: require ≥2 tags per detection
        self._constraints: Optional[List[PairwiseConstraint]] = None  # Cached pairwise constraints
        self._optimized_corrections: Optional[np.ndarray] = None  # Result: 6-DOF per camera
        self._result = None  # Optimization result (for covariance estimation)

    def add_capture(self, link_poses: Dict[str, np.ndarray], detections: List[Detection], capture_id: Optional[int] = None) -> None:
        """Add synchronized capture: link poses (from FK/SDK) + detections (from AprilTag)."""
        self.captures.append(Capture(capture_id or len(self.captures), link_poses, detections))
        self._constraints = None
        self._optimized_corrections = None
        self._result = None

    def _check_constraint_graph_connected(self, constraints: List[PairwiseConstraint]) -> bool:
        """BFS to verify all cameras connected via shared observations (ensures solvable system)."""
        if not constraints:
            return False

        # Build adjacency: each constraint connects two cameras observing same object
        camera_ids = [cam.camera_id for cam in self.robot_config.cameras]
        graph = {cid: set() for cid in camera_ids}
        for c in constraints:
            graph[c.camera_a_id].add(c.camera_b_id)
            graph[c.camera_b_id].add(c.camera_a_id)

        # BFS: can we reach all cameras from the first one?
        visited, queue = {camera_ids[0]}, deque([camera_ids[0]])
        while queue:
            for neighbor in graph[queue.popleft()]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == len(camera_ids)  # All cameras reachable?

    def optimize(
        self,
        verbose: bool = True,
        fine_tune: bool = False,
        fine_tune_lambda: float = 0.01,
        loss: Literal['linear', 'huber', 'cauchy'] = 'huber',
        f_scale: float = 0.01,
    ) -> Dict[str, np.ndarray]:
        """Run optimization to solve for camera transforms.

        Args:
            verbose: Print progress
            fine_tune: Second pass with reduced regularization for higher precision
            fine_tune_lambda: Regularization for fine-tune (default 0.01)
            loss: Loss function - 'huber' (default, robust), 'linear' (L2), or 'cauchy'
            f_scale: Outlier threshold for robust loss (residuals > f_scale are down-weighted)

        Returns:
            Dict[camera_id → optimized link_T_cam (4x4 SE3)]
        """
        if not self.captures:
            raise ValueError("No captures. Call add_capture() first.")

        # Generate pairwise constraints (camera pairs observing same object)
        self._constraints = generate_pairwise_constraints(self.captures, self.robot_config, self.min_tags)
        if not self._constraints:
            raise ValueError(f"No constraints generated. Need multiple cameras seeing same object with ≥{self.min_tags} tags.")
        if verbose:
            print(f"Generated {len(self._constraints)} constraints")

        if not self._check_constraint_graph_connected(self._constraints):
            warnings.warn("Constraint graph disconnected - some cameras lack shared observations.")

        # Use LM for linear loss (faster), TRF for robust loss (supports huber/cauchy)
        method = 'lm' if loss == 'linear' else 'trf'

        def solve(initial: np.ndarray, lambda_reg: float):
            result = least_squares(
                lambda x: compute_residuals(x, self._constraints, self.robot_config, self.w_rot, lambda_reg),
                initial, method=method, loss=loss, f_scale=f_scale, verbose=2 if verbose else 0,
            )
            if not result.success:
                warnings.warn(f"Optimization warning: {result.message}")
            return result

        # Main optimization pass
        self._result = solve(np.zeros(6 * len(self.robot_config.cameras)), self.lambda_reg)
        self._optimized_corrections = self._result.x

        # Optional fine-tuning with reduced regularization
        if fine_tune:
            if verbose:
                print(f"\nFine-tuning (lambda={fine_tune_lambda})...")
            self._result = solve(self._optimized_corrections, fine_tune_lambda)
            self._optimized_corrections = self._result.x

        return self.get_optimized_transforms()

    def get_optimized_transforms(self) -> Dict[str, np.ndarray]:
        """Get optimized camera transforms (4x4 SE3 matrices)."""
        if self._optimized_corrections is None:
            raise RuntimeError("Call optimize() first")

        return {cam.camera_id: apply_correction(cam.init_link_T_cam, self._optimized_corrections[i*6:(i+1)*6])
                for i, cam in enumerate(self.robot_config.cameras)}

    def get_corrections(self) -> Dict[str, np.ndarray]:
        """Get 6-DOF corrections [rx,ry,rz,tx,ty,tz] for each camera."""
        if self._optimized_corrections is None:
            raise RuntimeError("Call optimize() first")

        return {cam.camera_id: self._optimized_corrections[i*6:(i+1)*6]
                for i, cam in enumerate(self.robot_config.cameras)}

    def get_standard_errors(self) -> Dict[str, tuple]:
        """Estimate calibration uncertainty from Jacobian. Returns (rotation_std_deg, translation_std_mm)."""
        if self._result is None:
            raise RuntimeError("Call optimize() first")

        # Covariance = σ² (JᵀJ)⁻¹ where σ² = sum(residuals²) / dof
        J, r = self._result.jac, self._result.fun
        dof = max(len(r) - len(self._result.x), 1)
        cov = (np.sum(r**2) / dof) * np.linalg.pinv(J.T @ J)

        # Extract standard errors for each camera
        result = {}
        for i, cam in enumerate(self.robot_config.cameras):
            std = np.sqrt(np.diag(cov[i*6:(i+1)*6, i*6:(i+1)*6]))
            result[cam.camera_id] = (np.rad2deg(np.linalg.norm(std[:3])), np.linalg.norm(std[3:]) * 1000)
        return result

    def validate(
        self,
        test_captures: Optional[List[Capture]] = None,
        sanity_trans_mm: float = 20.0,
        sanity_rot_deg: float = 5.0,
    ) -> ValidationResult:
        """Validate calibration quality and sanity check correction magnitudes.

        Args:
            test_captures: Hold-out test set (uses training set if None)
            sanity_trans_mm: Max acceptable correction translation (mm)
            sanity_rot_deg: Max acceptable correction rotation (degrees)

        Returns:
            ValidationResult with residual errors and sanity check status
        """
        if self._optimized_corrections is None:
            raise RuntimeError("Call optimize() first")

        # Generate constraints for validation (same object seen by multiple cameras)
        captures = test_captures if test_captures is not None else self.captures
        constraints = generate_pairwise_constraints(captures, self.robot_config, self.min_tags)

        if not constraints:
            raise ValueError("No constraints for validation")

        # Compute pairwise constraint errors (should be small if calibration is good)
        trans_errors, rot_errors = compute_pairwise_errors(constraints, self.robot_config, self._optimized_corrections)
        trans_errors_mm = [e * 1000 for e in trans_errors]
        rot_errors_deg = [np.rad2deg(e) for e in rot_errors]

        # Compute correction magnitudes (sanity check: corrections shouldn't be huge)
        correction_magnitudes = {}
        for i, cam in enumerate(self.robot_config.cameras):
            corr = self._optimized_corrections[i*6:(i+1)*6]
            correction_magnitudes[cam.camera_id] = (
                np.linalg.norm(corr[3:]) * 1000,  # Translation magnitude (mm)
                np.rad2deg(np.linalg.norm(corr[:3]))  # Rotation magnitude (degrees)
            )

        # Sanity check: flag if any correction exceeds bounds (indicates bad initial estimate)
        passed_sanity = all(
            trans < sanity_trans_mm and rot < sanity_rot_deg
            for trans, rot in correction_magnitudes.values()
        )

        return ValidationResult(
            mean_translation_error_mm=np.mean(trans_errors_mm),
            mean_rotation_error_deg=np.mean(rot_errors_deg),
            max_translation_error_mm=np.max(trans_errors_mm),
            max_rotation_error_deg=np.max(rot_errors_deg),
            num_constraints=len(constraints),
            passed_sanity_check=passed_sanity,
            correction_magnitudes=correction_magnitudes,
        )
