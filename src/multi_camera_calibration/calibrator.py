"""Levenberg-Marquardt calibration engine for multi-camera systems."""

import numpy as np
from scipy.optimize import least_squares
from typing import Dict, List, Optional
import warnings

from .data_structures import RobotConfig, Capture, Detection, ValidationResult
from .cost_function import generate_pairwise_constraints, compute_residuals, compute_pairwise_errors, PairwiseConstraint
from .transforms import apply_correction


class Calibrator:
    """
    Multi-camera calibration via pairwise constraints.

    Pipeline:
      1. add_capture() - collect synchronized multi-camera observations
      2. optimize() - solve for link_T_cam via Levenberg-Marquardt
      3. validate() - check accuracy and sanity bounds

    Parameters:
      - w_rot: Rotation weight = d² (working distance²), default 0.25 for d=0.5m
      - lambda_reg: Regularization weight, default 1.0
      - min_tags: Minimum tags per detection, default 2
    """

    def __init__(self, robot_config: RobotConfig, w_rot: float = 0.25, lambda_reg: float = 1.0, min_tags: int = 2):
        self.robot_config = robot_config
        self.captures: List[Capture] = []
        self.w_rot = w_rot
        self.lambda_reg = lambda_reg
        self.min_tags = min_tags
        self._constraints: Optional[List[PairwiseConstraint]] = None
        self._optimized_corrections: Optional[np.ndarray] = None

    def add_capture(self, link_poses: Dict[str, np.ndarray], detections: List[Detection], capture_id: Optional[int] = None) -> None:
        """Add synchronized capture: link poses (from SDK) + detections (from AprilTag pipeline)."""
        self.captures.append(Capture(
            capture_id=capture_id if capture_id is not None else len(self.captures),
            link_poses=link_poses,
            detections=detections,
        ))
        self._constraints = None  # Invalidate cache
        self._optimized_corrections = None

    def _check_constraint_graph_connected(self, constraints: List[PairwiseConstraint]) -> bool:
        """BFS to verify all cameras are transitively connected via shared observations."""
        if not constraints:
            return False

        # Build adjacency graph
        camera_ids = [cam.camera_id for cam in self.robot_config.cameras]
        graph = {cam_id: set() for cam_id in camera_ids}
        for c in constraints:
            graph[c.camera_a_id].add(c.camera_b_id)
            graph[c.camera_b_id].add(c.camera_a_id)

        # BFS from first camera
        visited = {camera_ids[0]}
        queue = [camera_ids[0]]
        while queue:
            current = queue.pop(0)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == len(camera_ids)

    def optimize(self, verbose: bool = True, fine_tune: bool = False, fine_tune_lambda: float = 0.01) -> Dict[str, np.ndarray]:
        """
        Run Levenberg-Marquardt optimization to solve for camera transforms.

        Args:
          verbose: Print progress
          fine_tune: Second pass with reduced regularization for high precision
          fine_tune_lambda: Regularization for fine-tuning (default 0.01)

        Returns:
          Dict mapping camera_id -> optimized link_T_cam (4x4 SE3)
        """
        if not self.captures:
            raise ValueError("No captures. Call add_capture() first.")

        # Generate pairwise constraints
        if verbose:
            print("Generating pairwise constraints...")
        self._constraints = generate_pairwise_constraints(self.captures, self.robot_config, self.min_tags)

        if not self._constraints:
            raise ValueError(f"No constraints generated. Need multiple cameras seeing same object with ≥{self.min_tags} tags.")

        if verbose:
            print(f"Generated {len(self._constraints)} constraints")

        # Check graph connectivity
        if not self._check_constraint_graph_connected(self._constraints):
            warnings.warn("Constraint graph disconnected - some cameras lack shared observations.")

        # Optimize: corrections start at zero (identity)
        if verbose:
            print("Running Levenberg-Marquardt...")

        def residual_fn(corrections, reg_weight):
            return compute_residuals(corrections, self._constraints, self.robot_config, self.w_rot, reg_weight)

        # Initial optimization
        result = least_squares(
            lambda x: residual_fn(x, self.lambda_reg),
            np.zeros(6 * len(self.robot_config.cameras)),
            method='lm',
            verbose=2 if verbose else 0,
        )

        if not result.success:
            warnings.warn(f"Optimization failed: {result.message}")

        self._optimized_corrections = result.x

        if verbose:
            print(f"Done: {result.message}, cost={result.cost:.6f}")

        # Optional fine-tuning with reduced regularization
        if fine_tune:
            if verbose:
                print(f"\nFine-tuning (lambda={fine_tune_lambda})...")

            result_ft = least_squares(
                lambda x: residual_fn(x, fine_tune_lambda),
                self._optimized_corrections,  # Warm start
                method='lm',
                verbose=2 if verbose else 0,
            )

            if result_ft.success:
                self._optimized_corrections = result_ft.x
                if verbose:
                    print(f"Fine-tuned: {result_ft.message}, cost={result_ft.cost:.6f}")
            else:
                warnings.warn(f"Fine-tuning failed: {result_ft.message}")

        return self.get_optimized_transforms()

    def get_optimized_transforms(self) -> Dict[str, np.ndarray]:
        """Get optimized camera transforms (4x4 SE3 matrices)."""
        if self._optimized_corrections is None:
            raise RuntimeError("Call optimize() first")

        transforms = {}
        for i, camera in enumerate(self.robot_config.cameras):
            correction = self._optimized_corrections[i * 6 : (i + 1) * 6]
            transforms[camera.camera_id] = apply_correction(camera.init_link_T_cam, correction)
        return transforms

    def get_corrections(self) -> Dict[str, np.ndarray]:
        """Get 6-DOF corrections [rx,ry,rz,tx,ty,tz] for each camera."""
        if self._optimized_corrections is None:
            raise RuntimeError("Call optimize() first")

        corrections = {}
        for i, camera in enumerate(self.robot_config.cameras):
            corrections[camera.camera_id] = self._optimized_corrections[i * 6 : (i + 1) * 6]
        return corrections

    def validate(
        self,
        test_captures: Optional[List[Capture]] = None,
        sanity_trans_mm: float = 20.0,
        sanity_rot_deg: float = 5.0,
    ) -> ValidationResult:
        """
        Validate calibration results.

        Args:
          test_captures: Hold-out test set (uses training data if None)
          sanity_trans_mm: Max acceptable correction magnitude (mm)
          sanity_rot_deg: Max acceptable correction magnitude (degrees)

        Returns:
          ValidationResult with errors and sanity checks
        """
        if self._optimized_corrections is None:
            raise RuntimeError("Call optimize() first")

        # Generate constraints for validation set
        captures = test_captures if test_captures is not None else self.captures
        constraints = generate_pairwise_constraints(captures, self.robot_config, self.min_tags)

        if not constraints:
            raise ValueError("No constraints for validation")

        # Compute errors
        trans_errors, rot_errors = compute_pairwise_errors(constraints, self.robot_config, self._optimized_corrections)
        trans_errors_mm = [e * 1000 for e in trans_errors]
        rot_errors_deg = [np.rad2deg(e) for e in rot_errors]

        # Correction magnitudes
        correction_magnitudes = {}
        for i, camera in enumerate(self.robot_config.cameras):
            corr = self._optimized_corrections[i * 6 : (i + 1) * 6]
            trans_mm = np.linalg.norm(corr[3:]) * 1000
            rot_deg = np.rad2deg(np.linalg.norm(corr[:3]))
            correction_magnitudes[camera.camera_id] = (trans_mm, rot_deg)

        # Sanity check: all corrections within bounds?
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
