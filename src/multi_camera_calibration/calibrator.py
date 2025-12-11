"""Main calibration engine using Levenberg-Marquardt optimization."""

import numpy as np
from scipy.optimize import least_squares
from typing import Dict, List, Optional
import warnings

from .data_structures import (
    RobotConfig,
    Capture,
    Detection,
    ValidationResult,
)
from .cost_function import (
    generate_pairwise_constraints,
    compute_residuals,
    compute_pairwise_errors,
    PairwiseConstraint,
)
from .transforms import (
    apply_correction,
    transform_to_correction,
)


class Calibrator:
    """Multi-camera calibration using pairwise constraints.

    This class handles the full calibration pipeline:
    1. Collecting synchronized captures from multiple cameras
    2. Generating pairwise constraints from shared object observations
    3. Optimizing camera transforms using Levenberg-Marquardt
    4. Validating results

    Attributes:
        robot_config: Configuration of the robot's camera system
        captures: List of captured data
        w_rot: Weight for rotation errors (default: 0.25 for 0.5m working distance)
        lambda_reg: Regularization weight (default: 1.0)
        min_tags: Minimum tags for valid detection (default: 2)
    """

    def __init__(
        self,
        robot_config: RobotConfig,
        w_rot: float = 0.25,
        lambda_reg: float = 1.0,
        min_tags: int = 2,
    ):
        """Initialize the calibrator.

        Args:
            robot_config: Configuration of the robot's camera system
            w_rot: Weight for rotation errors (d² where d is working distance in meters)
            lambda_reg: Regularization weight to prevent large corrections
            min_tags: Minimum number of AprilTags required for valid detection
        """
        self.robot_config = robot_config
        self.captures: List[Capture] = []
        self.w_rot = w_rot
        self.lambda_reg = lambda_reg
        self.min_tags = min_tags
        self._constraints: Optional[List[PairwiseConstraint]] = None
        self._optimized_corrections: Optional[np.ndarray] = None

    def add_capture(
        self,
        link_poses: Dict[str, np.ndarray],
        detections: List[Detection],
        capture_id: Optional[int] = None,
    ) -> None:
        """Add a synchronized capture to the dataset.

        Args:
            link_poses: Dictionary mapping link names to base_T_link transforms
            detections: List of detections from all cameras in this capture
            capture_id: Optional ID for this capture (auto-assigned if not provided)
        """
        if capture_id is None:
            capture_id = len(self.captures)

        capture = Capture(
            capture_id=capture_id,
            link_poses=link_poses,
            detections=detections,
        )

        self.captures.append(capture)
        # Invalidate cached constraints since we added new data
        self._constraints = None
        self._optimized_corrections = None

    def _check_constraint_graph(self, constraints: List[PairwiseConstraint]) -> bool:
        """Check if the constraint graph is connected.

        Every camera must have at least one shared observation with another camera
        (directly or transitively) for the optimization to work.

        Args:
            constraints: List of pairwise constraints

        Returns:
            True if graph is connected, False otherwise
        """
        if not constraints:
            return False

        # Build adjacency list
        camera_ids = [cam.camera_id for cam in self.robot_config.cameras]
        graph = {cam_id: set() for cam_id in camera_ids}

        for constraint in constraints:
            graph[constraint.camera_a_id].add(constraint.camera_b_id)
            graph[constraint.camera_b_id].add(constraint.camera_a_id)

        # BFS to check connectivity
        visited = set()
        queue = [camera_ids[0]]
        visited.add(camera_ids[0])

        while queue:
            current = queue.pop(0)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == len(camera_ids)

    def optimize(
        self,
        verbose: bool = True,
        fine_tune: bool = False,
        fine_tune_lambda: float = 0.01,
    ) -> Dict[str, np.ndarray]:
        """Run calibration optimization.

        Args:
            verbose: If True, print optimization progress
            fine_tune: If True, run second optimization pass with reduced regularization
            fine_tune_lambda: Regularization weight for fine-tuning (default: 0.01)

        Returns:
            Dictionary mapping camera_id to optimized link_T_cam transform

        Raises:
            ValueError: If no captures have been added or constraint graph is disconnected
        """
        if not self.captures:
            raise ValueError("No captures added. Use add_capture() to add data.")

        # Generate pairwise constraints
        if verbose:
            print("Generating pairwise constraints...")

        self._constraints = generate_pairwise_constraints(
            self.captures,
            self.robot_config,
            self.min_tags,
        )

        if not self._constraints:
            raise ValueError(
                "No valid constraints generated. Ensure multiple cameras see "
                f"the same objects with at least {self.min_tags} tags."
            )

        if verbose:
            print(f"Generated {len(self._constraints)} pairwise constraints")

        # Check constraint graph connectivity
        if not self._check_constraint_graph(self._constraints):
            warnings.warn(
                "Constraint graph is not fully connected. Some cameras may not "
                "have shared observations with others. Calibration may be unreliable."
            )

        # Initialize corrections to zero (identity transform)
        num_cameras = len(self.robot_config.cameras)
        initial_corrections = np.zeros(6 * num_cameras)

        if verbose:
            print("Running Levenberg-Marquardt optimization...")

        # Define residual function for optimizer
        def residual_fn(corrections):
            return compute_residuals(
                corrections,
                self._constraints,
                self.robot_config,
                self.w_rot,
                self.lambda_reg,
            )

        # Run optimization
        result = least_squares(
            residual_fn,
            initial_corrections,
            method='lm',
            verbose=2 if verbose else 0,
        )

        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")

        self._optimized_corrections = result.x

        if verbose:
            print(f"Optimization finished: {result.message}")
            print(f"Final cost: {result.cost:.6f}")

        # Fine-tuning pass with reduced regularization
        if fine_tune:
            if verbose:
                print(f"\nFine-tuning with lambda={fine_tune_lambda}...")

            def fine_tune_residual_fn(corrections):
                return compute_residuals(
                    corrections,
                    self._constraints,
                    self.robot_config,
                    self.w_rot,
                    fine_tune_lambda,  # Reduced regularization
                )

            result_ft = least_squares(
                fine_tune_residual_fn,
                self._optimized_corrections,  # Start from previous result
                method='lm',
                verbose=2 if verbose else 0,
            )

            if not result_ft.success:
                warnings.warn(f"Fine-tuning did not converge: {result_ft.message}")
            else:
                self._optimized_corrections = result_ft.x
                if verbose:
                    print(f"Fine-tuning finished: {result_ft.message}")
                    print(f"Final cost: {result_ft.cost:.6f}")

        # Return optimized transforms
        return self.get_optimized_transforms()

    def get_optimized_transforms(self) -> Dict[str, np.ndarray]:
        """Get the optimized camera transforms.

        Returns:
            Dictionary mapping camera_id to optimized link_T_cam transform

        Raises:
            RuntimeError: If optimize() hasn't been called yet
        """
        if self._optimized_corrections is None:
            raise RuntimeError("Must call optimize() before getting transforms")

        transforms = {}
        for i, camera in enumerate(self.robot_config.cameras):
            correction = self._optimized_corrections[i * 6 : (i + 1) * 6]
            link_T_cam = apply_correction(camera.init_link_T_cam, correction)
            transforms[camera.camera_id] = link_T_cam

        return transforms

    def get_corrections(self) -> Dict[str, np.ndarray]:
        """Get the 6-DOF corrections applied to each camera.

        Returns:
            Dictionary mapping camera_id to correction vector [rx, ry, rz, tx, ty, tz]

        Raises:
            RuntimeError: If optimize() hasn't been called yet
        """
        if self._optimized_corrections is None:
            raise RuntimeError("Must call optimize() before getting corrections")

        corrections = {}
        for i, camera in enumerate(self.robot_config.cameras):
            correction = self._optimized_corrections[i * 6 : (i + 1) * 6]
            corrections[camera.camera_id] = correction

        return corrections

    def validate(
        self,
        test_captures: Optional[List[Capture]] = None,
        sanity_trans_mm: float = 20.0,
        sanity_rot_deg: float = 5.0,
    ) -> ValidationResult:
        """Validate the calibration results.

        Args:
            test_captures: Optional separate test captures. If None, uses training captures.
            sanity_trans_mm: Maximum acceptable correction magnitude in mm
            sanity_rot_deg: Maximum acceptable correction magnitude in degrees

        Returns:
            ValidationResult with error metrics and sanity checks

        Raises:
            RuntimeError: If optimize() hasn't been called yet
        """
        if self._optimized_corrections is None:
            raise RuntimeError("Must call optimize() before validation")

        # Use test captures if provided, otherwise use training captures
        captures_to_validate = test_captures if test_captures is not None else self.captures

        # Generate constraints for validation
        constraints = generate_pairwise_constraints(
            captures_to_validate,
            self.robot_config,
            self.min_tags,
        )

        if not constraints:
            raise ValueError("No constraints generated for validation")

        # Compute errors
        trans_errors, rot_errors = compute_pairwise_errors(
            constraints,
            self.robot_config,
            self._optimized_corrections,
        )

        # Convert to mm and degrees
        trans_errors_mm = [e * 1000 for e in trans_errors]
        rot_errors_deg = [np.rad2deg(e) for e in rot_errors]

        # Compute correction magnitudes
        correction_magnitudes = {}
        for i, camera in enumerate(self.robot_config.cameras):
            correction = self._optimized_corrections[i * 6 : (i + 1) * 6]
            trans_mag = np.linalg.norm(correction[3:]) * 1000  # mm
            rot_mag = np.rad2deg(np.linalg.norm(correction[:3]))  # degrees
            correction_magnitudes[camera.camera_id] = (trans_mag, rot_mag)

        # Sanity check: corrections should be within reasonable bounds
        passed_sanity = all(
            trans_mag < sanity_trans_mm and rot_mag < sanity_rot_deg
            for trans_mag, rot_mag in correction_magnitudes.values()
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
