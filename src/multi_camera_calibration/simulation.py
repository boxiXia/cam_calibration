"""Simulation framework for verification without hardware."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .data_structures import (
    CameraConfig,
    RobotConfig,
    Capture,
    Detection,
)
from .transforms import (
    compose_transforms,
    invert_transform,
    axis_angle_to_rotation_matrix,
    transform_to_correction,
)


class SimulationFramework:
    """Generate synthetic calibration data with known ground truth.

    This allows verification of the calibration algorithm without hardware.

    Attributes:
        robot_config: Robot configuration with initial (perturbed) estimates
        ground_truth_transforms: True link_T_cam for each camera
        noise_translation_m: Standard deviation of translation noise (meters)
        noise_rotation_rad: Standard deviation of rotation noise (radians)
    """

    def __init__(
        self,
        robot_config: RobotConfig,
        ground_truth_transforms: Dict[str, np.ndarray],
        noise_translation_m: float = 0.002,  # 2mm
        noise_rotation_rad: float = np.deg2rad(1.0),  # 1 degree
    ):
        """Initialize the simulation framework.

        Args:
            robot_config: Robot configuration with perturbed initial estimates
            ground_truth_transforms: Dictionary mapping camera_id to true link_T_cam
            noise_translation_m: Standard deviation of translation noise in detection
            noise_rotation_rad: Standard deviation of rotation noise in detection
        """
        self.robot_config = robot_config
        self.ground_truth_transforms = ground_truth_transforms
        self.noise_translation_m = noise_translation_m
        self.noise_rotation_rad = noise_rotation_rad

        # Verify all cameras have ground truth
        for camera in robot_config.cameras:
            if camera.camera_id not in ground_truth_transforms:
                raise ValueError(
                    f"Missing ground truth for camera {camera.camera_id}"
                )

    @staticmethod
    def create_from_scratch(
        camera_ids: List[str],
        parent_links: List[str],
        perturbation_translation_m: float = 0.020,  # 20mm
        perturbation_rotation_deg: float = 3.0,  # 3 degrees
        seed: Optional[int] = None,
    ) -> Tuple["SimulationFramework", RobotConfig, Dict[str, np.ndarray]]:
        """Create a simulation framework from scratch with random ground truth.

        This generates random ground truth transforms and perturbs them to create
        initial estimates for calibration.

        Args:
            camera_ids: List of camera IDs
            parent_links: List of parent link names (same length as camera_ids)
            perturbation_translation_m: How much to perturb translation (meters)
            perturbation_rotation_deg: How much to perturb rotation (degrees)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (SimulationFramework, RobotConfig, ground_truth_transforms)
        """
        if len(camera_ids) != len(parent_links):
            raise ValueError("camera_ids and parent_links must have same length")

        if seed is not None:
            np.random.seed(seed)

        # Generate random ground truth transforms
        ground_truth_transforms = {}
        for camera_id in camera_ids:
            # Random translation in [-0.2, 0.2] meters
            trans = np.random.uniform(-0.2, 0.2, 3)
            # Random rotation
            axis_angle = np.random.uniform(-np.pi / 4, np.pi / 4, 3)
            R = axis_angle_to_rotation_matrix(axis_angle)

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = trans

            ground_truth_transforms[camera_id] = T

        # Create perturbed initial estimates
        cameras = []
        for camera_id, parent_link in zip(camera_ids, parent_links):
            gt_transform = ground_truth_transforms[camera_id]

            # Add perturbation
            perturb_trans = np.random.uniform(
                -perturbation_translation_m,
                perturbation_translation_m,
                3,
            )
            perturb_rot = np.random.uniform(
                -np.deg2rad(perturbation_rotation_deg),
                np.deg2rad(perturbation_rotation_deg),
                3,
            )

            perturb_T = np.eye(4)
            perturb_T[:3, :3] = axis_angle_to_rotation_matrix(perturb_rot)
            perturb_T[:3, 3] = perturb_trans

            init_estimate = compose_transforms(gt_transform, perturb_T)

            camera = CameraConfig(
                camera_id=camera_id,
                parent_link=parent_link,
                init_link_T_cam=init_estimate,
            )
            cameras.append(camera)

        robot_config = RobotConfig(cameras=cameras)

        sim = SimulationFramework(
            robot_config=robot_config,
            ground_truth_transforms=ground_truth_transforms,
        )

        return sim, robot_config, ground_truth_transforms

    def generate_random_link_pose(self) -> np.ndarray:
        """Generate a random base_T_link transform.

        Returns:
            4x4 SE3 transform matrix
        """
        # Random translation
        trans = np.random.uniform(-1.0, 1.0, 3)
        # Random rotation
        axis_angle = np.random.uniform(-np.pi, np.pi, 3)
        R = axis_angle_to_rotation_matrix(axis_angle)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = trans

        return T

    def generate_random_object_pose(self) -> np.ndarray:
        """Generate a random base_T_object transform.

        Returns:
            4x4 SE3 transform matrix representing object pose in base frame
        """
        # Random translation in front of robot
        trans = np.random.uniform([0.3, -0.5, 0.0], [1.5, 0.5, 1.0])
        # Random rotation
        axis_angle = np.random.uniform(-np.pi / 4, np.pi / 4, 3)
        R = axis_angle_to_rotation_matrix(axis_angle)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = trans

        return T

    def add_noise_to_transform(self, T: np.ndarray) -> np.ndarray:
        """Add noise to a transform.

        Args:
            T: 4x4 SE3 transform matrix

        Returns:
            Noisy 4x4 SE3 transform matrix
        """
        # Add translation noise
        trans_noise = np.random.normal(0, self.noise_translation_m, 3)

        # Add rotation noise
        rot_noise = np.random.normal(0, self.noise_rotation_rad, 3)
        R_noise = axis_angle_to_rotation_matrix(rot_noise)

        # Apply noise
        T_noisy = T.copy()
        T_noisy[:3, 3] += trans_noise
        T_noisy[:3, :3] = T_noisy[:3, :3] @ R_noise

        return T_noisy

    def can_camera_see_object(
        self,
        base_T_link: np.ndarray,
        link_T_cam: np.ndarray,
        base_T_object: np.ndarray,
        max_distance: float = 3.0,
        min_distance: float = 0.1,
    ) -> bool:
        """Check if a camera can see an object.

        Simple check based on distance from camera to object.

        Args:
            base_T_link: Transform from base to camera's parent link
            link_T_cam: Transform from link to camera
            base_T_object: Transform from base to object
            max_distance: Maximum viewing distance (meters)
            min_distance: Minimum viewing distance (meters)

        Returns:
            True if object is within viewing range
        """
        base_T_cam = compose_transforms(base_T_link, link_T_cam)
        cam_T_base = invert_transform(base_T_cam)
        cam_T_object = compose_transforms(cam_T_base, base_T_object)

        distance = np.linalg.norm(cam_T_object[:3, 3])
        return min_distance <= distance <= max_distance

    def generate_capture(
        self,
        capture_id: int,
        num_objects: int = 3,
        link_poses: Optional[Dict[str, np.ndarray]] = None,
        object_poses: Optional[List[np.ndarray]] = None,
    ) -> Capture:
        """Generate a synthetic capture with detections.

        Args:
            capture_id: ID for this capture
            num_objects: Number of calibration objects
            link_poses: Optional pre-specified link poses (random if None)
            object_poses: Optional pre-specified object poses (random if None)

        Returns:
            Synthetic Capture with detections
        """
        # Generate random link poses if not provided
        if link_poses is None:
            link_poses = {}
            unique_links = set(cam.parent_link for cam in self.robot_config.cameras)
            for link_name in unique_links:
                link_poses[link_name] = self.generate_random_link_pose()

        # Generate random object poses if not provided
        if object_poses is None:
            object_poses = [
                self.generate_random_object_pose() for _ in range(num_objects)
            ]

        detections = []

        # For each camera, check which objects it can see
        for camera in self.robot_config.cameras:
            base_T_link = link_poses[camera.parent_link]
            link_T_cam_gt = self.ground_truth_transforms[camera.camera_id]

            for object_id, base_T_object in enumerate(object_poses):
                # Check if camera can see object
                if self.can_camera_see_object(
                    base_T_link, link_T_cam_gt, base_T_object
                ):
                    # Compute ground truth cam_T_object
                    base_T_cam = compose_transforms(base_T_link, link_T_cam_gt)
                    cam_T_base = invert_transform(base_T_cam)
                    cam_T_object_gt = compose_transforms(cam_T_base, base_T_object)

                    # Add noise
                    cam_T_object_noisy = self.add_noise_to_transform(cam_T_object_gt)

                    # Random number of tags (weighted toward higher values)
                    num_tags = np.random.choice([2, 3, 4, 5, 6], p=[0.1, 0.2, 0.3, 0.2, 0.2])

                    detection = Detection(
                        camera_id=camera.camera_id,
                        object_id=object_id,
                        cam_T_object=cam_T_object_noisy,
                        num_tags=num_tags,
                    )
                    detections.append(detection)

        return Capture(
            capture_id=capture_id,
            link_poses=link_poses,
            detections=detections,
        )

    def generate_dataset(
        self,
        num_captures: int,
        num_objects_per_capture: int = 3,
    ) -> List[Capture]:
        """Generate a full synthetic dataset.

        Args:
            num_captures: Number of captures to generate
            num_objects_per_capture: Number of calibration objects per capture

        Returns:
            List of synthetic captures
        """
        captures = []
        for i in range(num_captures):
            capture = self.generate_capture(i, num_objects_per_capture)
            captures.append(capture)

        return captures

    def compute_ground_truth_errors(
        self,
        optimized_transforms: Dict[str, np.ndarray],
    ) -> Dict[str, Tuple[float, float]]:
        """Compute errors between optimized and ground truth transforms.

        Args:
            optimized_transforms: Dictionary mapping camera_id to optimized link_T_cam

        Returns:
            Dictionary mapping camera_id to (translation_error_mm, rotation_error_deg)
        """
        errors = {}

        for camera_id, optimized_T in optimized_transforms.items():
            gt_T = self.ground_truth_transforms[camera_id]

            # Translation error
            trans_error = np.linalg.norm(optimized_T[:3, 3] - gt_T[:3, 3]) * 1000  # mm

            # Rotation error
            R_opt = optimized_T[:3, :3]
            R_gt = gt_T[:3, :3]
            R_diff = R_gt.T @ R_opt

            # Compute rotation angle
            trace = np.trace(R_diff)
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            rot_error = np.rad2deg(angle)

            errors[camera_id] = (trans_error, rot_error)

        return errors
