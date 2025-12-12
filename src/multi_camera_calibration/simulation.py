"""Simulation framework for verification using actual robot models."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .data_structures import CameraConfig, RobotConfig, Capture, Detection
from .transforms import compose_transforms, invert_transform, axis_angle_to_rotation_matrix


class SimulationFramework:
    """Generate synthetic calibration data with known ground truth from robot models."""

    def __init__(
        self,
        robot_config: RobotConfig,
        ground_truth_transforms: Dict[str, np.ndarray],
        noise_translation_m: float = 0.002,
        noise_rotation_rad: float = np.deg2rad(1.0),
        urdf_model=None,
    ):
        self.robot_config = robot_config
        self.ground_truth_transforms = ground_truth_transforms
        self.noise_translation_m = noise_translation_m
        self.noise_rotation_rad = noise_rotation_rad
        self.urdf_model = urdf_model

        for camera in robot_config.cameras:
            if camera.camera_id not in ground_truth_transforms:
                raise ValueError(f"Missing ground truth for camera {camera.camera_id}")

    @staticmethod
    def create_from_urdf(
        urdf_path: str,
        camera_specs: List[Dict],
        perturbation_translation_m: float = 0.020,
        perturbation_rotation_deg: float = 3.0,
        seed: Optional[int] = None,
    ) -> Tuple["SimulationFramework", RobotConfig, Dict[str, np.ndarray]]:
        """Create simulation from URDF file with camera specifications.

        Args:
            urdf_path: Path to URDF file (must exist)
            camera_specs: List of dicts with keys:
                - camera_id: str
                - parent_link: str (must exist in URDF)
                - xyz: [x, y, z] position relative to parent link
                - rpy: [r, p, y] orientation (roll, pitch, yaw) in radians
            perturbation_translation_m: Perturbation magnitude for initial estimate
            perturbation_rotation_deg: Perturbation magnitude for initial estimate
            seed: Random seed

        Returns:
            Tuple of (SimulationFramework, RobotConfig, ground_truth_transforms)

        Example:
            camera_specs = [
                {"camera_id": "chest", "parent_link": "torso_link",
                 "xyz": [0.15, 0, 0.3], "rpy": [0, 0, 0]},
                {"camera_id": "head", "parent_link": "head_link",
                 "xyz": [0.1, 0, 0.05], "rpy": [0, 0, 0]},
            ]
        """
        import yourdfpy

        if seed is not None:
            np.random.seed(seed)

        # Load URDF and verify all parent links exist (skip mesh loading)
        urdf_model = yourdfpy.URDF.load(urdf_path, load_meshes=False)

        # Verify all parent links exist in URDF
        urdf_links = set(urdf_model.link_map.keys())
        for spec in camera_specs:
            parent_link = spec["parent_link"]
            if parent_link not in urdf_links:
                raise ValueError(
                    f"Parent link '{parent_link}' not found in URDF. "
                    f"Available links: {sorted(urdf_links)}"
                )

        # Build ground truth link_T_cam transforms from camera specs
        # These are the transforms we're trying to calibrate
        ground_truth_transforms = {}
        cameras = []

        for spec in camera_specs:
            camera_id = spec["camera_id"]
            parent_link = spec["parent_link"]
            xyz = np.array(spec["xyz"])
            rpy = np.array(spec["rpy"])

            # Build ground truth link_T_cam from xyz + rpy
            # RPY: roll (X), pitch (Y), yaw (Z) in fixed frame (ZYX convention)
            R_x = axis_angle_to_rotation_matrix([rpy[0], 0, 0])
            R_y = axis_angle_to_rotation_matrix([0, rpy[1], 0])
            R_z = axis_angle_to_rotation_matrix([0, 0, rpy[2]])
            R = R_z @ R_y @ R_x  # ZYX rotation order

            link_T_cam = np.eye(4)
            link_T_cam[:3, :3] = R
            link_T_cam[:3, 3] = xyz
            ground_truth_transforms[camera_id] = link_T_cam

            # Create perturbed initial estimate
            perturb_trans = np.random.uniform(
                -perturbation_translation_m, perturbation_translation_m, 3
            )
            perturb_rot = np.random.uniform(
                -np.deg2rad(perturbation_rotation_deg),
                np.deg2rad(perturbation_rotation_deg),
                3,
            )
            perturb_T = np.eye(4)
            perturb_T[:3, :3] = axis_angle_to_rotation_matrix(perturb_rot)
            perturb_T[:3, 3] = perturb_trans

            init_estimate = link_T_cam @ perturb_T

            cameras.append(CameraConfig(
                camera_id=camera_id,
                parent_link=parent_link,
                init_link_T_cam=init_estimate,
            ))

        robot_config = RobotConfig(cameras=cameras)
        sim = SimulationFramework(
            robot_config,
            ground_truth_transforms,
            urdf_model=urdf_model
        )

        return sim, robot_config, ground_truth_transforms

    @staticmethod
    def create_from_scratch(
        camera_ids: List[str],
        parent_links: List[str],
        perturbation_translation_m: float = 0.020,
        perturbation_rotation_deg: float = 3.0,
        seed: Optional[int] = None,
    ) -> Tuple["SimulationFramework", RobotConfig, Dict[str, np.ndarray]]:
        """Create simulation with random ground truth (for generic testing)."""
        if len(camera_ids) != len(parent_links):
            raise ValueError("camera_ids and parent_links must have same length")

        if seed is not None:
            np.random.seed(seed)

        ground_truth_transforms = {}
        for camera_id in camera_ids:
            trans = np.random.uniform(-0.2, 0.2, 3)
            axis_angle = np.random.uniform(-np.pi / 4, np.pi / 4, 3)
            R = axis_angle_to_rotation_matrix(axis_angle)

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = trans
            ground_truth_transforms[camera_id] = T

        cameras = []
        for camera_id, parent_link in zip(camera_ids, parent_links):
            gt_transform = ground_truth_transforms[camera_id]

            perturb_trans = np.random.uniform(
                -perturbation_translation_m, perturbation_translation_m, 3
            )
            perturb_rot = np.random.uniform(
                -np.deg2rad(perturbation_rotation_deg),
                np.deg2rad(perturbation_rotation_deg),
                3,
            )
            perturb_T = np.eye(4)
            perturb_T[:3, :3] = axis_angle_to_rotation_matrix(perturb_rot)
            perturb_T[:3, 3] = perturb_trans

            init_estimate = gt_transform @ perturb_T

            cameras.append(CameraConfig(
                camera_id=camera_id,
                parent_link=parent_link,
                init_link_T_cam=init_estimate,
            ))

        robot_config = RobotConfig(cameras=cameras)
        sim = SimulationFramework(robot_config, ground_truth_transforms)

        return sim, robot_config, ground_truth_transforms

    def generate_urdf_link_poses(self) -> Dict[str, np.ndarray]:
        """Generate link poses using URDF forward kinematics.

        Samples random joint configuration and computes forward kinematics
        to get base_T_link for each parent link used by cameras.

        Returns:
            Dictionary mapping link_name to base_T_link transform
        """
        if self.urdf_model is None:
            raise ValueError("URDF model not loaded. Use create_from_urdf().")

        # Get unique parent links needed
        unique_links = set(cam.parent_link for cam in self.robot_config.cameras)

        # Sample random joint configuration
        cfg = {}
        for joint_name in self.urdf_model.actuated_joint_names:
            joint = self.urdf_model.joint_map[joint_name]
            if joint.limit is not None and joint.limit.lower is not None:
                # Use joint limits
                cfg[joint_name] = np.random.uniform(
                    joint.limit.lower, joint.limit.upper
                )
            else:
                # No limits, use reasonable range
                cfg[joint_name] = np.random.uniform(-np.pi / 2, np.pi / 2)

        # Update URDF configuration
        self.urdf_model.update_cfg(cfg)

        # Compute forward kinematics for parent links
        link_poses = {}
        for link_name in unique_links:
            try:
                link_poses[link_name] = self.urdf_model.get_transform(link_name)
            except Exception as e:
                raise ValueError(
                    f"Failed to get transform for link '{link_name}': {e}\n"
                    f"Available links: {sorted(self.urdf_model.link_map.keys())}"
                )

        return link_poses

    def generate_random_link_pose(self) -> np.ndarray:
        """Generate random base_T_link transform (fallback when no URDF)."""
        trans = np.random.uniform(-1.0, 1.0, 3)
        axis_angle = np.random.uniform(-np.pi, np.pi, 3)
        R = axis_angle_to_rotation_matrix(axis_angle)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = trans
        return T

    def generate_random_object_pose(self) -> np.ndarray:
        """Generate random base_T_object transform."""
        trans = np.random.uniform([0.3, -0.5, 0.0], [1.5, 0.5, 1.0])
        axis_angle = np.random.uniform(-np.pi / 4, np.pi / 4, 3)
        R = axis_angle_to_rotation_matrix(axis_angle)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = trans
        return T

    def add_noise_to_transform(self, T: np.ndarray) -> np.ndarray:
        """Add noise to transform."""
        trans_noise = np.random.normal(0, self.noise_translation_m, 3)
        rot_noise = np.random.normal(0, self.noise_rotation_rad, 3)
        R_noise = axis_angle_to_rotation_matrix(rot_noise)

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
        """Check if camera can see object (distance-based)."""
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
        """Generate synthetic capture with detections.

        Uses URDF forward kinematics for link poses if URDF model is loaded,
        otherwise generates random link poses.
        """
        if link_poses is None:
            if self.urdf_model is not None:
                # Use URDF forward kinematics to generate realistic link poses
                link_poses = self.generate_urdf_link_poses()
            else:
                # Fallback to random link poses
                link_poses = {}
                unique_links = set(cam.parent_link for cam in self.robot_config.cameras)
                for link_name in unique_links:
                    link_poses[link_name] = self.generate_random_link_pose()

        if object_poses is None:
            object_poses = [self.generate_random_object_pose() for _ in range(num_objects)]

        detections = []

        for camera in self.robot_config.cameras:
            base_T_link = link_poses[camera.parent_link]
            link_T_cam_gt = self.ground_truth_transforms[camera.camera_id]

            for object_id, base_T_object in enumerate(object_poses):
                if self.can_camera_see_object(base_T_link, link_T_cam_gt, base_T_object):
                    # Compute ground truth cam_T_object
                    base_T_cam = compose_transforms(base_T_link, link_T_cam_gt)
                    cam_T_base = invert_transform(base_T_cam)
                    cam_T_object_gt = compose_transforms(cam_T_base, base_T_object)

                    # Add noise
                    cam_T_object_noisy = self.add_noise_to_transform(cam_T_object_gt)

                    # Random number of tags (weighted toward higher values)
                    num_tags = np.random.choice([2, 3, 4, 5, 6], p=[0.1, 0.2, 0.3, 0.2, 0.2])

                    detections.append(Detection(
                        camera_id=camera.camera_id,
                        object_id=object_id,
                        cam_T_object=cam_T_object_noisy,
                        num_tags=num_tags,
                    ))

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
        """Generate full synthetic dataset."""
        captures = []
        for i in range(num_captures):
            capture = self.generate_capture(i, num_objects_per_capture)
            captures.append(capture)
        return captures

    def compute_ground_truth_errors(
        self,
        optimized_transforms: Dict[str, np.ndarray],
    ) -> Dict[str, Tuple[float, float]]:
        """Compute errors between optimized and ground truth transforms."""
        errors = {}

        for camera_id, optimized_T in optimized_transforms.items():
            gt_T = self.ground_truth_transforms[camera_id]

            # Translation error
            trans_error = np.linalg.norm(optimized_T[:3, 3] - gt_T[:3, 3]) * 1000  # mm

            # Rotation error
            R_opt = optimized_T[:3, :3]
            R_gt = gt_T[:3, :3]
            R_diff = R_gt.T @ R_opt

            trace = np.trace(R_diff)
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            rot_error = np.rad2deg(angle)

            errors[camera_id] = (trans_error, rot_error)

        return errors
