"""Simulation framework for verification using actual robot models."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .data_structures import CameraConfig, RobotConfig, Capture, Detection
from .transforms import compose_transforms, invert_transform, axis_angle_to_rotation_matrix, translation_error, rotation_error


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
            camera_id, parent_link = spec["camera_id"], spec["parent_link"]
            xyz, rpy = np.array(spec["xyz"]), np.array(spec["rpy"])

            # Build ground truth link_T_cam from xyz + rpy (ZYX convention: R = Rz*Ry*Rx)
            link_T_cam = np.eye(4)
            Rx = axis_angle_to_rotation_matrix([rpy[0], 0, 0])
            Ry = axis_angle_to_rotation_matrix([0, rpy[1], 0])
            Rz = axis_angle_to_rotation_matrix([0, 0, rpy[2]])
            link_T_cam[:3, :3] = Rz @ Ry @ Rx
            link_T_cam[:3, 3] = xyz
            ground_truth_transforms[camera_id] = link_T_cam

            # Perturb for initial estimate
            init_estimate = SimulationFramework._perturb_transform(
                link_T_cam, perturbation_translation_m, perturbation_rotation_deg
            )

            cameras.append(CameraConfig(camera_id=camera_id, parent_link=parent_link,
                                       init_link_T_cam=init_estimate))

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

        # Generate random ground truth transforms
        ground_truth_transforms = {
            cid: SimulationFramework._make_transform(
                np.random.uniform(-0.2, 0.2, 3),
                np.random.uniform(-np.pi / 4, np.pi / 4, 3)
            )
            for cid in camera_ids
        }

        # Perturb to create initial estimates
        cameras = [
            CameraConfig(
                camera_id=cid,
                parent_link=plink,
                init_link_T_cam=SimulationFramework._perturb_transform(
                    ground_truth_transforms[cid],
                    perturbation_translation_m,
                    perturbation_rotation_deg
                )
            )
            for cid, plink in zip(camera_ids, parent_links)
        ]

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

    @staticmethod
    def _make_transform(translation: np.ndarray, axis_angle: np.ndarray) -> np.ndarray:
        """Helper: Build 4x4 SE3 transform from translation + axis-angle rotation."""
        T = np.eye(4)
        T[:3, :3] = axis_angle_to_rotation_matrix(axis_angle)
        T[:3, 3] = translation
        return T

    @staticmethod
    def _perturb_transform(T: np.ndarray, trans_m: float, rot_deg: float) -> np.ndarray:
        """Add random perturbation to transform."""
        perturb_T = np.eye(4)
        perturb_T[:3, :3] = axis_angle_to_rotation_matrix(
            np.random.uniform(-np.deg2rad(rot_deg), np.deg2rad(rot_deg), 3)
        )
        perturb_T[:3, 3] = np.random.uniform(-trans_m, trans_m, 3)
        return T @ perturb_T

    def generate_capture(
        self,
        capture_id: int,
        num_objects: int = 3,
        link_poses: Optional[Dict[str, np.ndarray]] = None,
        object_poses: Optional[List[np.ndarray]] = None,
    ) -> Capture:
        """Generate synthetic capture with detections.

        Uses URDF FK for link poses if model loaded, otherwise random poses.
        """
        # Generate link poses: URDF FK if available, else random
        if link_poses is None:
            if self.urdf_model is not None:
                link_poses = self.generate_urdf_link_poses()
            else:
                # Random link poses (fallback when no URDF)
                link_poses = {
                    link: SimulationFramework._make_transform(
                        np.random.uniform(-1.0, 1.0, 3),
                        np.random.uniform(-np.pi, np.pi, 3)
                    )
                    for link in set(c.parent_link for c in self.robot_config.cameras)
                }

        # Generate random object poses if not provided (in front of robot for AprilTag-like target)
        if object_poses is None:
            object_poses = [
                SimulationFramework._make_transform(
                    np.random.uniform([0.3, -0.5, 0.0], [1.5, 0.5, 1.0]),  # In front, left/right, above ground
                    np.random.uniform(-np.pi / 4, np.pi / 4, 3)  # Small random tilt
                )
                for _ in range(num_objects)
            ]

        detections = []
        for camera in self.robot_config.cameras:
            base_T_link = link_poses[camera.parent_link]
            link_T_cam_gt = self.ground_truth_transforms[camera.camera_id]

            for object_id, base_T_object in enumerate(object_poses):
                # Check visibility: distance-based constraint
                base_T_cam = compose_transforms(base_T_link, link_T_cam_gt)
                cam_T_object = compose_transforms(invert_transform(base_T_cam), base_T_object)
                distance = np.linalg.norm(cam_T_object[:3, 3])
                if not (0.1 <= distance <= 3.0):  # min/max viewing distance
                    continue

                # Add detection noise (simulates AprilTag pose estimation error)
                cam_T_object_noisy = cam_T_object.copy()
                cam_T_object_noisy[:3, 3] += np.random.normal(0, self.noise_translation_m, 3)
                cam_T_object_noisy[:3, :3] @= axis_angle_to_rotation_matrix(
                    np.random.normal(0, self.noise_rotation_rad, 3)
                )

                # Random tag count (more tags → higher weight in optimization)
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

    def generate_dataset(self, num_captures: int, num_objects_per_capture: int = 3) -> List[Capture]:
        """Generate full synthetic dataset."""
        return [self.generate_capture(i, num_objects_per_capture) for i in range(num_captures)]

    def compute_ground_truth_errors(self, optimized_transforms: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float]]:
        """Compute errors between optimized and ground truth transforms. Returns (mm, degrees)."""
        return {
            camera_id: (
                translation_error(optimized_T, self.ground_truth_transforms[camera_id]) * 1000,
                np.rad2deg(rotation_error(optimized_T, self.ground_truth_transforms[camera_id]))
            )
            for camera_id, optimized_T in optimized_transforms.items()
        }
