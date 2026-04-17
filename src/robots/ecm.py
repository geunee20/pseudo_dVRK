from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.robots.robot import Robot
from src.utils.transforms import (
    normalize_vector,
    tool_transform_world as compute_tool_transform_world,
)
from src.utils.urdfParser import parse_urdf


@dataclass
class CameraPose:
    position: np.ndarray
    focal_point: np.ndarray
    viewup: np.ndarray


class ECM(Robot):
    def __init__(
        self,
        robot_root: str | Path = "../urdfs/ecm",
        world_link: str = "world",
        base_transform: Optional[np.ndarray] = None,
    ):
        super().__init__(
            name="ECM",
            robot_root=robot_root,
            base_link="base_link",
            tool_link="end_link",
            world_link=world_link,
        )
        self.base_transform: Optional[np.ndarray] = (
            None if base_transform is None else np.asarray(base_transform, dtype=float)
        )
        self.urdf_path = self.robot_root / "ecm.urdf"
        self.build()
        self.finalize()

    @classmethod
    def from_camera_pose(
        cls,
        camera_world_tf: np.ndarray,
        robot_root: str | Path = "../urdfs/ecm",
        world_link: str = "world",
        initial_q: Optional[np.ndarray] = None,
    ) -> "ECM":
        """
        Create an ECM whose base_transform places the tool tip
        at `camera_world_tf` when the joint configuration is `initial_q`.

        base_transform = camera_world_tf @ inv(FK_local(initial_q))
        """
        tmp = cls(robot_root=robot_root, world_link=world_link, base_transform=None)
        q = (
            np.zeros(tmp.dof)
            if initial_q is None
            else np.asarray(initial_q, dtype=float)
        )
        T_fk = tmp.tool_transform_world(theta=q)  # FK with identity base
        base_transform = np.asarray(camera_world_tf, dtype=float) @ np.linalg.inv(T_fk)
        return cls(
            robot_root=robot_root, world_link=world_link, base_transform=base_transform
        )

    def build(self) -> None:
        parse_urdf(self, self.urdf_path)

        self.active_joint_names = [
            name
            for name, joint in self.joints.items()
            if joint.mimic is None and joint.joint_type != "fixed"
        ]

    def tool_transform_world(
        self,
        theta: Optional[np.ndarray | List[float]] = None,
        base_transform: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return tool transform in world frame."""
        theta_arr = (
            None if theta is None else np.asarray(theta, dtype=float).reshape(-1)
        )
        bt = base_transform if base_transform is not None else self.base_transform
        return compute_tool_transform_world(
            self,
            theta=theta_arr,
            base_transform=bt,
        )

    def scope_camera_pose(
        self,
        theta: Optional[np.ndarray | List[float]] = None,
        base_transform: Optional[np.ndarray] = None,
        optical_tilt_deg: float = 0.0,
        focus_distance: float = 0.2,
    ) -> CameraPose:
        """
        Build a monocular camera pose from tool frame.

        Optical tilt rotates around local +X axis of the tool frame.
        """
        T_world_tool = self.tool_transform_world(
            theta=theta, base_transform=base_transform
        )

        p = T_world_tool[:3, 3]
        y_axis = normalize_vector(T_world_tool[:3, 1])
        z_axis = normalize_vector(T_world_tool[:3, 2])

        tilt_rad = np.deg2rad(float(optical_tilt_deg))
        c = float(np.cos(tilt_rad))
        s = float(np.sin(tilt_rad))

        # Rotate local basis around X axis: y' = c*y + s*z, z' = -s*y + c*z.
        cam_up = normalize_vector(c * y_axis + s * z_axis)
        cam_forward = normalize_vector(-s * y_axis + c * z_axis)

        if np.linalg.norm(cam_forward) < 1e-12:
            cam_forward = z_axis
        if np.linalg.norm(cam_up) < 1e-12:
            cam_up = y_axis

        focal = p + float(focus_distance) * cam_forward
        return CameraPose(position=p, focal_point=focal, viewup=cam_up)

    def scope_stereo_camera_poses(
        self,
        theta: Optional[np.ndarray | List[float]] = None,
        base_transform: Optional[np.ndarray] = None,
        optical_tilt_deg: float = 0.0,
        baseline: float = 0.008,
        focus_distance: float = 0.2,
    ) -> Tuple[CameraPose, CameraPose]:
        """
        Build a parallel stereo camera pair from tool frame.

        left/right are translated by +/- baseline/2 along local +X axis,
        while sharing the same forward and up directions.
        """
        center = self.scope_camera_pose(
            theta=theta,
            base_transform=base_transform,
            optical_tilt_deg=optical_tilt_deg,
            focus_distance=focus_distance,
        )

        T_world_tool = self.tool_transform_world(
            theta=theta, base_transform=base_transform
        )
        x_axis = normalize_vector(T_world_tool[:3, 0])

        half = 0.5 * float(baseline)
        left_pos = center.position - half * x_axis
        right_pos = center.position + half * x_axis

        forward = normalize_vector(center.focal_point - center.position)
        left = CameraPose(
            position=left_pos,
            focal_point=left_pos + float(focus_distance) * forward,
            viewup=center.viewup.copy(),
        )
        right = CameraPose(
            position=right_pos,
            focal_point=right_pos + float(focus_distance) * forward,
            viewup=center.viewup.copy(),
        )
        return left, right
