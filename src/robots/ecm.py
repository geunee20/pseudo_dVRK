from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.robots.urdf_robot import UrdfRobot
from src.utils.transforms import (
    normalize_vector,
    tool_transform_world as compute_tool_transform_world,
)


@dataclass
class CameraPose:
    position: np.ndarray
    focal_point: np.ndarray
    viewup: np.ndarray


class ECM(UrdfRobot):
    """Endoscope Camera Manipulator (ECM) robot with camera-pose helpers.

    Extends :class:`~src.robots.urdf_robot.UrdfRobot` with:

    * An optional ``base_transform`` that maps the robot's local FK frame
      into the world frame.
    * :meth:`scope_camera_pose` / :meth:`scope_stereo_camera_poses` for
      computing PyVista camera parameters directly from the ECM joint state.
    * :meth:`from_camera_pose` class method to construct an ECM whose
      base transform is chosen so the scope tip matches a desired world pose.

    Args:
        robot_root: Directory containing ``ecm.urdf`` and ``meshes/``.
        world_link: Name of the root link in the URDF tree.
        base_transform: Optional 4×4 transform from the robot base frame to
            the world frame.  If ``None``, identity is used.
    """

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
            urdf_filename="ecm.urdf",
            world_link=world_link,
        )
        self.base_transform: Optional[np.ndarray] = (
            None if base_transform is None else np.asarray(base_transform, dtype=float)
        )

    @classmethod
    def from_camera_pose(
        cls,
        camera_world_tf: np.ndarray,
        robot_root: str | Path = "../urdfs/ecm",
        world_link: str = "world",
        initial_q: Optional[np.ndarray] = None,
    ) -> "ECM":
        """Construct an ECM whose scope tip is placed at *camera_world_tf*.

        Computes the required ``base_transform`` by inverting the FK at
        *initial_q*:

        .. math::

            T_{\\text{base}}^{\\text{world}}
            = T_{\\text{cam}}^{\\text{world}} \\cdot
              \\bigl(T_{\\text{FK}}(q_0)\\bigr)^{-1}

        Args:
            camera_world_tf: 4×4 desired scope-tip pose in the world frame.
            robot_root: Directory containing the ECM URDF and meshes.
            world_link: Name of the root link in the URDF tree.
            initial_q: (dof,) joint configuration at which the scope tip should
                coincide with *camera_world_tf*.  Defaults to all-zeros.

        Returns:
            :class:`ECM` instance with ``base_transform`` set accordingly.
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

    def tool_transform_world(
        self,
        theta: Optional[np.ndarray | List[float]] = None,
        base_transform: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return the scope-tip (tool) pose in the world frame.

        Applies the optional *base_transform* on top of the URDF tree FK:

        .. math::

            T_{\\text{tool}}^{\\text{world}} = T_{\\text{base}}^{\\text{world}}
                \\cdot T_{\\text{FK}}(\\theta)

        Args:
            theta: (dof,) joint configuration.  Defaults to the current
                stored joint state.
            base_transform: Override for ``self.base_transform``.  If both
                are ``None``, the FK result is returned unchanged.

        Returns:
            4×4 homogeneous scope-tip pose in the world frame.
        """
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
        """Compute a monocular camera pose from the current ECM joint state.

        The camera is placed at the scope tip.  A tilt about the local
        :math:`+X` axis (the optical tilt) adjusts the viewing direction:

        .. math::

            \\hat{y}' &= c\\,\\hat{y} + s\\,\\hat{z} \\\\
            \\hat{z}' &= -s\\,\\hat{y} + c\\,\\hat{z}

        where :math:`c = \\cos(\\alpha)`, :math:`s = \\sin(\\alpha)`,
        :math:`\\alpha = \\text{optical\\_tilt\\_deg}`.

        Args:
            theta: (dof,) joint configuration.  Defaults to the stored state.
            base_transform: Optional base-to-world override.
            optical_tilt_deg: Optical tilt angle around the scope X axis
                (degrees).  Typically 0° or 30° for 0°/30° scopes.
            focus_distance: Distance (metres) from the camera origin to the
                focal point along the viewing direction.

        Returns:
            :class:`CameraPose` with ``position``, ``focal_point``, and
            ``viewup`` vectors.
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
        """Compute a parallel stereo camera pair from the current ECM joint state.

        Left and right cameras are translated by :math:`\\pm \\text{baseline}/2`
        along the local :math:`+X` axis of the scope tip while sharing the same
        forward direction and viewup:

        .. math::

            p_{\\text{left}}  &= p_{\\text{center}} - \\tfrac{d}{2}\\,\\hat{x} \\\\
            p_{\\text{right}} &= p_{\\text{center}} + \\tfrac{d}{2}\\,\\hat{x}

        where :math:`d` is *baseline*.

        Args:
            theta: (dof,) joint configuration.  Defaults to the stored state.
            base_transform: Optional base-to-world override.
            optical_tilt_deg: Optical tilt angle (degrees) applied before the
                stereo split.
            baseline: Stereo baseline (metres), i.e. inter-camera separation.
            focus_distance: Distance (metres) to the focal point along the
                shared viewing direction.

        Returns:
            ``(left_pose, right_pose)`` — each a :class:`CameraPose`.
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
