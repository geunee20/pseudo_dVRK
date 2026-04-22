from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from src.robots.protocols import CollisionRobotLike, RobotTreeLike, VisualRobotLike
from .se3 import exp_screw_axis, origin_transform

if TYPE_CHECKING:
    from src.robots.robot import JointInfo, LinkCollision, LinkVisual


@dataclass
class VisualPose:
    link_name: str
    visual: LinkVisual
    T_world: np.ndarray


@dataclass
class CollisionPose:
    link_name: str
    collision: LinkCollision
    T_world: np.ndarray


@dataclass
class JointFrame:
    joint_name: str
    joint: JointInfo
    T_world: np.ndarray


def joint_transform(joint_type: str, axis: np.ndarray, q: float) -> np.ndarray:
    """Return the joint-generated homogeneous transform for a single URDF joint.

    * **Revolute** joint with axis :math:`\\hat{a}`:

      .. math::

          T = e^{[\\hat{a},\\, 0]\\, q}

    * **Prismatic** joint with axis :math:`\\hat{a}`:

      .. math::

          T = e^{[0,\\, \\hat{a}]\\, q}

    * **Fixed** joint: returns :math:`I_4`.

    Args:
        joint_type: One of ``'revolute'``, ``'prismatic'``, or ``'fixed'``.
        axis: (3,) joint axis vector (expressed in the joint local frame).
        q: Scalar joint displacement (radians for revolute, metres for
            prismatic).

    Raises:
        ValueError: If *joint_type* is not one of the supported types.

    Returns:
        4×4 homogeneous transform :math:`T \\in SE(3)`.
    """
    axis = np.asarray(axis, dtype=float).reshape(
        3,
    )

    if joint_type == "fixed":
        return np.eye(4)

    if joint_type == "revolute":
        S = np.concatenate([axis, np.zeros(3)])
        return exp_screw_axis(S, q)

    if joint_type == "prismatic":
        S = np.concatenate([np.zeros(3), axis])
        return exp_screw_axis(S, q)

    raise ValueError(f"Unsupported joint type: {joint_type}")


def link_transforms(
    robot: RobotTreeLike,
    theta: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute world-frame transforms for every link via a depth-first tree traversal.

    Starting from the world link at :math:`T = I_4`, the function propagates
    transforms through the kinematic tree:

    .. math::

        T_{\\text{child}} = T_{\\text{parent}} \\cdot T_{\\text{joint origin}} \\cdot T_{\\text{joint}}(q)

    where :math:`T_{\\text{joint origin}}` comes from the URDF ``<origin>`` tag and
    :math:`T_{\\text{joint}}(q)` is the joint-generated transform from
    :func:`joint_transform`.

    Args:
        robot: Robot object implementing the :class:`~src.robots.protocols.RobotTreeLike`
            interface.
        theta: (dof,) active joint vector.  If ``None``, ``robot.get_theta()``
            is used.

    Returns:
        Dictionary mapping ``link_name`` → 4×4 world-frame transform.
    """
    if theta is None:
        theta = robot.get_theta()

    q_full = robot.expand_theta(theta)
    T_links = {robot.world_link: np.eye(4)}

    def _dfs(parent_link: str) -> None:
        T_parent = T_links[parent_link]

        for joint_name in robot.get_child_joints(parent_link):
            joint = robot.get_joint(joint_name)

            T_parent_joint = origin_transform(joint.origin_xyz, joint.origin_rpy)
            T_joint_child = joint_transform(
                joint.joint_type,
                joint.axis,
                q_full.get(joint.name, 0.0),
            )

            T_child = T_parent @ T_parent_joint @ T_joint_child
            T_links[joint.child] = T_child
            _dfs(joint.child)

    _dfs(robot.world_link)

    return T_links


def forward_kinematics(
    robot: RobotTreeLike,
    theta: np.ndarray | None = None,
    link_name: str | None = None,
) -> np.ndarray:
    """Return the world-frame pose of a single link (generic URDF tree FK).

    Internally calls :func:`link_transforms` and returns the transform for the
    requested link directly.

    Args:
        robot: Robot object implementing the
            :class:`~src.robots.protocols.RobotTreeLike` interface.
        theta: (dof,) active joint vector.  If ``None``, ``robot.get_theta()`` is used.
        link_name: Name of the link whose pose is requested.  Defaults to
            ``robot.tool_link``.

    Raises:
        KeyError: If *link_name* is not present in the computed transform tree.

    Returns:
        4×4 world-frame homogeneous transform of the requested link.
    """
    if link_name is None:
        link_name = robot.tool_link

    T_links = link_transforms(robot, theta)

    if link_name not in T_links:
        raise KeyError(f"Link '{link_name}' not found in transform tree.")

    return T_links[link_name]


def visual_transforms(
    robot: VisualRobotLike,
    theta: np.ndarray | None = None,
    base_transform: np.ndarray | None = None,
) -> list[VisualPose]:
    """
    Compute world-frame transforms of all visual geometries of a robot.

    This function evaluates forward kinematics for all links and composes
    each link transform with its associated visual origin to obtain the
    final pose of each mesh in the world frame.

    Inputs
    ------
    robot : Robot-like object
        Must implement:
            - link_names
            - get_link_visuals(link_name)
            - expand_theta()
            - get_theta()
            - get_child_joints()
            - get_joint()

    theta : (dof,) array-like, optional
        Active joint vector. If None, uses robot.get_theta().

    base_transform : (4,4) ndarray, optional
        Homogeneous transform from the robot base frame to the world frame.
        If None, identity is used.

    Returns
    -------
    list[VisualPose]
        A list of VisualPose objects, each containing:
            - link_name : name of the parent link
            - visual    : LinkVisual object (includes mesh_path and local origin)
            - T_world   : (4,4) homogeneous transform of the visual in world frame
    """
    if base_transform is None:
        base_transform = np.eye(4)
    else:
        base_transform = np.asarray(base_transform, dtype=float).reshape(4, 4)

    T_links = link_transforms(robot, theta)
    out: list[VisualPose] = []

    for link_name in robot.link_names:
        if link_name not in T_links:
            continue

        T_world_link = base_transform @ T_links[link_name]

        for visual in robot.get_link_visuals(link_name):
            T_link_visual = origin_transform(visual.origin_xyz, visual.origin_rpy)
            T_world_visual = T_world_link @ T_link_visual

            out.append(VisualPose(link_name, visual, T_world_visual))

    return out


def collision_transforms(
    robot: CollisionRobotLike,
    theta: np.ndarray | None = None,
    base_transform: np.ndarray | None = None,
) -> list[CollisionPose]:
    """Compute world-frame transforms of all collision geometries of a robot."""
    if base_transform is None:
        base_transform = np.eye(4)
    else:
        base_transform = np.asarray(base_transform, dtype=float).reshape(4, 4)

    T_links = link_transforms(robot, theta)
    out: list[CollisionPose] = []

    for link_name in robot.link_names:
        if link_name not in T_links:
            continue

        T_world_link = base_transform @ T_links[link_name]
        for collision in robot.get_link_collisions(link_name):
            T_link_collision = origin_transform(
                collision.origin_xyz,
                collision.origin_rpy,
            )
            out.append(
                CollisionPose(
                    link_name=link_name,
                    collision=collision,
                    T_world=T_world_link @ T_link_collision,
                )
            )

    return out


def joint_frames(
    robot: VisualRobotLike,
    theta: np.ndarray | None = None,
    base_transform: np.ndarray | None = None,
) -> list[JointFrame]:
    """
    Compute world-frame transforms of all active joint frames of a robot.

    This function evaluates forward kinematics for all links and composes
    each parent-link transform with the joint origin transform to obtain
    the pose of each active joint frame in the world frame.

    Inputs
    ------
    robot : Robot-like object
        Must implement:
            - active_joint_names
            - get_joint(joint_name)
            - get_theta()
            - expand_theta()
            - get_child_joints(link_name)
            - link_names

    theta : (dof,) array-like, optional
        Active joint vector. If None, uses robot.get_theta().

    base_transform : (4,4) ndarray, optional
        Homogeneous transform from the robot base/world reference used by
        the caller to the rendered world frame. If None, identity is used.

    Returns
    -------
    list[JointFrame]
        A list of JointFrame objects, each containing:
            - joint_name : name of the joint
            - joint      : JointInfo object
            - T_world    : (4,4) homogeneous transform of the joint frame
                           in the world frame

    Notes
    -----
    - The returned transform corresponds to the joint frame located at the
      URDF joint origin.
    - This function performs only geometric pose computation and does NOT
      perform any rendering.
    """
    if base_transform is None:
        base_transform = np.eye(4)
    else:
        base_transform = np.asarray(base_transform, dtype=float).reshape(4, 4)

    T_links = link_transforms(robot, theta)
    out: list[JointFrame] = []

    for joint_name in robot.active_joint_names:
        joint = robot.get_joint(joint_name)
        parent = joint.parent

        if parent not in T_links:
            continue

        T_world_parent = base_transform @ T_links[parent]
        T_parent_joint = origin_transform(joint.origin_xyz, joint.origin_rpy)
        T_world_joint = T_world_parent @ T_parent_joint

        out.append(
            JointFrame(
                joint_name=joint_name,
                joint=joint,
                T_world=T_world_joint,
            )
        )

    return out
