from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =========================
# Data containers
# =========================


@dataclass
class JointLimit:
    """URDF joint position limits.

    Attributes:
        lower: Lower joint position limit (radians or metres).
        upper: Upper joint position limit (radians or metres).
    """

    lower: float
    upper: float


@dataclass
class Mimic:
    """URDF ``<mimic>`` specification linking one joint to another.

    The mimicking joint value is computed as:

    .. math::

        q_{\\text{mimic}} = \\text{multiplier} \\cdot q_{\\text{source}} + \\text{offset}

    Attributes:
        joint: Name of the source (master) joint.
        multiplier: Linear gain applied to the source joint value.
        offset: Constant offset added after scaling.
    """

    joint: str
    multiplier: float = 1.0
    offset: float = 0.0


@dataclass
class JointInfo:
    """Parsed representation of a single URDF joint.

    Attributes:
        name: Joint name from the URDF.
        joint_type: One of ``'revolute'``, ``'prismatic'``, or ``'fixed'``.
        parent: Name of the parent link.
        child: Name of the child link.
        origin_xyz: (3,) translation of the joint frame relative to the parent
            link frame (metres).
        origin_rpy: (3,) roll-pitch-yaw orientation of the joint frame relative
            to the parent link frame (radians).
        axis: (3,) joint motion axis expressed in the joint local frame.
        limit: Optional joint position limits; ``None`` for fixed joints.
        mimic: Optional mimic relationship; ``None`` for non-mimic joints.
    """

    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    limit: Optional[JointLimit] = None
    mimic: Optional[Mimic] = None


@dataclass
class LinkVisual:
    """Path and local origin of a single visual mesh element for a URDF link.

    Attributes:
        link_name: Name of the parent link.
        mesh_path: Absolute path to the mesh file (STL preferred over DAE).
        origin_xyz: (3,) local translation of the visual in the link frame.
        origin_rpy: (3,) local roll-pitch-yaw orientation of the visual in the
            link frame (radians).
    """

    link_name: str
    mesh_path: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray


@dataclass
class LinkCollision:
    """Collision geometry attached to a URDF link.

    Attributes:
        link_name: Name of the parent link.
        geometry_type: One of ``"mesh"``, ``"box"``, ``"sphere"``, or
            ``"cylinder"``.
        origin_xyz: (3,) local translation of the collision shape in the link
            frame (metres).
        origin_rpy: (3,) local roll-pitch-yaw orientation of the collision
            shape in the link frame (radians).
        mesh_path: Absolute path to the collision mesh, when
            ``geometry_type == "mesh"``.
        size: Box side lengths ``(x, y, z)`` for box collisions.
        radius: Sphere/cylinder radius in metres.
        length: Cylinder length in metres.
    """

    link_name: str
    geometry_type: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    mesh_path: str | None = None
    size: np.ndarray | None = None
    radius: float | None = None
    length: float | None = None


# =========================
# Base Interface
# =========================


class Robot(ABC):
    """
    Base interface for dVRK robots (PSM, ECM, MTM).

    Responsibilities:
    - store robot structure (links, joints, visuals)
    - manage joint state (theta)
    - provide metadata access

    NOT responsible for:
    - forward kinematics
    - inverse kinematics
    - Jacobians
    - rendering
    """

    def __init__(
        self,
        name: str,
        robot_root: str | Path,
        base_link: str,
        tool_link: str,
        world_link: str = "world",
    ) -> None:
        self.name = name
        self.robot_root = Path(robot_root)
        self.mesh_root = self.robot_root / "meshes"

        self.world_link = world_link
        self.base_link = base_link
        self.tool_link = tool_link

        # topology
        self.links: Dict[str, dict] = {}
        self.joints: Dict[str, JointInfo] = {}
        self.visuals: Dict[str, List[LinkVisual]] = {}
        self.collisions: Dict[str, List[LinkCollision]] = {}
        self.children_of_link: Dict[str, List[str]] = {}

        # DOF
        self.active_joint_names: List[str] = []
        self.theta = np.zeros(0)

        # kinematics
        self.kinematic_model = None

    # =========================
    # Required builder
    # =========================

    @abstractmethod
    def build(self) -> None:
        """Populate links, joints, visuals, and active_joint_names."""
        raise NotImplementedError

    def finalize(self) -> None:
        """Call after build(). Initializes joint state."""
        self.theta = np.zeros(self.dof)
        from src.kinematics.kinematic_model import build_kinematic_model

        self.kinematic_model = build_kinematic_model(self)

    # =========================
    # Basic properties
    # =========================

    @property
    def dof(self) -> int:
        return len(self.active_joint_names)

    @property
    def link_names(self) -> List[str]:
        return list(self.links.keys())

    @property
    def joint_names(self) -> List[str]:
        return list(self.joints.keys())

    # =========================
    # Joint state
    # =========================

    def set_theta(self, theta: np.ndarray | List[float]) -> None:
        """Set the active joint vector, validating its size.

        Args:
            theta: (dof,) joint configuration.

        Raises:
            ValueError: If ``len(theta) != dof``.
        """
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != self.dof:
            raise ValueError(f"Expected {self.dof}, got {theta.size}")
        self.theta = theta.copy()

    def get_theta(self) -> np.ndarray:
        """Return a copy of the current active joint vector."""
        return self.theta.copy()

    def zero_theta(self) -> np.ndarray:
        """Reset all active joints to zero and return the resulting vector."""
        self.theta = np.zeros(self.dof)
        return self.get_theta()

    def clamp_theta(self, theta: np.ndarray | List[float]) -> np.ndarray:
        """Return a copy of *theta* with each joint clamped to its URDF limits.

        Joints without a ``<limit>`` tag are left unchanged.

        Args:
            theta: (dof,) joint configuration to clamp.

        Raises:
            ValueError: If ``len(theta) != dof``.

        Returns:
            (dof,) clamped joint configuration.
        """
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != self.dof:
            raise ValueError(f"Expected {self.dof}, got {theta.size}")

        out = theta.copy()
        for i, name in enumerate(self.active_joint_names):
            joint = self.joints[name]
            if joint.limit is not None:
                out[i] = np.clip(out[i], joint.limit.lower, joint.limit.upper)
        return out

    # =========================
    # Joint metadata
    # =========================

    def get_joint(self, name: str) -> JointInfo:
        return self.joints[name]

    def get_active_joints(self) -> List[JointInfo]:
        return [self.joints[n] for n in self.active_joint_names]

    def joint_limits(self) -> Dict[str, Tuple[float, float]]:
        limit = {}
        for name in self.active_joint_names:
            joint = self.joints[name]
            if joint.limit is not None:
                limit[name] = (joint.limit.lower, joint.limit.upper)
        return limit

    def expand_theta(
        self, theta: Optional[np.ndarray | List[float]] = None
    ) -> Dict[str, float]:
        """Expand active-joint vector to a full joint dictionary including mimic joints.

        Mimic joints are resolved iteratively using the formula:

        .. math::

            q_{\\text{mimic}} = \\text{multiplier} \\cdot q_{\\text{source}} + \\text{offset}

        Revolute and prismatic joints not covered by active or mimic rules
        default to 0.

        Args:
            theta: (dof,) active-joint vector.  Defaults to ``self.theta``.

        Raises:
            ValueError: If ``len(theta) != dof``.

        Returns:
            Dictionary mapping every non-fixed joint name to its float value.
        """
        if theta is None:
            theta = self.theta

        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != self.dof:
            raise ValueError(f"Expected {self.dof}, got {theta.size}")

        q = {n: float(v) for n, v in zip(self.active_joint_names, theta)}

        changed = True
        while changed:
            changed = False
            for name, joint in self.joints.items():
                if joint.mimic is None:
                    continue
                src = joint.mimic.joint
                if src not in q:
                    continue
                val = joint.mimic.multiplier * q[src] + joint.mimic.offset
                if name not in q or abs(q[name] - val) > 1e-12:
                    q[name] = val
                    changed = True

        # fill missing
        for name, joint in self.joints.items():
            if joint.joint_type in ("revolute", "prismatic") and name not in q:
                q[name] = 0.0

        return q

    # =========================
    # Link / tree
    # =========================

    def get_child_joints(self, link: str) -> List[str]:
        return list(self.children_of_link.get(link, []))

    def get_link_visuals(self, link: Optional[str] = None) -> List[LinkVisual]:
        if link is None:
            out = []
            for visuals in self.visuals.values():
                out.extend(visuals)
            return out
        return list(self.visuals.get(link, []))

    def get_link_collisions(self, link: Optional[str] = None) -> List[LinkCollision]:
        if link is None:
            out = []
            for collisions in self.collisions.values():
                out.extend(collisions)
            return out
        return list(self.collisions.get(link, []))

    # =========================
    # Registration helpers
    # =========================

    def add_link(self, name: str) -> None:
        self.links[name] = {"name": name}
        self.visuals.setdefault(name, [])
        self.collisions.setdefault(name, [])

    def add_joint(self, joint: JointInfo, active: bool = False) -> None:
        self.joints[joint.name] = joint
        self.children_of_link.setdefault(joint.parent, []).append(joint.name)
        if active:
            self.active_joint_names.append(joint.name)

    def add_visual(self, visual: LinkVisual) -> None:
        self.visuals.setdefault(visual.link_name, []).append(visual)

    def add_collision(self, collision: LinkCollision) -> None:
        self.collisions.setdefault(collision.link_name, []).append(collision)

    def resolve_mesh_path(self, filename: str) -> Path:
        return self.mesh_root / filename

    # =========================
    # Kinematics
    # =========================
    def build_kinematic_model(self):
        from src.kinematics.kinematic_model import build_kinematic_model

        return build_kinematic_model(self)

    def forward_kinematics(self, theta=None, link_name=None):
        from src.kinematics.fk import forward_kinematics

        return forward_kinematics(self, theta=theta, link_name=link_name)

    def link_transforms(self, theta=None):
        from src.kinematics.fk import link_transforms

        return link_transforms(self, theta=theta)

    # =========================
    # Debug
    # =========================

    def summary(self) -> dict:
        return {
            "name": self.name,
            "dof": self.dof,
            "base_link": self.base_link,
            "tool_link": self.tool_link,
            "active_joints": self.active_joint_names,
        }

    def visualize(
        self,
        theta: np.ndarray | None = None,
        show_frames: bool = False,
        alpha: float = 1.0,
    ) -> None:
        """Render this robot in a simple interactive viewer.

        Args:
            theta: Optional active-joint configuration, shape ``(dof,)``.
            show_frames: Whether to draw joint frames.
            alpha: Robot mesh opacity in ``[0, 1]``.
        """
        from src.utils.visualization.viewers import set_camera_view, visualize

        scene = visualize(self, theta=theta, show_frames=show_frames, alpha=alpha)
        set_camera_view(scene, eye=[1, 1, 1], target=[0, 0, 0])
        scene.show()
