from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Protocol, Sequence

import numpy as np

if TYPE_CHECKING:
    from src.robots.robot import JointInfo, LinkVisual


class JointStateLike(Protocol):
    """Minimal protocol for objects that expose a joint-angle sequence."""

    @property
    def joints(self) -> Sequence[float]: ...


class RobotTreeLike(Protocol):
    """Structural protocol for robots that expose a kinematic tree.

    Used by :func:`~src.kinematics.fk.link_transforms` and related FK helpers
    so they can operate on any compliant robot object without inheriting from
    a concrete base class.
    """

    @property
    def tool_link(self) -> str: ...

    @property
    def world_link(self) -> str: ...

    @property
    def dof(self) -> int: ...

    def get_theta(self) -> np.ndarray: ...

    def expand_theta(
        self,
        theta: np.ndarray | list[float] | None = None,
    ) -> Mapping[str, float]: ...

    def get_child_joints(self, link: str) -> Sequence[str]: ...

    def get_joint(self, name: str) -> JointInfo: ...


class VisualRobotLike(RobotTreeLike, Protocol):
    """Extended protocol for robots that also expose visual (mesh) information.

    Used by :func:`~src.kinematics.fk.visual_transforms` and the
    :class:`~src.utils.visualization.viewers.DvrkRealtimeViz` viewer.
    """

    @property
    def active_joint_names(self) -> Sequence[str]: ...

    @property
    def link_names(self) -> Sequence[str]: ...

    def get_link_visuals(self, link: str | None = None) -> Sequence[LinkVisual]: ...


class ToolKinematicsRobotLike(RobotTreeLike, Protocol):
    """Protocol for robots that can compute their own forward kinematics.

    Used by :func:`~src.utils.transforms.tool_transform_world`.
    """

    def forward_kinematics(
        self,
        theta: np.ndarray | None = None,
        link_name: str | None = None,
    ) -> np.ndarray: ...
