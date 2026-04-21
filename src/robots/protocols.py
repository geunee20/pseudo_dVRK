from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Protocol, Sequence

import numpy as np

if TYPE_CHECKING:
    from src.robots.robot import JointInfo, LinkVisual


class JointStateLike(Protocol):
    @property
    def joints(self) -> Sequence[float]: ...


class RobotTreeLike(Protocol):
    @property
    def world_link(self) -> str: ...

    @property
    def tool_link(self) -> str: ...

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
    @property
    def link_names(self) -> Sequence[str]: ...

    @property
    def active_joint_names(self) -> Sequence[str]: ...

    def get_link_visuals(self, link: str | None = None) -> Sequence[LinkVisual]: ...


class ToolKinematicsRobotLike(RobotTreeLike, Protocol):
    def forward_kinematics(
        self,
        theta: np.ndarray | None = None,
        link_name: str | None = None,
    ) -> np.ndarray: ...
