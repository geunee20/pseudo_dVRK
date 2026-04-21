from __future__ import annotations

from pathlib import Path

from src.robots.robot import Robot
from src.utils.urdf_parser import parse_urdf


class UrdfRobot(Robot):
    """Robot base class for URDF-backed robots with standard active-joint rules."""

    def __init__(
        self,
        *,
        name: str,
        robot_root: str | Path,
        base_link: str,
        tool_link: str,
        urdf_filename: str,
        world_link: str = "world",
    ) -> None:
        super().__init__(
            name=name,
            robot_root=robot_root,
            base_link=base_link,
            tool_link=tool_link,
            world_link=world_link,
        )
        self.urdf_path = self.robot_root / urdf_filename
        self.build()
        self.finalize()

    def build(self) -> None:
        parse_urdf(self, self.urdf_path)
        self.active_joint_names = self._default_active_joint_names()

    def _default_active_joint_names(self) -> list[str]:
        return [
            name
            for name, joint in self.joints.items()
            if joint.mimic is None and joint.joint_type != "fixed"
        ]
