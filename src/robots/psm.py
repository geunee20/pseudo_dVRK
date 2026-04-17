from pathlib import Path

from src.robots.robot import Robot
from src.utils.urdfParser import parse_urdf


class PSM(Robot):
    def __init__(
        self,
        robot_root: str | Path = "../urdfs/psm",
        world_link: str = "world",
    ):
        super().__init__(
            name="PSM",
            robot_root=robot_root,
            base_link="base_link",
            tool_link="tool_yaw_link",
            world_link=world_link,
        )
        self.urdf_path = self.robot_root / "psm.urdf"
        self.build()
        self.finalize()

    def build(self) -> None:
        parse_urdf(self, self.urdf_path)

        self.active_joint_names = [
            name
            for name, joint in self.joints.items()
            if joint.mimic is None and joint.joint_type != "fixed"
        ]
