from pathlib import Path

from src.robots.robot import Robot
from src.utils.urdfParser import parse_urdf


class Phantom(Robot):
    def __init__(
        self,
        robot_root: str | Path = "../urdfs/phantom_touch",
        world_link: str = "base",
    ):
        super().__init__(
            name="Phantom",
            robot_root=robot_root,
            base_link="base",
            tool_link="stylus_point",
            world_link=world_link,
        )
        self.urdf_path = self.robot_root / "phantom_touch.urdf"

        self.build()
        self.finalize()

    def build(self) -> None:
        parse_urdf(self, self.urdf_path)

        self.active_joint_names = [
            name
            for name, joint in self.joints.items()
            if joint.mimic is None and joint.joint_type != "fixed"
        ]
