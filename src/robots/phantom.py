from pathlib import Path

from src.robots.urdf_robot import UrdfRobot


class Phantom(UrdfRobot):
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
            urdf_filename="phantom_touch.urdf",
            world_link=world_link,
        )
