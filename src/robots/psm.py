from pathlib import Path

from src.robots.urdf_robot import UrdfRobot


class PSM(UrdfRobot):
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
            urdf_filename="psm.urdf",
            world_link=world_link,
        )
