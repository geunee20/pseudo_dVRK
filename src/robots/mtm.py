from pathlib import Path

from src.robots.urdf_robot import UrdfRobot


class MTM(UrdfRobot):
    def __init__(
        self,
        robot_root: str | Path = "../urdfs/mtm",
        world_link: str = "world",
    ):
        super().__init__(
            name="MTM",
            robot_root=robot_root,
            base_link="top_panel",
            tool_link="ee_link",
            urdf_filename="mtm.urdf",
            world_link=world_link,
        )
