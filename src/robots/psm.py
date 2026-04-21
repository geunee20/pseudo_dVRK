from pathlib import Path

from src.robots.urdf_robot import UrdfRobot


class PSM(UrdfRobot):
    """Patient Side Manipulator (PSM) robot.

    Loads ``psm.urdf`` from *robot_root*, sets ``tool_yaw_link`` as the
    end-effector link, and uses the default active-joint selection (all
    non-fixed, non-mimic joints).

    Args:
        robot_root: Directory containing ``psm.urdf`` and ``meshes/``.
        world_link: Name of the root/world link in the URDF tree.
    """

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
