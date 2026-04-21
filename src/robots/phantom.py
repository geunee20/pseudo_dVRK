from pathlib import Path

from src.robots.urdf_robot import UrdfRobot


class Phantom(UrdfRobot):
    """Phantom Touch haptic device modelled as a passive robot.

    Loads ``phantom_touch.urdf`` from *robot_root* and uses
    ``stylus_point`` as the end-effector link.  The world link is set to
    ``'base'`` to match the Phantom URDF convention.

    Args:
        robot_root: Directory containing ``phantom_touch.urdf`` and ``meshes/``.
        world_link: Name of the root link in the URDF tree (default ``'base'``).
    """

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
