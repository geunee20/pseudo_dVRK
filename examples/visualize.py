from __future__ import annotations

import numpy as np

from src.robots.psm import PSM
from src.robots.ecm import ECM
from src.robots.mtm import MTM
from src.utils.script_common import (
    DEFAULT_ECM_ROOT,
    DEFAULT_MTM_ROOT,
    DEFAULT_PSM_ROOT,
)


def main() -> None:
    psm = PSM(robot_root=DEFAULT_PSM_ROOT)
    q = np.zeros(psm.dof)
    q[0] = np.pi / 4
    q[1] = np.pi / 4
    q[2] = np.pi / 4
    q[3] = 0.05
    q[4] = np.pi / 4
    q[5] = np.pi / 4
    q[6] = np.pi / 4
    q[7] = np.pi / 2
    scene = psm.visualize(theta=q, show_frames=True, alpha=0.5)

    ecm = ECM(robot_root=DEFAULT_ECM_ROOT)
    q = np.zeros(ecm.dof)
    # q[0] = np.pi / 4
    # q[1] = np.pi / 4
    # q[2] = np.pi / 4
    # q[3] = np.pi / 4
    scene = ecm.visualize(theta=q, show_frames=True, alpha=0.5)

    mtm = MTM(robot_root=DEFAULT_MTM_ROOT)
    q = np.zeros(mtm.dof)
    # q[0] = np.pi / 4
    # q[1] = np.pi / 4
    # q[2] = np.pi / 4
    # q[3] = np.pi / 4
    # q[4] = np.pi / 4
    # q[5] = np.pi / 4
    # q[6] = np.pi / 4
    scene = mtm.visualize(theta=q, show_frames=True, alpha=0.5)


if __name__ == "__main__":
    main()
