from __future__ import annotations

import time

import numpy as np

from src.robots.mtm import MTM
from src.robots.psm import PSM
from src.robots.ecm import ECM
from src.utils.real_time_viz import DvrkRealtimeViz
from src.robots.phantom import Phantom
from src.kinematics.so3 import Rx, Ry, Rz
from src.utils.script_common import (
    DEFAULT_ECM_ROOT,
    DEFAULT_MTM_ROOT,
    DEFAULT_PHANTOM_ROOT,
    DEFAULT_PSM_ROOT,
)


def main() -> None:
    mtm = MTM(robot_root=DEFAULT_MTM_ROOT)
    psm = PSM(robot_root=DEFAULT_PSM_ROOT)
    ecm = ECM(robot_root=DEFAULT_ECM_ROOT)
    phantom = Phantom(robot_root=DEFAULT_PHANTOM_ROOT)

    viz = DvrkRealtimeViz(
        title="dVRK Real-Time Viz",
        window_size=(1600, 1000),
        background="white",
        show_frames=False,
        alpha=0.5,
        frame_scale=0.05,
        marker_radius=0.01,
    )

    T_psm = np.eye(4)
    T_psm[:3, 3] = np.array([0.0, 0.0, 0.0])
    q_psm = np.zeros(psm.dof)
    viz.add_robot("psm", psm, theta=q_psm, base_transform=T_psm, color="orange")

    T_mtm = np.eye(4)
    T_mtm[:3, :3] = np.array(
        [
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ]
    )
    T_mtm[:3, 3] = np.array([1.0, -1.0, 0.3])
    q_mtm = np.zeros(mtm.dof)
    viz.add_robot("mtm", mtm, theta=q_mtm, base_transform=T_mtm, color="lightsteelblue")

    T_ecm = np.eye(4)
    T_ecm[:3, :3] = Rz(np.pi / 2)
    T_ecm[:3, 3] = np.array([0.0, -1.0, -0.1])
    q_ecm = np.zeros(ecm.dof)
    viz.add_robot("ecm", ecm, theta=q_ecm, base_transform=T_ecm, color="lightcoral")

    T_phantom = np.eye(4)
    T_phantom[:3, :3] = Rz(np.pi / 2)
    T_phantom[:3, 3] = np.array([1.0, 0.0, 0.0])
    q_phantom = np.zeros(phantom.dof)
    viz.add_robot(
        "phantom",
        phantom,
        theta=q_phantom,
        base_transform=T_phantom,
        color="lightsteelblue",
    )

    viz.set_camera(
        position=(2.0, 2.0, 2.0), focal_point=(0.7, 0.2, 0.5), viewup=(0.0, 0.0, 1.0)
    )
    viz.plotter.show(auto_close=False, interactive_update=True)

    t0 = time.perf_counter()

    try:
        while True:
            t = time.perf_counter() - t0
            # -----------------------------
            # Example PSM motion
            # Replace this later with your MTM->PSM controller
            # -----------------------------
            q_psm = np.zeros(psm.dof)
            if psm.dof >= 8:
                q_psm[0] = 0.10 * np.sin(0.8 * t)
                q_psm[1] = 0.25 * np.sin(0.6 * t)
                q_psm[2] = -0.20 * np.sin(0.6 * t + 0.4)
                q_psm[3] = 0.08 * (1.0 + np.sin(0.9 * t))
                q_psm[4] = 0.40 * np.sin(1.0 * t)
                q_psm[5] = 0.25 * np.sin(0.7 * t + 0.2)
                q_psm[6] = 0.30 * np.sin(0.9 * t + 0.8)
                q_psm[7] = 0.15 * (1.0 + np.sin(1.2 * t))
            viz.update_robot("psm", q_psm, base_transform=T_psm)

            # -----------------------------
            # Example MTM motion
            # -----------------------------
            q_mtm = np.zeros(mtm.dof)
            if mtm.dof >= 4:
                q_mtm[0] = 0.15 * np.sin(0.8 * t)
                q_mtm[1] = 0.20 * np.sin(0.6 * t + 0.5)
                q_mtm[2] = 0.05 * (1.0 + np.sin(0.9 * t))
                q_mtm[3] = 0.30 * np.sin(0.7 * t)
                q_mtm[4] = 0.20 * np.sin(0.6 * t + 0.5)
                q_mtm[5] = 0.03 * np.sin(0.9 * t)
                q_mtm[6] = 0.30 * np.sin(0.7 * t)
            viz.update_robot("mtm", q_mtm, base_transform=T_mtm)

            # -----------------------------
            # Example ECM motion
            # -----------------------------
            q_ecm = np.zeros(ecm.dof)
            if ecm.dof >= 3:
                q_ecm[0] = 0.10 * np.sin(0.8 * t)
                q_ecm[1] = 0.10 * np.sin(0.6 * t + 0.5)
                q_ecm[2] = 0.05 * (1.0 + np.sin(0.9 * t))
            viz.update_robot("ecm", q_ecm, base_transform=T_ecm)

            # -----------------------------
            # Example Phantom motion
            # -----------------------------
            q_phantom = np.zeros(phantom.dof)
            if phantom.dof >= 6:
                q_phantom[0] = 0.15 * np.sin(0.8 * t)
                q_phantom[1] = 0.20 * np.sin(0.6 * t + 0.5)
                q_phantom[2] = 0.05 * (1.0 + np.sin(0.9 * t))
                q_phantom[3] = 0.30 * np.sin(0.7 * t)
                q_phantom[4] = 0.20 * np.sin(0.6 * t + 0.5)
                q_phantom[5] = 0.03 * np.sin(0.9 * t)
            viz.update_robot("phantom", q_phantom, base_transform=T_phantom)

            viz.plotter.update()
            # time.sleep(0.02)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        viz.plotter.close()


if __name__ == "__main__":
    main()
