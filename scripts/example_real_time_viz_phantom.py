from __future__ import annotations

import time

import numpy as np

from src.robots.phantom import Phantom
from src.robots.psm import PSM
from src.utils.real_time_viz import DvrkRealtimeViz
from src.robots.phantom import Phantom
from src.kinematics.so3 import Rx, Ry, Rz
from src.utils.script_common import DEFAULT_PHANTOM_ROOT, DEFAULT_PSM_ROOT


def main() -> None:
    viz = DvrkRealtimeViz(
        title="dVRK Real-Time Viz",
        window_size=(1600, 1000),
        background="white",
        show_frames=False,
        alpha=1.0,
        frame_scale=0.05,
        marker_radius=0.01,
    )

    phantom = Phantom(robot_root=DEFAULT_PHANTOM_ROOT)
    T_phantom = np.eye(4)
    T_phantom[:3, :3] = Rz(np.pi / 2)
    T_phantom[:3, 3] = np.array([0.5, 0.0, 0.0])
    q_phantom = np.zeros(phantom.dof)
    viz.add_robot(
        "phantom",
        phantom,
        theta=q_phantom,
        base_transform=T_phantom,
        color="lightsteelblue",
    )

    psm = PSM(robot_root=DEFAULT_PSM_ROOT)
    T_psm = np.eye(4)
    T_psm[:3, 3] = np.array([0.0, 0.0, 0.0])
    q_psm = np.zeros(psm.dof)
    viz.add_robot("psm", psm, theta=q_psm, base_transform=T_psm, color="orange")

    viz.set_camera(
        position=(1.5, 1.5, 1.2), focal_point=(0.15, 0.0, 0.0), viewup=(0.0, 0.0, 1.0)
    )
    viz.plotter.show(auto_close=False, interactive_update=True)

    t0 = time.perf_counter()

    try:
        while True:
            t = time.perf_counter() - t0

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

            # -----------------------------
            # Example PSM motion
            # Replace this later with your Phantom->PSM controller
            # -----------------------------
            q_psm = np.zeros(psm.dof)
            if psm.dof >= 8:
                q_psm[0] = 0
                q_psm[1] = 0.25 * np.sin(0.6 * t)
                q_psm[2] = 0.40 * np.sin(0.6 * t)
                q_psm[3] = 0.08 * (1.0 + np.sin(0.9 * t))
                q_psm[4] = 0.40 * np.sin(1.0 * t)
                q_psm[5] = 0.25 * np.sin(0.7 * t + 0.2)
                q_psm[6] = 0.30 * np.sin(0.9 * t + 0.8)
                q_psm[7] = 0.15 * (1.0 + np.sin(1.2 * t))
            viz.update_robot("psm", q_psm, base_transform=T_psm)

            viz.plotter.update()
            # time.sleep(0.02)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        viz.plotter.close()
    return


if __name__ == "__main__":
    main()
