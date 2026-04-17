from __future__ import annotations

import time

import numpy as np
import pyvista as pv

from src.robots.phantom import Phantom
from src.utils.haptics import DeviceState, state_to_q
from src.utils.transforms import make_base_transform
from src.utils.real_time_viz import DvrkRealtimeViz
from src.utils.script_common import DEFAULT_PHANTOM_ROOT, run_with_dual_devices
from src.kinematics.fk import link_transforms

LEFT_Y_OFFSET = -0.25
RIGHT_Y_OFFSET = +0.25


def main() -> None:
    viz = DvrkRealtimeViz(
        title="dVRK Real-Time Viz - Both Phantom Devices",
        window_size=(1600, 1000),
        background="white",
        show_frames=False,
        alpha=0.8,
        frame_scale=0.05,
        marker_radius=0.01,
    )

    phantom_left = Phantom(robot_root=DEFAULT_PHANTOM_ROOT)
    phantom_right = Phantom(robot_root=DEFAULT_PHANTOM_ROOT)

    T_left = make_base_transform(LEFT_Y_OFFSET)
    T_right = make_base_transform(RIGHT_Y_OFFSET)

    viz.add_robot(
        "left",
        phantom_left,
        theta=np.zeros(phantom_left.dof),
        base_transform=T_left,
        color="lightsteelblue",
    )
    viz.add_robot(
        "right",
        phantom_right,
        theta=np.zeros(phantom_right.dof),
        base_transform=T_right,
        color="salmon",
    )

    fk_left = viz.add_fk_marker(color="blue")
    fk_right = viz.add_fk_marker(color="red")

    viz.set_camera(
        position=(1.5, 0.0, 1.0),
        focal_point=(0.0, 0.0, 0.0),
        viewup=(0.0, 0.0, 1.0),
    )
    viz.plotter.show(auto_close=False, interactive_update=True)

    t0 = time.perf_counter()

    try:
        while True:
            q_left = state_to_q(phantom_left, left_state)
            q_right = state_to_q(phantom_right, right_state)

            viz.update_robot("left", q_left, base_transform=T_left)
            viz.update_robot("right", q_right, base_transform=T_right)

            T_links_left = link_transforms(phantom_left, q_left)
            T_tool_world_left = T_left @ T_links_left[phantom_left.tool_link]
            p_left = T_tool_world_left[:3, 3]

            T_links_right = link_transforms(phantom_right, q_right)
            T_tool_world_right = T_right @ T_links_right[phantom_right.tool_link]
            p_right = T_tool_world_right[:3, 3]

            fk_left.points = np.array([p_left], dtype=float)
            fk_right.points = np.array([p_right], dtype=float)
            fk_left.Modified()
            fk_right.Modified()

            viz.plotter.update()
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        viz.plotter.close()


if __name__ == "__main__":
    left_state = DeviceState()
    right_state = DeviceState()
    run_with_dual_devices(left_state=left_state, right_state=right_state, callback=main)
