from __future__ import annotations

import time

import numpy as np

from src.robots.phantom import Phantom
from src.utils.device_runtime import (
    DEFAULT_PHANTOM_ROOT,
    DeviceState,
    run_with_dual_devices,
    state_to_q,
)
from src.utils.transforms import make_base_transform
from src.utils.visualization import (
    DvrkRealtimeViz,
    create_point_poly,
    tool_position_world,
    update_point_poly,
)
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))
from settings import PHANTOM_BOTH_Y_DISTANCE


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

    T_left = make_base_transform(-PHANTOM_BOTH_Y_DISTANCE / 2.0)
    T_right = make_base_transform(PHANTOM_BOTH_Y_DISTANCE / 2.0)

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

            p_left = tool_position_world(phantom_left, q_left, base_transform=T_left)
            p_right = tool_position_world(
                phantom_right, q_right, base_transform=T_right
            )

            update_point_poly(fk_left, p_left)
            update_point_poly(fk_right, p_right)

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
