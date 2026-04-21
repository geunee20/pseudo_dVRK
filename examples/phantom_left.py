from __future__ import annotations

import numpy as np

from src.robots.phantom import Phantom
from src.utils.device_runtime import (
    DEFAULT_PHANTOM_ROOT,
    DeviceState,
    run_with_single_device,
    state_to_q,
)
from src.utils.visualization import DvrkRealtimeViz
from src.utils.visualization import (
    create_point_poly,
    tool_position_world,
    update_point_poly,
)


def main() -> None:
    viz = DvrkRealtimeViz(
        title="Left Phantom Real-Time Viz",
        window_size=(1600, 1000),
        background="white",
        show_frames=False,
        alpha=0.8,
        frame_scale=0.05,
        marker_radius=0.01,
    )

    phantom = Phantom(robot_root=DEFAULT_PHANTOM_ROOT)

    T_phantom = np.eye(4)
    T_phantom[1, 3] = 0.0

    q_phantom = np.zeros(phantom.dof)
    viz.add_robot(
        "left",
        phantom,
        theta=q_phantom,
        base_transform=T_phantom,
        color="lightsteelblue",
    )

    p_fk = tool_position_world(phantom, q_phantom, base_transform=T_phantom)
    fk_points = create_point_poly(p_fk)

    viz.plotter.add_points(
        fk_points,
        color="blue",
        point_size=10,
        render_points_as_spheres=True,
    )

    viz.set_camera(
        position=(1.5, 1.5, 1.2),
        focal_point=(0.15, 0.0, 0.0),
        viewup=(0.0, 0.0, 1.0),
    )
    viz.plotter.show(auto_close=False, interactive_update=True)

    try:
        while True:
            q_phantom = state_to_q(phantom, device_state)

            viz.update_robot("left", q_phantom, base_transform=T_phantom)

            p_fk = tool_position_world(phantom, q_phantom, base_transform=T_phantom)
            update_point_poly(fk_points, p_fk)

            viz.plotter.update()

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        viz.plotter.close()


if __name__ == "__main__":
    device_state = DeviceState()
    run_with_single_device(device_state=device_state, side="left", callback=main)
