from __future__ import annotations

import numpy as np
import pyvista as pv

from src.robots.phantom import Phantom
from src.utils.real_time_viz import DvrkRealtimeViz
from src.utils.haptics import DeviceState, state_to_q
from src.utils.script_common import DEFAULT_PHANTOM_ROOT, run_with_single_device

from src.kinematics.fk import link_transforms


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

    T_links = link_transforms(phantom, q_phantom)
    T_tool_world = T_phantom @ T_links[phantom.tool_link]
    p_fk = T_tool_world[:3, 3]

    fk_points = pv.PolyData(np.array([p_fk], dtype=float))

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

            T_links = link_transforms(phantom, q_phantom)
            T_tool_world = T_phantom @ T_links[phantom.tool_link]
            p_fk = T_tool_world[:3, 3]

            fk_points.points = np.array([p_fk], dtype=float)
            fk_points.Modified()

            viz.plotter.update()

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        viz.plotter.close()


if __name__ == "__main__":
    device_state = DeviceState()
    run_with_single_device(device_state=device_state, side="left", callback=main)
