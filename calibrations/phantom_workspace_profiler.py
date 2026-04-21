from __future__ import annotations

import time

import numpy as np
import pyvista as pv

from src.robots.phantom import Phantom
from src.utils.device_runtime import (
    DEFAULT_PHANTOM_ROOT,
    DeviceState,
    run_with_single_device,
    state_to_q,
)
from src.utils.visualization import DvrkRealtimeViz

from src.kinematics.fk import link_transforms


def local_to_world(T_base, p_local):
    p_local_h = np.array([p_local[0], p_local[1], p_local[2], 1.0], dtype=float)
    p_world_h = T_base @ p_local_h
    return p_world_h[:3]


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
        position=(1.0, 0.0, 0.2),
        focal_point=(0.0, 0.0, 0.2),
        viewup=(0.0, 0.0, 1.0),
    )
    viz.plotter.show(auto_close=False, interactive_update=True)

    t0 = time.perf_counter()

    # observed workspace limits on x-z plane
    y_min, y_max = float("inf"), -float("inf")
    z_min, z_max = float("inf"), -float("inf")

    line_ymin = viz.make_line()
    line_ymax = viz.make_line()
    line_zmin = viz.make_line()
    line_zmax = viz.make_line()

    viz.plotter.add_mesh(line_ymin, color="red", line_width=3)
    viz.plotter.add_mesh(line_ymax, color="red", line_width=3)
    viz.plotter.add_mesh(line_zmin, color="red", line_width=3)
    viz.plotter.add_mesh(line_zmax, color="red", line_width=3)

    try:
        while True:
            q_phantom = state_to_q(phantom, device_state)

            viz.update_robot("left", q_phantom, base_transform=T_phantom)

            T_links = link_transforms(phantom, q_phantom)
            T_tool_world = T_phantom @ T_links[phantom.tool_link]
            p_fk = T_tool_world[:3, 3]

            # --- line update (y_min ~ y_max at current z) ---
            p_local = T_links[phantom.tool_link][:3, 3]
            y_local = float(p_local[1])
            z_local = float(p_local[2])

            # update local limits
            y_min = min(y_min, y_local)
            y_max = max(y_max, y_local)
            z_min = min(z_min, z_local)
            z_max = max(z_max, z_local)

            # define 4 local boundary lines on local x-z plane (y=0)
            p1_local = [0.0, y_min, z_min]
            p2_local = [0.0, y_min, z_max]

            p3_local = [0.0, y_max, z_min]
            p4_local = [0.0, y_max, z_max]

            # convert local line endpoints to world for visualization
            line_ymin.points[0] = p1_local
            line_ymin.points[1] = p2_local

            line_ymax.points[0] = p3_local
            line_ymax.points[1] = p4_local

            line_zmin.points[0] = p1_local
            line_zmin.points[1] = p3_local

            line_zmax.points[0] = p2_local
            line_zmax.points[1] = p4_local

            line_ymin.Modified()
            line_ymax.Modified()
            line_zmin.Modified()
            line_zmax.Modified()

            fk_points.points = np.array([p_fk], dtype=float)
            fk_points.Modified()

            viz.plotter.update()

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        print("=== Workspace (x, z) limit ===")
        print(f"y_min: {y_min}, y_max: {y_max}")
        print(f"z_min: {z_min}, z_max: {z_max}")
        viz.plotter.close()


if __name__ == "__main__":
    device_state = DeviceState()
    run_with_single_device(device_state=device_state, side="right", callback=main)
