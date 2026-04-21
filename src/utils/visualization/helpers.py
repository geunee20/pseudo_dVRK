from __future__ import annotations

from typing import Any

import numpy as np
import pyvista as pv


def add_world_floor_and_object(
    plotter: pv.Plotter,
    object_type: str = "cube",
    center: tuple[float, float, float] = (0.0, 0.0, 0.035),
    color: str = "tomato",
    cube_size: tuple[float, float, float] = (0.05, 0.04, 0.06),
    sphere_radius: float = 0.02,
    floor_size: tuple[float, float] = (0.35, 0.35),
    floor_color: str = "lightgray",
    floor_opacity: float = 0.9,
) -> None:
    """Add a simple floor and one target object in world frame."""
    floor = pv.Plane(
        center=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0),
        i_size=float(floor_size[0]),
        j_size=float(floor_size[1]),
    )

    obj_type = str(object_type).strip().lower()
    if obj_type == "sphere":
        obj = pv.Sphere(radius=float(sphere_radius), center=tuple(center))
    else:
        obj = pv.Cube(
            center=tuple(center),
            x_length=float(cube_size[0]),
            y_length=float(cube_size[1]),
            z_length=float(cube_size[2]),
        )

    plotter.add_mesh(floor, color=floor_color, opacity=float(floor_opacity))
    plotter.add_mesh(obj, color=color, smooth_shading=True)
    plotter.add_axes(line_width=2)  # type: ignore[misc]


def hfov_to_vfov_deg(hfov_deg: float, width_px: int, height_px: int) -> float:
    aspect = float(width_px) / float(height_px)
    hfov_rad = np.deg2rad(float(hfov_deg))
    vfov_rad = 2.0 * np.arctan(np.tan(hfov_rad / 2.0) / max(aspect, 1e-12))
    return float(np.rad2deg(vfov_rad))


def apply_camera_pose(
    plotter: pv.Plotter,
    pose: Any,
    vfov_deg: float,
    near: float,
    far: float,
) -> None:
    cam = plotter.camera
    cam.position = tuple(np.asarray(pose.position, dtype=float).tolist())
    cam.focal_point = tuple(np.asarray(pose.focal_point, dtype=float).tolist())
    cam.up = tuple(np.asarray(pose.viewup, dtype=float).tolist())
    cam.view_angle = float(vfov_deg)
    cam.clipping_range = (float(near), float(far))
