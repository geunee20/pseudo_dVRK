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
    """Add a floor plane and one target object to a PyVista plotter.

    The floor lies in the XY plane at z = 0.  The target object is either a
    cube or a sphere centred at *center*.  World-frame XYZ axes are also drawn.

    Args:
        plotter: PyVista plotter to add meshes to.
        object_type: ``"cube"`` or ``"sphere"``.
        center: World-frame centre of the target object ``(x, y, z)`` in metres.
        color: Colour of the target object.
        cube_size: ``(x_length, y_length, z_length)`` for cube objects.
        sphere_radius: Radius in metres for sphere objects.
        floor_size: ``(i_size, j_size)`` of the XY floor plane in metres.
        floor_color: Colour of the floor plane.
        floor_opacity: Opacity ``[0, 1]`` of the floor plane.
    """
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
    """Convert a horizontal field-of-view angle to a vertical field-of-view angle.

    Uses the pinhole camera relationship:

    .. math::

        \\text{VFOV} = 2 \\arctan\\!\\left(\\frac{\\tan(\\text{HFOV}/2)}{w/h}\\right)

    Args:
        hfov_deg: Horizontal field of view in degrees.
        width_px: Image width in pixels.
        height_px: Image height in pixels.

    Returns:
        Vertical field of view in degrees.
    """
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
    """Apply a camera pose to a PyVista plotter.

    Sets position, focal point, up vector, view angle, and clipping range
    from the fields of *pose*, which must expose:
    ``position``, ``focal_point``, and ``viewup`` attributes.

    Args:
        plotter: PyVista plotter whose camera will be configured.
        pose: Camera pose object with ``position``, ``focal_point``, and
            ``viewup`` array-like attributes.
        vfov_deg: Vertical field of view in degrees (camera view angle).
        near: Near clipping plane distance in metres.
        far: Far clipping plane distance in metres.
    """
    cam = plotter.camera
    cam.position = tuple(np.asarray(pose.position, dtype=float).tolist())
    cam.focal_point = tuple(np.asarray(pose.focal_point, dtype=float).tolist())
    cam.up = tuple(np.asarray(pose.viewup, dtype=float).tolist())
    cam.view_angle = float(vfov_deg)
    cam.clipping_range = (float(near), float(far))
