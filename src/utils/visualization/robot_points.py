from __future__ import annotations

from typing import Optional

import numpy as np
import pyvista as pv

from src.kinematics.fk import link_transforms
from src.robots.protocols import RobotTreeLike


def tool_position_world(
    robot: RobotTreeLike,
    theta: np.ndarray,
    base_transform: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute tool-tip position in world coordinates.

    Runs forward kinematics up to ``robot.tool_link`` and applies the
    optional base transform to produce a 3-vector in world frame.

    Args:
        robot: Robot model exposing :attr:`tool_link` and joint structure.
        theta: Joint configuration, shape ``(dof,)``.
        base_transform: Optional ``(4, 4)`` world-from-base transform.
            Defaults to identity.

    Returns:
        Position vector :math:`\\mathbf{p} \\in \\mathbb{R}^3`.
    """
    if base_transform is None:
        T_base = np.eye(4)
    else:
        T_base = np.asarray(base_transform, dtype=float).reshape(4, 4)

    T_links = link_transforms(robot, np.asarray(theta, dtype=float).reshape(-1))
    return (T_base @ T_links[robot.tool_link])[:3, 3]


def create_point_poly(point: np.ndarray) -> pv.PolyData:
    """Create a single-point :class:`pyvista.PolyData` marker.

    Args:
        point: 3-D point coordinates, shape ``(3,)``.

    Returns:
        :class:`pyvista.PolyData` containing exactly one point.
    """
    return pv.PolyData(np.array([np.asarray(point, dtype=float)], dtype=float))


def update_point_poly(poly: pv.PolyData, point: np.ndarray) -> None:
    """Update an existing single-point :class:`pyvista.PolyData` marker in-place.

    Replaces ``poly.points`` and calls ``poly.Modified()`` so that the
    PyVista renderer picks up the change without re-adding the actor.

    Args:
        poly: Marker previously created by :func:`create_point_poly`.
        point: New 3-D point coordinates, shape ``(3,)``.
    """
    poly.points = np.array([np.asarray(point, dtype=float)], dtype=float)
    poly.Modified()
