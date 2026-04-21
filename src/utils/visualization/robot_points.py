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
    """Compute tool-link position in world coordinates."""
    if base_transform is None:
        T_base = np.eye(4)
    else:
        T_base = np.asarray(base_transform, dtype=float).reshape(4, 4)

    T_links = link_transforms(robot, np.asarray(theta, dtype=float).reshape(-1))
    return (T_base @ T_links[robot.tool_link])[:3, 3]


def create_point_poly(point: np.ndarray) -> pv.PolyData:
    """Create a single-point PolyData marker."""
    return pv.PolyData(np.array([np.asarray(point, dtype=float)], dtype=float))


def update_point_poly(poly: pv.PolyData, point: np.ndarray) -> None:
    """Update an existing single-point PolyData marker in-place."""
    poly.points = np.array([np.asarray(point, dtype=float)], dtype=float)
    poly.Modified()
