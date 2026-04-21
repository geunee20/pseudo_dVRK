from __future__ import annotations

from typing import Optional

import numpy as np

from src.robots.protocols import ToolKinematicsRobotLike


def make_base_transform(y_offset: float) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[1, 3] = y_offset
    return T


def make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def inv_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    p = T[:3, 3]
    T_inv = np.eye(4, dtype=float)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p
    return T_inv


def scaled_relative_transform(
    T_ref: np.ndarray,
    T_cur: np.ndarray,
    translation_gain: float = 1.0,
) -> np.ndarray:
    """Return relative transform from ref -> cur with scaled translation."""
    T_delta = inv_transform(T_ref) @ T_cur

    T_scaled = np.eye(4, dtype=float)
    T_scaled[:3, :3] = T_delta[:3, :3]
    T_scaled[:3, 3] = T_delta[:3, 3] * translation_gain
    return T_scaled


def project_to_canvas_point(p_world: np.ndarray) -> np.ndarray:
    """Match the canvas mapping used in the original tracking scripts."""
    p_canvas = np.asarray(p_world, dtype=float).copy()
    p_canvas[0] = 0.0
    p_canvas[1] += 2.0
    return p_canvas


def normalize_vector(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return unit vector; return zeros when norm is too small."""
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def tool_transform_world(
    robot: ToolKinematicsRobotLike,
    theta: Optional[np.ndarray] = None,
    base_transform: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute tool pose in world frame from robot FK and optional base transform."""
    T_tool = robot.forward_kinematics(theta=theta, link_name=robot.tool_link)

    if base_transform is None:
        return T_tool

    T_base = np.asarray(base_transform, dtype=float).reshape(4, 4)
    return T_base @ T_tool
