from __future__ import annotations

from typing import Optional

import numpy as np

from src.robots.protocols import ToolKinematicsRobotLike


def make_base_transform(y_offset: float) -> np.ndarray:
    """Return a pure Y-axis translation as a 4×4 homogeneous transform.

    .. math::

        T = \\begin{bmatrix} I_3 & [0, y, 0]^\\top \\\\ 0 & 1 \\end{bmatrix}

    Args:
        y_offset: Translation along the Y axis (metres).

    Returns:
        4×4 homogeneous transform with identity rotation.
    """
    T = np.eye(4, dtype=float)
    T[1, 3] = y_offset
    return T


def make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Assemble a 4×4 homogeneous transform from a rotation matrix and translation.

    .. math::

        T = \\begin{bmatrix} R & p \\\\ 0 & 1 \\end{bmatrix}

    Args:
        R: 3×3 rotation matrix :math:`R \\in SO(3)`.
        p: 3-vector translation.

    Returns:
        4×4 homogeneous transform :math:`T \\in SE(3)`.
    """
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def inv_transform(T: np.ndarray) -> np.ndarray:
    """Compute the closed-form inverse of a homogeneous transform.

    .. math::

        T^{-1} = \\begin{bmatrix} R^\\top & -R^\\top p \\\\ 0 & 1 \\end{bmatrix}

    Args:
        T: 4×4 homogeneous transform :math:`T \\in SE(3)`.

    Returns:
        4×4 inverse :math:`T^{-1} \\in SE(3)`.
    """
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
    """Return the relative transform from *T_ref* to *T_cur* with scaled translation.

    .. math::

        T_{\\Delta} = T_{\\text{ref}}^{-1}\\, T_{\\text{cur}},
        \\qquad
        T_{\\text{scaled}} = \\begin{bmatrix}
            R_\\Delta & k\\, p_\\Delta \\\\ 0 & 1
        \\end{bmatrix}

    Args:
        T_ref: 4×4 reference transform.
        T_cur: 4×4 current transform.
        translation_gain: Scalar gain :math:`k` applied to the relative
            translation (default ``1.0``).

    Returns:
        4×4 scaled relative transform.
    """
    T_delta = inv_transform(T_ref) @ T_cur

    T_scaled = np.eye(4, dtype=float)
    T_scaled[:3, :3] = T_delta[:3, :3]
    T_scaled[:3, 3] = T_delta[:3, 3] * translation_gain
    return T_scaled


def project_to_canvas_point(p_world: np.ndarray) -> np.ndarray:
    """Project a world-frame point onto the 2-D visualisation canvas.

    Zeroes out the X component and shifts Y by +2 m to match the canvas
    coordinate origin used in the tracking scripts:

    .. math::

        p_{\\text{canvas}} = [0,\\; p_y + 2,\\; p_z]^\\top

    Args:
        p_world: 3-vector in the world frame.

    Returns:
        3-vector in the canvas frame (X zeroed, Y shifted).
    """
    p_canvas = np.asarray(p_world, dtype=float).copy()
    p_canvas[0] = 0.0
    p_canvas[1] += 2.0
    return p_canvas


def normalize_vector(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return the unit vector of *v*, or the zero vector when *v* is near-zero.

    .. math::

        \\hat{v} = \\begin{cases} v / \\|v\\| & \\|v\\| \\ge \\varepsilon \\\\ 0 & \\text{otherwise} \\end{cases}

    Args:
        v: Input vector of arbitrary shape.
        eps: Threshold below which *v* is treated as zero.

    Returns:
        Unit vector with the same shape as *v*, or zeros.
    """
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def tool_transform_world(
    robot: ToolKinematicsRobotLike,
    theta: Optional[np.ndarray] = None,
    base_transform: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the tool-link pose in the world frame.

    .. math::

        T_{\\text{tool}}^{\\text{world}} =
        \\begin{cases}
            T_{\\text{base}}^{\\text{world}} \\cdot T_{\\text{FK}}(\\theta)
                & \\text{if base_transform is given} \\\\
            T_{\\text{FK}}(\\theta) & \\text{otherwise}
        \\end{cases}

    Args:
        robot: Robot object implementing
            :class:`~src.robots.protocols.ToolKinematicsRobotLike`.
        theta: (dof,) joint configuration.  Defaults to the robot's stored
            state when ``None``.
        base_transform: Optional 4×4 base-to-world transform.

    Returns:
        4×4 tool-link pose in the world frame.
    """
    T_tool = robot.forward_kinematics(theta=theta, link_name=robot.tool_link)

    if base_transform is None:
        return T_tool

    T_base = np.asarray(base_transform, dtype=float).reshape(4, 4)
    return T_base @ T_tool
