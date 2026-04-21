from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

import numpy as np

from src.kinematics.so3 import Rx, Rz


def build_dual_phantom_base_transforms(
    phantom_dual_y_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return left/right phantom base transforms for dual setup."""
    T_left = np.eye(4)
    T_left[1, 3] = -float(phantom_dual_y_distance) / 2.0

    T_right = np.eye(4)
    T_right[1, 3] = float(phantom_dual_y_distance) / 2.0
    return T_left, T_right


def build_dual_psm_base_transforms(
    psm_base_x: float,
    psm_base_y_distance: float,
    psm_base_z: float,
    psm_base_x_rotation_split_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return left/right PSM base transforms for dual setup."""
    psm_x_half_rad = np.deg2rad(float(psm_base_x_rotation_split_deg)) / 2.0

    T_left = np.eye(4)
    T_left[:3, :3] = Rx(+psm_x_half_rad) @ Rz(-np.pi / 2)
    T_left[:3, 3] = np.array(
        [float(psm_base_x), -float(psm_base_y_distance) / 2.0, float(psm_base_z)]
    )

    T_right = np.eye(4)
    T_right[:3, :3] = Rx(-psm_x_half_rad) @ Rz(-np.pi / 2)
    T_right[:3, 3] = np.array(
        [float(psm_base_x), float(psm_base_y_distance) / 2.0, float(psm_base_z)]
    )
    return T_left, T_right


def camera_pose_from_psm_tools(
    p_left_world: np.ndarray,
    p_right_world: np.ndarray,
    camera_z_m: float,
) -> np.ndarray:
    """Compute camera pose from PSM home midpoint and fixed camera z."""
    p_mid = 0.5 * (p_left_world + p_right_world)

    T = np.eye(4, dtype=float)
    T[0, 0] = 1.0
    T[1, 1] = -1.0
    T[2, 2] = -1.0
    T[:3, 3] = np.array([p_mid[0], p_mid[1], float(camera_z_m)])
    return T


class ClutchEvent(str, Enum):
    NONE = "none"
    PRESSED = "pressed"
    RELEASED = "released"


def update_clutch_state(
    clutch_pressed: bool,
    prev_clutch_pressed: bool,
    clutch_active: bool,
) -> tuple[bool, bool, ClutchEvent]:
    """Return updated clutch state and event.

    Returns:
        (new_clutch_active, new_prev_clutch_pressed, clutch_event)
    """
    on_press_edge = clutch_pressed and (not prev_clutch_pressed)
    on_release_edge = (not clutch_pressed) and prev_clutch_pressed
    event = ClutchEvent.NONE

    if on_press_edge:
        clutch_active = True
        event = ClutchEvent.PRESSED
    elif on_release_edge:
        clutch_active = False
        event = ClutchEvent.RELEASED

    return clutch_active, clutch_pressed, event


def set_robot_mesh_color(
    robots_by_name: Mapping[str, Any],
    robot_name: str,
    color: str,
) -> None:
    """Apply a single color to all mesh actors for one robot in a viewer."""
    robot_state = robots_by_name.get(robot_name)
    if robot_state is None:
        return
    for item in robot_state.mesh_items.values():
        item.actor.prop.color = color  # type: ignore[attr-defined]


def compute_desired_position_world(
    *,
    clutch_active: bool,
    desired_hold_world: np.ndarray,
    phantom_tool_world: np.ndarray,
    phantom_home_world: np.ndarray,
    psm_home_world: np.ndarray,
    teleoperation_gain: float,
) -> np.ndarray:
    """Compute desired PSM tool position in world frame for position teleoperation."""
    if clutch_active:
        return np.asarray(desired_hold_world, dtype=float).copy()
    delta = np.asarray(phantom_tool_world, dtype=float) - np.asarray(
        phantom_home_world, dtype=float
    )
    return np.asarray(psm_home_world, dtype=float) + float(teleoperation_gain) * delta


def compute_desired_pose_world(
    *,
    clutch_active: bool,
    desired_hold_world: np.ndarray,
    phantom_tool_world: np.ndarray,
    phantom_home_world: np.ndarray,
    psm_base_world: np.ndarray,
    psm_home_local: np.ndarray,
    teleoperation_gain: float,
) -> np.ndarray:
    """Compute desired PSM tool pose in world frame for pose teleoperation."""
    if clutch_active:
        return np.asarray(desired_hold_world, dtype=float).copy()

    T_psm_home_world = np.asarray(psm_base_world, dtype=float) @ np.asarray(
        psm_home_local, dtype=float
    )
    p_delta = (
        np.asarray(phantom_tool_world, dtype=float)[:3, 3]
        - np.asarray(phantom_home_world, dtype=float)[:3, 3]
    )
    p_new = T_psm_home_world[:3, 3] + float(teleoperation_gain) * p_delta

    R_delta = (
        np.asarray(phantom_home_world, dtype=float)[:3, :3].T
        @ np.asarray(phantom_tool_world, dtype=float)[:3, :3]
    )
    R_new = T_psm_home_world[:3, :3] @ R_delta

    T_desired_world = np.eye(4, dtype=float)
    T_desired_world[:3, :3] = R_new
    T_desired_world[:3, 3] = p_new
    return T_desired_world


def update_jaw_command(
    jaw_cmd: float,
    gripper_pressed: bool,
    close_speed: float,
    open_speed: float,
    jaw_min: float,
    jaw_max: float,
) -> float:
    """Integrate jaw command from gripper button and clamp to limits."""
    if gripper_pressed:
        jaw_cmd -= float(close_speed)
    else:
        jaw_cmd += float(open_speed)
    return float(np.clip(jaw_cmd, float(jaw_min), float(jaw_max)))
