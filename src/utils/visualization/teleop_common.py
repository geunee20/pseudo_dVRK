from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

import numpy as np

from src.kinematics.so3 import Rx, Rz


def build_dual_phantom_base_transforms(
    phantom_dual_y_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return left/right Phantom base transforms for a symmetric dual-device setup.

    Both devices are placed symmetrically about the world origin along the Y axis:

    .. math::

        T_{\\text{left}}  = \\begin{bmatrix} I_{3} & [0,\\ -d/2,\\ 0]^\\top \\\\ 0 & 1 \\end{bmatrix}

        T_{\\text{right}} = \\begin{bmatrix} I_{3} & [0,\\ +d/2,\\ 0]^\\top \\\\ 0 & 1 \\end{bmatrix}

    where :math:`d` is *phantom_dual_y_distance*.

    Args:
        phantom_dual_y_distance: Total Y separation (metres) between the two
            Phantom device bases.

    Returns:
        ``(T_left, T_right)`` — each a 4×4 homogeneous transform in world frame.
    """
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
    """Return left/right PSM base transforms for a symmetric dual-arm setup.

    Each PSM is rotated about the X axis by ±half of *psm_base_x_rotation_split_deg*
    and then about the Z axis by :math:`-\\pi/2` to align the instrument shaft
    downward into the surgical field:

    .. math::

        R_{\\text{left}}  &= R_x(+\\alpha/2)\\, R_z(-\\pi/2) \\\\
        R_{\\text{right}} &= R_x(-\\alpha/2)\\, R_z(-\\pi/2)

    where :math:`\\alpha = \\text{psm\\_base\\_x\\_rotation\\_split\\_deg}` in radians.

    The translation is:

    .. math::

        t_{\\text{left}}  = [x,\\ -d/2,\\ z]^\\top, \\quad
        t_{\\text{right}} = [x,\\ +d/2,\\ z]^\\top

    where :math:`d` is *psm_base_y_distance*.

    Args:
        psm_base_x: X offset of both PSM bases in the world frame (metres).
        psm_base_y_distance: Total Y separation (metres) between the two PSM bases.
        psm_base_z: Z offset of both PSM bases in the world frame (metres).
        psm_base_x_rotation_split_deg: Total X-axis rotation spread (degrees)
            shared symmetrically between the two arms.

    Returns:
        ``(T_left, T_right)`` — each a 4×4 homogeneous base transform in world frame.
    """
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
    """Compute an ECM-style camera pose from the PSM tool home positions.

    The camera is placed at the midpoint of the two PSM home positions projected
    onto the XY plane, at a fixed height *camera_z_m*:

    .. math::

        p_{\\text{cam}} = \\left[\\,
            \\frac{x_L + x_R}{2},\\;
            \\frac{y_L + y_R}{2},\\;
            z_{\\text{cam}}
        \\,\\right]^\\top

    The camera orientation is set so that the image plane faces downward
    (i.e., the Z axis of the camera frame points toward the surgical field):

    .. math::

        R_{\\text{cam}} = \\begin{bmatrix} 1 & 0 & 0 \\\\ 0 & -1 & 0 \\\\ 0 & 0 & -1 \\end{bmatrix}

    Args:
        p_left_world: 3-vector (x, y, z) of the left PSM home position in world frame.
        p_right_world: 3-vector (x, y, z) of the right PSM home position in world frame.
        camera_z_m: Fixed camera height above the world origin (metres).

    Returns:
        4×4 homogeneous camera pose :math:`T_{\\text{cam}}^{\\text{world}}`.
    """
    p_mid = 0.5 * (p_left_world + p_right_world)

    T = np.eye(4, dtype=float)
    T[0, 0] = 1.0
    T[1, 1] = -1.0
    T[2, 2] = -1.0
    T[:3, 3] = np.array([p_mid[0], p_mid[1], float(camera_z_m)])
    return T


class ClutchEvent(str, Enum):
    """Edge-triggered clutch event detected on a single control loop tick.

    Attributes:
        NONE: No state transition occurred; clutch button state is unchanged.
        PRESSED: Rising edge — the clutch button was just pressed this tick.
        RELEASED: Falling edge — the clutch button was just released this tick.
    """

    NONE = "none"
    PRESSED = "pressed"
    RELEASED = "released"


def update_clutch_state(
    clutch_pressed: bool,
    prev_clutch_pressed: bool,
    clutch_active: bool,
) -> tuple[bool, bool, ClutchEvent]:
    """Detect rising/falling edges of the clutch button and update clutch state.

    Edge detection is performed on the transition between the previous and current
    button readings:

    * **Rising edge** (``clutch_pressed and not prev_clutch_pressed``):
      clutch becomes active; event is :attr:`ClutchEvent.PRESSED`.
    * **Falling edge** (``not clutch_pressed and prev_clutch_pressed``):
      clutch becomes inactive; event is :attr:`ClutchEvent.RELEASED`.
    * **No edge**: clutch active state is unchanged; event is :attr:`ClutchEvent.NONE`.

    Args:
        clutch_pressed: Current (this-tick) state of the clutch button.
        prev_clutch_pressed: State of the clutch button on the previous tick.
        clutch_active: Current clutch active flag before this update.

    Returns:
        ``(new_clutch_active, new_prev_clutch_pressed, clutch_event)``

        * *new_clutch_active* — updated clutch active flag.
        * *new_prev_clutch_pressed* — value to store as ``prev_clutch_pressed``
          for the next call (equals *clutch_pressed*).
        * *clutch_event* — the edge event that occurred this tick.
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
    """Apply a uniform color to every mesh actor belonging to one robot.

    Iterates over all mesh items stored in the robot's visualization state and
    sets their PyVista actor property color.  A no-op if *robot_name* is not
    present in *robots_by_name*.

    Args:
        robots_by_name: Mapping from robot name to its visualization state
            object (e.g., ``DvrkRealtimeViz._robots``).
        robot_name: Key identifying the target robot (e.g., ``"left"``).
        color: Color string accepted by PyVista / VTK (e.g., ``"red"``,
            ``"#FF0000"``).
    """
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
    """Compute the desired PSM tool position in world frame (position-only teleoperation).

    When the clutch is **active** the robot holds its last commanded position:

    .. math::

        p_{\\text{desired}} = p_{\\text{hold}}

    When the clutch is **inactive** a scaled displacement from the Phantom home
    position is added to the PSM home position:

    .. math::

        p_{\\text{desired}} = p_{\\text{PSM home}} + k \\,(p_{\\text{Phantom}} - p_{\\text{Phantom home}})

    where :math:`k` is *teleoperation_gain*.

    Args:
        clutch_active: When ``True`` the robot holds *desired_hold_world*.
        desired_hold_world: 3-vector — position to hold while clutch is active.
        phantom_tool_world: 3-vector — current Phantom stylus tip position in
            world frame.
        phantom_home_world: 3-vector — Phantom stylus position at the moment
            teleoperation began (reference frame origin).
        psm_home_world: 3-vector — PSM tool tip position in world frame at the
            moment teleoperation began.
        teleoperation_gain: Scalar gain :math:`k` mapping Phantom workspace
            displacements to PSM workspace displacements.

    Returns:
        3-vector representing the desired PSM tool position in world frame.
    """
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
    """Compute the desired PSM tool pose in world frame (full-pose teleoperation).

    When the clutch is **active** the robot holds its last commanded pose:

    .. math::

        T_{\\text{desired}} = T_{\\text{hold}}

    When the clutch is **inactive** the desired pose is computed in two parts:

    **Translate** — scaled displacement from the Phantom home position:

    .. math::

        T_{\\text{PSM home}} &= T_{\\text{PSM base}}^{\\text{world}} \\cdot T_{\\text{PSM home}}^{\\text{local}} \\\\
        p_{\\text{desired}} &= p_{\\text{PSM home}} + k\\,\\bigl(p_{\\text{Phantom}} - p_{\\text{Phantom home}}\\bigr)

    where :math:`k` is *teleoperation_gain* and all :math:`p` terms are the
    translation columns of their respective 4×4 transforms.

    **Rotate** — relative rotation from Phantom home to current Phantom pose,
    applied to the PSM home orientation:

    .. math::

        R_{\\Delta} &= R_{\\text{Phantom home}}^\\top \\cdot R_{\\text{Phantom}} \\\\
        R_{\\text{desired}} &= R_{\\text{PSM home}} \\cdot R_{\\Delta}

    The desired pose is then assembled as:

    .. math::

        T_{\\text{desired}} = \\begin{bmatrix} R_{\\text{desired}} & p_{\\text{desired}} \\\\ 0 & 1 \\end{bmatrix}

    Args:
        clutch_active: When ``True`` the robot holds *desired_hold_world*.
        desired_hold_world: 4×4 transform — pose to hold while clutch is active.
        phantom_tool_world: 4×4 homogeneous transform of the Phantom stylus tip
            in world frame (current).
        phantom_home_world: 4×4 homogeneous transform of the Phantom stylus tip
            in world frame at teleoperation start.
        psm_base_world: 4×4 homogeneous transform of the PSM base in world frame
            (:math:`T_{\\text{PSM base}}^{\\text{world}}`).
        psm_home_local: 4×4 homogeneous transform of the PSM tool tip in the
            PSM base frame at teleoperation start
            (:math:`T_{\\text{PSM home}}^{\\text{local}}`).
        teleoperation_gain: Scalar gain :math:`k` mapping Phantom workspace
            displacements to PSM workspace displacements.

    Returns:
        4×4 homogeneous transform representing the desired PSM tool pose in
        world frame.
    """
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
    """Integrate jaw angle command from the gripper button and clamp to joint limits.

    On each control tick the jaw command is incremented or decremented based on
    the gripper button state, then clamped to ``[jaw_min, jaw_max]``:

    .. math::

        q_{\\text{jaw}}^{\\text{new}} =
        \\text{clip}\\!\\left(
            q_{\\text{jaw}} +
            \\begin{cases}
                -s_{\\text{close}} & \\text{if gripper pressed} \\\\
                +s_{\\text{open}}  & \\text{otherwise}
            \\end{cases},
        \\; q_{\\min}, q_{\\max}
        \\right)

    Args:
        jaw_cmd: Current jaw joint angle command (radians).
        gripper_pressed: ``True`` when the operator is squeezing the gripper
            button (close command).
        close_speed: Angular rate (rad / tick) at which the jaw closes.
        open_speed: Angular rate (rad / tick) at which the jaw opens.
        jaw_min: Minimum allowed jaw angle (radians).
        jaw_max: Maximum allowed jaw angle (radians).

    Returns:
        Updated jaw joint angle command clamped to ``[jaw_min, jaw_max]``.
    """
    if gripper_pressed:
        jaw_cmd -= float(close_speed)
    else:
        jaw_cmd += float(open_speed)
    return float(np.clip(jaw_cmd, float(jaw_min), float(jaw_max)))
