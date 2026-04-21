from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.robots.phantom import Phantom
from src.robots.protocols import JointStateLike
from src.pyOpenHaptics.hd import (
    get_buttons,
    get_gimbal_angles,
    get_joint_angles,
    set_force,
)
from src.pyOpenHaptics.hd_define import HD_DEVICE_BUTTON_1, HD_DEVICE_BUTTON_2
from src.pyOpenHaptics.hd_device import HapticDevice


@dataclass
class DeviceState:
    """Shared mutable state updated by the haptic-device scheduler callback.

    All fields are written from the haptics scheduler thread and read from the
    main rendering thread; no explicit locking is provided — callers rely on
    Python's GIL for atomicity on individual attribute reads.

    Attributes:
        clutch_button: ``True`` while the clutch (Button 1) is held down.
        gripper_button: ``True`` while the gripper (Button 2) is held down.
        button: Convenience alias for ``clutch_button``.
        position: Current stylus tip position ``[x, y, z]`` in device space
            (metres).
        joints: 6-element joint angle list in the order
            ``[q0, q1, q2_corrected, gimbal0, gimbal1, gimbal2]`` (radians).
        gimbals: Raw gimbal angle list as read from the device.
        force: Force command ``[fx, fy, fz]`` (newtons) sent to the device
            on each scheduler tick.
    """

    clutch_button: bool = False
    gripper_button: bool = False
    button: bool = False
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    joints: list[float] = field(default_factory=lambda: [0.0] * 6)
    gimbals: list[float] = field(default_factory=list)
    force: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


def make_state_callback(device_state: DeviceState, joint2_coeff: float):
    """Build the haptic-device scheduler callback that populates *device_state*.

    The callback is designed to be called by the OpenHaptics scheduler at
    a fixed servo rate (typically 1 kHz).  It performs three tasks:

    1. Reads raw joint and gimbal angles and assembles the corrected 6-DOF
       joint vector with the joint-2 parallelogram correction:

       .. math::

           q_2^{\\text{corr}} = q_2^{\\text{raw}} - c_{J2} \\cdot q_1

       where :math:`c_{J2}` is *joint2_coeff*.

    2. Sends the current force command back to the device via
       :func:`~src.pyOpenHaptics.hd.set_force`.

    3. Updates the button flags from the device button bitmask.

    Args:
        device_state: Shared :class:`DeviceState` object to update.
        joint2_coeff: Parallelogram coupling coefficient for joint 2.

    Returns:
        Zero-argument callable suitable for
        :class:`~src.pyOpenHaptics.hd_device.HapticDevice`.
    """

    def state_callback():
        joints = get_joint_angles()
        gimbals = get_gimbal_angles()

        device_state.joints = [
            joints[0],
            joints[1],
            joints[2] - joint2_coeff * joints[1],
            gimbals[0],
            gimbals[1],
            gimbals[2],
        ]

        set_force(device_state.force)

        button_mask = get_buttons()
        device_state.clutch_button = (button_mask & HD_DEVICE_BUTTON_1) != 0
        device_state.gripper_button = (button_mask & HD_DEVICE_BUTTON_2) != 0
        device_state.button = device_state.clutch_button

    return state_callback


def _build_device(
    *,
    device_state: DeviceState,
    device_name: str,
    joint2_coeff: float,
) -> HapticDevice:
    callback = make_state_callback(device_state, joint2_coeff)
    return HapticDevice(device_name=device_name, callback=callback)


def setup_device(
    device_state: DeviceState,
    device_name: str,
    joint2_coeff: float,
) -> HapticDevice:
    """Create and initialise a single :class:`~src.pyOpenHaptics.hd_device.HapticDevice`.

    Args:
        device_state: :class:`DeviceState` that the callback will update.
        device_name: OpenHaptics device name string (e.g. ``'Default PHANToM'``).
        joint2_coeff: Parallelogram coupling coefficient for joint 2.

    Returns:
        Initialised :class:`~src.pyOpenHaptics.hd_device.HapticDevice`.
    """
    return _build_device(
        device_state=device_state,
        device_name=device_name,
        joint2_coeff=joint2_coeff,
    )


def setup_devices(
    left_state: DeviceState,
    right_state: DeviceState,
    left_device_name: str,
    right_device_name: str,
    left_joint2_coeff: float,
    right_joint2_coeff: float,
) -> tuple[HapticDevice, HapticDevice]:
    """Create and initialise a matched left/right pair of haptic devices.

    Args:
        left_state: :class:`DeviceState` for the left device.
        right_state: :class:`DeviceState` for the right device.
        left_device_name: OpenHaptics device name for the left device.
        right_device_name: OpenHaptics device name for the right device.
        left_joint2_coeff: Joint-2 correction coefficient for the left device.
        right_joint2_coeff: Joint-2 correction coefficient for the right device.

    Returns:
        ``(left_device, right_device)`` — each an initialised
        :class:`~src.pyOpenHaptics.hd_device.HapticDevice`.
    """
    left_device = _build_device(
        device_state=left_state,
        device_name=left_device_name,
        joint2_coeff=left_joint2_coeff,
    )
    right_device = _build_device(
        device_state=right_state,
        device_name=right_device_name,
        joint2_coeff=right_joint2_coeff,
    )
    return left_device, right_device


def state_to_q(phantom: Phantom, state: JointStateLike) -> np.ndarray:
    """Convert a :class:`DeviceState` joint list to a Phantom joint vector.

    Copies ``min(phantom.dof, len(state.joints))`` values from *state.joints*
    into a zero-initialised (dof,) float array.

    Args:
        phantom: :class:`~src.robots.phantom.Phantom` instance defining the DOF.
        state: Any object exposing a ``joints`` sequence (e.g. :class:`DeviceState`).

    Returns:
        (dof,) float64 joint vector.
    """
    n = min(phantom.dof, len(state.joints))
    q = np.zeros(phantom.dof, dtype=float)
    q[:n] = np.asarray(state.joints[:n], dtype=float)
    return q
