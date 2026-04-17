from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from src.robots.phantom import Phantom
from src.pyOpenHaptics.hd import (
    get_joint_angles,
    get_gimbal_angles,
    get_buttons,
    set_force,
)
from src.pyOpenHaptics.hd_define import HD_DEVICE_BUTTON_1, HD_DEVICE_BUTTON_2
from src.pyOpenHaptics.hd_device import HapticDevice


@dataclass
class DeviceState:
    clutch_button: bool = False
    gripper_button: bool = False
    button: bool = False  # for backward compatibility with calibration scripts
    position: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    joints: list = field(default_factory=lambda: [0.0] * 6)
    gimbals: list = field(default_factory=list)
    force: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


class _JointStateLike(Protocol):
    joints: list


def make_state_callback(device_state: DeviceState, joint2_coeff: float):
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
        device_state.button = device_state.clutch_button  # for backward compatibility

    return state_callback


def setup_device(
    device_state: DeviceState,
    device_name: str,
    joint2_coeff: float,
):
    callback = make_state_callback(device_state, joint2_coeff)
    return HapticDevice(device_name=device_name, callback=callback)


def setup_devices(
    left_state: DeviceState,
    right_state: DeviceState,
    left_device_name: str,
    right_device_name: str,
    left_joint2_coeff: float,
    right_joint2_coeff: float,
):
    left_callback = make_state_callback(left_state, left_joint2_coeff)
    right_callback = make_state_callback(right_state, right_joint2_coeff)

    left_device = HapticDevice(device_name=left_device_name, callback=left_callback)
    right_device = HapticDevice(device_name=right_device_name, callback=right_callback)
    return left_device, right_device


def state_to_q(phantom: Phantom, state: _JointStateLike) -> np.ndarray:
    q = np.zeros(phantom.dof, dtype=float)
    n = min(phantom.dof, len(state.joints))
    q[:n] = np.asarray(state.joints[:n], dtype=float)
    return q
