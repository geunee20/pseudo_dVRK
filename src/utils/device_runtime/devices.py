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
    clutch_button: bool = False
    gripper_button: bool = False
    button: bool = False
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    joints: list[float] = field(default_factory=lambda: [0.0] * 6)
    gimbals: list[float] = field(default_factory=list)
    force: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


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
):
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
):
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
    q = np.zeros(phantom.dof, dtype=float)
    n = min(phantom.dof, len(state.joints))
    q[:n] = np.asarray(state.joints[:n], dtype=float)
    return q
