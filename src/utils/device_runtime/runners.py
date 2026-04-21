from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Literal

from src.utils.device_runtime.devices import DeviceState, setup_device, setup_devices
from src.pyOpenHaptics.hd import start_scheduler, stop_scheduler

import sys

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))
from settings import (
    DEFAULT_ECM_ROOT,
    DEFAULT_MTM_ROOT,
    DEFAULT_PHANTOM_ROOT,
    DEFAULT_PSM_ROOT,
    LEFT_DEVICE_NAME,
    LEFT_J2_COEFF,
    RIGHT_DEVICE_NAME,
    RIGHT_J2_COEFF,
)

Side = Literal["left", "right"]


def _run_with_scheduler(callback: Callable[[], None], *devices) -> None:
    try:
        start_scheduler()
        time.sleep(0.2)
        callback()
    finally:
        stop_scheduler()
        for device in reversed(devices):
            if device is not None:
                device.close()


def run_with_single_device(
    *,
    device_state: DeviceState,
    side: Side,
    callback: Callable[[], None],
) -> None:
    if side == "left":
        device_name = LEFT_DEVICE_NAME
        joint2_coeff = LEFT_J2_COEFF
    else:
        device_name = RIGHT_DEVICE_NAME
        joint2_coeff = RIGHT_J2_COEFF

    device = setup_device(
        device_state=device_state,
        device_name=device_name,
        joint2_coeff=joint2_coeff,
    )
    _run_with_scheduler(callback, device)


def run_with_dual_devices(
    *,
    left_state: DeviceState,
    right_state: DeviceState,
    callback: Callable[[], None],
) -> None:
    left_device, right_device = setup_devices(
        left_state=left_state,
        right_state=right_state,
        left_device_name=LEFT_DEVICE_NAME,
        right_device_name=RIGHT_DEVICE_NAME,
        left_joint2_coeff=LEFT_J2_COEFF,
        right_joint2_coeff=RIGHT_J2_COEFF,
    )
    _run_with_scheduler(callback, left_device, right_device)
