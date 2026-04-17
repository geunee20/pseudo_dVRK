from __future__ import annotations

import time
from typing import Callable, Literal

from src.pyOpenHaptics.hd import start_scheduler, stop_scheduler
from src.utils.haptics import DeviceState, setup_device, setup_devices

# Import common settings from project root
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))
from settings import (
    DEFAULT_PHANTOM_ROOT,
    DEFAULT_PSM_ROOT,
    DEFAULT_ECM_ROOT,
    DEFAULT_MTM_ROOT,
    LEFT_DEVICE_NAME,
    RIGHT_DEVICE_NAME,
    LEFT_J2_COEFF,
    RIGHT_J2_COEFF,
)

Side = Literal["left", "right"]


def run_with_single_device(
    *,
    device_state: DeviceState,
    side: Side,
    callback: Callable[[], None],
) -> None:
    """Set up one haptic device, run callback, then clean up safely."""
    if side == "left":
        device_name = LEFT_DEVICE_NAME
        joint2_coeff = LEFT_J2_COEFF
    else:
        device_name = RIGHT_DEVICE_NAME
        joint2_coeff = RIGHT_J2_COEFF

    device = None
    try:
        device = setup_device(
            device_state=device_state,
            device_name=device_name,
            joint2_coeff=joint2_coeff,
        )
        start_scheduler()
        time.sleep(0.2)
        callback()
    finally:
        stop_scheduler()
        if device is not None:
            device.close()


def run_with_dual_devices(
    *,
    left_state: DeviceState,
    right_state: DeviceState,
    callback: Callable[[], None],
) -> None:
    """Set up both haptic devices, run callback, then clean up safely."""
    left_device = None
    right_device = None
    try:
        left_device, right_device = setup_devices(
            left_state=left_state,
            right_state=right_state,
            left_device_name=LEFT_DEVICE_NAME,
            right_device_name=RIGHT_DEVICE_NAME,
            left_joint2_coeff=LEFT_J2_COEFF,
            right_joint2_coeff=RIGHT_J2_COEFF,
        )
        start_scheduler()
        time.sleep(0.2)
        callback()
    finally:
        stop_scheduler()
        if right_device is not None:
            right_device.close()
        if left_device is not None:
            left_device.close()
