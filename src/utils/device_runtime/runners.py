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
    LEFT_DEVICE_NAME,
    LEFT_J2_COEFF,
    RIGHT_DEVICE_NAME,
    RIGHT_J2_COEFF,
)

Side = Literal["left", "right"]


def _run_with_scheduler(callback: Callable[[], None], *devices) -> None:
    """Start the haptics scheduler, run *callback*, then cleanly shut down.

    Ensures the scheduler is stopped and all devices are closed (in reverse
    order) even if *callback* raises an exception.

    Args:
        callback: Zero-argument callable containing the main teleoperation loop.
        *devices: :class:`~src.pyOpenHaptics.hd_device.HapticDevice` instances
            to close on exit.
    """
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
    """Set up a single haptic device and run the teleoperation loop.

    Reads device name and joint-2 coefficient from ``settings.py`` based on
    *side*, constructs the :class:`~src.pyOpenHaptics.hd_device.HapticDevice`,
    starts the scheduler, calls *callback*, then shuts down cleanly.

    Args:
        device_state: :class:`~src.utils.device_runtime.devices.DeviceState`
            to be updated by the device callback.
        side: ``'left'`` or ``'right'`` — selects device name and J2 coefficient
            from settings.
        callback: Main teleoperation loop (runs between scheduler start/stop).
    """
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
    """Set up a left/right pair of haptic devices and run the teleoperation loop.

    Reads device names and joint-2 coefficients for both sides from
    ``settings.py``, constructs both devices, starts the scheduler, calls
    *callback*, then shuts down cleanly.

    Args:
        left_state: :class:`~src.utils.device_runtime.devices.DeviceState`
            for the left device.
        right_state: :class:`~src.utils.device_runtime.devices.DeviceState`
            for the right device.
        callback: Main teleoperation loop.
    """
    left_device, right_device = setup_devices(
        left_state=left_state,
        right_state=right_state,
        left_device_name=LEFT_DEVICE_NAME,
        right_device_name=RIGHT_DEVICE_NAME,
        left_joint2_coeff=LEFT_J2_COEFF,
        right_joint2_coeff=RIGHT_J2_COEFF,
    )
    _run_with_scheduler(callback, left_device, right_device)
