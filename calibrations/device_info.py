"""
Query and print hardware limits and calibration values from connected Phantom devices.

Run with:
    python -m calibrations.device_info

Output can be used to set MAX_FORCE_NORM, FORCE_KP, FORCE_KD, etc. in experiment
examples. Key relationships:

  MAX_FORCE_NORM      <= nominal_max_continuous_force   (safe sustained limit)
  FORCE_KP  (N/m)    <= nominal_max_stiffness           (hard stiffness ceiling)
  FORCE_KD  (N·s/m)  <= nominal_max_damping             (hard damping ceiling)
"""

from __future__ import annotations

import time

from src.utils.device_runtime import DeviceState, setup_devices
from src.pyOpenHaptics.hd import (
    start_scheduler,
    stop_scheduler,
    get_update_rate,
    get_instantaneous_update_rate,
    get_nominal_max_force,
    get_nominal_max_continuous_force,
    get_nominal_max_stiffness,
    get_nominal_max_damping,
    make_current_device,
)
from src.pyOpenHaptics.hd_device import HapticDevice

LEFT_DEVICE_NAME = "Left Device"
RIGHT_DEVICE_NAME = "Right Device"
LEFT_J2_COEFF = 0.993545966786979
RIGHT_J2_COEFF = 0.995028312512455


def _print_device_info(label: str, device: HapticDevice) -> None:
    make_current_device(device.id)
    print(f"\n=== {label} ===")
    print(f"  Update rate (nominal)        : {get_update_rate()} Hz")
    print(f"  Update rate (instantaneous)  : {get_instantaneous_update_rate()} Hz")
    print()
    print(f"  Nominal max force            : {get_nominal_max_force():.2f} N")
    print(
        f"  Nominal max continuous force : {get_nominal_max_continuous_force():.2f} N"
    )
    print()
    print(f"  Nominal max stiffness        : {get_nominal_max_stiffness():.2f} N/m")
    print(f"  Nominal max damping          : {get_nominal_max_damping():.4f} N·s/m")
    print()
    print("  Suggested experiment constants:")
    max_cont = get_nominal_max_continuous_force()
    max_stiff = get_nominal_max_stiffness()
    max_damp = get_nominal_max_damping()
    print(
        f"    MAX_FORCE_NORM  = {max_cont * 0.8:.1f}  # 80% of nominal max continuous force"
    )
    print(
        f"    FORCE_KP        = {max_stiff * 0.5:.1f}  # 50% of nominal max stiffness"
    )
    print(f"    FORCE_KD        = {max_damp * 0.3:.4f}  # 30% of nominal max damping")


def main() -> None:
    left_state = DeviceState()
    right_state = DeviceState()
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
        time.sleep(0.3)  # let scheduler settle

        _print_device_info("Left Device", left_device)
        _print_device_info("Right Device", right_device)

    finally:
        stop_scheduler()
        if right_device is not None:
            right_device.close()
        if left_device is not None:
            left_device.close()


if __name__ == "__main__":
    main()
