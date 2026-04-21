from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pyvista as pv

from src.robots.phantom import Phantom
from src.utils.device_runtime import (
    DEFAULT_PHANTOM_ROOT,
    DeviceState,
    run_with_single_device,
    state_to_q,
)
from src.utils.visualization import DvrkRealtimeViz
from src.kinematics.so3 import Rz
from src.kinematics.fk import link_transforms

from src.pyOpenHaptics import hd

# Global variables for calibration data collection
calibration_data = {
    "j1_data": [],
    "j2_raw_data": [],
    "calibration_duration": 10.0,
    "countdown_duration": 3.0,
    "state": "waiting",  # waiting, countdown, collecting, done
    "button_pressed_time": None,
    "collecting_start_time": None,
    "prev_button": False,
}


def main() -> None:
    global calibration_data, device_state

    viz = DvrkRealtimeViz(
        title="Joint2 Calibration - Press button to start (3s countdown, 10s collection)",
        window_size=(1600, 1000),
        background="white",
        show_frames=False,
        alpha=0.8,
        frame_scale=0.05,
        marker_radius=0.01,
    )

    phantom = Phantom(robot_root=DEFAULT_PHANTOM_ROOT)

    T_phantom = np.eye(4)
    T_phantom[:3, :3] = Rz(np.pi / 2)
    T_phantom[:3, 3] = np.array([0.0, 0.0, 0.0], dtype=float)

    q_phantom = np.zeros(phantom.dof, dtype=float)
    viz.add_robot(
        "phantom",
        phantom,
        theta=q_phantom,
        base_transform=T_phantom,
        color="lightsteelblue",
    )

    device_points = pv.PolyData(np.array([[0.0, 0.0, 0.0]], dtype=float))
    fk_points = pv.PolyData(np.array([[0.0, 0.0, 0.0]], dtype=float))

    viz.plotter.add_points(
        device_points,
        color="red",
        point_size=20,
        render_points_as_spheres=True,
    )
    viz.plotter.add_points(
        fk_points,
        color="blue",
        point_size=20,
        render_points_as_spheres=True,
    )

    viz.set_camera(
        position=(1.5, 1.5, 1.2),
        focal_point=(0.15, 0.0, 0.0),
        viewup=(0.0, 0.0, 1.0),
    )
    viz.plotter.show(auto_close=False, interactive_update=True)

    try:
        while True:
            q_phantom = state_to_q(phantom, device_state)
            viz.update_robot("phantom", q_phantom, base_transform=T_phantom)

            # State machine for button-triggered calibration
            button_current = device_state.button

            if calibration_data["state"] == "waiting":
                # Detect button press (rising edge)
                if button_current and not calibration_data["prev_button"]:
                    calibration_data["button_pressed_time"] = time.perf_counter()
                    calibration_data["state"] = "countdown"
                    print("Button pressed! Starting countdown...")

            elif calibration_data["state"] == "countdown":
                elapsed = time.perf_counter() - calibration_data["button_pressed_time"]
                countdown_remaining = calibration_data["countdown_duration"] - elapsed

                if countdown_remaining <= 0:
                    calibration_data["state"] = "collecting"
                    calibration_data["collecting_start_time"] = time.perf_counter()
                    print("Starting data collection for 10 seconds!")
                else:
                    # Update title to show countdown
                    print(
                        f"  Countdown: {countdown_remaining:.1f}s remaining...",
                        end="\r",
                    )

            elif calibration_data["state"] == "collecting":
                elapsed = (
                    time.perf_counter() - calibration_data["collecting_start_time"]
                )

                # Collect raw joints[1] and joints[2]
                j1_raw = float(device_state.joints[1])
                j2_raw = hd.get_joint_angles()[2]
                calibration_data["j1_data"].append(j1_raw)
                calibration_data["j2_raw_data"].append(j2_raw)

                if elapsed >= calibration_data["calibration_duration"]:
                    calibration_data["state"] = "done"
                    print("\nData collection complete!")
                else:
                    print(
                        f"  Collecting: {elapsed:.1f}s / {calibration_data['calibration_duration']:.1f}s",
                        end="\r",
                    )

            elif calibration_data["state"] == "done":
                # Exit loop after data collection
                break

            device_scale = 0.001
            p_device = np.array(
                [
                    *(device_scale * np.array(device_state.position, dtype=float)),
                    1.0,
                ],
                dtype=float,
            )
            p_world = (T_phantom @ p_device)[:3]

            T_links = link_transforms(phantom, q_phantom)
            T_tool_world = T_phantom @ T_links[phantom.tool_link]
            p_fk = T_tool_world[:3, 3]

            device_points.points = np.array([p_world], dtype=float)
            fk_points.points = np.array([p_fk], dtype=float)
            device_points.Modified()
            fk_points.Modified()

            calibration_data["prev_button"] = button_current
            viz.plotter.update()

    except KeyboardInterrupt:
        print(f"\nStopped by user.")
    finally:
        viz.plotter.close()

    # Compute joint2_coeff via linear regression
    j1_data = calibration_data["j1_data"]
    j2_raw_data = calibration_data["j2_raw_data"]

    if len(j1_data) > 1 and len(j2_raw_data) > 1:
        j1_array = np.array(j1_data, dtype=float)
        j2_array = np.array(j2_raw_data, dtype=float)

        # Linear fit: j2_raw = coeff * j1 + intercept
        coeffs = np.polyfit(j1_array, j2_array, 1)
        joint2_coeff = coeffs[0]
        intercept = coeffs[1]

        print("\n" + "=" * 60)
        print("JOINT2 CALIBRATION RESULT")
        print("=" * 60)
        print(
            f"Collected {len(j1_data)} data points over {calibration_data['calibration_duration']:.1f} seconds"
        )
        print(f"\nLinear regression: joints[2]_raw = coeff * joints[1] + intercept")
        print(f"  Slope (joint2_coeff):  {joint2_coeff:.15f}")
        print(f"  Intercept:             {intercept:.15f}")
        print(
            f"  R-squared:             {np.corrcoef(j1_array, j2_array)[0, 1]**2:.6f}"
        )
        print("\nUsage in haptics.py:")
        print(f"  joints[2] = joints[2] - {joint2_coeff:.15f} * joints[1]")
        print("=" * 60 + "\n")
    else:
        print(
            f"Warning: Not enough data collected for calibration ({len(j1_data)} points)"
        )


if __name__ == "__main__":
    device_state = DeviceState()
    run_with_single_device(device_state=device_state, side="left", callback=main)
