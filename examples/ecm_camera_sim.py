from __future__ import annotations

import numpy as np
import pyvista as pv

from src.utils.visualization import (
    BASELINE_M,
    FAR,
    HFOV_DEG,
    NEAR,
    build_ecm_control_from_camera_pose,
    build_arg_parser,
    register_keys,
    update_cameras,
)
from src.utils.device_runtime import DEFAULT_ECM_ROOT
from src.utils.visualization import add_world_floor_and_object, hfov_to_vfov_deg


def main() -> None:
    args = build_arg_parser().parse_args()

    q0 = np.zeros(4, dtype=float)
    q0[2] = 0.12

    # Desired camera pose: at (0, 0, 0.15) looking straight down (-Z).
    # Rx(π) flips local Y and Z so local +Z points toward world -Z.
    T_cam = np.eye(4, dtype=float)
    T_cam[1, 1] = -1.0  # Rx(π): cos π = -1 on Y,Z diag
    T_cam[2, 2] = -1.0
    T_cam[:3, 3] = [0.0, 0.0, 0.15]

    ecm, ctrl = build_ecm_control_from_camera_pose(
        camera_world_tf=T_cam,
        robot_root=DEFAULT_ECM_ROOT,
        initial_q=q0,
    )

    if args.cameras == 1:
        plotter = pv.Plotter(title="ECM Mono Camera Sim", window_size=[900, 900])
        add_world_floor_and_object(
            plotter,
            object_type="cube",
            center=(0.0, 0.0, 0.0),
            color="tomato",
            cube_size=(0.05, 0.04, 0.06),
            sphere_radius=0.02,
            floor_size=(0.35, 0.35),
        )
        vfov_deg = hfov_to_vfov_deg(HFOV_DEG, 900, 900)
    else:
        plotter = pv.Plotter(
            shape=(1, 2),
            title="ECM Stereo Camera Sim (Left | Right)",
            window_size=[1400, 700],
        )
        plotter.subplot(0, 0)
        add_world_floor_and_object(
            plotter,
            object_type="cube",
            center=(0.0, 0.0, 0.035),
            color="tomato",
            cube_size=(0.05, 0.04, 0.06),
            sphere_radius=0.02,
            floor_size=(0.35, 0.35),
        )
        plotter.add_text("Left", font_size=14)

        plotter.subplot(0, 1)
        add_world_floor_and_object(
            plotter,
            object_type="cube",
            center=(0.0, 0.0, 0.035),
            color="tomato",
            cube_size=(0.05, 0.04, 0.06),
            sphere_radius=0.02,
            floor_size=(0.35, 0.35),
        )
        plotter.add_text("Right", font_size=14)

        vfov_deg = hfov_to_vfov_deg(HFOV_DEG, 700, 700)

    update_cameras(
        plotter,
        ecm,
        ctrl,
        optical_tilt_deg=float(args.scope_deg),
        n_cameras=args.cameras,
        vfov_deg=vfov_deg,
    )

    register_keys(
        plotter,
        ecm,
        ctrl,
        optical_tilt_deg=float(args.scope_deg),
        n_cameras=args.cameras,
        vfov_deg=vfov_deg,
    )

    print("Controls:")
    print("  Left / Right  : ECM yaw")
    print("  Up / Down     : ECM pitch")
    print("  PageUp / PageDown: ECM insertion")
    print(f"  Cameras       : {args.cameras}")
    print(f"  Scope tilt    : {args.scope_deg:.0f} deg")
    print(f"  HFOV          : {HFOV_DEG:.1f} deg")
    print(f"  near/far      : {NEAR:.2f} / {FAR:.1f}")
    if args.cameras == 2:
        print(f"  baseline      : {BASELINE_M:.3f} m")

    plotter.show()


if __name__ == "__main__":
    main()
