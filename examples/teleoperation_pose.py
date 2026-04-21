from __future__ import annotations

import numpy as np

from src.robots.phantom import Phantom
from src.robots.psm import PSM

from src.kinematics.fk import link_transforms
from src.kinematics.so3 import Rz
from src.kinematics.pinocchio_ik import PinocchioIK

from src.utils.device_runtime import (
    DEFAULT_PHANTOM_ROOT,
    DEFAULT_PSM_ROOT,
    DeviceState,
    run_with_single_device,
)
from src.utils.transforms import inv_transform
from src.utils.visualization import (
    ClutchEvent,
    DvrkRealtimeViz,
    compute_desired_pose_world,
    create_point_poly,
    set_robot_mesh_color,
    tool_position_world,
    update_clutch_state,
    update_jaw_command,
    update_point_poly,
)

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))
from settings import (
    TELEOPERATION_GAIN,
    JAW_IDX,
    JAW_CLOSE_SPEED,
    JAW_OPEN_SPEED,
    JAW_MIN,
    JAW_MAX,
    PHANTOM_BOTH_Y_DISTANCE,
    PSM_BASE_X,
    PSM_BASE_Y_DISTANCE,
    PSM_BASE_Z,
)

# Translation scaling only. Rotation is mapped 1:1 in this version.


def main() -> None:
    viz = DvrkRealtimeViz(
        title="Left Phantom Pose Teleoperation",
        window_size=(1600, 1000),
        background="white",
        show_frames=False,
        alpha=1.0,
        frame_scale=0.05,
        marker_radius=0.01,
    )

    # --------------------- Phantom ---------------------
    phantom = Phantom(robot_root=DEFAULT_PHANTOM_ROOT)
    T_phantom_base = np.eye(4)
    T_phantom_base[1, 3] = -PHANTOM_BOTH_Y_DISTANCE / 2

    viz.add_robot(
        "left",
        phantom,
        theta=np.asarray(device_state.joints),
        base_transform=T_phantom_base,
        color="lightsteelblue",
    )

    p_phantom_home = tool_position_world(
        phantom,
        np.asarray(device_state.joints),
        base_transform=T_phantom_base,
    )

    fk_phantom = create_point_poly(p_phantom_home)
    viz.plotter.add_points(
        fk_phantom,
        color="blue",
        point_size=10,
        render_points_as_spheres=True,
    )

    # --------------------- PSM ---------------------
    psm = PSM(robot_root=DEFAULT_PSM_ROOT)
    T_psm_base = np.eye(4)
    T_psm_base[:3, :3] = Rz(-np.pi / 2)
    T_psm_base[:3, 3] = np.array([PSM_BASE_X, PSM_BASE_Y_DISTANCE / 2.0, PSM_BASE_Z])
    q_psm = np.array(
        [
            0.0,
            0.0,
            0.3,
            0.1,
            0.0,
            -0.8,
            0.0,
            1.0,
        ]
    )

    viz.add_robot("psm", psm, theta=q_psm, base_transform=T_psm_base, color="orange")

    psm_ik = PinocchioIK(
        robot=psm,
        urdf_path=DEFAULT_PSM_ROOT / "psm.urdf",
        ee_frame_name="tool_yaw_link",
    )

    T_psm_tool_local = psm_ik.forward_pose(q_psm)
    T_psm_tool_world = T_psm_base @ T_psm_tool_local
    p_psm_home = T_psm_tool_world[:3, 3]

    target_psm = create_point_poly(p_psm_home)
    viz.plotter.add_points(
        target_psm,
        color="magenta",
        point_size=15,
        render_points_as_spheres=True,
    )

    fk_psm = create_point_poly(p_psm_home)
    viz.plotter.add_points(
        fk_psm,
        color="red",
        point_size=8,
        render_points_as_spheres=True,
    )

    # ---------------------- Visualization Setup ---------------------
    viz.set_camera(
        position=(1.5, 0.0, 1.5),
        focal_point=(0.0, 0.0, 0.0),
        viewup=(0.0, 0.0, 1.0),
    )
    viz.plotter.show(auto_close=False, interactive_update=True)

    # Home references for relative pose teleoperation
    T_links_phantom = link_transforms(phantom, np.asarray(device_state.joints))
    T_phantom_home = T_phantom_base @ T_links_phantom[phantom.tool_link]

    T_psm_home_local = psm_ik.forward_pose(q_psm)
    T_psm_desired_world = T_psm_base @ T_psm_home_local

    prev_clutch_button = False
    clutch_active = False
    T_psm_desired_hold = T_psm_desired_world.copy()

    jaw_cmd = float(q_psm[JAW_IDX])

    try:
        while True:
            # ---------------------- Update Phantom State ---------------------
            viz.update_robot(
                "left", np.asarray(device_state.joints), base_transform=T_phantom_base
            )

            T_links_phantom = link_transforms(phantom, np.asarray(device_state.joints))
            T_phantom_tool_world = T_phantom_base @ T_links_phantom[phantom.tool_link]
            p_phantom = T_phantom_tool_world[:3, 3]

            update_point_poly(fk_phantom, p_phantom)

            # ------------------------ Input State -----------------------
            clutch_button = device_state.clutch_button
            gripper_button = device_state.gripper_button

            # ---------- clutch logic ----------
            (
                clutch_active,
                prev_clutch_button,
                clutch_event,
            ) = update_clutch_state(
                clutch_pressed=clutch_button,
                prev_clutch_pressed=prev_clutch_button,
                clutch_active=clutch_active,
            )
            if clutch_event == ClutchEvent.PRESSED:
                T_psm_desired_hold = T_psm_desired_world.copy()
                set_robot_mesh_color(viz._robots, "left", "red")

            elif clutch_event == ClutchEvent.RELEASED:
                T_phantom_home = T_phantom_tool_world.copy()
                T_psm_home_local = psm_ik.forward_pose(q_psm)
                T_psm_desired_world = T_psm_base @ T_psm_home_local
                set_robot_mesh_color(viz._robots, "left", "lightsteelblue")

            # ---------- pose teleoperation ----------
            T_psm_desired_world = compute_desired_pose_world(
                clutch_active=clutch_active,
                desired_hold_world=T_psm_desired_hold,
                phantom_tool_world=T_phantom_tool_world,
                phantom_home_world=T_phantom_home,
                psm_base_world=T_psm_base,
                psm_home_local=T_psm_home_local,
                teleoperation_gain=TELEOPERATION_GAIN,
            )

            # desired target point in world frame
            update_point_poly(target_psm, T_psm_desired_world[:3, 3])

            # world -> PSM local
            T_psm_desired_local = inv_transform(T_psm_base) @ T_psm_desired_world

            q_psm, ok = psm_ik.solve_pose(
                T_des=T_psm_desired_local,
                q_init=q_psm,
                max_iters=40,
                tol_rot=1e-3,
                tol_pos=1e-4,
                damping=1e-3,
                step_size=0.4,
                weight_rot=0.2,
                weight_pos=1.0,
            )

            # ---------- jaw logic ----------
            jaw_cmd = update_jaw_command(
                jaw_cmd=jaw_cmd,
                gripper_pressed=gripper_button,
                close_speed=JAW_CLOSE_SPEED,
                open_speed=JAW_OPEN_SPEED,
                jaw_min=JAW_MIN,
                jaw_max=JAW_MAX,
            )
            q_psm[JAW_IDX] = jaw_cmd

            viz.update_robot("psm", q_psm, base_transform=T_psm_base)

            T_psm_tool_local = psm_ik.forward_pose(q_psm)
            T_psm_tool_world = T_psm_base @ T_psm_tool_local

            update_point_poly(fk_psm, T_psm_tool_world[:3, 3])

            viz.plotter.update()

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        viz.plotter.close()


if __name__ == "__main__":
    device_state = DeviceState()
    run_with_single_device(device_state=device_state, side="left", callback=main)
