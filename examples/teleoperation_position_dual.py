from __future__ import annotations

import time

import numpy as np

from src.robots.phantom import Phantom
from src.robots.psm import PSM

from src.kinematics.fk import link_transforms
from src.kinematics.so3 import Rx, Rz
from src.kinematics.pinocchio_ik import PinocchioIK

from src.utils.device_runtime import (
    DEFAULT_PHANTOM_ROOT,
    DEFAULT_PSM_ROOT,
    DeviceState,
    run_with_dual_devices,
)
from src.utils.visualization import (
    ClutchEvent,
    DvrkRealtimeViz,
    compute_desired_position_world,
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
    PHANTOM_DUAL_Y_DISTANCE,
    PSM_BASE_X,
    PSM_BASE_Y_DISTANCE,
    PSM_BASE_Z,
    PSM_BASE_X_ROTATION_SPLIT_DEG,
)


def main() -> None:
    viz = DvrkRealtimeViz(
        title="Dual Phantom Position Teleoperation",
        window_size=(1800, 1000),
        background="white",
        show_frames=False,
        alpha=1.0,
        frame_scale=0.05,
        marker_radius=0.01,
    )

    # --------------------- Phantoms ---------------------
    phantom_left = Phantom(robot_root=DEFAULT_PHANTOM_ROOT)
    phantom_right = Phantom(robot_root=DEFAULT_PHANTOM_ROOT)

    T_phantom_left = np.eye(4)
    T_phantom_left[1, 3] = -PHANTOM_DUAL_Y_DISTANCE / 2.0

    T_phantom_right = np.eye(4)
    T_phantom_right[1, 3] = PHANTOM_DUAL_Y_DISTANCE / 2.0

    viz.add_robot(
        "left",
        phantom_left,
        theta=np.asarray(left_state.joints),
        base_transform=T_phantom_left,
        color="lightsteelblue",
    )
    viz.add_robot(
        "right",
        phantom_right,
        theta=np.asarray(right_state.joints),
        base_transform=T_phantom_right,
        color="salmon",
    )

    p_phantom_left_home = tool_position_world(
        phantom_left,
        np.asarray(left_state.joints),
        base_transform=T_phantom_left,
    )

    p_phantom_right_home = tool_position_world(
        phantom_right,
        np.asarray(right_state.joints),
        base_transform=T_phantom_right,
    )

    fk_phantom_left = create_point_poly(p_phantom_left_home)
    fk_phantom_right = create_point_poly(p_phantom_right_home)

    viz.plotter.add_points(
        fk_phantom_left,
        color="blue",
        point_size=10,
        render_points_as_spheres=True,
    )
    viz.plotter.add_points(
        fk_phantom_right,
        color="darkred",
        point_size=10,
        render_points_as_spheres=True,
    )

    # --------------------- PSMs ---------------------
    psm_left = PSM(robot_root=DEFAULT_PSM_ROOT)
    psm_right = PSM(robot_root=DEFAULT_PSM_ROOT)
    psm_x_half_rad = np.deg2rad(PSM_BASE_X_ROTATION_SPLIT_DEG) / 2.0

    T_psm_left = np.eye(4)
    T_psm_left[:3, :3] = Rx(+psm_x_half_rad) @ Rz(-np.pi / 2)
    T_psm_left[:3, 3] = np.array([PSM_BASE_X, -PSM_BASE_Y_DISTANCE / 2.0, PSM_BASE_Z])

    T_psm_right = np.eye(4)
    T_psm_right[:3, :3] = Rx(-psm_x_half_rad) @ Rz(-np.pi / 2)
    T_psm_right[:3, 3] = np.array([PSM_BASE_X, PSM_BASE_Y_DISTANCE / 2.0, PSM_BASE_Z])

    q_psm_left = np.array([0.0, 0.0, 0.3, 0.1, 0.0, 0.0, 0.0, 1.0])
    q_psm_right = np.array([0.0, 0.0, 0.3, 0.1, 0.0, 0.0, 0.0, 1.0])

    viz.add_robot(
        "psm_left",
        psm_left,
        theta=q_psm_left,
        base_transform=T_psm_left,
        color="orange",
    )
    viz.add_robot(
        "psm_right",
        psm_right,
        theta=q_psm_right,
        base_transform=T_psm_right,
        color="gold",
    )

    psm_ik_left = PinocchioIK(
        robot=psm_left,
        urdf_path=DEFAULT_PSM_ROOT / "psm.urdf",
        ee_frame_name="tool_gripper2_joint",
    )
    psm_ik_right = PinocchioIK(
        robot=psm_right,
        urdf_path=DEFAULT_PSM_ROOT / "psm.urdf",
        ee_frame_name="tool_gripper2_joint",
    )

    T_links_psm_left = link_transforms(psm_left, q_psm_left)
    p_psm_left_home = (T_psm_left @ T_links_psm_left[psm_left.tool_link])[:3, 3]

    T_links_psm_right = link_transforms(psm_right, q_psm_right)
    p_psm_right_home = (T_psm_right @ T_links_psm_right[psm_right.tool_link])[:3, 3]

    target_psm_left = create_point_poly(p_psm_left_home)
    target_psm_right = create_point_poly(p_psm_right_home)

    viz.plotter.add_points(
        target_psm_left,
        color="magenta",
        point_size=5,
        render_points_as_spheres=True,
    )
    viz.plotter.add_points(
        target_psm_right,
        color="purple",
        point_size=5,
        render_points_as_spheres=True,
    )

    fk_psm_left = create_point_poly(p_psm_left_home)
    fk_psm_right = create_point_poly(p_psm_right_home)

    viz.plotter.add_points(
        fk_psm_left,
        color="red",
        point_size=5,
        render_points_as_spheres=True,
    )
    viz.plotter.add_points(
        fk_psm_right,
        color="brown",
        point_size=5,
        render_points_as_spheres=True,
    )

    viz.set_camera(
        position=(2.0, -0.2, 1.6),
        focal_point=(0.0, 0.0, 0.0),
        viewup=(0.0, 0.0, 1.0),
    )
    viz.plotter.show(auto_close=False, interactive_update=True)

    p_psm_left_desired = p_psm_left_home.copy()
    p_psm_right_desired = p_psm_right_home.copy()

    prev_clutch_left = False
    prev_clutch_right = False
    clutch_left_active = False
    clutch_right_active = False

    p_psm_left_desired_hold = p_psm_left_home.copy()
    p_psm_right_desired_hold = p_psm_right_home.copy()

    jaw_left_cmd = float(q_psm_left[JAW_IDX])
    jaw_right_cmd = float(q_psm_right[JAW_IDX])

    try:
        while True:
            # ---------------------- Update Phantom State ---------------------
            viz.update_robot(
                "left", np.asarray(left_state.joints), base_transform=T_phantom_left
            )
            viz.update_robot(
                "right", np.asarray(right_state.joints), base_transform=T_phantom_right
            )

            p_phantom_left = tool_position_world(
                phantom_left,
                np.asarray(left_state.joints),
                base_transform=T_phantom_left,
            )
            update_point_poly(fk_phantom_left, p_phantom_left)

            p_phantom_right = tool_position_world(
                phantom_right,
                np.asarray(right_state.joints),
                base_transform=T_phantom_right,
            )
            update_point_poly(fk_phantom_right, p_phantom_right)

            # ------------------------ Input State -----------------------
            clutch_left = left_state.clutch_button
            clutch_right = right_state.clutch_button
            gripper_left = left_state.gripper_button
            gripper_right = right_state.gripper_button

            (
                clutch_left_active,
                prev_clutch_left,
                clutch_left_event,
            ) = update_clutch_state(
                clutch_pressed=clutch_left,
                prev_clutch_pressed=prev_clutch_left,
                clutch_active=clutch_left_active,
            )
            if clutch_left_event == ClutchEvent.PRESSED:
                p_psm_left_desired_hold = p_psm_left_desired.copy()
                set_robot_mesh_color(viz._robots, "left", "red")
            elif clutch_left_event == ClutchEvent.RELEASED:
                p_phantom_left_home = p_phantom_left.copy()
                p_psm_left_home = p_psm_left_desired.copy()
                set_robot_mesh_color(viz._robots, "left", "lightsteelblue")

            (
                clutch_right_active,
                prev_clutch_right,
                clutch_right_event,
            ) = update_clutch_state(
                clutch_pressed=clutch_right,
                prev_clutch_pressed=prev_clutch_right,
                clutch_active=clutch_right_active,
            )
            if clutch_right_event == ClutchEvent.PRESSED:
                p_psm_right_desired_hold = p_psm_right_desired.copy()
                set_robot_mesh_color(viz._robots, "right", "red")
            elif clutch_right_event == ClutchEvent.RELEASED:
                p_phantom_right_home = p_phantom_right.copy()
                p_psm_right_home = p_psm_right_desired.copy()
                set_robot_mesh_color(viz._robots, "right", "salmon")

            p_psm_left_desired = compute_desired_position_world(
                clutch_active=clutch_left_active,
                desired_hold_world=p_psm_left_desired_hold,
                phantom_tool_world=p_phantom_left,
                phantom_home_world=p_phantom_left_home,
                psm_home_world=p_psm_left_home,
                teleoperation_gain=TELEOPERATION_GAIN,
            )

            p_psm_right_desired = compute_desired_position_world(
                clutch_active=clutch_right_active,
                desired_hold_world=p_psm_right_desired_hold,
                phantom_tool_world=p_phantom_right,
                phantom_home_world=p_phantom_right_home,
                psm_home_world=p_psm_right_home,
                teleoperation_gain=TELEOPERATION_GAIN,
            )

            update_point_poly(target_psm_left, p_psm_left_desired)

            update_point_poly(target_psm_right, p_psm_right_desired)

            R_psm_left = T_psm_left[:3, :3]
            t_psm_left = T_psm_left[:3, 3]
            p_psm_left_desired_local = R_psm_left.T @ (p_psm_left_desired - t_psm_left)

            R_psm_right = T_psm_right[:3, :3]
            t_psm_right = T_psm_right[:3, 3]
            p_psm_right_desired_local = R_psm_right.T @ (
                p_psm_right_desired - t_psm_right
            )

            q_psm_left, _ = psm_ik_left.solve_position(
                p_des=p_psm_left_desired_local,
                q_init=q_psm_left,
                max_iters=40,
                tol=1e-4,
                damping=1e-3,
                step_size=0.5,
            )
            q_psm_right, _ = psm_ik_right.solve_position(
                p_des=p_psm_right_desired_local,
                q_init=q_psm_right,
                max_iters=40,
                tol=1e-4,
                damping=1e-3,
                step_size=0.5,
            )

            jaw_left_cmd = update_jaw_command(
                jaw_cmd=jaw_left_cmd,
                gripper_pressed=gripper_left,
                close_speed=JAW_CLOSE_SPEED,
                open_speed=JAW_OPEN_SPEED,
                jaw_min=JAW_MIN,
                jaw_max=JAW_MAX,
            )
            q_psm_left[JAW_IDX] = jaw_left_cmd

            jaw_right_cmd = update_jaw_command(
                jaw_cmd=jaw_right_cmd,
                gripper_pressed=gripper_right,
                close_speed=JAW_CLOSE_SPEED,
                open_speed=JAW_OPEN_SPEED,
                jaw_min=JAW_MIN,
                jaw_max=JAW_MAX,
            )
            q_psm_right[JAW_IDX] = jaw_right_cmd

            viz.update_robot("psm_left", q_psm_left, base_transform=T_psm_left)
            viz.update_robot("psm_right", q_psm_right, base_transform=T_psm_right)

            T_links_psm_left = link_transforms(psm_left, q_psm_left)
            p_fk_left = (T_psm_left @ T_links_psm_left[psm_left.tool_link])[:3, 3]
            update_point_poly(fk_psm_left, p_fk_left)

            T_links_psm_right = link_transforms(psm_right, q_psm_right)
            p_fk_right = (T_psm_right @ T_links_psm_right[psm_right.tool_link])[:3, 3]
            update_point_poly(fk_psm_right, p_fk_right)

            viz.plotter.update()
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        viz.plotter.close()


if __name__ == "__main__":
    left_state = DeviceState()
    right_state = DeviceState()
    run_with_dual_devices(left_state=left_state, right_state=right_state, callback=main)
