from __future__ import annotations

import time

import numpy as np
import pyvista as pv

from src.robots.phantom import Phantom
from src.robots.psm import PSM

from src.kinematics.fk import link_transforms
from src.kinematics.pinocchio_ik import PinocchioIK

from src.utils.device_runtime import (
    DEFAULT_ECM_ROOT,
    DEFAULT_PHANTOM_ROOT,
    DEFAULT_PSM_ROOT,
    DeviceState,
    run_with_dual_devices,
)
from src.utils.visualization import (
    ClutchEvent,
    DvrkRealtimeViz,
    PlotterRobotMeshState,
    add_robot_meshes_to_plotter,
    build_dual_phantom_base_transforms,
    build_dual_psm_base_transforms,
    build_ecm_control_from_camera_pose,
    camera_pose_from_psm_tools,
    compute_desired_position_world,
    create_point_poly,
    create_camera_plotter,
    hfov_to_vfov_deg,
    register_keys,
    set_robot_mesh_color,
    tool_position_world,
    update_cameras,
    update_clutch_state,
    update_jaw_command,
    update_point_poly,
    update_robot_meshes_on_plotter,
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
    ECM_CAMERA_HFOV_DEG,
    ECM_CAMERA_DEFAULT_COUNT,
    ECM_CAMERA_DEFAULT_SCOPE_DEG,
    CAMERA_ROLL_DEG,
    CAMERA_Z_M,
)


def main() -> None:
    # ===================== 3rd-person viz =====================
    viz = DvrkRealtimeViz(
        title="Dual Position Teleoperation (3rd Person)",
        window_size=(1800, 1000),
        background="white",
        show_frames=False,
        alpha=1.0,
        frame_scale=0.05,
        marker_radius=0.01,
    )

    # ----- Phantoms -----
    phantom_left = Phantom(robot_root=DEFAULT_PHANTOM_ROOT)
    phantom_right = Phantom(robot_root=DEFAULT_PHANTOM_ROOT)

    T_phantom_left, T_phantom_right = build_dual_phantom_base_transforms(
        PHANTOM_DUAL_Y_DISTANCE
    )

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
        fk_phantom_left, color="blue", point_size=10, render_points_as_spheres=True
    )
    viz.plotter.add_points(
        fk_phantom_right, color="darkred", point_size=10, render_points_as_spheres=True
    )

    # ----- PSMs -----
    psm_left = PSM(robot_root=DEFAULT_PSM_ROOT)
    psm_right = PSM(robot_root=DEFAULT_PSM_ROOT)
    T_psm_left, T_psm_right = build_dual_psm_base_transforms(
        psm_base_x=PSM_BASE_X,
        psm_base_y_distance=PSM_BASE_Y_DISTANCE,
        psm_base_z=PSM_BASE_Z,
        psm_base_x_rotation_split_deg=PSM_BASE_X_ROTATION_SPLIT_DEG,
    )

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

    T_psm_left_ee_local = psm_ik_left.forward_pose(q_psm_left)
    T_psm_left_ee_world = T_psm_left @ T_psm_left_ee_local
    p_psm_left_home = T_psm_left_ee_world[:3, 3]

    T_psm_right_ee_local = psm_ik_right.forward_pose(q_psm_right)
    T_psm_right_ee_world = T_psm_right @ T_psm_right_ee_local
    p_psm_right_home = T_psm_right_ee_world[:3, 3]

    target_psm_left = create_point_poly(p_psm_left_home)
    target_psm_right = create_point_poly(p_psm_right_home)
    viz.plotter.add_points(
        target_psm_left, color="magenta", point_size=5, render_points_as_spheres=True
    )
    viz.plotter.add_points(
        target_psm_right, color="purple", point_size=5, render_points_as_spheres=True
    )

    fk_psm_left = create_point_poly(p_psm_left_home)
    fk_psm_right = create_point_poly(p_psm_right_home)
    viz.plotter.add_points(
        fk_psm_left, color="red", point_size=5, render_points_as_spheres=True
    )
    viz.plotter.add_points(
        fk_psm_right, color="brown", point_size=5, render_points_as_spheres=True
    )

    # ----- ECM setup from computed camera pose -----
    T_cam = camera_pose_from_psm_tools(
        p_left_world=p_psm_left_home,
        p_right_world=p_psm_right_home,
        camera_z_m=CAMERA_Z_M,
    )

    ecm, ecm_ctrl = build_ecm_control_from_camera_pose(
        camera_world_tf=T_cam,
        robot_root=DEFAULT_ECM_ROOT,
        initial_q=np.zeros(4, dtype=float),
    )
    vfov_deg = hfov_to_vfov_deg(ECM_CAMERA_HFOV_DEG, 700, 700)

    # ----- ECM in 3rd-person view -----
    viz.add_robot(
        "ecm",
        ecm,
        theta=ecm_ctrl.q,
        base_transform=ecm.base_transform,
        color="gray",
    )

    viz.set_camera(
        position=(2.0, -0.2, 1.6),
        focal_point=(0.0, 0.0, 0.0),
        viewup=(0.0, 0.0, 1.0),
    )
    viz.plotter.show(auto_close=False, interactive_update=True)

    # ===================== Camera view window =====================
    cam_plotter, is_stereo = create_camera_plotter(ECM_CAMERA_DEFAULT_COUNT)

    # PSM tool tip markers in camera view
    cam_plotter.subplot(0, 0)
    cam_left_psm_left_mesh = add_robot_meshes_to_plotter(
        cam_plotter,
        name="cam_left_psm_left",
        robot=psm_left,
        theta=q_psm_left,
        base_transform=T_psm_left,
        color="orange",
        alpha=1.0,
    )
    cam_left_psm_right_mesh = add_robot_meshes_to_plotter(
        cam_plotter,
        name="cam_left_psm_right",
        robot=psm_right,
        theta=q_psm_right,
        base_transform=T_psm_right,
        color="gold",
        alpha=1.0,
    )
    cam_left_psm_left = create_point_poly(p_psm_left_home)
    cam_left_psm_right = create_point_poly(p_psm_right_home)
    cam_plotter.add_points(
        cam_left_psm_left, color="orange", point_size=12, render_points_as_spheres=True
    )
    cam_plotter.add_points(
        cam_left_psm_right, color="gold", point_size=12, render_points_as_spheres=True
    )

    cam_right_psm_left_mesh: PlotterRobotMeshState | None = None
    cam_right_psm_right_mesh: PlotterRobotMeshState | None = None
    cam_right_psm_left: pv.PolyData | None = None
    cam_right_psm_right: pv.PolyData | None = None

    if is_stereo:
        cam_plotter.subplot(0, 1)
        cam_right_psm_left_mesh = add_robot_meshes_to_plotter(
            cam_plotter,
            name="cam_right_psm_left",
            robot=psm_left,
            theta=q_psm_left,
            base_transform=T_psm_left,
            color="orange",
            alpha=1.0,
        )
        cam_right_psm_right_mesh = add_robot_meshes_to_plotter(
            cam_plotter,
            name="cam_right_psm_right",
            robot=psm_right,
            theta=q_psm_right,
            base_transform=T_psm_right,
            color="gold",
            alpha=1.0,
        )
        cam_right_psm_left = create_point_poly(p_psm_left_home)
        cam_right_psm_right = create_point_poly(p_psm_right_home)
        cam_plotter.add_points(
            cam_right_psm_left,
            color="orange",
            point_size=12,
            render_points_as_spheres=True,
        )
        cam_plotter.add_points(
            cam_right_psm_right,
            color="gold",
            point_size=12,
            render_points_as_spheres=True,
        )

    register_keys(
        cam_plotter,
        ecm,
        ecm_ctrl,
        optical_tilt_deg=ECM_CAMERA_DEFAULT_SCOPE_DEG,
        n_cameras=ECM_CAMERA_DEFAULT_COUNT,
        vfov_deg=vfov_deg,
        camera_roll_deg=CAMERA_ROLL_DEG,
    )

    update_cameras(
        cam_plotter,
        ecm,
        ecm_ctrl,
        optical_tilt_deg=ECM_CAMERA_DEFAULT_SCOPE_DEG,
        n_cameras=ECM_CAMERA_DEFAULT_COUNT,
        vfov_deg=vfov_deg,
        camera_roll_deg=CAMERA_ROLL_DEG,
    )
    cam_plotter.show(auto_close=False, interactive_update=True)

    # ===================== Teleoperation loop =====================
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
            # ---- Phantom state ----
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

            # ---- Clutch ----
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

            # ---- Position teleoperation ----
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

            # ---- Update 3rd-person viz ----
            viz.update_robot("psm_left", q_psm_left, base_transform=T_psm_left)
            viz.update_robot("psm_right", q_psm_right, base_transform=T_psm_right)
            viz.update_robot("ecm", ecm_ctrl.q, base_transform=ecm.base_transform)
            update_robot_meshes_on_plotter(
                cam_left_psm_left_mesh, q_psm_left, base_transform=T_psm_left
            )
            update_robot_meshes_on_plotter(
                cam_left_psm_right_mesh, q_psm_right, base_transform=T_psm_right
            )
            if is_stereo:
                assert cam_right_psm_left_mesh is not None
                assert cam_right_psm_right_mesh is not None
                update_robot_meshes_on_plotter(
                    cam_right_psm_left_mesh, q_psm_left, base_transform=T_psm_left
                )
                update_robot_meshes_on_plotter(
                    cam_right_psm_right_mesh, q_psm_right, base_transform=T_psm_right
                )

            T_links_psm_left = link_transforms(psm_left, q_psm_left)
            p_fk_left = (T_psm_left @ T_links_psm_left[psm_left.tool_link])[:3, 3]
            update_point_poly(fk_psm_left, p_fk_left)

            T_links_psm_right = link_transforms(psm_right, q_psm_right)
            p_fk_right = (T_psm_right @ T_links_psm_right[psm_right.tool_link])[:3, 3]
            update_point_poly(fk_psm_right, p_fk_right)

            # ---- Update camera view: PSM tool tip markers ----
            update_point_poly(cam_left_psm_left, p_fk_left)
            update_point_poly(cam_left_psm_right, p_fk_right)
            if is_stereo:
                assert cam_right_psm_left is not None
                assert cam_right_psm_right is not None
                update_point_poly(cam_right_psm_left, p_fk_left)
                update_point_poly(cam_right_psm_right, p_fk_right)

            # ---- Refresh ECM camera pose ----
            update_cameras(
                cam_plotter,
                ecm,
                ecm_ctrl,
                optical_tilt_deg=ECM_CAMERA_DEFAULT_SCOPE_DEG,
                n_cameras=ECM_CAMERA_DEFAULT_COUNT,
                vfov_deg=vfov_deg,
                camera_roll_deg=CAMERA_ROLL_DEG,
            )

            viz.plotter.update()
            cam_plotter.update()
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        viz.plotter.close()
        cam_plotter.close()


if __name__ == "__main__":
    left_state = DeviceState()
    right_state = DeviceState()
    run_with_dual_devices(left_state=left_state, right_state=right_state, callback=main)
