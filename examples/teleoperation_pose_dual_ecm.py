from __future__ import annotations

import time

import numpy as np
import pyvista as pv

from src.robots.phantom import Phantom
from src.robots.psm import PSM

from src.kinematics.fk import link_transforms
from src.kinematics.so3 import Rx, Rz
from src.kinematics.pinocchio_ik import PinocchioIK

from src.utils.real_time_viz import (
    DvrkRealtimeViz,
    add_robot_meshes_to_plotter,
    hfov_to_vfov_deg,
    update_robot_meshes_on_plotter,
)
from src.utils.ecm_camera import (
    build_ecm_control_from_camera_pose,
    create_camera_plotter,
    register_keys,
    update_cameras,
)
from src.utils.haptics import DeviceState
from src.utils.transforms import inv_transform
from src.utils.script_common import (
    DEFAULT_ECM_ROOT,
    DEFAULT_PHANTOM_ROOT,
    DEFAULT_PSM_ROOT,
    run_with_dual_devices,
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


def _camera_pose_from_psm_tools(
    p_left_world: np.ndarray,
    p_right_world: np.ndarray,
) -> np.ndarray:
    """Compute camera pose from PSM home midpoint with fixed z=0.35."""
    p_mid = 0.5 * (p_left_world + p_right_world)

    T = np.eye(4, dtype=float)
    # Keep camera forward on world -Z, then rotate around world Z by 180deg.
    T[0, 0] = 1.0
    T[1, 1] = -1.0
    T[2, 2] = -1.0
    T[:3, 3] = np.array([p_mid[0], p_mid[1], CAMERA_Z_M])
    return T


def main() -> None:
    # ===================== 3rd-person viz =====================
    viz = DvrkRealtimeViz(
        title="Dual Pose Teleoperation (3rd Person)",
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

    T_links_left = link_transforms(phantom_left, np.asarray(left_state.joints))
    T_phantom_left_tool_world = T_phantom_left @ T_links_left[phantom_left.tool_link]
    p_phantom_left_home = T_phantom_left_tool_world[:3, 3]

    T_links_right = link_transforms(phantom_right, np.asarray(right_state.joints))
    T_phantom_right_tool_world = (
        T_phantom_right @ T_links_right[phantom_right.tool_link]
    )
    p_phantom_right_home = T_phantom_right_tool_world[:3, 3]

    fk_phantom_left = pv.PolyData(np.array([p_phantom_left_home], dtype=float))
    fk_phantom_right = pv.PolyData(np.array([p_phantom_right_home], dtype=float))
    viz.plotter.add_points(
        fk_phantom_left, color="blue", point_size=10, render_points_as_spheres=True
    )
    viz.plotter.add_points(
        fk_phantom_right, color="darkred", point_size=10, render_points_as_spheres=True
    )

    # ----- PSMs -----
    psm_left = PSM(robot_root=DEFAULT_PSM_ROOT)
    psm_right = PSM(robot_root=DEFAULT_PSM_ROOT)
    psm_x_half_rad = np.deg2rad(PSM_BASE_X_ROTATION_SPLIT_DEG) / 2.0

    T_psm_left = np.eye(4)
    T_psm_left[:3, :3] = Rx(+psm_x_half_rad) @ Rz(-np.pi / 2)
    T_psm_left[:3, 3] = np.array([PSM_BASE_X, -PSM_BASE_Y_DISTANCE / 2.0, PSM_BASE_Z])
    T_psm_right = np.eye(4)
    T_psm_right[:3, :3] = Rx(-psm_x_half_rad) @ Rz(-np.pi / 2)
    T_psm_right[:3, 3] = np.array([PSM_BASE_X, PSM_BASE_Y_DISTANCE / 2.0, PSM_BASE_Z])

    q_psm_left = np.array([0.0, 0.0, 0.3, 0.1, 0.0, -0.8, 0.0, 1.0])
    q_psm_right = np.array([0.0, 0.0, 0.3, 0.1, 0.0, -0.8, 0.0, 1.0])

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
        ee_frame_name="tool_yaw_link",
    )
    psm_ik_right = PinocchioIK(
        robot=psm_right,
        urdf_path=DEFAULT_PSM_ROOT / "psm.urdf",
        ee_frame_name="tool_yaw_link",
    )

    T_psm_left_tool_local = psm_ik_left.forward_pose(q_psm_left)
    T_psm_left_tool_world = T_psm_left @ T_psm_left_tool_local
    p_psm_left_home = T_psm_left_tool_world[:3, 3]

    T_psm_right_tool_local = psm_ik_right.forward_pose(q_psm_right)
    T_psm_right_tool_world = T_psm_right @ T_psm_right_tool_local
    p_psm_right_home = T_psm_right_tool_world[:3, 3]

    target_psm_left = pv.PolyData(np.array([p_psm_left_home], dtype=float))
    target_psm_right = pv.PolyData(np.array([p_psm_right_home], dtype=float))
    viz.plotter.add_points(
        target_psm_left, color="magenta", point_size=15, render_points_as_spheres=True
    )
    viz.plotter.add_points(
        target_psm_right, color="purple", point_size=15, render_points_as_spheres=True
    )

    fk_psm_left = pv.PolyData(np.array([p_psm_left_home], dtype=float))
    fk_psm_right = pv.PolyData(np.array([p_psm_right_home], dtype=float))
    viz.plotter.add_points(
        fk_psm_left, color="red", point_size=8, render_points_as_spheres=True
    )
    viz.plotter.add_points(
        fk_psm_right, color="brown", point_size=8, render_points_as_spheres=True
    )

    # ----- ECM setup from computed camera pose -----
    T_cam = _camera_pose_from_psm_tools(p_psm_left_home, p_psm_right_home)

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
    cam_left_psm_left = pv.PolyData(np.array([p_psm_left_home], dtype=float))
    cam_left_psm_right = pv.PolyData(np.array([p_psm_right_home], dtype=float))
    cam_plotter.add_points(
        cam_left_psm_left, color="orange", point_size=12, render_points_as_spheres=True
    )
    cam_plotter.add_points(
        cam_left_psm_right, color="gold", point_size=12, render_points_as_spheres=True
    )

    cam_right_psm_left_mesh = None
    cam_right_psm_right_mesh = None
    cam_right_psm_left = None
    cam_right_psm_right = None

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
        cam_right_psm_left = pv.PolyData(np.array([p_psm_left_home], dtype=float))
        cam_right_psm_right = pv.PolyData(np.array([p_psm_right_home], dtype=float))
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
    # Home references for relative pose teleoperation
    T_links_left = link_transforms(phantom_left, np.asarray(left_state.joints))
    T_phantom_left_home = T_phantom_left @ T_links_left[phantom_left.tool_link]
    T_links_right = link_transforms(phantom_right, np.asarray(right_state.joints))
    T_phantom_right_home = T_phantom_right @ T_links_right[phantom_right.tool_link]

    T_psm_left_home_local = psm_ik_left.forward_pose(q_psm_left)
    T_psm_left_desired_world = T_psm_left @ T_psm_left_home_local
    T_psm_right_home_local = psm_ik_right.forward_pose(q_psm_right)
    T_psm_right_desired_world = T_psm_right @ T_psm_right_home_local

    prev_clutch_left = False
    prev_clutch_right = False
    clutch_left_active = False
    clutch_right_active = False

    T_psm_left_desired_hold = T_psm_left_desired_world.copy()
    T_psm_right_desired_hold = T_psm_right_desired_world.copy()

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

            T_links_left = link_transforms(phantom_left, np.asarray(left_state.joints))
            T_phantom_left_tool_world = (
                T_phantom_left @ T_links_left[phantom_left.tool_link]
            )
            p_phantom_left = T_phantom_left_tool_world[:3, 3]
            fk_phantom_left.points = np.array([p_phantom_left], dtype=float)
            fk_phantom_left.Modified()

            T_links_right = link_transforms(
                phantom_right, np.asarray(right_state.joints)
            )
            T_phantom_right_tool_world = (
                T_phantom_right @ T_links_right[phantom_right.tool_link]
            )
            p_phantom_right = T_phantom_right_tool_world[:3, 3]
            fk_phantom_right.points = np.array([p_phantom_right], dtype=float)
            fk_phantom_right.Modified()

            # ---- Clutch ----
            clutch_left = left_state.clutch_button
            clutch_right = right_state.clutch_button
            gripper_left = left_state.gripper_button
            gripper_right = right_state.gripper_button

            if clutch_left and not prev_clutch_left:
                clutch_left_active = True
                T_psm_left_desired_hold = T_psm_left_desired_world.copy()
                for item in viz._robots["left"].mesh_items.values():
                    item.actor.prop.color = "red"  # type: ignore
            elif (not clutch_left) and prev_clutch_left:
                clutch_left_active = False
                T_phantom_left_home = T_phantom_left_tool_world.copy()
                T_psm_left_home_local = psm_ik_left.forward_pose(q_psm_left)
                T_psm_left_desired_world = T_psm_left @ T_psm_left_home_local
                for item in viz._robots["left"].mesh_items.values():
                    item.actor.prop.color = "lightsteelblue"  # type: ignore

            if clutch_right and not prev_clutch_right:
                clutch_right_active = True
                T_psm_right_desired_hold = T_psm_right_desired_world.copy()
                for item in viz._robots["right"].mesh_items.values():
                    item.actor.prop.color = "red"  # type: ignore
            elif (not clutch_right) and prev_clutch_right:
                clutch_right_active = False
                T_phantom_right_home = T_phantom_right_tool_world.copy()
                T_psm_right_home_local = psm_ik_right.forward_pose(q_psm_right)
                T_psm_right_desired_world = T_psm_right @ T_psm_right_home_local
                for item in viz._robots["right"].mesh_items.values():
                    item.actor.prop.color = "salmon"  # type: ignore

            prev_clutch_left = clutch_left
            prev_clutch_right = clutch_right

            # ---- Pose teleoperation ----
            if clutch_left_active:
                T_psm_left_desired_world = T_psm_left_desired_hold.copy()
            else:
                T_psm_left_home_world = T_psm_left @ T_psm_left_home_local
                delta_left = (
                    T_phantom_left_tool_world[:3, 3] - T_phantom_left_home[:3, 3]
                )
                p_left_new = (
                    T_psm_left_home_world[:3, 3] + TELEOPERATION_GAIN * delta_left
                )
                R_left_delta = (
                    T_phantom_left_home[:3, :3].T @ T_phantom_left_tool_world[:3, :3]
                )
                R_left_new = T_psm_left_home_world[:3, :3] @ R_left_delta
                T_psm_left_desired_world = np.eye(4)
                T_psm_left_desired_world[:3, :3] = R_left_new
                T_psm_left_desired_world[:3, 3] = p_left_new

            if clutch_right_active:
                T_psm_right_desired_world = T_psm_right_desired_hold.copy()
            else:
                T_psm_right_home_world = T_psm_right @ T_psm_right_home_local
                delta_right = (
                    T_phantom_right_tool_world[:3, 3] - T_phantom_right_home[:3, 3]
                )
                p_right_new = (
                    T_psm_right_home_world[:3, 3] + TELEOPERATION_GAIN * delta_right
                )
                R_right_delta = (
                    T_phantom_right_home[:3, :3].T @ T_phantom_right_tool_world[:3, :3]
                )
                R_right_new = T_psm_right_home_world[:3, :3] @ R_right_delta
                T_psm_right_desired_world = np.eye(4)
                T_psm_right_desired_world[:3, :3] = R_right_new
                T_psm_right_desired_world[:3, 3] = p_right_new

            target_psm_left.points = np.array(
                [T_psm_left_desired_world[:3, 3]], dtype=float
            )
            target_psm_left.Modified()
            target_psm_right.points = np.array(
                [T_psm_right_desired_world[:3, 3]], dtype=float
            )
            target_psm_right.Modified()

            T_psm_left_desired_local = (
                inv_transform(T_psm_left) @ T_psm_left_desired_world
            )
            T_psm_right_desired_local = (
                inv_transform(T_psm_right) @ T_psm_right_desired_world
            )

            q_psm_left, _ = psm_ik_left.solve_pose(
                T_des=T_psm_left_desired_local,
                q_init=q_psm_left,
                max_iters=40,
                tol_rot=1e-3,
                tol_pos=1e-4,
                damping=1e-3,
                step_size=0.4,
                weight_rot=0.2,
                weight_pos=1.0,
            )
            q_psm_right, _ = psm_ik_right.solve_pose(
                T_des=T_psm_right_desired_local,
                q_init=q_psm_right,
                max_iters=40,
                tol_rot=1e-3,
                tol_pos=1e-4,
                damping=1e-3,
                step_size=0.4,
                weight_rot=0.2,
                weight_pos=1.0,
            )

            if gripper_left:
                jaw_left_cmd -= JAW_CLOSE_SPEED
            else:
                jaw_left_cmd += JAW_OPEN_SPEED
            jaw_left_cmd = np.clip(jaw_left_cmd, JAW_MIN, JAW_MAX)
            q_psm_left[JAW_IDX] = jaw_left_cmd

            if gripper_right:
                jaw_right_cmd -= JAW_CLOSE_SPEED
            else:
                jaw_right_cmd += JAW_OPEN_SPEED
            jaw_right_cmd = np.clip(jaw_right_cmd, JAW_MIN, JAW_MAX)
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
                if cam_right_psm_left_mesh is not None:
                    update_robot_meshes_on_plotter(
                        cam_right_psm_left_mesh, q_psm_left, base_transform=T_psm_left
                    )
                if cam_right_psm_right_mesh is not None:
                    update_robot_meshes_on_plotter(
                        cam_right_psm_right_mesh,
                        q_psm_right,
                        base_transform=T_psm_right,
                    )

            T_psm_left_tool_local = psm_ik_left.forward_pose(q_psm_left)
            T_psm_left_tool_world = T_psm_left @ T_psm_left_tool_local
            fk_psm_left.points = np.array([T_psm_left_tool_world[:3, 3]], dtype=float)
            fk_psm_left.Modified()

            T_psm_right_tool_local = psm_ik_right.forward_pose(q_psm_right)
            T_psm_right_tool_world = T_psm_right @ T_psm_right_tool_local
            fk_psm_right.points = np.array([T_psm_right_tool_world[:3, 3]], dtype=float)
            fk_psm_right.Modified()

            # ---- Update camera view: PSM tool tip markers ----
            cam_left_psm_left.points = np.array(
                [T_psm_left_tool_world[:3, 3]], dtype=float
            )
            cam_left_psm_left.Modified()
            cam_left_psm_right.points = np.array(
                [T_psm_right_tool_world[:3, 3]], dtype=float
            )
            cam_left_psm_right.Modified()
            if is_stereo:
                if cam_right_psm_left is not None:
                    cam_right_psm_left.points = np.array(
                        [T_psm_left_tool_world[:3, 3]], dtype=float
                    )
                    cam_right_psm_left.Modified()
                if cam_right_psm_right is not None:
                    cam_right_psm_right.points = np.array(
                        [T_psm_right_tool_world[:3, 3]], dtype=float
                    )
                    cam_right_psm_right.Modified()

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
