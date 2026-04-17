from __future__ import annotations

import time

import numpy as np
import pyvista as pv

from src.robots.phantom import Phantom
from src.robots.psm import PSM

from src.kinematics.fk import link_transforms
from src.kinematics.so3 import Rz
from src.kinematics.pinocchio_ik import PinocchioIK

from src.utils.real_time_viz import DvrkRealtimeViz
from src.utils.haptics import DeviceState
from src.utils.transforms import inv_transform
from src.utils.script_common import (
    DEFAULT_PHANTOM_ROOT,
    DEFAULT_PSM_ROOT,
    run_with_single_device,
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
    - PHANTOM_BOTH_Y_DISTANCE / 2,
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
    T_phantom_base[1, 3] = - PHANTOM_BOTH_Y_DISTANCE / 2

    viz.add_robot(
        "left",
        phantom,
        theta=np.asarray(device_state.joints),
        base_transform=T_phantom_base,
        color="lightsteelblue",
    )

    T_links_phantom = link_transforms(phantom, np.asarray(device_state.joints))
    T_phantom_tool_world = T_phantom_base @ T_links_phantom[phantom.tool_link]
    p_phantom_home = T_phantom_tool_world[:3, 3]

    fk_phantom = pv.PolyData(np.array([p_phantom_home], dtype=float))
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

    target_psm = pv.PolyData(np.array([p_psm_home], dtype=float))
    viz.plotter.add_points(
        target_psm,
        color="magenta",
        point_size=15,
        render_points_as_spheres=True,
    )

    fk_psm = pv.PolyData(np.array([p_psm_home], dtype=float))
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

            fk_phantom.points = np.array([p_phantom], dtype=float)
            fk_phantom.Modified()

            # ------------------------ Input State -----------------------
            clutch_button = device_state.clutch_button
            gripper_button = device_state.gripper_button

            # ---------- clutch logic ----------
            if clutch_button and not prev_clutch_button:
                clutch_active = True
                T_psm_desired_hold = T_psm_desired_world.copy()
                for item in viz._robots["left"].mesh_items.values():
                    item.actor.prop.color = "red"  # type: ignore

            elif (not clutch_button) and prev_clutch_button:
                clutch_active = False
                T_phantom_home = T_phantom_tool_world.copy()
                T_psm_home_local = psm_ik.forward_pose(q_psm)
                T_psm_desired_world = T_psm_base @ T_psm_home_local
                for item in viz._robots["left"].mesh_items.values():
                    item.actor.prop.color = "lightsteelblue"  # type: ignore

            prev_clutch_button = clutch_button

            # ---------- pose teleoperation ----------
            if clutch_active:
                T_psm_desired_world = T_psm_desired_hold.copy()
            else:
                T_psm_home_world = T_psm_base @ T_psm_home_local

                # position: world delta
                delta_phantom = T_phantom_tool_world[:3, 3] - T_phantom_home[:3, 3]
                p_new = T_psm_home_world[:3, 3] + TELEOPERATION_GAIN * delta_phantom

                # orientation: relative rotation
                R_phantom_home = T_phantom_home[:3, :3]
                R_phantom_cur = T_phantom_tool_world[:3, :3]
                R_delta = R_phantom_home.T @ R_phantom_cur

                R_psm_home = T_psm_home_world[:3, :3]
                R_new = R_psm_home @ R_delta

                T_psm_desired_world = np.eye(4)
                T_psm_desired_world[:3, :3] = R_new
                T_psm_desired_world[:3, 3] = p_new

            # desired target point in world frame
            target_psm.points = np.array([T_psm_desired_world[:3, 3]], dtype=float)
            target_psm.Modified()

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
            if gripper_button:
                jaw_cmd -= JAW_CLOSE_SPEED
            else:
                jaw_cmd += JAW_OPEN_SPEED

            jaw_cmd = np.clip(jaw_cmd, JAW_MIN, JAW_MAX)
            q_psm[JAW_IDX] = jaw_cmd

            viz.update_robot("psm", q_psm, base_transform=T_psm_base)

            T_psm_tool_local = psm_ik.forward_pose(q_psm)
            T_psm_tool_world = T_psm_base @ T_psm_tool_local

            fk_psm.points = np.array([T_psm_tool_world[:3, 3]], dtype=float)
            fk_psm.Modified()

            viz.plotter.update()

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        viz.plotter.close()


if __name__ == "__main__":
    device_state = DeviceState()
    run_with_single_device(device_state=device_state, side="left", callback=main)
