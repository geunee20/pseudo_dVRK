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
)


def main() -> None:
    viz = DvrkRealtimeViz(
        title="Dual Phantom Pose Teleoperation",
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
    T_phantom_left[1, 3] = -0.5

    T_phantom_right = np.eye(4)
    T_phantom_right[1, 3] = 0.5

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

    T_psm_left = np.eye(4)
    T_psm_left[:3, :3] = Rz(-np.pi / 2)
    T_psm_left[:3, 3] = np.array([-0.3, -0.25, 0.1])

    T_psm_right = np.eye(4)
    T_psm_right[:3, :3] = Rz(-np.pi / 2)
    T_psm_right[:3, 3] = np.array([-0.3, 0.25, 0.1])

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
        target_psm_left,
        color="magenta",
        point_size=15,
        render_points_as_spheres=True,
    )
    viz.plotter.add_points(
        target_psm_right,
        color="purple",
        point_size=15,
        render_points_as_spheres=True,
    )

    fk_psm_left = pv.PolyData(np.array([p_psm_left_home], dtype=float))
    fk_psm_right = pv.PolyData(np.array([p_psm_right_home], dtype=float))

    viz.plotter.add_points(
        fk_psm_left,
        color="red",
        point_size=8,
        render_points_as_spheres=True,
    )
    viz.plotter.add_points(
        fk_psm_right,
        color="brown",
        point_size=8,
        render_points_as_spheres=True,
    )

    viz.set_camera(
        position=(2.0, -0.2, 1.6),
        focal_point=(0.0, 0.0, 0.0),
        viewup=(0.0, 0.0, 1.0),
    )
    viz.plotter.show(auto_close=False, interactive_update=True)

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
            # ---------------------- Update Phantom State ---------------------
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

            viz.update_robot("psm_left", q_psm_left, base_transform=T_psm_left)
            viz.update_robot("psm_right", q_psm_right, base_transform=T_psm_right)

            T_psm_left_tool_local = psm_ik_left.forward_pose(q_psm_left)
            T_psm_left_tool_world = T_psm_left @ T_psm_left_tool_local
            fk_psm_left.points = np.array([T_psm_left_tool_world[:3, 3]], dtype=float)
            fk_psm_left.Modified()

            T_psm_right_tool_local = psm_ik_right.forward_pose(q_psm_right)
            T_psm_right_tool_world = T_psm_right @ T_psm_right_tool_local
            fk_psm_right.points = np.array([T_psm_right_tool_world[:3, 3]], dtype=float)
            fk_psm_right.Modified()

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
