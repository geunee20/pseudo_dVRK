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
    PHANTOM_BOTH_Y_DISTANCE,
    PSM_BASE_X,
    PSM_BASE_Y_DISTANCE,
    PSM_BASE_Z,
)


def main() -> None:
    viz = DvrkRealtimeViz(
        title="Left Phantom Real-Time Viz",
        window_size=(1600, 1000),
        background="white",
        show_frames=False,
        alpha=1.0,
        frame_scale=0.05,
        marker_radius=0.01,
    )
    # --------------------- Phantom ---------------------
    phantom = Phantom(robot_root=DEFAULT_PHANTOM_ROOT)
    T_phantom = np.eye(4)
    T_phantom[1, 3] = -PHANTOM_BOTH_Y_DISTANCE / 2
    viz.add_robot(
        "left",
        phantom,
        theta=np.asarray(device_state.joints),
        base_transform=T_phantom,
        color="lightsteelblue",
    )

    p_phantom_home = tool_position_world(
        phantom,
        np.asarray(device_state.joints),
        base_transform=T_phantom,
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
    T_psm = np.eye(4)
    T_psm[:3, :3] = Rz(-np.pi / 2)
    T_psm[:3, 3] = np.array([PSM_BASE_X, PSM_BASE_Y_DISTANCE / 2.0, PSM_BASE_Z])
    q_psm = np.array(
        [
            0.0,
            0.0,
            0.3,
            0.1,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    )
    viz.add_robot("psm", psm, theta=q_psm, base_transform=T_psm, color="orange")

    psm_ik = PinocchioIK(
        robot=psm,
        urdf_path=DEFAULT_PSM_ROOT / "psm.urdf",
        ee_frame_name="tool_gripper2_joint",
    )

    T_links_psm_home = link_transforms(psm, q_psm)
    p_psm_home = (T_psm @ T_links_psm_home[psm.tool_link])[:3, 3]

    # desired target point
    target_psm = create_point_poly(p_psm_home)
    viz.plotter.add_points(
        target_psm,
        color="magenta",
        point_size=5,
        render_points_as_spheres=True,
    )

    # actual FK point
    fk_psm = create_point_poly(p_psm_home)
    viz.plotter.add_points(
        fk_psm,
        color="red",
        point_size=5,
        render_points_as_spheres=True,
    )

    # ---------------------- Visualization Setup ---------------------
    viz.set_camera(
        position=(1.5, -1.5, 1.5),
        focal_point=(0.0, 0.25, 0.0),
        viewup=(0.0, 0.0, 1.0),
    )
    viz.plotter.show(auto_close=False, interactive_update=True)

    p_psm_desired = p_psm_home.copy()
    prev_clutch_button = False
    clutch_active = False
    p_psm_desired_hold = p_psm_home.copy()
    jaw_cmd = float(q_psm[JAW_IDX])

    try:
        while True:

            # ---------------------- Update Phantom State ---------------------
            viz.update_robot(
                "left", np.asarray(device_state.joints), base_transform=T_phantom
            )
            p_phantom = tool_position_world(
                phantom,
                np.asarray(device_state.joints),
                base_transform=T_phantom,
            )
            update_point_poly(fk_phantom, p_phantom)

            # ------------------------ Update PSM State -----------------------
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
                p_psm_desired_hold = p_psm_desired.copy()
                # Change phantom color to red when clutch is active
                set_robot_mesh_color(viz._robots, "left", "red")

            elif clutch_event == ClutchEvent.RELEASED:
                p_phantom_home = p_phantom.copy()
                p_psm_home = p_psm_desired.copy()
                # Change phantom color back to original when clutch is deactivated
                set_robot_mesh_color(viz._robots, "left", "lightsteelblue")

            p_psm_desired = compute_desired_position_world(
                clutch_active=clutch_active,
                desired_hold_world=p_psm_desired_hold,
                phantom_tool_world=p_phantom,
                phantom_home_world=p_phantom_home,
                psm_home_world=p_psm_home,
                teleoperation_gain=TELEOPERATION_GAIN,
            )

            # desired target in world frame
            update_point_poly(target_psm, p_psm_desired)

            # world -> PSM local
            R_psm = T_psm[:3, :3]
            t_psm = T_psm[:3, 3]
            p_psm_desired_local = R_psm.T @ (p_psm_desired - t_psm)

            q_psm, ok = psm_ik.solve_position(
                p_des=p_psm_desired_local,
                q_init=q_psm,
                max_iters=40,
                tol=1e-4,
                damping=1e-3,
                step_size=0.5,
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

            viz.update_robot("psm", q_psm, base_transform=T_psm)

            # actual FK in world frame
            T_links_psm = link_transforms(psm, q_psm)
            p_fk_psm = (T_psm @ T_links_psm[psm.tool_link])[:3, 3]
            update_point_poly(fk_psm, p_fk_psm)

            viz.plotter.update()

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        viz.plotter.close()


if __name__ == "__main__":
    device_state = DeviceState()
    run_with_single_device(device_state=device_state, side="left", callback=main)
