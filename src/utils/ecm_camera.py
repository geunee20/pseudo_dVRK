from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyvista as pv
from settings import (
    ECM_CAMERA_DEFAULT_COUNT,
    ECM_CAMERA_DEFAULT_SCOPE_DEG,
    ECM_CAMERA_FAR,
    ECM_CAMERA_FOCUS_DISTANCE_M,
    ECM_CAMERA_HFOV_DEG,
    ECM_CAMERA_INSERTION_STEP_M,
    ECM_CAMERA_NEAR,
    ECM_CAMERA_PITCH_STEP_DEG,
    ECM_CAMERA_YAW_STEP_DEG,
    ECM_STEREO_BASELINE_M,
)
from src.utils.real_time_viz import apply_camera_pose

if TYPE_CHECKING:
    from src.robots.ecm import ECM


HFOV_DEG = ECM_CAMERA_HFOV_DEG
NEAR = ECM_CAMERA_NEAR
FAR = ECM_CAMERA_FAR
BASELINE_M = ECM_STEREO_BASELINE_M
FOCUS_DISTANCE_M = ECM_CAMERA_FOCUS_DISTANCE_M


@dataclass
class EcmControl:
    q: np.ndarray
    yaw_idx: int | None
    pitch_idx: int | None
    insertion_idx: int | None
    yaw_step_rad: float = np.deg2rad(ECM_CAMERA_YAW_STEP_DEG)
    pitch_step_rad: float = np.deg2rad(ECM_CAMERA_PITCH_STEP_DEG)
    insertion_step_m: float = ECM_CAMERA_INSERTION_STEP_M


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ECM camera simulation with keyboard control. "
            "Arrow keys adjust yaw/pitch, PageUp/PageDown adjust insertion."
        )
    )
    parser.add_argument(
        "--cameras",
        type=int,
        default=ECM_CAMERA_DEFAULT_COUNT,
        choices=[1, 2],
        help="Number of cameras: 1 (mono) or 2 (stereo)",
    )
    parser.add_argument(
        "--scope-deg",
        type=float,
        default=ECM_CAMERA_DEFAULT_SCOPE_DEG,
        choices=[0.0, 30.0],
        help="Optical tilt angle in degrees (0 or 30)",
    )
    return parser


def find_joint_indices(ecm: ECM) -> tuple[int | None, int | None, int | None]:
    index = {name: i for i, name in enumerate(ecm.active_joint_names)}
    return (
        index.get("yaw_joint"),
        index.get("pitch_front_joint"),
        index.get("main_insertion_joint"),
    )


def create_camera_plotter(n_cameras: int) -> tuple[pv.Plotter, bool]:
    """Create camera-view plotter configured for mono or stereo mode."""
    is_stereo = int(n_cameras) == 2
    plotter = pv.Plotter(
        shape=(1, 2) if is_stereo else (1, 1),
        title=(
            "ECM Stereo Camera View (Left | Right)"
            if is_stereo
            else "ECM Monocular Camera View"
        ),
        window_size=[1400, 700] if is_stereo else [700, 700],
    )

    plotter.subplot(0, 0)
    plotter.add_text("Left" if is_stereo else "Mono", font_size=14)
    if is_stereo:
        plotter.subplot(0, 1)
        plotter.add_text("Right", font_size=14)

    return plotter, is_stereo


def build_ecm_control_from_camera_pose(
    camera_world_tf: np.ndarray,
    robot_root: str | Path,
    initial_q: np.ndarray | None = None,
) -> tuple["ECM", EcmControl]:
    """Build ECM + control state so the tool pose matches the desired camera pose."""
    # Local import avoids a runtime import cycle and keeps this utility lightweight.
    from src.robots.ecm import ECM

    ecm_tmp = ECM(robot_root=robot_root)
    q = np.zeros(ecm_tmp.dof, dtype=float)
    if initial_q is not None:
        q[:] = np.asarray(initial_q, dtype=float).reshape(-1)

    T_zero = ecm_tmp.tool_transform_world(theta=q)
    T_ecm_base = np.asarray(camera_world_tf, dtype=float).reshape(4, 4) @ np.linalg.inv(
        T_zero
    )

    ecm = ECM(robot_root=robot_root, base_transform=T_ecm_base)
    yaw_idx, pitch_idx, insertion_idx = find_joint_indices(ecm)
    ctrl = EcmControl(
        q=q,
        yaw_idx=yaw_idx,
        pitch_idx=pitch_idx,
        insertion_idx=insertion_idx,
    )
    return ecm, ctrl


def update_cameras(
    plotter: pv.Plotter,
    ecm: ECM,
    ctrl: EcmControl,
    optical_tilt_deg: float,
    n_cameras: int,
    vfov_deg: float,
    camera_roll_deg: float = 0.0,
) -> None:
    def _apply_roll(pose: Any) -> Any:
        if abs(float(camera_roll_deg)) < 1e-12:
            return pose
        forward = np.asarray(pose.focal_point, dtype=float) - np.asarray(
            pose.position, dtype=float
        )
        n = float(np.linalg.norm(forward))
        if n < 1e-12:
            return pose
        axis = forward / n
        up = np.asarray(pose.viewup, dtype=float)
        ang = np.deg2rad(float(camera_roll_deg))
        c = float(np.cos(ang))
        s = float(np.sin(ang))
        up_rot = c * up + s * np.cross(axis, up) + (1.0 - c) * np.dot(axis, up) * axis
        pose.viewup = up_rot
        return pose

    if n_cameras == 1:
        pose = ecm.scope_camera_pose(
            theta=ctrl.q,
            optical_tilt_deg=optical_tilt_deg,
            focus_distance=FOCUS_DISTANCE_M,
        )
        pose = _apply_roll(pose)
        apply_camera_pose(plotter, pose, vfov_deg=vfov_deg, near=NEAR, far=FAR)
        return

    left_pose, right_pose = ecm.scope_stereo_camera_poses(
        theta=ctrl.q,
        optical_tilt_deg=optical_tilt_deg,
        baseline=BASELINE_M,
        focus_distance=FOCUS_DISTANCE_M,
    )
    left_pose = _apply_roll(left_pose)
    right_pose = _apply_roll(right_pose)

    plotter.subplot(0, 0)
    apply_camera_pose(plotter, left_pose, vfov_deg=vfov_deg, near=NEAR, far=FAR)

    plotter.subplot(0, 1)
    apply_camera_pose(plotter, right_pose, vfov_deg=vfov_deg, near=NEAR, far=FAR)


def register_keys(
    plotter: pv.Plotter,
    ecm: ECM,
    ctrl: EcmControl,
    optical_tilt_deg: float,
    n_cameras: int,
    vfov_deg: float,
    camera_roll_deg: float = 0.0,
) -> None:
    def _apply_joint_delta(idx: int | None, delta: float) -> None:
        if idx is None:
            return
        ctrl.q[idx] += float(delta)
        ctrl.q[:] = ecm.clamp_theta(ctrl.q)
        update_cameras(
            plotter,
            ecm,
            ctrl,
            optical_tilt_deg=optical_tilt_deg,
            n_cameras=n_cameras,
            vfov_deg=vfov_deg,
            camera_roll_deg=camera_roll_deg,
        )
        plotter.render()

    add_key_event = getattr(plotter, "add_key_event")
    add_key_event_any: Any = add_key_event

    add_key_event_any(
        "Left", lambda: _apply_joint_delta(ctrl.pitch_idx, -ctrl.pitch_step_rad)
    )  # pyright: ignore[reportCallIssue]
    add_key_event_any(
        "Right", lambda: _apply_joint_delta(ctrl.pitch_idx, ctrl.pitch_step_rad)
    )  # pyright: ignore[reportCallIssue]
    add_key_event_any(
        "Up", lambda: _apply_joint_delta(ctrl.yaw_idx, ctrl.yaw_step_rad)
    )  # pyright: ignore[reportCallIssue]
    add_key_event_any(
        "Down", lambda: _apply_joint_delta(ctrl.yaw_idx, -ctrl.yaw_step_rad)
    )  # pyright: ignore[reportCallIssue]

    # VTK keysyms: Prior=PageUp, Next=PageDown
    add_key_event_any(
        "Prior", lambda: _apply_joint_delta(ctrl.insertion_idx, ctrl.insertion_step_m)
    )  # pyright: ignore[reportCallIssue]
    add_key_event_any(
        "Next", lambda: _apply_joint_delta(ctrl.insertion_idx, -ctrl.insertion_step_m)
    )  # pyright: ignore[reportCallIssue]
