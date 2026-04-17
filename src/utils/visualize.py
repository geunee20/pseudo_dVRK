from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from trimesh.visual import ColorVisuals

from src.kinematics.fk import link_transforms, origin_transform


def _load_mesh(mesh_path: Path) -> Optional[trimesh.Trimesh]:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            return None
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        return None
    return mesh


def _set_mesh_alpha(mesh: trimesh.Trimesh, alpha: float) -> None:
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return

    face_colors = getattr(visual, "face_colors", None)
    if face_colors is None:
        return

    face_colors = np.asarray(face_colors)
    if face_colors.ndim != 2 or face_colors.shape[1] < 4:
        return

    face_colors[:, 3] = int(np.clip(alpha, 0.0, 1.0) * 255)


def _set_mesh_color(mesh: trimesh.Trimesh, rgba: list[int]) -> None:
    mesh.visual = ColorVisuals(mesh=mesh, face_colors=rgba)


def visualize(
    robot,
    theta: Optional[np.ndarray] = None,
    show_frames: bool = False,
    alpha: float = 1.0,
) -> trimesh.Scene:
    """
    Generic dVRK visualization for any robot implementing the Robot interface.

    Inputs
    ------
    robot : Robot-like object
    theta : (dof,) joint vector. If None, robot.get_theta() is used.
    show_frames : if True, adds small axis markers at each link frame

    Returns
    -------
    scene : trimesh.Scene
    """
    if theta is None:
        theta = robot.get_theta()

    T_links = link_transforms(robot, theta)
    scene = trimesh.Scene()

    for link_name in robot.link_names:
        if link_name not in T_links:
            continue

        T_world_link = T_links[link_name]

        for visual in robot.get_link_visuals(link_name):
            mesh_path = Path(visual.mesh_path)
            if not mesh_path.exists():
                print(f"[warn] missing mesh: {mesh_path}")
                continue

            mesh = _load_mesh(mesh_path)
            if mesh is None:
                continue

            T_link_visual = origin_transform(visual.origin_xyz, visual.origin_rpy)
            T_world_visual = T_world_link @ T_link_visual

            mesh = mesh.copy()
            _set_mesh_alpha(mesh, alpha)
            mesh.apply_transform(T_world_visual)
            scene.add_geometry(mesh, node_name=f"{link_name}:{mesh_path.name}")

        if show_frames:
            q_full = robot.expand_theta(theta)

            for joint_name in robot.active_joint_names:
                joint = robot.get_joint(joint_name)

                parent = joint.parent
                if parent not in T_links:
                    continue

                T_parent = T_links[parent]
                T_parent_joint = origin_transform(joint.origin_xyz, joint.origin_rpy)

                T_world_joint = T_parent @ T_parent_joint

                # ----------- frame (axis) -----------
                axis = trimesh.creation.axis(origin_size=0.001, axis_length=0.05)
                axis.apply_transform(T_world_joint)

                scene.add_geometry(axis, node_name=f"{joint_name}:frame")

                # ----------- origin highlight -----------
                q_val = q_full.get(joint_name, 0.0)

                if abs(q_val) > 1e-6:
                    # red sphere for active joint (non-zero)
                    sphere = trimesh.creation.icosphere(radius=0.003)
                    _set_mesh_color(sphere, [255, 0, 0, 255])
                else:
                    # gray sphere for zero joint
                    sphere = trimesh.creation.icosphere(radius=0.003)
                    _set_mesh_color(sphere, [255, 255, 255, 255])

                sphere.apply_transform(T_world_joint)
                scene.add_geometry(sphere, node_name=f"{joint_name}:origin")

    return scene


def set_camera_view(scene, eye, target):
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)

    forward = target - eye
    forward /= np.linalg.norm(forward)

    up = np.array([0, 0, 1], dtype=float)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)

    R = np.column_stack((right, up, -forward))

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = eye

    scene.camera_transform = T


def demo_two_robots(ecm, psm, theta_ecm=None, theta_psm=None) -> None:
    """Quick visualization demo for displaying two robots together."""
    from src.utils.real_time_viz import DvrkRealtimeViz

    viz = DvrkRealtimeViz(show_frames=True, alpha=0.6)

    if theta_ecm is None:
        theta_ecm = np.zeros(ecm.dof)
    if theta_psm is None:
        theta_psm = np.zeros(psm.dof)

    T_ecm = np.eye(4)
    T_psm = np.eye(4)
    T_psm[:3, 3] = np.array([0.25, 0.0, 0.0])

    viz.add_robot(
        "ecm", ecm, theta=theta_ecm, base_transform=T_ecm, color="lightsteelblue"
    )
    viz.add_robot("psm", psm, theta=theta_psm, base_transform=T_psm, color="orange")
    viz.set_camera(
        position=(1.0, 1.0, 1.0), focal_point=(0.0, 0.0, 0.0), viewup=(0.0, 0.0, 1.0)
    )
    viz.show()
