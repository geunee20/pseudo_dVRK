from __future__ import annotations

from typing import Optional

import numpy as np
import pyvista as pv

from src.utils.real_time_viz import DvrkRealtimeViz


def visualize(
    robot,
    theta: Optional[np.ndarray] = None,
    show_frames: bool = False,
    alpha: float = 1.0,
) -> DvrkRealtimeViz:
    """
    Generic dVRK visualization for any robot implementing the Robot interface.

    Inputs
    ------
    robot : Robot-like object
    theta : (dof,) joint vector. If None, robot.get_theta() is used.
    show_frames : if True, adds small axis markers at each link frame

    Returns
    -------
    viz : DvrkRealtimeViz
    """
    if theta is None:
        theta = robot.get_theta()
    viz = DvrkRealtimeViz(show_frames=show_frames, alpha=float(alpha))
    viz.add_robot("robot", robot, theta=np.asarray(theta, dtype=float).reshape(-1))
    return viz


def set_camera_view(scene, eye, target):
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)

    if hasattr(scene, "set_camera"):
        scene.set_camera(
            position=tuple(eye.tolist()),
            focal_point=tuple(target.tolist()),
            viewup=(0.0, 0.0, 1.0),
        )
        return

    if isinstance(scene, pv.Plotter):
        scene.camera_position = [
            tuple(eye.tolist()),
            tuple(target.tolist()),
            (0.0, 0.0, 1.0),
        ]
        scene.render()
        return

    if hasattr(scene, "camera_position"):
        scene.camera_position = [
            tuple(eye.tolist()),
            tuple(target.tolist()),
            (0.0, 0.0, 1.0),
        ]
        if hasattr(scene, "render"):
            scene.render()


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
