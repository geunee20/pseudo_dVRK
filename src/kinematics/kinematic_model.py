from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .fk import forward_kinematics, joint_frames
from .se3 import adjoint_inverse, screw_axis_from_w_q
from .poe import body_product_of_exponentials


@dataclass
class KinematicModel:
    M_ee: np.ndarray
    S_list: List[np.ndarray]
    B_list: List[np.ndarray]
    q_min: np.ndarray
    q_max: np.ndarray


def compute_joint_limit_arrays(robot) -> tuple[np.ndarray, np.ndarray]:
    """
    Build active-joint limit arrays ordered according to robot.active_joint_names.
    """
    q_min = []
    q_max = []

    for joint_name in robot.active_joint_names:
        joint = robot.get_joint(joint_name)

        if joint.limit is None:
            q_min.append(-np.inf)
            q_max.append(np.inf)
        else:
            q_min.append(float(joint.limit.lower))
            q_max.append(float(joint.limit.upper))

    return np.array(q_min, dtype=float), np.array(q_max, dtype=float)


def compute_home_ee_pose(robot) -> np.ndarray:
    """
    Home end-effector pose at theta = 0 in the robot/world frame used by FK.
    """
    theta0 = np.zeros(robot.dof, dtype=float)
    M_ee = forward_kinematics(robot, theta=theta0, link_name=robot.tool_link)
    return np.asarray(M_ee, dtype=float).reshape(4, 4)


def compute_screw_axes(
    robot, M_ee: np.ndarray | None = None
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Compute space-frame and body-frame screw axes for the active joints.

    Notes
    -----
    - URDF joint.axis is expressed in the joint local frame.
    - We rotate it into the space/world frame using the home joint frame rotation.
    - For revolute joints:
        S = [w; -w x q]
      where q is a point on the axis in the space frame.
    - For prismatic joints:
        S = [0; v]
      where v is the axis direction in the space frame.
    """
    if M_ee is None:
        M_ee = compute_home_ee_pose(robot)

    theta0 = np.zeros(robot.dof, dtype=float)
    frames = joint_frames(robot, theta=theta0)
    frame_map = {frame.joint_name: frame.T_world for frame in frames}

    S_list: List[np.ndarray] = []
    B_list: List[np.ndarray] = []

    for joint_name in robot.active_joint_names:
        joint = robot.get_joint(joint_name)

        if joint_name not in frame_map:
            raise KeyError(f"Joint frame for '{joint_name}' not found.")

        T_joint = frame_map[joint_name]
        R_joint = T_joint[:3, :3]
        p_joint = T_joint[:3, 3]

        axis_local = np.asarray(joint.axis, dtype=float).reshape(3)
        axis_space = R_joint @ axis_local

        axis_norm = np.linalg.norm(axis_space)
        if axis_norm < 1e-12:
            raise ValueError(f"Joint '{joint_name}' has near-zero axis.")
        axis_space = axis_space / axis_norm

        if joint.joint_type == "revolute":
            S = screw_axis_from_w_q(axis_space, p_joint)
        elif joint.joint_type == "prismatic":
            S = np.concatenate([np.zeros(3), axis_space])
        else:
            raise ValueError(
                f"Active joint '{joint_name}' has unsupported type '{joint.joint_type}'."
            )

        B = adjoint_inverse(M_ee) @ S

        S_list.append(S.astype(float))
        B_list.append(B.astype(float))

    return S_list, B_list


def build_kinematic_model(robot) -> KinematicModel:
    """
    Build all PoE-ready quantities needed for Jacobian/IK.
    """
    M_ee = compute_home_ee_pose(robot)
    S_list, B_list = compute_screw_axes(robot, M_ee=M_ee)
    q_min, q_max = compute_joint_limit_arrays(robot)

    return KinematicModel(
        M_ee=M_ee,
        S_list=S_list,
        B_list=B_list,
        q_min=q_min,
        q_max=q_max,
    )


def validate_kinematic_model(
    robot,
    model: KinematicModel,
    atol: float = 1e-8,
) -> None:
    """
    Basic consistency check:
    - number of screw axes matches DOF
    - PoE home pose equals tree FK home pose
    - limit array shapes are correct
    """
    if len(model.S_list) != robot.dof:
        raise ValueError(f"S_list length {len(model.S_list)} != robot.dof {robot.dof}")

    if len(model.B_list) != robot.dof:
        raise ValueError(f"B_list length {len(model.B_list)} != robot.dof {robot.dof}")

    if model.q_min.shape != (robot.dof,):
        raise ValueError(f"q_min shape {model.q_min.shape} != ({robot.dof},)")

    if model.q_max.shape != (robot.dof,):
        raise ValueError(f"q_max shape {model.q_max.shape} != ({robot.dof},)")

    T_fk_home = forward_kinematics(
        robot, theta=np.zeros(robot.dof), link_name=robot.tool_link
    )
    T_poe_home = body_product_of_exponentials(
        model.M_ee,
        model.B_list,
        np.zeros(robot.dof, dtype=float),
    )

    if not np.allclose(T_fk_home, T_poe_home, atol=atol):
        raise ValueError(
            "Tree FK home pose and PoE home pose do not match.\n"
            f"T_fk_home=\n{T_fk_home}\n\nT_poe_home=\n{T_poe_home}"
        )
