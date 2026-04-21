from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .fk import forward_kinematics, joint_frames
from .se3 import adjoint_inverse, screw_axis_from_w_q
from .poe import body_product_of_exponentials


@dataclass
class KinematicModel:
    """Pre-computed PoE quantities for Jacobian and IK solvers.

    Attributes:
        M_ee: 4×4 home configuration of the end-effector at :math:`\\theta = 0`.
        S_list: List of *n* 6-vector screw axes in the **space** frame.
        B_list: List of *n* 6-vector screw axes in the **body** (end-effector) frame.
        q_min: (n,) lower joint limits (may contain ``-inf``).
        q_max: (n,) upper joint limits (may contain ``+inf``).
    """

    M_ee: np.ndarray
    S_list: List[np.ndarray]
    B_list: List[np.ndarray]
    q_min: np.ndarray
    q_max: np.ndarray


def compute_joint_limit_arrays(robot) -> tuple[np.ndarray, np.ndarray]:
    """Build (n,) joint-limit arrays ordered by ``robot.active_joint_names``.

    For joints without a ``<limit>`` element in the URDF the bounds are set
    to :math:`(-\\infty, +\\infty)`.

    Args:
        robot: Any robot object that exposes ``active_joint_names`` and
            ``get_joint(name)``.

    Returns:
        ``(q_min, q_max)`` — each a float64 array of length *dof*.
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
    """Return the home end-effector pose :math:`M_{\\text{ee}}` at :math:`\\theta = 0`.

    The pose is obtained by running tree FK with all joints set to zero
    and is expressed in the world/space frame used by the robot's FK.

    Args:
        robot: Robot object with attributes ``dof`` and ``tool_link`` and
            implementing :func:`~src.kinematics.fk.forward_kinematics`.

    Returns:
        4×4 homogeneous transform :math:`M_{\\text{ee}} \\in SE(3)`.
    """
    theta0 = np.zeros(robot.dof, dtype=float)
    M_ee = forward_kinematics(robot, theta=theta0, link_name=robot.tool_link)
    return np.asarray(M_ee, dtype=float).reshape(4, 4)


def compute_screw_axes(
    robot, M_ee: np.ndarray | None = None
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """Compute space-frame and body-frame screw axes for all active joints.

    For each active joint the URDF joint axis (expressed in the **local** joint
    frame) is rotated into the **space** frame using the home joint-frame
    rotation :math:`R_{\\text{joint}}`:

    .. math::

        \\hat{a}_{\\text{space}} = R_{\\text{joint}}\\, \\hat{a}_{\\text{local}}

    The space screw axis is then constructed as:

    * **Revolute**: :math:`\\mathcal{S} = [\\hat{a}_{\\text{space}}; -\\hat{a}_{\\text{space}} \\times q]`
    * **Prismatic**: :math:`\\mathcal{S} = [0; \\hat{a}_{\\text{space}}]`

    The body screw axis is converted via the inverse adjoint of :math:`M_{\\text{ee}}`:

    .. math::

        \\mathcal{B} = [\\text{Ad}_{M_{\\text{ee}}^{-1}}]\\, \\mathcal{S}

    Args:
        robot: Robot object exposing ``active_joint_names``, ``get_joint``, ``dof``,
            and ``tool_link``.
        M_ee: Pre-computed home end-effector pose.  If ``None``, computed via
            :func:`compute_home_ee_pose`.

    Raises:
        KeyError: If a joint frame is missing or a mimic-source joint is not found.
        ValueError: If an active joint has a near-zero axis or an unsupported type.

    Returns:
        ``(S_list, B_list)`` — each a list of *dof* 6-vector screw axes.
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
    """Build all PoE quantities needed for Jacobian and IK computation.

    Convenience wrapper that calls :func:`compute_home_ee_pose`,
    :func:`compute_screw_axes`, and :func:`compute_joint_limit_arrays` in
    sequence.

    Args:
        robot: Any robot object satisfying the interface expected by the
            three sub-functions.

    Returns:
        :class:`KinematicModel` populated with ``M_ee``, ``S_list``,
        ``B_list``, ``q_min``, and ``q_max``.
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
    """Assert self-consistency of a :class:`KinematicModel` against a robot.

    Checks performed:

    1. ``len(S_list) == len(B_list) == robot.dof``
    2. ``q_min.shape == q_max.shape == (robot.dof,)``
    3. Tree FK home pose agrees with body-PoE home pose:

       .. math::

           T_{\\text{FK}}(0) \\approx M_{\\text{ee}}\\, e^{[B_1] \\cdot 0} \\cdots
               e^{[B_n] \\cdot 0} = M_{\\text{ee}}

    Args:
        robot: Robot object used to build *model*.
        model: The :class:`KinematicModel` to validate.
        atol: Absolute tolerance passed to :func:`numpy.allclose`.

    Raises:
        ValueError: If any of the above checks fail.
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
