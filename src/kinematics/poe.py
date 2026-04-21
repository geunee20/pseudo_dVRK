import numpy as np
from src.kinematics.se3 import exp_screw_axis


def space_product_of_transforms(M: np.ndarray, T_list: list) -> np.ndarray:
    """Chain a list of joint transforms in the space frame and apply the home config.

    .. math::

        T_{\\text{ee}} = T_1\\, T_2 \\cdots T_n\\, M

    Args:
        M: 4×4 home configuration of the end-effector in the space frame.
        T_list: List of 4×4 joint-generated transforms, ordered from joint 1 to joint *n*.

    Returns:
        4×4 end-effector pose in the space frame.
    """
    T = np.eye(4)

    for T_i in T_list:
        T = T @ T_i

    T_ee = T @ M
    return T_ee


def body_product_of_transforms(M: np.ndarray, T_list: list) -> np.ndarray:
    """Chain a list of joint transforms in the body frame, starting from the home config.

    .. math::

        T_{\\text{ee}} = M\\, T_1\\, T_2 \\cdots T_n

    Args:
        M: 4×4 home configuration of the end-effector.
        T_list: List of 4×4 joint-generated transforms, ordered from joint 1 to joint *n*.

    Returns:
        4×4 end-effector pose in the space frame.
    """
    T = np.asarray(M, dtype=float).reshape(4, 4)

    for T_i in T_list:
        T = T @ T_i

    T_ee = T
    return T_ee


def space_product_of_exponentials(
    M: np.ndarray, S_list: list, theta: np.ndarray
) -> np.ndarray:
    """Forward kinematics via the **space-frame** Product of Exponentials formula.

    .. math::

        T(\\theta) = e^{[\\mathcal{S}_1]\\theta_1} \\cdots
                     e^{[\\mathcal{S}_n]\\theta_n}\, M

    Args:
        M: 4×4 home configuration of the end-effector (at :math:`\\theta = 0`).
        S_list: List of 6-vector screw axes expressed in the **space** frame.
        theta: Array of length *n* containing joint displacements.

    Returns:
        4×4 end-effector pose in the space frame.
    """
    T = np.eye(4)

    for S, theta_i in zip(S_list, theta):
        T = T @ exp_screw_axis(S, theta_i)

    T_ee = T @ M
    return T_ee


def body_product_of_exponentials(
    M: np.ndarray, B_list: list, theta: np.ndarray
) -> np.ndarray:
    """Forward kinematics via the **body-frame** Product of Exponentials formula.

    .. math::

        T(\\theta) = M\, e^{[\\mathcal{B}_1]\\theta_1} \\cdots
                         e^{[\\mathcal{B}_n]\\theta_n}

    Args:
        M: 4×4 home configuration of the end-effector (at :math:`\\theta = 0`).
        B_list: List of 6-vector screw axes expressed in the **body** (end-effector)
            frame.
        theta: Array of length *n* containing joint displacements.

    Returns:
        4×4 end-effector pose in the space frame.
    """
    T = np.asarray(M, dtype=float).reshape(4, 4)

    for B, theta_i in zip(B_list, theta):
        T = T @ exp_screw_axis(B, theta_i)

    T_ee = T
    return T_ee


### followings are just for visualizing the system.
### Used ChatGPT to create this functions
def space_link_poses(M_list: list, S_list: list, theta: np.ndarray) -> list:
    """Compute world-frame poses of all link frames for visualization.

    For each link *i* (0-indexed), the pose is accumulated as:

    .. math::

        T_i = e^{[\\mathcal{S}_1]\\theta_1} \\cdots e^{[\\mathcal{S}_i]\\theta_i}\, M_i

    The zeroth frame (base) is returned as :math:`M_0` (no joint transform applied).

    Args:
        M_list: List of 4×4 home configurations, one per link frame including the
            base (length *n + 1*).
        S_list: List of *n* 6-vector screw axes in the space frame.
        theta: Array of length *n* containing joint displacements.

    Returns:
        List of *n + 1* 4×4 homogeneous poses, one per link frame in the space
        frame.
    """
    T_list = []
    T_prefix = np.eye(4)

    T_list.append(T_prefix @ np.asarray(M_list[0], dtype=float).reshape(4, 4))

    for i in range(len(S_list)):
        T_prefix = T_prefix @ exp_screw_axis(S_list[i], theta[i])
        T_i = T_prefix @ np.asarray(M_list[i + 1], dtype=float).reshape(4, 4)
        T_list.append(T_i)

    return T_list
