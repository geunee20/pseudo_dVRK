import numpy as np
from typing import Any, Callable, Mapping

from .jacobian import (
    body_jacobian,
    damped_least_square_inverse,
    manipulability,
    pseudoinverse_jacobian,
)
from .poe import body_product_of_exponentials
from .se3 import inv_SE3, log_screw_axis


def finite_difference_grad(w_func, theta, eps=1e-6):
    """Approximate the gradient of a scalar objective using central finite differences.

    .. math::

        \\frac{\\partial w}{\\partial \\theta_i}
        \\approx \\frac{w(\\theta + \\varepsilon e_i) - w(\\theta - \\varepsilon e_i)}{2\\varepsilon}

    where :math:`e_i` is the *i*-th standard basis vector.

    .. note::
        This function was generated with AI assistance.

    Args:
        w_func: Scalar-valued objective :math:`w(\\theta)` that accepts an
            (n,) array and returns a float.
        theta: (n,) array of current joint coordinates.
        eps: Finite-difference step size :math:`\\varepsilon` (default ``1e-6``).

    Returns:
        (n,) gradient vector :math:`\\nabla_\\theta w`.
    """
    grad = np.zeros_like(theta)

    for i in range(len(theta)):
        t1 = theta.copy()
        t2 = theta.copy()

        t1[i] += eps
        t2[i] -= eps

        grad[i] = (w_func(t1) - w_func(t2)) / (2 * eps)

    return grad


def numerical_inverse_kinematics_position(
    M_ee: np.ndarray,
    B_list: list,
    theta_init: np.ndarray,
    p_des: np.ndarray,
    max_iters: int = 100,
    tol_converge: float = 1e-6,
    tol_manipulability: float = 1e-3,
    q_min: np.ndarray | None = None,
    q_max: np.ndarray | None = None,
    objective_func: Callable | None = None,
    objective_args: Mapping[str, Any] | None = None,
    k_null: float = 0.1,
    k_damping: float = 0.01,
    print_iterations: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerical position-only IK via the body Jacobian pseudoinverse.

    **Primary step** (minimise end-effector position error :math:`e = p_{\\text{des}} - p_{\\text{ee}}`):

    .. math::

        \\delta\\theta = J_v^\\dagger\\, e

    where :math:`J_v` is the 3×n linear-velocity block of the body Jacobian.

    **Null-space projection** (optional, secondary objective):

    .. math::

        \\delta\\theta \\mathrel{+}=
            \\underbrace{(I - J_v^\\dagger J_v)}_{\\text{null-space projector}}
            k_{\\text{null}}\\, \\nabla_\\theta w(\\theta)

    Near singularities the DLS inverse
    :math:`J_v^* = J_v^\\top (J_v J_v^\\top + k^2 I)^{-1}` is substituted.

    Args:
        M_ee: 4×4 home configuration of the end-effector.
        B_list: List of *n* 6-vector body-frame screw axes.
        theta_init: (n,) initial joint vector :math:`\\theta_0`.
        p_des: (3,) desired end-effector position.
        max_iters: Maximum number of Newton iterations.
        tol_converge: Convergence threshold on :math:`\\|e\\|`.
        tol_manipulability: If :func:`manipulability` of :math:`J_v` falls
            below this value, the DLS inverse is used instead of the
            pseudoinverse.
        q_min: (n,) lower joint limits (``None`` = unconstrained).
        q_max: (n,) upper joint limits (``None`` = unconstrained).
        objective_func: Optional scalar secondary objective
            :math:`w(\\theta, **\\text{args})`.
        objective_args: Keyword arguments forwarded to *objective_func*.
        k_null: Gain for the null-space secondary objective.
        k_damping: Damping factor :math:`k` for the DLS inverse.
        print_iterations: If ``True``, print per-iteration diagnostics.

    Returns:
        ``(theta, theta_history)`` where *theta* is the (n,) solution and
        *theta_history* is an (iters+1, n) array of intermediate joint vectors.
    """
    theta = np.asarray(theta_init, dtype=float).reshape(-1)
    p_des = np.asarray(p_des, dtype=float).reshape(3)
    objective_kwargs = objective_args if objective_args is not None else {}

    theta_history = [theta.copy()]

    for i in range(max_iters):

        T_ee = body_product_of_exponentials(M_ee, B_list, theta)
        p_ee = T_ee[:3, 3]

        error = p_des - p_ee

        if print_iterations:
            theta_str = ", ".join(
                f"θ{j+1}={np.degrees(t):.2f}°" for j, t in enumerate(theta)
            )
            print(
                f"Iteration {i}: ({theta_str}), "
                f"(x,y,z)=({p_ee[0]:.3f}, {p_ee[1]:.3f}, {p_ee[2]:.3f}), "
                f"||error||={np.linalg.norm(error):.3e}"
            )

        if np.linalg.norm(error) < tol_converge:
            break

        J_b = body_jacobian(B_list, theta)
        J_v = J_b[3:, :]
        if tol_manipulability < manipulability(J_v):
            J_dagger = pseudoinverse_jacobian(J_v)
        else:
            J_dagger = damped_least_square_inverse(J_v, k=k_damping)

        # primary IK objective: minimize position error
        dq = J_dagger @ error

        # secondary objective: maximize criteria
        if objective_func is not None:
            P = np.eye(len(theta)) - J_dagger @ J_v
            w_func = lambda th: objective_func(th, **objective_kwargs)
            dot_q_0 = k_null * finite_difference_grad(w_func, theta)
            dq += P @ dot_q_0

        theta = theta + dq
        if q_min is not None and q_max is not None:
            theta = np.clip(theta, q_min, q_max)

        theta_history.append(theta.copy())

    return theta, np.array(theta_history)


def numerical_inverse_kinematics_pose(
    M_ee: np.ndarray,
    B_list: list,
    theta_init: np.ndarray,
    T_sd: np.ndarray,
    max_iters: int = 100,
    tol_w: float = 1e-6,
    tol_v: float = 1e-6,
    tol_manipulability: float = 1e-3,
    q_min: np.ndarray | None = None,
    q_max: np.ndarray | None = None,
    objective_func: Callable | None = None,
    objective_args: Mapping[str, Any] | None = None,
    k_null: float = 0.1,
    k_damping: float = 0.01,
    print_iterations: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerical full-pose IK via the body Jacobian and matrix logarithm error.

    **Error twist** (from current body frame to desired frame):

    .. math::

        \\mathcal{V}_b = \\left[\\log\\!\\left(T_{bs}\\, T_{sd}\\right)\\right]^\\vee

    where :math:`T_{bs} = T_{sb}^{-1}` and :math:`T_{sb}` is the current
    end-effector pose from body-PoE FK.

    **Primary step**:

    .. math::

        \\delta\\theta = J_b^\\dagger\\,\\mathcal{V}_b

    **Null-space projection** and DLS fallback work identically to
    :func:`numerical_inverse_kinematics_position`.

    Args:
        M_ee: 4×4 home configuration of the end-effector.
        B_list: List of *n* 6-vector body-frame screw axes.
        theta_init: (n,) initial joint vector :math:`\\theta_0`.
        T_sd: 4×4 desired end-effector pose in the space frame.
        max_iters: Maximum number of Newton iterations.
        tol_w: Convergence threshold on :math:`\\|\\omega_b\\|`.
        tol_v: Convergence threshold on :math:`\\|v_b\\|`.
        tol_manipulability: Manipulability threshold for DLS fallback.
        q_min: (n,) lower joint limits (``None`` = unconstrained).
        q_max: (n,) upper joint limits (``None`` = unconstrained).
        objective_func: Optional scalar secondary objective.
        objective_args: Keyword arguments forwarded to *objective_func*.
        k_null: Gain for the null-space secondary objective.
        k_damping: Damping factor :math:`k` for the DLS inverse.
        print_iterations: If ``True``, print per-iteration diagnostics.

    Returns:
        ``(theta, theta_history)`` where *theta* is the (n,) solution and
        *theta_history* is an (iters+1, n) array of intermediate joint vectors.
    """
    theta = theta_init.copy()
    objective_kwargs = objective_args if objective_args is not None else {}
    theta_history = [theta.copy()]

    for i in range(max_iters):

        T_sb = body_product_of_exponentials(M_ee, B_list, theta)
        T_bs = inv_SE3(T_sb)

        T_bd = T_bs @ T_sd

        S, th = log_screw_axis(T_bd)
        V_b = S * th

        w_b = V_b[:3]
        v_b = V_b[3:]

        theta_deg = np.degrees(theta)

        if print_iterations:
            theta_str = ", ".join(
                f"θ{j+1}={angle:.2f}°" for j, angle in enumerate(theta_deg)
            )
            print(
                f"Iteration {i}: ({theta_str}), "
                f"(x, y, z)=({T_sb[0,3]:.3f}, {T_sb[1,3]:.3f}, {T_sb[2,3]:.3f}), "
                f"||w_b||={np.linalg.norm(w_b):.3f}, "
                f"||v_b||={np.linalg.norm(v_b):.3f}"
            )

        if np.linalg.norm(w_b) <= tol_w and np.linalg.norm(v_b) <= tol_v:
            break

        J_b = body_jacobian(B_list, theta)
        J_dagger = pseudoinverse_jacobian(J_b)

        if tol_manipulability < manipulability(J_b):
            J_dagger = pseudoinverse_jacobian(J_b)
        else:
            J_dagger = damped_least_square_inverse(J_b, k=k_damping)

        # primary IK objective: minimize position error
        dq = J_dagger @ V_b

        # secondary objective: maximize criteria
        if objective_func is not None:
            P = np.eye(len(theta)) - J_dagger @ J_b
            w_func = lambda th: objective_func(th, **objective_kwargs)
            dot_q_0 = k_null * finite_difference_grad(w_func, theta)
            dq += P @ dot_q_0

        theta = theta + dq

        if q_min is not None and q_max is not None:
            theta = np.clip(theta, q_min, q_max)

        theta_history.append(theta.copy())

    return theta, np.array(theta_history)


def manipulability_objective(theta: np.ndarray, B_list: list) -> float:
    """Compute Yoshikawa's manipulability as a secondary IK objective.

    .. math::

        w(\\theta) = \\sqrt{\\det(J_b(\\theta)\\, J_b(\\theta)^\\top)}

    Maximising *w* in the null-space of the primary task steers the robot
    away from kinematic singularities.

    Args:
        theta: (n,) current joint coordinates.
        B_list: List of *n* 6-vector body-frame screw axes.

    Returns:
        Scalar manipulability value :math:`w \\ge 0`.
    """
    J = body_jacobian(B_list, theta)
    w = manipulability(J)
    return float(w)


def joint_limit_objective(
    theta: np.ndarray, q_min: np.ndarray, q_max: np.ndarray
) -> float:
    """Compute a joint-limit avoidance objective for use in null-space control.

    Returns a negative value (to be maximised) that penalises configurations
    far from the joint-range midpoints:

    .. math::

        w(\\theta) = -\\frac{1}{2n} \\sum_{i=1}^{n}
            \\left(\\frac{\\theta_i - \\bar{q}_i}{q_{\\max,i} - q_{\\min,i}}\\right)^2,
        \\qquad \\bar{q}_i = \\frac{q_{\\min,i} + q_{\\max,i}}{2}.

    The gradient :math:`\\nabla_\\theta w` is used by
    :func:`numerical_inverse_kinematics_position` /
    :func:`numerical_inverse_kinematics_pose` via
    :func:`finite_difference_grad`.

    Args:
        theta: (n,) current joint coordinates.
        q_min: (n,) lower joint limits.
        q_max: (n,) upper joint limits.

    Returns:
        Scalar objective value :math:`w \\le 0`; closer to 0 means closer
        to the midpoint of each joint range.
    """
    span = q_max - q_min
    q_bar = (q_min + q_max) / 2
    w = -0.5 * np.mean(((theta - q_bar) / span) ** 2)
    return float(w)


def ik_jacobian_transpose_position(
    M_ee: np.ndarray,
    B_list: list,
    theta_init: np.ndarray,
    p_des: np.ndarray,
    max_iters: int = 100,
    tol_converge: float = 1e-6,
    q_min: np.ndarray | None = None,
    q_max: np.ndarray | None = None,
    K: np.ndarray = np.eye(3),
    print_iterations: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerical position-only IK using the Jacobian-transpose method.

    Unlike the pseudoinverse, this method uses the transpose of the linear
    Jacobian weighted by a gain matrix :math:`K`:

    .. math::

        \\delta\\theta = J_v^\\top K\\, e, \\qquad e = p_{\\text{des}} - p_{\\text{ee}}

    The Jacobian transpose method avoids the matrix inversion and is
    unconditionally stable, but typically converges more slowly and does not
    exploit the null space.

    Args:
        M_ee: 4×4 home configuration of the end-effector.
        B_list: List of *n* 6-vector body-frame screw axes.
        theta_init: (n,) initial joint vector :math:`\\theta_0`.
        p_des: (3,) desired end-effector position.
        max_iters: Maximum number of iterations.
        tol_converge: Convergence threshold on :math:`\\|e\\|`.
        q_min: (n,) lower joint limits (``None`` = unconstrained).
        q_max: (n,) upper joint limits (``None`` = unconstrained).
        K: 3×3 positive-definite gain matrix (default: identity).
        print_iterations: If ``True``, print per-iteration diagnostics.

    Returns:
        ``(theta, theta_history)`` where *theta* is the (n,) solution and
        *theta_history* is an (iters+1, n) array of intermediate joint vectors.
    """
    theta = np.asarray(theta_init, dtype=float).reshape(-1)
    p_des = np.asarray(p_des, dtype=float).reshape(3)

    theta_history = [theta.copy()]

    for i in range(max_iters):

        T_ee = body_product_of_exponentials(M_ee, B_list, theta)
        p_ee = T_ee[:3, 3]

        error = p_des - p_ee

        if print_iterations:
            theta_str = ", ".join(
                f"θ{j+1}={np.degrees(t):.2f}°" for j, t in enumerate(theta)
            )
            print(
                f"Iteration {i}: ({theta_str}), "
                f"(x,y,z)=({p_ee[0]:.3f}, {p_ee[1]:.3f}, {p_ee[2]:.3f}), "
                f"||error||={np.linalg.norm(error):.3e}"
            )

        if np.linalg.norm(error) < tol_converge:
            break

        J_b = body_jacobian(B_list, theta)
        J_v = J_b[3:, :]
        dq = J_v.T @ K @ error

        theta = theta + dq
        if q_min is not None and q_max is not None:
            theta = np.clip(theta, q_min, q_max)

        theta_history.append(theta.copy())

    return theta, np.array(theta_history)
