import numpy as np
from .se3 import adjoint, exp_screw_axis


def space_jacobian(S_list: list, theta: np.ndarray) -> np.ndarray:
    """Compute the 6×n space Jacobian using the PoE forward recursion.

    Column *i* (0-indexed) is the *i*-th screw axis transformed into the space
    frame by the product of all preceding joint exponentials:

    .. math::

        J_s^{(i)} = [\\text{Ad}_{e^{[\\mathcal{S}_1]\\theta_1}
                       \\cdots e^{[\\mathcal{S}_i]\\theta_i}}]\\,\\mathcal{S}_{i+1},
        \\quad J_s^{(0)} = \\mathcal{S}_1.

    Args:
        S_list: List of *n* 6-vector screw axes in the space frame.
        theta: Array of length *n* containing joint displacements.

    Returns:
        6×n space Jacobian :math:`J_s`.
    """

    n = len(S_list)
    J = np.zeros((6, n))
    J[:, 0] = S_list[0]

    T = np.eye(4)
    for i in range(1, n):
        T = T @ exp_screw_axis(S_list[i - 1], theta[i - 1])
        J[:, i] = adjoint(T) @ S_list[i]

    return J


def body_jacobian(B_list: list, theta: np.ndarray) -> np.ndarray:
    """Compute the 6×n body Jacobian using the PoE backward recursion.

    Column *i* (0-indexed) is the *i*-th body screw axis transformed by the
    inverse adjoint of all **following** joint exponentials:

    .. math::

        J_b^{(i)} = [\\text{Ad}_{e^{-[\\mathcal{B}_{i+1}]\\theta_{i+1}}
                       \\cdots e^{-[\\mathcal{B}_n]\\theta_n}}]\\,\\mathcal{B}_{i},
        \\quad J_b^{(n-1)} = \\mathcal{B}_n.

    Args:
        B_list: List of *n* 6-vector screw axes in the body (end-effector) frame.
        theta: Array of length *n* containing joint displacements.

    Returns:
        6×n body Jacobian :math:`J_b`.
    """
    n = len(B_list)
    J_b = np.zeros((6, n), dtype=float)
    J_b[:, n - 1] = B_list[n - 1]

    T = np.eye(4)
    for i in range(n - 2, -1, -1):
        T = T @ exp_screw_axis(B_list[i + 1], -theta[i + 1])
        J_b[:, i] = adjoint(T) @ B_list[i]

    return J_b


def check_singularity(J: np.ndarray) -> bool:
    """Return ``True`` if the Jacobian is rank-deficient (singular).

    A Jacobian is singular when it does not have full row rank (for
    redundant/square manipulators) or full column rank (for non-redundant):

    .. math::

        \\text{singular} \\iff \\operatorname{rank}(J) < \\min(m, n)

    Args:
        J: *m*×*n* Jacobian matrix.

    Returns:
        ``True`` if *J* is rank-deficient, ``False`` otherwise.
    """
    J = np.asarray(J, dtype=float)
    is_singular = np.linalg.matrix_rank(J) < min(J.shape)
    return is_singular


def manipulability(J: np.ndarray) -> float:
    """Compute Yoshikawa's manipulability measure.

    .. math::

        \\mu = \\sqrt{\\det(J J^\\top)}

    The measure is zero at a singularity and larger values indicate greater
    dexterity.  For a square, full-rank Jacobian this equals
    :math:`|\\det(J)|`.

    Args:
        J: *m*×*n* Jacobian matrix (can be the full 6×n body/space Jacobian or
           a submatrix such as the 3×n linear-velocity block).

    Returns:
        Non-negative scalar manipulability :math:`\\mu`.
    """
    J = np.asarray(J, dtype=float)
    A = J @ J.T
    mu = np.sqrt(np.linalg.det(A))
    return mu


def manipulability_ellipsoid(J: np.ndarray) -> list:
    """Compute the manipulability ellipsoid data for the angular and linear subspaces.

    For each of the angular (:math:`J_\\omega`, rows 0–2) and linear
    (:math:`J_v`, rows 3–5) blocks the following are computed:

    .. math::

        A_* = J_* J_*^\\top

    The eigendecomposition :math:`A_* = V \\Lambda V^\\top` yields the principal
    axes and semi-axis lengths.  Three scalar measures are also returned:

    .. math::

        \\mu_1 &= \\sqrt{\\lambda_{\\max} / \\lambda_{\\min}}  &&\\text{(isotropy)} \\\\
        \\mu_2 &= \\lambda_{\\max} / \\lambda_{\\min}          &&\\text{(condition number)} \\\\
        \\mu_3 &= \\sqrt{\\det(A_*)}                          &&\\text{(ellipsoid volume)}

    Args:
        J: 6×n Jacobian matrix (full body or space Jacobian).

    Returns:
        ``[[A_w, eigvals_w, eigvecs_w, mu1_w, mu2_w, mu3_w],
           [A_v, eigvals_v, eigvecs_v, mu1_v, mu2_v, mu3_v]]``
        where ``*_w`` is the angular subspace and ``*_v`` is the linear subspace.
        :math:`\\mu_1` and :math:`\\mu_2` are ``inf`` when :math:`\\lambda_{\\min} \\approx 0`.
    """
    J = np.asarray(J, dtype=float)
    J_w = J[:3, :]
    J_v = J[3:, :]

    A_w = J_w @ J_w.T
    eigvals_w, eigvecs_w = np.linalg.eigh(A_w)
    if np.min(eigvals_w) < 1e-12:
        mu1_w = np.inf
        mu2_w = np.inf
    else:
        mu1_w = np.sqrt(np.max(eigvals_w) / np.min(eigvals_w))
        mu2_w = np.max(eigvals_w) / np.min(eigvals_w)
    mu3_w = np.sqrt(np.linalg.det(A_w))

    A_v = J_v @ J_v.T
    eigvals_v, eigvecs_v = np.linalg.eigh(A_v)
    if np.min(eigvals_v) < 1e-12:
        mu1_v = np.inf
        mu2_v = np.inf
    else:
        mu1_v = np.sqrt(np.max(eigvals_v) / np.min(eigvals_v))
        mu2_v = np.max(eigvals_v) / np.min(eigvals_v)
    mu3_v = np.sqrt(np.linalg.det(A_v))

    return [
        [A_w, eigvals_w, eigvecs_w, mu1_w, mu2_w, mu3_w],
        [A_v, eigvals_v, eigvecs_v, mu1_v, mu2_v, mu3_v],
    ]


def pseudoinverse_jacobian(J: np.ndarray) -> np.ndarray:
    """Compute the Moore-Penrose pseudoinverse of a Jacobian matrix.

    Two cases:

    * **Fat Jacobian** (:math:`n > m`, full row rank):
      right pseudoinverse minimising :math:`\\|\\delta\\theta\\|`:

      .. math::

          J^\\dagger = J^\\top (J J^\\top)^{-1}

    * **Tall Jacobian** (:math:`n \\le m`, full column rank):
      left pseudoinverse minimising the least-squares residual:

      .. math::

          J^\\dagger = (J^\\top J)^{-1} J^\\top

    Args:
        J: *m*×*n* Jacobian matrix (must have full row or column rank).

    Raises:
        ValueError: If *J* does not satisfy the required rank condition.

    Returns:
        *n*×*m* pseudoinverse :math:`J^\\dagger`.
    """
    J = np.asarray(J, dtype=float)
    m, n = J.shape
    r = np.linalg.matrix_rank(J)

    if n > m:  # fat Jacobian, use right pseudo-inverse
        if r != m:
            raise ValueError("Jacobian does not have full row rank")
        J_dagger = J.T @ np.linalg.inv(J @ J.T)
    else:  # tall Jacobian, use left pseudo-inverse
        if r != n:
            raise ValueError("Jacobian does not have full column rank")
        J_dagger = np.linalg.inv(J.T @ J) @ J.T

    return J_dagger


def damped_least_square_inverse(J, k=0.01):
    """Compute the Damped Least-Squares (DLS) pseudoinverse of a Jacobian.

    The DLS inverse avoids numerical blow-up near singularities by adding a
    regularisation term :math:`k^2 I` to the Gram matrix:

    .. math::

        J^* = J^\\top (J J^\\top + k^2 I)^{-1}

    As :math:`k \\to 0` this converges to the Moore-Penrose pseudoinverse;
    larger :math:`k` trades accuracy for stability near singularities.

    Args:
        J: *m*×*n* Jacobian matrix.
        k: Damping factor (default ``0.01``).

    Returns:
        *n*×*m* DLS pseudoinverse :math:`J^*`.
    """
    JJT = J @ J.T
    J_star = J.T @ np.linalg.inv((JJT + k**2 * np.eye(JJT.shape[0])))
    return J_star
