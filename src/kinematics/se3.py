import numpy as np
from .so3 import RToAxisAngle, rpy_to_R, skew


def screw_axis_from_w_q(w: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Build a revolute screw axis from a rotation axis and a point on the axis.

    A revolute screw axis has zero pitch, so the linear component is
    determined entirely by the cross-product rule:

    .. math::

        \\mathcal{S} = \\begin{bmatrix} \\hat{\\omega} \\\\ v \\end{bmatrix},
        \\quad v = -\\hat{\\omega} \\times q = -[\\hat{\\omega}]\\, q

    Args:
        w: Unit rotation axis :math:`\\hat{\\omega} \\in \\mathbb{R}^3`.
        q: Any point on the rotation axis :math:`q \\in \\mathbb{R}^3`.

    Returns:
        6-vector :math:`\\mathcal{S} = [\\hat{\\omega}^\\top,\\, v^\\top]^\\top`.
    """
    w = np.asarray(w, dtype=float).reshape(
        3,
    )
    q = np.asarray(q, dtype=float).reshape(
        3,
    )
    return np.concatenate([w, -skew(w) @ q])


def vec_to_se3(V: np.ndarray) -> np.ndarray:
    """Lift a 6-vector twist to its 4×4 :math:`\\mathfrak{se}(3)` matrix (hat map).

    .. math::

        \\mathcal{V} = \\begin{bmatrix} \\omega \\\\ v \\end{bmatrix}
        \\;\\longmapsto\\;
        [\\mathcal{V}] = \\begin{bmatrix} [\\omega] & v \\\\ 0 & 0 \\end{bmatrix}
        \\in \\mathfrak{se}(3)

    Args:
        V: 6-vector :math:`[\\omega^\\top, v^\\top]^\\top` of twist coordinates.

    Returns:
        4×4 :math:`\\mathfrak{se}(3)` matrix :math:`[\\mathcal{V}]`.
    """
    hat_V = np.zeros((4, 4))
    hat_V[:3, :3] = skew(V[:3])
    hat_V[:3, 3] = V[3:]
    return hat_V


def exp_screw_hat(hat_S: np.ndarray, theta: float) -> np.ndarray:
    """Compute the matrix exponential :math:`e^{[\\mathcal{S}]\\theta} \\in SE(3)`.

    For a revolute-type screw axis (:math:`\\|\\omega\\| > 0`):

    .. math::

        T = e^{[\\mathcal{S}]\\theta} =
        \\begin{bmatrix}
            e^{[\\omega]\\theta} & G(\\theta)\\, v \\\\
            0 & 1
        \\end{bmatrix}

    where :math:`e^{[\\omega]\\theta}` uses Rodrigues' formula and

    .. math::

        G(\\theta) = I\\theta + (1-\\cos\\theta)[\\omega]
                    + (\\theta - \\sin\\theta)[\\omega]^2.

    For a pure-translation axis (:math:`\\omega = 0`):

    .. math::

        T = \\begin{bmatrix} I & v\\,\\theta \\\\ 0 & 1 \\end{bmatrix}.

    Args:
        hat_S: 4×4 :math:`\\mathfrak{se}(3)` matrix :math:`[\\mathcal{S}]`.
        theta: Joint displacement (radians for revolute, metres for prismatic).

    Returns:
        4×4 homogeneous transformation :math:`T \\in SE(3)`.
    """
    hat_w = hat_S[:3, :3]
    v = hat_S[:3, 3]

    w = np.array([hat_w[2, 1], hat_w[0, 2], hat_w[1, 0]])
    wn = np.linalg.norm(w)

    T = np.eye(4)

    if wn < 1e-12:
        # pure translation
        T[:3, :3] = np.eye(3)
        T[:3, 3] = v * theta
        return T

    R = np.eye(3) + np.sin(theta) * hat_w + (1 - np.cos(theta)) * (hat_w @ hat_w)
    G = (
        np.eye(3) * theta
        + (1 - np.cos(theta)) * hat_w
        + (theta - np.sin(theta)) * (hat_w @ hat_w)
    )

    T[:3, :3] = R
    T[:3, 3] = G @ v
    return T


def exp_screw_axis(S: np.ndarray, theta: float) -> np.ndarray:
    """Compute the matrix exponential :math:`e^{[\\mathcal{S}]\\theta} \\in SE(3)`.

    Convenience wrapper around :func:`exp_screw_hat` that accepts the 6-vector
    form of the screw axis directly.

    .. math::

        T = e^{[\\mathcal{S}]\\theta}, \\quad \\mathcal{S} \\in \\mathbb{R}^6.

    Args:
        S: 6-vector :math:`[\\omega^\\top, v^\\top]^\\top` screw axis.
        theta: Joint displacement (radians for revolute, metres for prismatic).

    Returns:
        4×4 homogeneous transformation :math:`T \\in SE(3)`.
    """
    hat_S = vec_to_se3(S)
    return exp_screw_hat(hat_S, theta)


def log_screw_axis(T: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute the matrix logarithm of an SE(3) element (inverse of :func:`exp_screw_axis`).

    Returns the screw axis :math:`\\mathcal{S}` and displacement :math:`\\theta`
    such that :math:`T = e^{[\\mathcal{S}]\\theta}`.

    For the revolute case (:math:`\\|\\omega\\| > 0`):

    .. math::

        \\theta = \\arccos\\!\\left(\\frac{\\operatorname{tr}(R)-1}{2}\\right),
        \\qquad
        v = G^{-1}(\\theta)\\, p

    where

    .. math::

        G^{-1}(\\theta) = \\frac{I}{\\theta}
            - \\frac{[\\omega]}{2}
            + \\left(\\frac{1}{\\theta} - \\frac{1}{2}\\cot\\frac{\\theta}{2}\\right)[\\omega]^2.

    For the pure-translation case (:math:`\\omega = 0`):

    .. math::

        \\mathcal{S} = [0,\\, \\hat{v}]^\\top, \\quad \\theta = \\|p\\|.

    Args:
        T: 4×4 homogeneous transformation :math:`T \\in SE(3)`.

    Returns:
        ``(S, theta)`` where *S* is the 6-vector screw axis and *theta* is
        the scalar displacement.
    """
    R = T[:3, :3]
    p = T[:3, 3]

    w, theta = RToAxisAngle(R)

    if np.linalg.norm(w * theta) < 1e-12:
        # pure translation
        v = p / np.linalg.norm(p)
        return np.concatenate([np.zeros(3), v]), float(np.linalg.norm(p))

    G_inv = (
        np.eye(3) / theta
        - 0.5 * skew(w)
        + (1 / theta - 0.5 / np.tan(theta / 2)) * (skew(w) @ skew(w))
    )
    v = G_inv @ p

    return np.concatenate([w, v]), float(theta)


def inv_SE3(T: np.ndarray) -> np.ndarray:
    """Compute the closed-form inverse of a homogeneous transformation.

    Exploits the structure of :math:`SE(3)` to avoid a generic matrix inversion:

    .. math::

        T^{-1} = \\begin{bmatrix} R^\\top & -R^\\top p \\\\ 0 & 1 \\end{bmatrix}

    Args:
        T: 4×4 homogeneous transformation :math:`T \\in SE(3)`.

    Returns:
        4×4 inverse :math:`T^{-1} \\in SE(3)`.
    """
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    p = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p

    return T_inv


def adjoint(T: np.ndarray) -> np.ndarray:
    """Compute the 6×6 adjoint representation :math:`[\\text{Ad}_T]` of an SE(3) element.

    .. math::

        [\\text{Ad}_T] =
        \\begin{bmatrix} R & 0 \\\\ [p]R & R \\end{bmatrix}

    where :math:`[p]` is the skew-symmetric matrix of the translation :math:`p`.

    Args:
        T: 4×4 homogeneous transformation :math:`T \\in SE(3)`.

    Returns:
        6×6 adjoint matrix :math:`[\\text{Ad}_T]`.
    """
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    p = T[:3, 3]
    return np.block([[R, np.zeros((3, 3))], [skew(p) @ R, R]])


def adjoint_inverse(T: np.ndarray) -> np.ndarray:
    """Compute the 6×6 inverse adjoint :math:`[\\text{Ad}_{T^{-1}}]`.

    .. math::

        [\\text{Ad}_{T^{-1}}] =
        \\begin{bmatrix} R^\\top & 0 \\\\ -R^\\top[p] & R^\\top \\end{bmatrix}

    This is equivalent to :math:`[\\text{Ad}_T]^{-1}` but computed without
    a matrix inversion.

    Args:
        T: 4×4 homogeneous transformation :math:`T \\in SE(3)`.

    Returns:
        6×6 inverse adjoint matrix :math:`[\\text{Ad}_{T^{-1}}]`.
    """
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    p = T[:3, 3]
    Ad_T_inv = np.block([[R.T, np.zeros((3, 3))], [-R.T @ skew(p), R.T]])
    return Ad_T_inv


def adjoint_transform(
    T: np.ndarray, X: np.ndarray, to_space: bool = True
) -> np.ndarray:
    """Transform a twist between body and space frames using the adjoint map.

    * **Body → Space** (``to_space=True``):

      .. math::

          \\mathcal{V}_s = [\\text{Ad}_{T_{sb}}]\\, \\mathcal{V}_b

    * **Space → Body** (``to_space=False``):

      .. math::

          \\mathcal{V}_b = [\\text{Ad}_{T_{sb}^{-1}}]\\, \\mathcal{V}_s

    Args:
        T: 4×4 pose of the body frame in the space frame (:math:`T_{sb}`).
        X: 6-vector twist to transform.
        to_space: If ``True``, transforms from body to space frame;
            if ``False``, from space to body frame.

    Returns:
        Transformed 6-vector twist.
    """
    T = np.asarray(T, dtype=float).reshape(4, 4)
    X = np.asarray(X, dtype=float).reshape(
        6,
    )

    if to_space:
        return adjoint(T) @ X
    return adjoint_inverse(T) @ X


def adjoint_transform_list(T: np.ndarray, X_list: list, to_space: bool = True) -> list:
    """Apply :func:`adjoint_transform` to a list of twists.

    Applies the same adjoint transformation elementwise:

    .. math::

        \\mathcal{V}'_i = [\\text{Ad}_{T_{sb}}^{\\pm 1}]\\, \\mathcal{V}_i
        \\quad \\forall\\, i.

    Args:
        T: 4×4 pose of the body frame in the space frame (:math:`T_{sb}`).
        X_list: List of 6-vector twists to transform.
        to_space: If ``True``, transforms from body to space frame;
            if ``False``, from space to body frame.

    Returns:
        List of transformed 6-vector twists in the same order as *X_list*.
    """
    return [adjoint_transform(T, X, to_space) for X in X_list]


def origin_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    """Build a 4×4 homogeneous transform from a URDF ``<origin>`` tag.

    .. math::

        T = \\begin{bmatrix} R(\\text{rpy}) & \\text{xyz} \\\\ 0 & 1 \\end{bmatrix}

    where :math:`R(\\text{rpy}) = R_z(y)\\,R_y(p)\\,R_x(r)` (ZYX convention).

    Args:
        xyz: 3-vector translation :math:`[x, y, z]^\\top` (metres).
        rpy: 3-vector :math:`[\\text{roll}, \\text{pitch}, \\text{yaw}]^\\top`
            (radians).

    Returns:
        4×4 homogeneous transform :math:`T \\in SE(3)`.
    """
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(
        np.asarray(rpy, dtype=float).reshape(
            3,
        )
    )
    T[:3, 3] = np.asarray(xyz, dtype=float).reshape(
        3,
    )
    return T
