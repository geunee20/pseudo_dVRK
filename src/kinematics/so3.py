import numpy as np
import warnings


def Rx(a: float) -> np.ndarray:
    """Return the 3×3 rotation matrix for a rotation about the X axis.

    .. math::

        R_x(a) = \\begin{bmatrix}
            1 & 0 & 0 \\\\
            0 & \\cos a & -\\sin a \\\\
            0 & \\sin a &  \\cos a
        \\end{bmatrix}

    Args:
        a: Rotation angle in radians.

    Returns:
        3×3 rotation matrix :math:`R_x(a) \\in SO(3)`.
    """
    c, s = np.cos(a), np.sin(a)
    return np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ]
    )


def Ry(a: float) -> np.ndarray:
    """Return the 3×3 rotation matrix for a rotation about the Y axis.

    .. math::

        R_y(a) = \\begin{bmatrix}
             \\cos a & 0 & \\sin a \\\\
             0      & 1 & 0 \\\\
            -\\sin a & 0 & \\cos a
        \\end{bmatrix}

    Args:
        a: Rotation angle in radians.

    Returns:
        3×3 rotation matrix :math:`R_y(a) \\in SO(3)`.
    """
    c, s = np.cos(a), np.sin(a)
    return np.array(
        [
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ]
    )


def Rz(a: float) -> np.ndarray:
    """Return the 3×3 rotation matrix for a rotation about the Z axis.

    .. math::

        R_z(a) = \\begin{bmatrix}
            \\cos a & -\\sin a & 0 \\\\
            \\sin a &  \\cos a & 0 \\\\
            0      &  0      & 1
        \\end{bmatrix}

    Args:
        a: Rotation angle in radians.

    Returns:
        3×3 rotation matrix :math:`R_z(a) \\in SO(3)`.
    """
    c, s = np.cos(a), np.sin(a)
    return np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ]
    )


def rpy_to_R(rpy: np.ndarray) -> np.ndarray:
    """Convert roll-pitch-yaw angles to a 3×3 rotation matrix (ZYX convention).

    The rotation is applied in the order roll → pitch → yaw:

    .. math::

        R = R_z(y)\\, R_y(p)\\, R_x(r)

    Args:
        rpy: Array of shape (3,) containing ``[roll, pitch, yaw]`` in radians.

    Returns:
        3×3 rotation matrix :math:`R \\in SO(3)`.
    """
    r, p, y = rpy
    return Rz(y) @ Ry(p) @ Rx(r)


def RToAxisAngle(R: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, float]:
    """Extract the rotation axis and angle from a rotation matrix (matrix logarithm).

    Three cases are handled:

    * **Identity** (:math:`\\|R - I\\| < \\varepsilon`): axis is undefined;
      returns :math:`\\hat{z}` and :math:`\\theta = 0`.
    * **:math:`\\theta = \\pi`** (:math:`\\operatorname{tr}(R) = -1`):
      extracts the axis from the column with the largest diagonal entry.
    * **Generic case**: uses

      .. math::

          \\theta = \\arccos\\!\\left(\\frac{\\operatorname{tr}(R)-1}{2}\\right),
          \\qquad
          [\\hat{\\omega}] = \\frac{R - R^\\top}{2\\sin\\theta}

    Args:
        R: 3×3 rotation matrix.
        eps: Numerical tolerance for detecting special cases.

    Returns:
        ``(w_hat, theta)`` where *w_hat* is the unit rotation axis (shape (3,))
        and *theta* is the rotation angle in radians :math:`\\in [0, \\pi]`.
    """
    R = np.asarray(R, dtype=float)
    tr_R = np.trace(R)
    I = np.eye(3)
    # Case I: R = I
    if np.linalg.norm(R - I) < eps:
        warnings.warn("Rotation axis is undefined", UserWarning)
        return np.array([0.0, 0.0, 1.0]), 0.0

    # Case II: tr(R) = -1
    if abs(np.trace(R) + 1) < eps:
        diag = 1 + np.diag(R)
        i = int(np.argmax(diag))

        theta = np.pi

        w_hat = R[:, i].copy()
        w_hat[i] += 1.0
        w_hat = w_hat / np.sqrt(2 * (1 + R[i, i]))

    else:  # Case III: Otherwise
        theta = np.arccos((tr_R - 1) / 2)
        theta = np.clip(theta, 0, np.pi)

        W = (R - np.transpose(R)) / (2 * np.sin(theta))
        w_hat = np.array([W[2, 1], W[0, 2], W[1, 0]])

    w_hat /= np.linalg.norm(w_hat)
    return w_hat, theta


def AxisAngleToR(w: np.ndarray, theta: float) -> np.ndarray:
    """Convert a rotation axis–angle pair to a 3×3 rotation matrix (Rodrigues' formula).

    .. math::

        R = e^{[\\omega]\\theta} = I
            + \\sin\\theta\\,[\\omega]
            + (1 - \\cos\\theta)\\,[\\omega]^2

    where :math:`[\\omega]` is the :func:`skew`-symmetric (hat) matrix of
    the unit vector :math:`\\hat{\\omega}`.

    Args:
        w: Rotation axis as a 3-vector (need not be a unit vector; normalised
           internally).
        theta: Rotation angle in radians.

    Returns:
        3×3 rotation matrix :math:`R \\in SO(3)`.  Returns :math:`I` when
        :math:`\\|w\\| < 10^{-12}`.
    """
    w = np.asarray(w, dtype=float)

    wn = np.linalg.norm(w)

    if wn < 1e-12:
        return np.eye(3)

    w = w / wn
    w_hat = skew(w)

    R = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * (w_hat @ w_hat)
    return R


def skew(w: np.ndarray) -> np.ndarray:
    """Build the 3×3 skew-symmetric (hat) matrix of a 3-vector.

    The hat map :math:`(\\cdot)^\\wedge : \\mathbb{R}^3 \\to \\mathfrak{so}(3)`
    is defined as:

    .. math::

        [\\omega]^{} = \\begin{bmatrix}
             0    & -w_3 &  w_2 \\\\
             w_3  &  0   & -w_1 \\\\
            -w_2  &  w_1 &  0
        \\end{bmatrix}

    so that :math:`[\\omega] v = \\omega \\times v` for any :math:`v \\in \\mathbb{R}^3`.

    Args:
        w: 3-vector :math:`\\omega = [w_1, w_2, w_3]^\\top`.

    Returns:
        3×3 skew-symmetric matrix :math:`[\\omega] \\in \\mathfrak{so}(3)`.
    """
    w1, w2, w3 = w
    return np.array([[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]])


def unskew(hat_w: np.ndarray) -> np.ndarray:
    """Extract the 3-vector from a skew-symmetric matrix (vee map).

    Inverse of :func:`skew`:

    .. math::

        \\omega = [\\omega]^{\\vee}
        = \\begin{bmatrix} [\\omega]_{32} \\\\ [\\omega]_{13} \\\\ [\\omega]_{21} \\end{bmatrix}

    Args:
        hat_w: 3×3 skew-symmetric matrix :math:`[\\omega] \\in \\mathfrak{so}(3)`.

    Returns:
        3-vector :math:`\\omega`.
    """
    return np.array([hat_w[2, 1], hat_w[0, 2], hat_w[1, 0]])
