from __future__ import annotations

from pathlib import Path
import numpy as np
import pinocchio as pin


_REF_LOCAL = getattr(getattr(pin, "ReferenceFrame", pin), "LOCAL", None)
_REF_LOCAL_WORLD_ALIGNED = getattr(
    getattr(pin, "ReferenceFrame", pin), "LOCAL_WORLD_ALIGNED", None
)


class PinocchioIK:
    """Pinocchio-backed IK solver for URDF robots with mimic joints.

    This class wraps a Pinocchio model built from the robot's URDF and exposes
    position-only and full-pose IK solvers that work entirely in the
    **active-joint** space (i.e. the subspace spanned by
    ``robot.active_joint_names``).  Mimic joints are handled via an affine
    projection:

    .. math::

        q_{\\text{full}} = b + A\\, q_{\\text{act}}

    where :math:`A` and :math:`b` are computed from the ``<mimic>`` tags in
    the URDF.

    Args:
        robot: Robot object exposing ``active_joint_names``, ``joints``, and
            ``get_joint``.
        urdf_path: Path to the URDF file used to build the Pinocchio model.
        ee_frame_name: Name of the end-effector frame as it appears in the URDF.

    Raises:
        ValueError: If *ee_frame_name* is not found in the Pinocchio model, or
            if the Pinocchio model has undefined ``nq``/``nv``.
        KeyError: If the active/mimic joint mapping is inconsistent between the
            custom robot model and the Pinocchio model.
    """

    def __init__(
        self,
        robot,
        urdf_path: str | Path,
        ee_frame_name: str,
    ) -> None:
        """Build Pinocchio model/data and pre-compute active-joint mappings.

        Args:
            robot: Robot object exposing ``active_joint_names``, ``joints``, and
                ``get_joint``.
            urdf_path: Path to URDF file used by Pinocchio.
            ee_frame_name: End-effector frame name in the Pinocchio model.
        """
        self.robot = robot
        self.urdf_path = str(urdf_path)

        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()

        if not self.model.existFrame(ee_frame_name):
            raise ValueError(f"End-effector frame '{ee_frame_name}' not found in URDF.")

        self.ee_frame_name = ee_frame_name
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)

        names = self.model.names
        self.pin_joint_names = list(names)[1:] if names else []  # skip universe
        if self.model.nq is None or self.model.nv is None:
            raise ValueError("Pinocchio model has undefined nq/nv.")
        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)

        self.A, self.b = self._build_active_to_full_mapping()
        self.q_min, self.q_max = self._build_active_limits()

    def _build_active_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract joint limits for all active joints in order.

        Joints without a ``<limit>`` tag receive :math:`(-\\infty, +\\infty)`.

        Returns:
            ``(q_min, q_max)`` — each a float64 array of length ``len(active_joint_names)``.
        """
        q_min = []
        q_max = []
        for joint_name in self.robot.active_joint_names:
            joint = self.robot.get_joint(joint_name)
            if joint.limit is None:
                q_min.append(-np.inf)
                q_max.append(np.inf)
            else:
                q_min.append(joint.limit.lower)
                q_max.append(joint.limit.upper)
        return np.array(q_min, dtype=float), np.array(q_max, dtype=float)

    def _build_active_to_full_mapping(self) -> tuple[np.ndarray, np.ndarray]:
        """Build the affine map from active-joint space to the full Pinocchio q vector.

        .. math::

            q_{\\text{full}} = b + A\\, q_{\\text{act}}

        For non-mimic joint *i* (in Pinocchio order) the row is a one-hot
        indicator for its position in ``active_joint_names``.
        For a mimic joint mimicking source joint *s*:

        .. math::

            q_{\\text{full},i} = \\text{multiplier} \\cdot q_{\\text{act},s}
                               + \\text{offset}

        Returns:
            ``(A, b)`` where *A* has shape ``(nq, dof)`` and *b* has shape ``(nq,)``.
        """
        active_names = list(self.robot.active_joint_names)
        active_index = {name: i for i, name in enumerate(active_names)}

        # q_full = b + A @ q_act
        A: np.ndarray = np.zeros((self.nq, len(active_names)), dtype=float)
        b: np.ndarray = np.zeros(self.nq, dtype=float)

        for row, joint_name in enumerate(self.pin_joint_names):
            joint = self.robot.joints.get(joint_name, None)
            if joint is None:
                raise KeyError(
                    f"Joint '{joint_name}' exists in Pinocchio model but not in custom robot."
                )

            if joint.mimic is None:
                if joint_name not in active_index:
                    raise KeyError(
                        f"Non-mimic joint '{joint_name}' not found in active_joint_names."
                    )
                A[row, active_index[joint_name]] = 1.0
            else:
                src = joint.mimic.joint
                if src not in active_index:
                    raise KeyError(
                        f"Mimic source joint '{src}' for '{joint_name}' not found in active_joint_names."
                    )
                A[row, active_index[src]] = joint.mimic.multiplier
                b[row] = joint.mimic.offset

        return A, b

    def active_to_full(self, q_act: np.ndarray) -> np.ndarray:
        """Map the active-joint vector to the full Pinocchio configuration vector.

        .. math::

            q_{\\text{full}} = b + A\\, q_{\\text{act}}

        Args:
            q_act: (dof,) active-joint vector.

        Returns:
            (nq,) full Pinocchio configuration vector.
        """
        return self.b + self.A @ q_act

    def clamp_active(self, q_act: np.ndarray) -> np.ndarray:
        """Clamp an active-joint vector to the robot's joint limits.

        Args:
            q_act: (dof,) active-joint vector.

        Returns:
            (dof,) clamped active-joint vector within ``[q_min, q_max]``.
        """
        return np.minimum(np.maximum(q_act, self.q_min), self.q_max)

    def forward_pose(self, q_act: np.ndarray) -> np.ndarray:
        """Return the current end-effector pose from Pinocchio FK.

        Runs :func:`pinocchio.forwardKinematics` and
        :func:`pinocchio.updateFramePlacements` with the full configuration
        obtained via :meth:`active_to_full`, then reads off the end-effector
        placement.

        Args:
            q_act: (dof,) active-joint vector.

        Returns:
            4×4 homogeneous end-effector pose in the robot local/base frame.
        """
        q_full = self.active_to_full(q_act)

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        if self.data.oMf is None:
            raise RuntimeError("Pinocchio frame placements are unavailable.")
        oMf = self.data.oMf[self.ee_frame_id]

        T = np.eye(4)
        T[:3, :3] = oMf.rotation
        T[:3, 3] = oMf.translation
        return T

    def solve_position(
        self,
        p_des: np.ndarray,
        q_init: np.ndarray,
        max_iters: int = 50,
        tol: float = 1e-4,
        damping: float = 1e-3,
        step_size: float = 0.5,
    ) -> tuple[np.ndarray, bool]:
        """Position-only IK in active-joint space (Damped Least-Squares).

        Iterates the DLS update rule on the 3×dof linear-velocity Jacobian:

        .. math::

            \\delta q_{\\text{act}} = J_{\\text{act}}^\\top
                \\left(J_{\\text{act}} J_{\\text{act}}^\\top + k^2 I\\right)^{-1}
                e, \\qquad e = p_{\\text{des}} - p_{\\text{ee}}

        where :math:`J_{\\text{act}} = J_{\\text{full}} A` maps from active
        joints to the Cartesian position error.

        Args:
            p_des: (3,) desired end-effector position expressed in the robot
                local/base frame.
            q_init: (dof,) initial active-joint vector.
            max_iters: Maximum number of iterations.
            tol: Convergence threshold on :math:`\\|e\\|`.
            damping: DLS damping factor :math:`k` (default ``1e-3``).
            step_size: Step-size scaling :math:`\\alpha \\in (0, 1]`
                (default ``0.5``).

        Returns:
            ``(q_act, converged)`` where *q_act* is the (dof,) solution and
            *converged* is ``True`` if the tolerance was met.
        """
        q_act = self.clamp_active(q_init)

        for _ in range(max_iters):
            q_full = self.active_to_full(q_act)

            pin.forwardKinematics(self.model, self.data, q_full)
            pin.updateFramePlacements(self.model, self.data)

            if self.data.oMf is None:
                raise RuntimeError("Pinocchio frame placements are unavailable.")
            oMf = self.data.oMf[self.ee_frame_id]
            p_cur = oMf.translation.copy()

            err = np.asarray(p_des, dtype=float).reshape(3) - p_cur
            if np.linalg.norm(err) < tol:
                return q_act, True

            J6_full = pin.computeFrameJacobian(
                self.model,
                self.data,
                q_full,
                self.ee_frame_id,
                _REF_LOCAL_WORLD_ALIGNED,
            )
            J_full = J6_full[:3, :]  # 3 x nq
            J_act = J_full @ self.A  # 3 x dof

            A_dls = J_act @ J_act.T + (damping**2) * np.eye(3)
            dq_act = J_act.T @ np.linalg.solve(A_dls, err)

            q_act = self.clamp_active(q_act + step_size * dq_act)

        return q_act, False

    def solve_pose(
        self,
        T_des: np.ndarray,
        q_init: np.ndarray,
        max_iters: int = 50,
        tol_rot: float = 1e-3,
        tol_pos: float = 1e-4,
        damping: float = 1e-3,
        step_size: float = 0.5,
        weight_rot: float = 1.0,
        weight_pos: float = 1.0,
    ) -> tuple[np.ndarray, bool]:
        """Full-pose IK in active-joint space using Pinocchio's SE(3) error.

        The pose error is computed in the **local** (end-effector) frame via
        Pinocchio's :func:`pin.log6`:

        .. math::

            e = \\log\\!\\left(T_{\\text{cur}}^{-1}\\, T_{\\text{des}}\\right)^\\vee
            \\in \\mathbb{R}^6

        A weighted DLS step is applied to the 6×dof Jacobian:

        .. math::

            \\delta q_{\\text{act}} = (W J_{\\text{act}})^\\top
                \\left((W J_{\\text{act}})(W J_{\\text{act}})^\\top
                 + k^2 I\\right)^{-1} (W e)

        where :math:`W = \\operatorname{diag}(w_\\omega I_3,\\, w_v I_3)`.

        Args:
            T_des: 4×4 desired end-effector pose in the robot local/base frame.
            q_init: (dof,) initial active-joint vector.
            max_iters: Maximum number of iterations.
            tol_rot: Convergence threshold on the rotational component
                :math:`\\|e_{[:3]}\\|`.
            tol_pos: Convergence threshold on the positional component
                :math:`\\|e_{[3:]}\\|`.
            damping: DLS damping factor :math:`k`.
            step_size: Step-size scaling :math:`\\alpha \\in (0, 1]`.
            weight_rot: Weight :math:`w_\\omega` on the rotational error.
            weight_pos: Weight :math:`w_v` on the positional error.

        Returns:
            ``(q_act, converged)`` where *q_act* is the (dof,) solution and
            *converged* is ``True`` if both tolerances were met.
        """
        q_act = self.clamp_active(q_init)

        R_des = T_des[:3, :3]
        p_des = T_des[:3, 3]
        oMd = pin.SE3(R_des, p_des)

        W = np.diag(
            [weight_rot, weight_rot, weight_rot, weight_pos, weight_pos, weight_pos]
        )

        for _ in range(max_iters):
            q_full = self.active_to_full(q_act)

            pin.forwardKinematics(self.model, self.data, q_full)
            pin.updateFramePlacements(self.model, self.data)

            if self.data.oMf is None:
                raise RuntimeError("Pinocchio frame placements are unavailable.")
            oMf = self.data.oMf[self.ee_frame_id]

            # current frame -> desired frame error
            fMd = oMf.actInv(oMd)
            err = np.asarray(pin.log6(fMd).vector, dtype=float).reshape(6)

            if np.linalg.norm(err[:3]) < tol_rot and np.linalg.norm(err[3:]) < tol_pos:
                return q_act, True

            J6_full = pin.computeFrameJacobian(
                self.model,
                self.data,
                q_full,
                self.ee_frame_id,
                _REF_LOCAL,
            )  # 6 x nq

            J6_act = J6_full @ self.A  # 6 x dof

            JW = W @ J6_act
            eW = W @ err

            A_dls = JW @ JW.T + (damping**2) * np.eye(6)
            dq_act = JW.T @ np.linalg.solve(A_dls, eW)

            q_act = self.clamp_active(q_act + step_size * dq_act)

        return q_act, False
