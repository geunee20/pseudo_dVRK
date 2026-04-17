from __future__ import annotations

from pathlib import Path
import numpy as np
import pinocchio as pin
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt


class PinocchioIK:
    def __init__(self, robot, urdf_path: str | Path, ee_frame_name: str):
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
        self.nq = int(self.model.nq)  # type: ignore
        self.nv = int(self.model.nv)  # type: ignore

        self.A, self.b = self._build_active_to_full_mapping()
        self.q_min, self.q_max = self._build_active_limits()

    def _build_active_limits(self) -> tuple[np.ndarray, np.ndarray]:
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
        q_act = np.asarray(q_act, dtype=float).reshape(-1)
        return self.b + self.A @ q_act

    def clamp_active(self, q_act: np.ndarray) -> np.ndarray:
        q_act = np.asarray(q_act, dtype=float).reshape(-1)
        return np.minimum(np.maximum(q_act, self.q_min), self.q_max)

    def forward_pose(self, q_act: np.ndarray) -> np.ndarray:
        """
        Return current end-effector pose as a 4x4 transform
        in the robot local/base frame.
        """
        q_full = self.active_to_full(q_act)

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        oMf = self.data.oMf[self.ee_frame_id]  # type: ignore

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
        """
        Position-only IK in active-joint space.
        p_des must be expressed in the robot local/base frame.
        q_init is active-joint vector (length = robot.dof).
        """
        q_act = self.clamp_active(q_init)

        for _ in range(max_iters):
            q_full = self.active_to_full(q_act)

            pin.forwardKinematics(self.model, self.data, q_full)
            pin.updateFramePlacements(self.model, self.data)

            oMf = self.data.oMf[self.ee_frame_id]  # type: ignore
            p_cur = oMf.translation.copy()

            err = np.asarray(p_des, dtype=float).reshape(3) - p_cur
            if np.linalg.norm(err) < tol:
                return q_act, True

            J6_full = pin.computeFrameJacobian(
                self.model,
                self.data,
                q_full,
                self.ee_frame_id,
                pin.ReferenceFrame.WORLD,  # type: ignore
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
        """
        Full pose IK in active-joint space.

        Parameters
        ----------
        T_des : (4,4) ndarray
            Desired end-effector pose expressed in the robot local/base frame.
        q_init : (dof,) ndarray
            Initial active joint vector.
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

            oMf = self.data.oMf[self.ee_frame_id]  # type: ignore

            # current frame -> desired frame error
            fMd = oMf.actInv(oMd)
            err_vec = pin.log6(fMd).vector  # type: ignore  # shape (6,)
            err = err_vec  # type: ignore

            if np.linalg.norm(err[:3]) < tol_rot and np.linalg.norm(err[3:]) < tol_pos:  # type: ignore
                return q_act, True

            J6_full = pin.computeFrameJacobian(
                self.model,
                self.data,
                q_full,
                self.ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,  # type: ignore
            )  # 6 x nq

            J6_act = J6_full @ self.A  # 6 x dof

            JW = W @ J6_act
            eW = W @ err

            A_dls = JW @ JW.T + (damping**2) * np.eye(6)
            dq_act = JW.T @ np.linalg.solve(A_dls, eW)

            q_act = self.clamp_active(q_act + step_size * dq_act)

        return q_act, False
