from ctypes import CDLL, POINTER, c_char_p
from typing import Any, List, Optional
from sys import platform

from .hd_define import (
    HDint,
    HDenum,
    HHD,
    HDErrorInfo,
    HDstring,
    HD_BAD_HANDLE,
    HD_SUCCESS,
    HD_INVALID_ENUM,
    HD_INVALID_VALUE,
    HD_INVALID_OPERATION,
    HD_INVALID_HANDLE,
    HD_FORCE_ERROR,
    HD_WARM_MOTORS,
    HD_EXCEED_MAX_FORCE,
    HD_EXCEEDED_MAX_FORCE_IMPULSE,
    HD_EXCEEDED_MAX_VELOCITY,
    HD_DEVICE_FAULT,
    HD_DEVICE_ALREADY_INITIATED,
    HD_COMM_ERROR,
    HD_COMM_CONFIG_ERROR,
    HD_TIMER_ERROR,
    HD_ILLEGAL_BEGIN,
    HD_ILLEGAL_END,
    HD_FRAME_ERROR,
    HD_INVALID_PRIORITY,
    HD_SCHEDULER_FULL,
    HD_INVALID_LICENSE,
    HDint,
    HDdouble,
    HD_CURRENT_BUTTONS,
    HD_CURRENT_TRANSFORM,
    HD_CURRENT_VELOCITY,
    HD_CURRENT_JOINT_ANGLES,
    HD_CURRENT_GIMBAL_ANGLES,
    HD_JOINT_ANGLE_REFERENCES,
    HD_CURRENT_FORCE,
    HD_FORCE_OUTPUT,
    HD_DEVICE_MODEL_TYPE,
    HD_DEVICE_VENDOR,
    HD_DEVICE_SERIAL_NUMBER,
    HD_UPDATE_RATE,
    HD_INSTANTANEOUS_UPDATE_RATE,
    HD_NOMINAL_MAX_FORCE,
    HD_NOMINAL_MAX_CONTINUOUS_FORCE,
    HD_NOMINAL_MAX_STIFFNESS,
    HD_NOMINAL_MAX_DAMPING,
)
from .hdu_matrix import hduMatrix, hduVector3Dd
from .exceptions import *
import numpy as np

_scheduler_started = False

# OpenHaptics axis mapping (logical XYZ -> device XYZ)
_axis_perm = np.array([1, 2, 0], dtype=int)
_axis_sign = np.array([1.0, 1.0, 1.0], dtype=float)

exception_dict = {
    HD_BAD_HANDLE: DeviceInitException,
    HD_INVALID_ENUM: InvalidEnumException,
    HD_INVALID_VALUE: InvalidValueException,
    HD_INVALID_OPERATION: InvalidOperationException,
    HD_INVALID_HANDLE: DeviceInitException,
    HD_FORCE_ERROR: ForceTypeExceptions,
    HD_WARM_MOTORS: ForceTypeExceptions,
    HD_EXCEED_MAX_FORCE: ForceTypeExceptions,
    HD_EXCEEDED_MAX_FORCE_IMPULSE: ForceTypeExceptions,
    HD_EXCEEDED_MAX_VELOCITY: ForceTypeExceptions,
    HD_DEVICE_FAULT: DeviceInitException,
    HD_DEVICE_ALREADY_INITIATED: DeviceInitException,
    HD_COMM_ERROR: DeviceInitException,
    HD_COMM_CONFIG_ERROR: DeviceInitException,
    HD_TIMER_ERROR: DeviceInitException,
    HD_ILLEGAL_BEGIN: InvalidOperationException,
    HD_ILLEGAL_END: InvalidOperationException,
    HD_FRAME_ERROR: InvalidOperationException,
    HD_INVALID_PRIORITY: InvalidOperationException,
    HD_SCHEDULER_FULL: InvalidOperationException,
    HD_INVALID_LICENSE: DeviceInitException,
}

error_name_dict = {
    HD_SUCCESS: "HD_SUCCESS",
    HD_INVALID_ENUM: "HD_INVALID_ENUM",
    HD_INVALID_VALUE: "HD_INVALID_VALUE",
    HD_INVALID_OPERATION: "HD_INVALID_OPERATION",
    HD_BAD_HANDLE: "HD_BAD_HANDLE",
    HD_WARM_MOTORS: "HD_WARM_MOTORS",
    HD_EXCEED_MAX_FORCE: "HD_EXCEED_MAX_FORCE",
    HD_EXCEEDED_MAX_FORCE_IMPULSE: "HD_EXCEEDED_MAX_FORCE_IMPULSE",
    HD_EXCEEDED_MAX_VELOCITY: "HD_EXCEEDED_MAX_VELOCITY",
    HD_FORCE_ERROR: "HD_FORCE_ERROR",
    HD_DEVICE_FAULT: "HD_DEVICE_FAULT",
    HD_DEVICE_ALREADY_INITIATED: "HD_DEVICE_ALREADY_INITIATED",
    HD_COMM_ERROR: "HD_COMM_ERROR",
    HD_COMM_CONFIG_ERROR: "HD_COMM_CONFIG_ERROR",
    HD_TIMER_ERROR: "HD_TIMER_ERROR",
    HD_ILLEGAL_BEGIN: "HD_ILLEGAL_BEGIN",
    HD_ILLEGAL_END: "HD_ILLEGAL_END",
    HD_FRAME_ERROR: "HD_FRAME_ERROR",
    HD_INVALID_PRIORITY: "HD_INVALID_PRIORITY",
    HD_SCHEDULER_FULL: "HD_SCHEDULER_FULL",
    HD_INVALID_LICENSE: "HD_INVALID_LICENSE",
}


if platform == "linux" or platform == "linux2":
    _lib_hd = CDLL("libHD.so")
elif platform == "win32":
    _lib_hd = CDLL("HD.dll")
else:
    raise OSError(f"Unsupported platform: {platform}")


def _get_doublev(code: int, dtype):
    data = dtype()
    _lib_hd.hdGetDoublev.argtypes = [HDenum, POINTER(dtype)]
    _lib_hd.hdGetDoublev.restype = None
    _lib_hd.hdGetDoublev(code, data)
    return data


def _get_integerv(code: int, dtype):
    data = dtype()
    _lib_hd.hdGetIntegerv.argtypes = [HDenum, POINTER(dtype)]
    _lib_hd.hdGetIntegerv.restype = None
    _lib_hd.hdGetIntegerv(code, data)
    return data


def _set_doublev(code: int, value, dtype):
    data = dtype(*value)
    _lib_hd.hdSetDoublev.argtypes = [HDenum, POINTER(dtype)]
    _lib_hd.hdSetDoublev.restype = None
    _lib_hd.hdSetDoublev(code, data)


def _get_error() -> HDErrorInfo:
    _lib_hd.hdGetError.restype = HDErrorInfo
    return _lib_hd.hdGetError()


def error_code_name(error_code: int) -> str:
    return error_name_dict.get(error_code, f"UNKNOWN_HD_ERROR_0x{int(error_code):04X}")


def init_device(name: str = "Default Device") -> int:
    """Initialise a PHANTOM haptic device and return its handle.

    Wraps ``hdInitDevice``.  Raises :class:`~src.pyOpenHaptics.exceptions.DeviceInitException`
    if the device cannot be opened (handle == ``HD_BAD_HANDLE``).

    Args:
        name: Device name string as configured in the OpenHaptics driver.
            Pass ``"Default Device"`` to open the first available device.

    Returns:
        Integer device handle (``HHD``) to be passed to ``hdMakeCurrentDevice``.

    Raises:
        DeviceInitException: If ``hdInitDevice`` returns ``HD_BAD_HANDLE``.
    """
    _lib_hd.hdInitDevice.argtypes = [c_char_p]
    _lib_hd.hdInitDevice.restype = HHD
    device_id = _lib_hd.hdInitDevice(name.encode())
    if int(device_id) == HD_BAD_HANDLE:
        raise DeviceInitException(
            f"Unable to initialize device '{name}'. hdInitDevice returned HD_BAD_HANDLE."
        )
    return int(device_id)


def get_buttons() -> int:
    """Return the current button bitmask of the active haptic device.

    Bit 0 = button 1, bit 1 = button 2, etc.  Must be called inside an
    ``hdBeginFrame`` / ``hdEndFrame`` block.

    Returns:
        Integer bitmask where each set bit indicates a pressed button.
    """
    return _get_integerv(HD_CURRENT_BUTTONS, HDint).value


def get_transform() -> Any:  # hduMatrix
    """Return the current 4×4 column-major homogeneous transform of the stylus.

    Returns:
        ``hduMatrix`` (ctypes array of 16 doubles, column-major) representing
        the stylus pose in device millimetre coordinates.
    """
    return _get_doublev(HD_CURRENT_TRANSFORM, hduMatrix)


def get_velocity() -> np.ndarray:
    """Return the current stylus velocity in logical (canvas) frame, in mm/s.

    Applies the inverse of the axis permutation set by :func:`set_axis_mapping`
    so that the velocity components match the logical coordinate frame of the
    forces commanded via :func:`set_force`.

    Returns:
        Velocity vector ``[vx, vy, vz]``, shape ``(3,)``, in mm/s.
    """
    raw = _get_doublev(HD_CURRENT_VELOCITY, hduVector3Dd)
    raw_np = np.array([raw[0], raw[1], raw[2]], dtype=float)
    # Apply inverse of the axis mapping used in set_force so that velocity is in
    # the same logical (canvas) frame as the forces we command.
    result = np.empty(3, dtype=float)
    result[_axis_perm] = raw_np * _axis_sign
    return result


def get_joint_angles() -> Any:  # hduVector3Dd
    """Return the current joint angles of the device arm (shoulder, elbow, yaw).

    Returns:
        ``hduVector3Dd`` with three joint angles in radians.
    """
    return _get_doublev(HD_CURRENT_JOINT_ANGLES, hduVector3Dd)


def get_gimbal_angles() -> Any:  # hduVector3Dd
    """Return the current gimbal (wrist) angles of the device stylus.

    Returns:
        ``hduVector3Dd`` with three gimbal angles in radians.
    """
    return _get_doublev(HD_CURRENT_GIMBAL_ANGLES, hduVector3Dd)


def get_joint_angle_references() -> Any:  # hduVector3Dd
    """Return factory-calibrated joint angle reference offsets.

    Returns:
        ``hduVector3Dd`` with three reference angle values in radians.
    """
    return _get_doublev(HD_JOINT_ANGLE_REFERENCES, hduVector3Dd)


def get_last_error_info() -> HDErrorInfo:
    """Return the most recent OpenHaptics error info structure.

    Returns:
        :class:`~src.pyOpenHaptics.hd_define.HDErrorInfo` with ``errorCode``,
        ``internalErrorCode``, and ``hHD`` fields.
    """
    return _get_error()


def get_error(raise_on_error: bool = False) -> bool:
    """Check for a pending OpenHaptics error and optionally raise an exception.

    Args:
        raise_on_error: If ``True``, raise the appropriate exception class
            (from :mod:`src.pyOpenHaptics.exceptions`) when an error is detected.

    Returns:
        ``True`` if an error was present, ``False`` if ``HD_SUCCESS``.
    """
    info = _get_error()
    error = int(info.errorCode)
    if error == HD_SUCCESS:
        return False

    exc_type = exception_dict.get(error, RuntimeError)
    error_name = error_code_name(error)
    msg = (
        f"OpenHaptics error: {error_name} (0x{error:04X}, {error}). "
        f"internalErrorCode={int(info.internalErrorCode)}, hHD={int(info.hHD)}"
    )
    print(msg)
    if raise_on_error:
        raise exc_type(msg)
    return True


def set_force(feedback: List[float]) -> None:
    """Command the haptic device to render a force vector.

    The force is specified in the logical (canvas) coordinate frame and
    re-mapped to the device frame using the current axis permutation / sign.

    Args:
        feedback: Desired force ``[Fx, Fy, Fz]`` in Newtons.

    Raises:
        ValueError: If *feedback* does not have exactly 3 elements.
    """
    feedback_arr = np.asarray(feedback, dtype=float)
    if feedback_arr.shape != (3,):
        raise ValueError("feedback must be length-3: [Fx, Fy, Fz]")

    mapped = feedback_arr[_axis_perm] * _axis_sign
    _set_doublev(HD_CURRENT_FORCE, mapped.tolist(), hduVector3Dd)


def set_axis_mapping(perm: List[int], sign: List[float] | None = None) -> None:
    """Configure the axis permutation and sign flip applied to forces and velocities.

    Sets the global ``_axis_perm`` / ``_axis_sign`` used by :func:`set_force`
    and :func:`get_velocity` to remap between device and canvas frames.

    Args:
        perm: Length-3 permutation of ``[0, 1, 2]``, e.g. ``[1, 2, 0]``
            maps device axis 0 → canvas axis 1, etc.
        sign: Optional length-3 sign vector, each ±1.  Defaults to all ones.

    Raises:
        ValueError: If *perm* or *sign* have wrong shape or *perm* is not a
            valid permutation.
    """
    global _axis_perm, _axis_sign

    perm_np = np.asarray(perm, dtype=int)
    if perm_np.shape != (3,):
        raise ValueError("perm must be length-3, e.g. [1, 2, 0]")
    if sorted(perm_np.tolist()) != [0, 1, 2]:
        raise ValueError("perm must be a permutation of [0, 1, 2]")

    if sign is None:
        sign_np = np.array([1.0, 1.0, 1.0], dtype=float)
    else:
        sign_np = np.asarray(sign, dtype=float)
        if sign_np.shape != (3,):
            raise ValueError("sign must be length-3, e.g. [1, -1, 1]")

    _axis_perm = perm_np
    _axis_sign = sign_np


def get_axis_mapping() -> tuple[list[int], list[float]]:
    """Return the current axis permutation and sign mapping.

    Returns:
        ``(perm, sign)`` — copies of the permutation and sign arrays as Python lists.
    """
    return _axis_perm.tolist(), _axis_sign.tolist()


def get_update_rate() -> int:
    """Return the configured servo-loop update rate in Hz.

    Returns:
        Integer update rate (e.g. 1000).
    """
    return int(_get_integerv(HD_UPDATE_RATE, HDint).value)


def get_instantaneous_update_rate() -> int:
    """Return the measured instantaneous update rate in Hz.

    Returns:
        Integer instantaneous rate measured in the last servo tick.
    """
    return int(_get_integerv(HD_INSTANTANEOUS_UPDATE_RATE, HDint).value)


def get_nominal_max_force() -> float:
    """Return the device's nominal maximum force in Newtons.

    Returns:
        Float peak force limit (N).
    """
    return float(_get_doublev(HD_NOMINAL_MAX_FORCE, HDdouble).value)


def get_nominal_max_continuous_force() -> float:
    """Return the device's nominal maximum continuous force in Newtons.

    Returns:
        Float continuous force limit (N).
    """
    return float(_get_doublev(HD_NOMINAL_MAX_CONTINUOUS_FORCE, HDdouble).value)


def get_nominal_max_stiffness() -> float:
    """Return the device's nominal maximum stiffness in N/mm.

    Returns:
        Float stiffness limit (N/mm).
    """
    return float(_get_doublev(HD_NOMINAL_MAX_STIFFNESS, HDdouble).value)


def get_nominal_max_damping() -> float:
    """Return the device's nominal maximum damping in N·s/mm.

    Returns:
        Float damping limit (N·s/mm).
    """
    return float(_get_doublev(HD_NOMINAL_MAX_DAMPING, HDdouble).value)


def close_device(device_id: int) -> None:
    _lib_hd.hdDisableDevice.argtypes = [HHD]
    _lib_hd.hdDisableDevice(device_id)


def get_current_device() -> HHD:
    _lib_hd.hdGetCurrentDevice.restype = HHD
    return _lib_hd.hdGetCurrentDevice()


def make_current_device(device_id: int) -> None:
    _lib_hd.hdMakeCurrentDevice.argtypes = [HHD]
    _lib_hd.hdMakeCurrentDevice(device_id)


def start_scheduler() -> None:
    global _scheduler_started
    if _scheduler_started:
        return
    _lib_hd.hdStartScheduler()
    _scheduler_started = True


def stop_scheduler() -> None:
    global _scheduler_started
    if not _scheduler_started:
        return
    _lib_hd.hdStopScheduler()
    _scheduler_started = False


def enable_force() -> None:
    _lib_hd.hdEnable.argtypes = [HDenum]
    _lib_hd.hdEnable.restype = None
    _lib_hd.hdEnable(HD_FORCE_OUTPUT)
    print("Force feedback enabled!")


def begin_frame(device_id: int) -> None:
    _lib_hd.hdBeginFrame.argtypes = [HHD]
    _lib_hd.hdBeginFrame.restype = None
    _lib_hd.hdBeginFrame(device_id)


def end_frame(device_id: int) -> None:
    _lib_hd.hdEndFrame.argtypes = [HHD]
    _lib_hd.hdEndFrame.restype = None
    _lib_hd.hdEndFrame(device_id)


def _get_string(code: int) -> Optional[str]:
    _lib_hd.hdGetString.argtypes = [HDenum]
    _lib_hd.hdGetString.restype = HDstring
    value = _lib_hd.hdGetString(code)
    if value is None:
        return None
    try:
        return value.decode()
    except Exception:
        return None


def get_model() -> Optional[str]:
    return _get_string(HD_DEVICE_MODEL_TYPE)


def get_vendor() -> Optional[str]:
    return _get_string(HD_DEVICE_VENDOR)


def get_serial_number() -> Optional[str]:
    return _get_string(HD_DEVICE_SERIAL_NUMBER)
