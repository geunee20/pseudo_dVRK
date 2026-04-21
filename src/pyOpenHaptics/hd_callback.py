from ctypes import CDLL, CFUNCTYPE, POINTER, byref, c_void_p
from .hd_define import (
    HDCallbackCode,
    HD_CALLBACK_DONE,
    HD_CALLBACK_CONTINUE,
    HD_MAX_SCHEDULER_PRIORITY,
)
import functools
from .hd import (
    get_current_device,
    begin_frame,
    end_frame,
    get_error,
    make_current_device,
)
from sys import platform

_hd_callbacks: set[int] = set()  # tracks ids of functions decorated with @hd_callback

if platform == "linux" or platform == "linux2":
    _lib_hd = CDLL("libHD.so")
elif platform == "win32":
    _lib_hd = CDLL("HD.dll")


def _call_user_function(func):
    """Call either:
    1) a plain Python callback with no args, or
    2) an already-decorated CFUNCTYPE callback expecting one pUserData arg.
    """
    try:
        return func()
    except TypeError as e:
        msg = str(e)
        if "argument" in msg and "0 given" in msg:
            return func(None)
        raise


def hd_callback(input_function):
    """Backward-compatible decorator.

    This preserves the original behavior used by existing scripts:
    the decorated function can be passed directly to HapticDevice.
    """

    @functools.wraps(input_function)
    @CFUNCTYPE(HDCallbackCode, POINTER(c_void_p))
    def _callback(pUserData):
        device_id = int(get_current_device())
        begin_frame(device_id)
        _call_user_function(input_function)
        end_frame(device_id)
        if get_error():
            return HD_CALLBACK_DONE
        return HD_CALLBACK_CONTINUE

    _hd_callbacks.add(id(_callback))
    return _callback


def bind_device_callback(input_function, device_id):
    """Bind a callback to a specific device handle.

    Works with both:
    - plain Python functions (no args)
    - callbacks already decorated with @hd_callback
    """
    """Bind a servo callback to a specific haptic device handle.

    Wraps *input_function* in a ``CFUNCTYPE`` that calls
    ``hdMakeCurrentDevice(device_id)`` before each tick, ensuring the
    correct device is active in multi-device setups.

    Works with both plain Python functions (no args) and callbacks already
    decorated with ``@hd_callback`` (single ``pUserData`` arg).

    Args:
        input_function: Servo-loop function to call each haptic tick.
        device_id: Integer device handle (``HHD``) returned by :func:`~hd.init_device`.

    Returns:
        ``CFUNCTYPE``-wrapped callback suitable for the OpenHaptics scheduler.
    """

    @functools.wraps(input_function)
    @CFUNCTYPE(HDCallbackCode, POINTER(c_void_p))
    def _callback(pUserData):
        make_current_device(device_id)
        begin_frame(device_id)

        try:
            input_function()
        except TypeError:
            input_function(None)

        end_frame(device_id)

        if get_error():
            return HD_CALLBACK_DONE
        return HD_CALLBACK_CONTINUE

    return _callback


def hdAsyncSheduler(callback):
    """Schedule a callback on the OpenHaptics asynchronous (high-rate) scheduler.

    Args:
        callback: ``CFUNCTYPE``-wrapped function pointer compatible with
            ``hdScheduleAsynchronous``.
    """
    pUserData = c_void_p()
    _lib_hd.hdScheduleAsynchronous(
        callback, byref(pUserData), HD_MAX_SCHEDULER_PRIORITY
    )


def hdSyncSheduler(callback):
    """Schedule a callback on the OpenHaptics synchronous scheduler.

    Args:
        callback: ``CFUNCTYPE``-wrapped function pointer compatible with
            ``hdScheduleSynchronous``.
    """
    pUserData = c_void_p()
    _lib_hd.hdScheduleSynchronous(callback, byref(pUserData), HD_MAX_SCHEDULER_PRIORITY)
