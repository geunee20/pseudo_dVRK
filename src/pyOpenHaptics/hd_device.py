from .hd import (
    get_current_device,
    init_device,
    make_current_device,
    get_vendor,
    get_model,
    get_serial_number,
    enable_force,
    get_error,
    close_device,
)
from .hd_callback import bind_device_callback, hdAsyncSheduler, hdSyncSheduler
from .hd_define import HD_BAD_HANDLE, HD_INVALID_HANDLE


class HapticDevice(object):
    """High-level wrapper around a single PHANTOM haptic device.

    Initialises the device via ``hdInitDevice``, enables force output,
    and schedules a C-compatible servo callback using either the
    asynchronous or synchronous OpenHaptics scheduler.

    Args:
        callback: Servo-loop function to call at haptic-rate (typically 1 kHz).
            May be a plain Python function or a ``@hd_callback``-decorated callable.
        device_name: Device name as configured in the OpenHaptics driver.
        scheduler_type: ``"async"`` for asynchronous scheduling (default)
            or ``"sync"`` for synchronous.
        bind_callback_to_device: If ``True`` (default), wrap *callback* so it
            explicitly makes this device current before each tick.
    """

    def __init__(
        self,
        callback,
        device_name: str = "Default Device",
        scheduler_type: str = "async",
        bind_callback_to_device: bool = True,
    ):
        """Initialise the haptic device, enable force output, and start the scheduler.

        Args:
            callback: Servo-loop function invoked each haptic tick.
            device_name: OpenHaptics device name (default ``"Default Device"``).
            scheduler_type: ``"async"`` (high-rate) or ``"sync"`` (application-rate) scheduler.
            bind_callback_to_device: If ``True``, wrap *callback* with
                :func:`~hd_callback.bind_device_callback` so the correct device
                is made current before every tick.

        Raises:
            RuntimeError: If the device handle is invalid or vendor/model cannot be queried.
        """
        print("Initializing haptic device with name {}".format(device_name))
        current_id = get_current_device()

        self.id = init_device(device_name)
        if self.id in (None, HD_BAD_HANDLE, HD_INVALID_HANDLE):
            raise RuntimeError(f"Unable to initialize device '{device_name}'")

        if current_id == self.id:
            raise RuntimeError(f"Device '{device_name}' is already initialized.")

        make_current_device(self.id)

        vendor = self.__vendor__()
        model = self.__model__()
        if vendor is None or model is None:
            raise RuntimeError(
                f"Initialized '{device_name}' handle={self.id}, but vendor/model could not be queried."
            )

        serial = self.__serial__()
        if serial:
            print(f"Intialized device! {vendor}/{model} (serial={serial})")
        else:
            print(f"Intialized device! {vendor}/{model}")

        enable_force()
        if get_error():
            raise RuntimeError(
                f"OpenHaptics reported an error after starting scheduler for '{device_name}'"
            )

        if bind_callback_to_device:
            self.callback = bind_device_callback(callback, self.id)
        else:
            self.callback = callback
        self.scheduler(self.callback, scheduler_type)

    def close(self):
        """Close the device handle and release driver resources."""
        close_device(self.id)

    def scheduler(self, callback, scheduler_type):
        """Schedule the haptic callback using the specified OpenHaptics scheduler.

        Args:
            callback: C-compatible ``CFUNCTYPE`` callback pointer.
            scheduler_type: ``"async"`` or ``"sync"``.
        """
        if scheduler_type == "async":
            hdAsyncSheduler(callback)
        else:
            hdSyncSheduler(callback)

    @staticmethod
    def __vendor__() -> str | None:
        """Return the vendor string of this device."""
        return get_vendor()

    @staticmethod
    def __model__() -> str | None:
        """Return the model type string of this device."""
        return get_model()

    @staticmethod
    def __serial__() -> str | None:
        """Return the serial number string of this device, or ``None`` if unavailable."""
        try:
            return get_serial_number()
        except Exception:
            return None
