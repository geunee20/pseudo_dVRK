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
    def __init__(
        self,
        callback,
        device_name: str = "Default Device",
        scheduler_type: str = "async",
        bind_callback_to_device: bool = True,
    ):
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
        close_device(self.id)

    def scheduler(self, callback, scheduler_type):
        if scheduler_type == "async":
            hdAsyncSheduler(callback)
        else:
            hdSyncSheduler(callback)

    @staticmethod
    def __vendor__() -> str | None:
        return get_vendor()

    @staticmethod
    def __model__() -> str | None:
        return get_model()

    @staticmethod
    def __serial__() -> str | None:
        try:
            return get_serial_number()
        except Exception:
            return None
