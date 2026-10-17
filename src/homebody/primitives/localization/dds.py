"""Process-wide Unitree DDS domain initialization."""
from __future__ import annotations

import threading


class _Domain:
    """The one DDS domain this process joins, on one robot-LAN interface."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.interface: str | None = None


_DOMAIN = _Domain()


def initialize(interface_name: str) -> None:
    interface = str(interface_name).strip()
    if not interface or interface == "auto":
        raise ValueError("DDS requires one resolved robot-LAN interface")
    with _DOMAIN.lock:
        if _DOMAIN.interface is not None:
            if _DOMAIN.interface != interface:
                raise RuntimeError(
                    f"DDS already uses {_DOMAIN.interface!r}; refusing a second interface "
                    f"{interface!r}"
                )
            return
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        ChannelFactoryInitialize(0, interface)
        _DOMAIN.interface = interface
