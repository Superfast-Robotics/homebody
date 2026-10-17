"""The G1 remote's buttons as Holosoma lifecycle commands: Y initial pose, A deploy, B stop.

Rising-edge, key for key; L1+R1 kills. The sticks are ignored: the skills own the base
velocity through the lower_body channel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class WirelessPort(Protocol):
    def wireless_message(self): ...

    @staticmethod
    def wireless_key(message) -> str | None: ...


@dataclass(frozen=True)
class WirelessSample:
    """The commands newly pressed in one SDK sample, in the remote's own words."""

    commands: tuple[str, ...]
    key: str | None
    neutral: bool = False


# Remote key -> vendored ControllerCommand member.
COMMAND_BY_KEY = {
    "A": "start",
    "B": "stop",
    "Y": "initialize",
    "up": "stiffness_up",
    "down": "stiffness_down",
    "left": "stiffness_down_fine",
    "right": "stiffness_up_fine",
    "F1": "stiffness_reset",
    "L1+R1": "kill",
    "start": "toggle_walk",
    "L2": "zero_velocity",
}


class UnitreeWirelessInput:
    """Rising-edge detection over the remote's key codes."""

    def __init__(self) -> None:
        self._key_states: dict[str, bool] = {}

    def poll(self, port: WirelessPort) -> WirelessSample:
        message = port.wireless_message()
        previous = self._key_states.copy()
        key = None
        if message is not None:
            key = port.wireless_key(message)
            if key:
                # Pressed keys stay set until a no-button sample.
                self._key_states[key] = True
            else:
                self._key_states = dict.fromkeys(self._key_states, False)
        commands = tuple(
            command for name, command in COMMAND_BY_KEY.items()
            if self._key_states.get(name, False) and not previous.get(name, False)
        )
        return WirelessSample(commands=commands, key=key,
                              neutral=message is not None and getattr(message, "keys", None) == 0)
