"""Recording an event never fails a skill: plans holding frames, read-only mappings or large arrays serialise."""
import json
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from homebody.session.recorder import write_json


@dataclass(frozen=True)
class Frameish:
    hands: object
    rgb: np.ndarray


@dataclass
class Plan:
    frame: Frameish
    side: str


def test_a_plan_holding_a_frame_serialises(tmp_path):
    frame = Frameish(MappingProxyType({"left": {"closure": 0.1}}), np.zeros((480, 640, 3), np.uint8))
    write_json(tmp_path / "event.json", {"plan": Plan(frame, "left"), "kind": object()})
    written = json.loads((tmp_path / "event.json").read_text())
    assert written["plan"]["side"] == "left" and written["plan"]["frame"]["hands"]["left"]["closure"] == 0.1
    assert written["plan"]["frame"]["rgb"].startswith("array") and isinstance(written["kind"], str)
