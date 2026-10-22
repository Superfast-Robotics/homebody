"""Help the open models get through a session, each part an agent setting off by default (Sol never uses them):

local_reach_mask    the pointer sees the head camera with the pixels beyond the arm's reach greyed out
                    (rgb_masked.png beside the frame's images; vlm/composite.py's pointer takes it when present)
local_place_reach   a place click beyond reach is moved toward the robot, on the same surface, before the skill runs
local_repeat_guard  a decision identical to the one that just failed is refused with the reason, not run again

The bench (evolve/agent_place.py) measured the first two on the reference's successful placements: a usable click
(on the surface and within reach) rose from 19% to 39% (mask) and 45% (moved within reach) for Molmo2-ER."""
import json
from dataclasses import replace
from pathlib import Path

from PIL import Image

from homebody.helpers.reach import beyond, greyed, pull_within

POINT = "point_normalized_1000"
MASKED = "rgb_masked.png"
ACCEPT_MARGIN_M = 0.03  # a moved click counts as within reach up to this beyond agent.local_reach_m


def write_reach_mask(frame, folder: Path, reach_m: float) -> Path:
    """The head image with every pixel beyond REACH_M (and a little more) greyed out, saved as rgb_masked.png."""
    path = folder / MASKED
    if not path.exists():
        mask = beyond(frame.depth, frame.intrinsics, frame.world_camera, frame.world_torso, reach_m + 0.08)
        Image.fromarray(greyed(frame.rgb, mask)).save(path)
    return path


def pull_place_point(decision, frame, reach_m: float):
    """DECISION with a place click beyond reach moved within it on the same surface (unchanged otherwise), and the
    original click when it moved."""
    point = decision.arguments.get(POINT)
    if decision.skill != "place" or not isinstance(point, list) or len(point) != 2:
        return decision, None
    moved = pull_within(point, frame.depth, frame.intrinsics, frame.world_camera, frame.world_torso, frame.labels,
                        reach_m, reach_m + ACCEPT_MARGIN_M)
    if moved == list(point):
        return decision, None
    return replace(decision, arguments={**decision.arguments, POINT: moved}), list(point)


class RepeatGuard:
    """Remembers the last decision that failed; the same decision again is refused with what failed."""

    def __init__(self):
        self.failed: tuple[str, str] | None = None
        self.code = ""

    @staticmethod
    def key(decision) -> tuple[str, str]:
        return decision.skill, json.dumps(decision.arguments, sort_keys=True)

    def refusal(self, decision) -> dict | None:
        if self.failed is None or self.key(decision) != self.failed:
            return None
        return {"skill": decision.skill, "code": "INVALID_ARGUMENT",
                "message": (f"This exact {decision.skill} just failed ({self.code}); repeating it changes nothing. "
                            "Change what the result points to: stand closer or elsewhere, click another point, use "
                            "the other hand, or mark the goal failed.")}

    def observe(self, decision, result: dict):
        failed = result.get("code") not in ("OK", None)
        self.failed, self.code = (self.key(decision), result.get("code", "")) if failed else (None, "")
