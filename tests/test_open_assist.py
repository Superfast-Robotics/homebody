from dataclasses import dataclass

import numpy as np

from homebody.helpers import reach
from homebody.session.open_assist import RepeatGuard, pull_place_point, write_reach_mask
from homebody.vlm.provider import Decision


@dataclass
class View:
    """A camera at the torso looking down at a table 0.7 m below: image rows run from 0.9 m ahead (top) to 0.2 m
    (bottom), all of it the table (label 3)."""
    h: int = 48
    w: int = 64

    def __post_init__(self):
        self.intrinsics = np.array([[60.0, 0, self.w / 2], [0, 60.0, self.h / 2], [0, 0, 1]])
        # optical axis straight down; image up is world +x (ahead), image right is world -y
        self.world_camera = np.array([[0, -1, 0, 0.55], [-1, 0, 0, 0], [0, 0, -1, 0.7], [0, 0, 0, 1]], float)
        self.world_torso = np.eye(4)
        self.depth = np.full((self.h, self.w), 0.7, np.float32)
        self.labels = np.full((self.h, self.w), 3, np.int32)
        self.rgb = np.full((self.h, self.w, 3), 200, np.uint8)


def test_the_view_geometry_and_the_beyond_reach_mask(tmp_path):
    view = View()
    near, far = reach.world_of([500, 1000], view.depth, view.intrinsics, view.world_camera), reach.world_of(
        [500, 0], view.depth, view.intrinsics, view.world_camera)
    assert reach.reach_of(near, view.world_torso) < 0.3 < 0.8 < reach.reach_of(far, view.world_torso)
    image = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(write_reach_mask(view, tmp_path, 0.42)))
    assert image[0, 32].tolist() != [200, 200, 200] and image[-1, 32].tolist() == [200, 200, 200]


def test_a_far_place_click_moves_within_reach_on_the_same_surface():
    view = View()
    decision = Decision("put it there", "place", {"hand": "right", "point_normalized_1000": [500, 100]})
    moved, original = pull_place_point(decision, view, 0.42)
    assert original == [500, 100] and moved.arguments["hand"] == "right"
    seen = reach.world_of(moved.arguments["point_normalized_1000"], view.depth, view.intrinsics, view.world_camera)
    assert reach.reach_of(seen, view.world_torso) <= 0.45
    near = Decision("put it here", "place", {"hand": "right", "point_normalized_1000": [500, 900]})
    assert pull_place_point(near, view, 0.42) == (near, None)
    assert pull_place_point(Decision("go", "pick", {"point_normalized_1000": [500, 100]}), view, 0.42)[1] is None


def test_the_repeat_guard_forgets_after_a_success():
    guard, failing = RepeatGuard(), Decision("x", "place", {"hand": "right"})
    guard.observe(failing, {"code": "ARM_NOT_REACHED"})
    assert "ARM_NOT_REACHED" in guard.refusal(failing)["message"]
    assert guard.refusal(Decision("x", "place", {"hand": "left"})) is None
    guard.observe(Decision("y", "navigate", {}), {"code": "OK"})
    assert guard.refusal(failing) is None


def test_the_repeat_guard_lets_a_retryable_failure_be_retried():
    guard, retry = RepeatGuard(), Decision("x", "navigate", {"goal_xy_m": [1, 2]})
    guard.observe(retry, {"code": "STALL"})
    assert guard.refusal(retry) is None
    guard.observe(retry, {"code": "NO_ROUTE"})
    assert "NO_ROUTE" in guard.refusal(retry)["message"]
