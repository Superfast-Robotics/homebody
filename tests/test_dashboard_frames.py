"""The camera frame check: invalid, suspected invalid and valid frames, by explicit configured rules."""
import numpy as np
from fakes import fake_dashboard

from homebody.session.config import Dashboard
from homebody.ui.dashboard.frames import FrameCheck


def scene(seed=0):
    return np.random.default_rng(seed).integers(0, 255, (64, 80, 3), dtype=np.uint8)


def test_blank_dimensionless_and_undecodable_frames_are_invalid_and_a_textured_one_valid():
    check = FrameCheck(Dashboard())
    assert check.quality.state == "unknown"
    assert check.check(scene(), 1.0, 0.1).state == "valid"
    assert check.check(np.zeros((64, 80, 3), np.uint8), 2.0, 0.2).state == "invalid"
    assert check.check(np.zeros((0, 80, 3), np.uint8), 3.0, 0.3).reason == "Frame without dimensions"
    check.undecodable(4.0, "bad zlib")
    assert check.quality.state == "invalid" and "bad zlib" in check.quality.reason
    assert check.check(scene(1), 5.0, 0.5).state == "valid" and check.quality.problems == 3


def test_a_nearly_uniform_or_frozen_image_is_only_suspected():
    check = FrameCheck(Dashboard(frame_uniform_std=2.0, frame_repeat_s=10.0))
    flat = np.full((64, 80, 3), 120, np.uint8)
    flat[0, 0] = 121  # not blank: one pixel differs
    assert check.check(flat, 1.0, 0.0).state == "suspected_invalid"
    frozen = scene(2)
    assert check.check(frozen, 2.0, 1.0).state == "valid"
    assert check.check(frozen, 3.0, 5.0).state == "valid"  # repeating, but not yet for 10 s of simulated time
    quality = check.check(frozen, 4.0, 12.0)
    assert quality.state == "suspected_invalid" and "renderer may have stopped" in quality.reason
    assert check.check(frozen, 5.0, None).state == "valid"  # without simulated time, repetition proves nothing


def test_the_dashboard_checks_robot_frames_and_stamps_readings_with_the_quality():
    dashboard = fake_dashboard()
    dashboard.remote_frame("robot", np.zeros((32, 32, 3), np.uint8), {"sim": 1.0})
    assert dashboard._snapshot()["frame_quality"]["state"] == "invalid"
    dashboard.monitor({"level": "expected", "kind": "observation", "source": "m", "risk": 0.1})
    assert dashboard.store.readings[-1].frame_quality == "invalid"
    dashboard.remote_frame("observer", np.zeros((32, 32, 3), np.uint8), {})  # only the robot camera counts
    dashboard.remote_frame("robot", scene(), {"sim": 2.0})
    assert dashboard._snapshot()["frame_quality"]["state"] == "valid"
