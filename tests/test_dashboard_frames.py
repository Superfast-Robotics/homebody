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


def test_a_reading_carries_the_frame_quality_at_its_own_observation_time_not_the_current_one():
    dashboard = fake_dashboard()
    dashboard.remote_frame("robot", np.zeros((32, 32, 3), np.uint8), {"sim": 1.0})
    dashboard.remote_frame("robot", scene(), {"sim": 2.0})
    dashboard.remote_frame("robot", scene(3), {"sim": 3.0})
    assert dashboard._snapshot()["frame_quality"]["state"] == "valid"
    dashboard.monitor({"level": "anomaly", "kind": "observation", "source": "m", "risk": 0.9, "sim_time": 1.02})
    late = dashboard.store.readings[-1]
    assert late.frame_quality == "invalid" and "sim" in late.quality_source and late.quality_reason.startswith("Blank")
    dashboard.monitor({"level": "expected", "kind": "observation", "source": "m", "risk": 0.1})
    unmatched = dashboard.store.readings[-1]  # no simulation time: the check at receipt, said so
    assert unmatched.frame_quality == "valid" and "receipt" in unmatched.quality_source
    dashboard.monitor({"level": "expected", "kind": "observation", "source": "m", "risk": 0.1, "sim_time": 40.0})
    assert dashboard.store.readings[-1].frame_quality == "unknown"  # no checked frame near that time
    assert late.id != unmatched.id and late.id.startswith("o")


def test_the_check_keeps_each_frames_state_by_simulation_time():
    check = FrameCheck(Dashboard())
    check.check(scene(), 1.0, 10.0)
    check.check(np.zeros((64, 80, 3), np.uint8), 2.0, 10.5)
    check.check(scene(1), 3.0, 11.0)
    assert check.at(10.45).state == "invalid" and check.at(10.9).state == "valid"
    assert check.at(10.45).sim_time == 10.5 and check.at(10.45).how == "nearest"
    assert check.at(30.0) is None and check.at(None) is None


def test_an_undecodable_frame_breaks_its_run_so_a_reading_in_that_gap_is_not_called_valid():
    check = FrameCheck(Dashboard())
    for i in range(11):
        check.check(scene(i), float(i), 10.0 + i * 0.2)  # valid frames from sim 10 to 12
    check.undecodable(20.0, "bad zlib")
    for i in range(6):
        check.check(scene(i), 21.0 + i, 12.6 + i * 0.2)  # valid again from sim 12.6
    gap = check.at(12.3)
    assert gap.state == "invalid" and gap.how == "undecodable" and "bad zlib" in gap.reason
    assert check.at(11.0).how == "covered" and check.at(11.0).state == "valid"
    check.check(scene(9), 30.0, 15.0)  # 1.4 s after the last frame: a new run, the time between is not covered
    assert check.at(14.5).how == "nearest"


def test_an_incident_keeps_its_detecting_observation_and_the_page_gets_what_it_needs_to_find_evidence():
    dashboard = fake_dashboard()
    dashboard.remote_frame("robot", scene(), {"sim": 1.0})
    dashboard.monitor({"level": "anomaly", "kind": "observation", "source": "m", "risk": 0.9, "sim_time": 1.0})
    dashboard.monitor({"level": "anomaly", "kind": "observation", "source": "m", "risk": 1.0, "sim_time": 1.0})
    (incident,) = dashboard.store.incidents.values()
    first, newest = dashboard.store.readings
    assert incident.evidence["first_observation"]["id"] == first.id == incident.evidence["observation_id"]
    assert incident.observation["id"] == newest.id and incident.evidence["risk"] == 1.0  # the peak, kept apart
    snapshot = dashboard._snapshot()
    assert snapshot["history_period"] > 0 and snapshot["keep_s"] == 600
