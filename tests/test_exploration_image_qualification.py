"""Current-camera sampling and immutable gallery evidence have separate qualifications."""
import io
import json
import math
import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from homebody.primitives.navigation.errors import Stopped
from homebody.real2sim.exploration import decision_context, settings, vlm_pick
from homebody.real2sim.exploration import explorer as explorer_module
from homebody.real2sim.exploration.livemap import Boundary
from homebody.real2sim.exploration.visual_history import completion_refusal


class Clock:
    value = 100.

    def time(self):
        return self.value

    monotonic = time

    def wait(self, seconds):
        self.value += seconds
        return False


def frozen_decision(tmp_path, monkeypatch, sequences, *, navigation_pose=None,
                    previous_action=None, continuation=None):
    clock = Clock()
    monkeypatch.setattr(explorer_module, "time", clock)
    received = {}
    frames = []
    sequence_iterator = iter(sequences)

    def camera():
        sequence = next(sequence_iterator)
        decoded = received.setdefault(sequence, clock.time() - .01)
        rgb = np.zeros((30, 40, 3), dtype=np.uint8)
        rgb[::2, ::2] = 30 * sequence
        frames.append(rgb)
        return rgb, {"status": "fresh", "sequence": sequence,
                     "receiver_decode_time_s": decoded,
                     "receive_age_s": clock.time() - decoded}

    source = SimpleNamespace(
        stop_flag=SimpleNamespace(is_set=lambda: False, wait=clock.wait),
        pause_flag=threading.Event(), attach=lambda *_: None,
        pose=lambda: (0., 0., 0.), base_xy=lambda: (0., 0.), now=clock.time,
        odom_age_s=lambda: .1, health_stable_s=lambda: math.inf,
        context_image=camera, camera_description="uncalibrated iPhone RGB")
    owner = explorer_module.Explorer(
        source, Boundary(((-2., -2.), (2., -2.), (2., 2.), (-2., 2.))), tmp_path)
    owner.brief = "Explore the workspace."
    owner.map_age_s = lambda: .1
    if navigation_pose is not None:
        source.navigation_pose = navigation_pose
    owner.previous_action_result = previous_action
    overview = np.full((30, 40, 3), 80, np.uint8)
    local = np.full((20, 30, 3), 90, np.uint8)
    transform = np.eye(3)
    monkeypatch.setattr(vlm_pick, "render", lambda *_: (overview, transform, None))
    monkeypatch.setattr(vlm_pick, "local_map", lambda *_, **__: (local, transform))
    evidence = decision_context.build_context(
        owner, 0., 0., 0., clock=clock, continuation=continuation)
    prompt = evidence.request.prompt
    observation = json.loads(prompt.split("Current observation:\n")[1].split("\n\nViews:")[0])
    history = json.loads(prompt.split("Views:\n")[1].split("\n\nRecent results:")[0])
    return owner, evidence, observation, history, frames, (overview, local)


def test_live_fallback_does_not_disqualify_independent_gallery(tmp_path, monkeypatch):
    owner, evidence, observation, history, frames, maps = frozen_decision(
        tmp_path, monkeypatch, [1, 2, 3, 4, 5, 5, 5])
    assert observation["current_camera"] == {
        "image_index": 0, "selection": {
            "status": "latest_fallback", "reason": "insufficient distinct phone frames"}}
    assert "current_view_unqualified" not in observation
    assert history["current_rgb_is_separate"]
    assert history["admission"] == "saved new immutable image"
    key = history["current_viewpoint"]
    assert history["view_ids"] == [key]
    assert history["readable_ids"] == []
    record, = history["images"]
    assert record["id"] == key and record["usability"] == "pending"
    assert not record["advisory_only"]
    assert record["selection"]["status"] == "sharpest_stable"
    assert len(evidence.request.images) == 4  # current, gallery sheet, overview, local
    decoded = [np.asarray(Image.open(io.BytesIO(data))) for data in evidence.request.images]
    np.testing.assert_array_equal(decoded[0], frames[-1])
    for actual, expected in zip(decoded[-2:], maps, strict=True):
        np.testing.assert_array_equal(actual, expected)
    saved = json.loads((tmp_path / "vlm/context_leg_1.json").read_text())
    assert saved["history"]["view_ids"] == history["view_ids"]
    assert saved["images"] == ["leg_1_camera.png", "leg_1_history_1.png",
                               "leg_1.png", "leg_1_local.png"]
    assert owner.visual_history.records[0]["usability"] == "pending"


def test_insufficient_gallery_samples_remain_unadmitted(tmp_path, monkeypatch):
    _, evidence, observation, history, _, _ = frozen_decision(
        tmp_path, monkeypatch, [1] * 7)
    assert observation["current_camera"]["selection"]["status"] == "latest_fallback"
    assert history["current_viewpoint"] is None
    assert history["view_ids"] == history["images"] == history["readable_ids"] == []
    assert history["admission"] == "gallery needs a stable held sample; live fallback stays separate"
    assert len(evidence.request.images) == 3


def test_post_capture_departure_rechecks_navigation_centre_not_gallery_pose(tmp_path, monkeypatch):
    calls = []

    def navigation_pose():
        calls.append("navigation_pose")
        return (.4, .5, .6)

    previous = {"kind": "recovery", "code": None, "why": "departed",
                "details": {"departure_verdict": "admitted", "outcome": "departure_restored"}}
    _, evidence, observation, history, _, _ = frozen_decision(
        tmp_path, monkeypatch, [1, 2, 3, 4, 5, 5, 5],
        navigation_pose=navigation_pose, previous_action=previous)
    assert calls == ["navigation_pose"]
    assert evidence.position == (.4, .5, .6)
    assert history["images"][0]["robot_pose_xyyaw"] == [0., 0., 0.]
    assert observation["departure"]["status"] == "starting_footprint_blocked"
    results = json.loads(evidence.request.prompt.split("Recent results:\n")[1])
    assert results["departure_handoff"]["usable_now"] is False
    assert results["departure_handoff"]["checked_after_capture"]
    assert previous["details"]["departure_verdict"] == "admitted"  # Preserve the historical result.
    saved = json.loads((tmp_path / "vlm/context_leg_1.json").read_text())
    assert saved["robot_pose_xyyaw"] == [.4, .5, .6]
    assert saved["current_departure"] == observation["departure"]


@pytest.mark.parametrize("delayed_pose_read", [1, 2])
def test_frozen_current_image_expires_during_pose_recovery_but_gallery_is_retained(
        tmp_path, monkeypatch, delayed_pose_read):
    pose_reads = []

    def navigation_pose():
        pose_reads.append(explorer_module.time.time())
        if len(pose_reads) == delayed_pose_read:
            explorer_module.time.wait(10.)
        return (0., 0., 0.)

    with pytest.raises(Stopped, match="selected current camera view expired"):
        frozen_decision(tmp_path, monkeypatch, [1, 2, 3, 4, 5, 5, 5],
                        navigation_pose=navigation_pose,
                        continuation={"reason": "held reassessment"})
    assert len(pose_reads) == delayed_pose_read
    saved = json.loads(next((tmp_path / "views").glob("epoch_*/index.json")).read_text())
    record, = saved["views"]
    assert record["phone"]["selection"]["status"] == "sharpest_stable"
    assert record["usability"] == "pending" and not record["advisory_only"]
    assert len(list((tmp_path / "views").glob("epoch_*/*.png"))) == 1


@pytest.mark.parametrize(("delay", "accepted"), [(.1, True), (.2, False)])
def test_frozen_image_uses_configured_camera_age_without_resampling(
        tmp_path, monkeypatch, delay, accepted):
    configured = settings.current()
    monkeypatch.setitem(settings._installed, "settings", replace(
        configured, iphone=replace(configured.iphone, max_age_s=.6)))

    def navigation_pose():
        explorer_module.time.wait(delay)
        return (0., 0., 0.)

    if not accepted:
        with pytest.raises(Stopped, match="limit 0.6 s"):
            frozen_decision(tmp_path, monkeypatch, [1, 2, 3, 4, 5, 5, 5],
                            navigation_pose=navigation_pose)
        return
    _, evidence, _, _, frames, _ = frozen_decision(
        tmp_path, monkeypatch, [1, 2, 3, 4, 5, 5, 5], navigation_pose=navigation_pose)
    assert evidence.camera_metadata["receive_age_s"] == pytest.approx(.41 + delay)
    assert len(frames) == 7


def test_admission_requires_assessment_and_corrected_ledger_retains_requested_areas(
        tmp_path, monkeypatch):
    owner, _, _, history, _, _ = frozen_decision(
        tmp_path, monkeypatch, [1, 2, 3, 4, 5, 5, 5])
    gallery = owner.visual_history
    key = history["current_viewpoint"]
    area = {"area": "room", "requirement": "walls and openings",
            "complementary_required": False, "scope": "included", "status": "covered",
            "evidence_views": [key], "complementary_views": [], "gaps": [],
            "resolution": "Walls and openings captured."}
    assert gallery.apply_coverage([area], [key]) == (
        "covered area requires readable, non-advisory images")
    assert gallery.apply_image_assessments(
        [{"id": key, "usability": "readable", "reason": "Walls and openings are visible."}],
        [key]) == ""
    old = area | {"requirement": "all sides", "status": "needs_view",
                  "gaps": ["far side"], "resolution": ""}
    assert gallery.apply_coverage([old], [key]) == ""
    assert gallery.apply_coverage([area | {"resolution": ""}], [key]) == (
        "scope or requirement correction requires an explicit resolution")
    assert gallery.apply_coverage([area], [key]) == ""
    completion = {"coverage_summary": "Room geometry captured.",
                  "evidence_views": [key], "remaining_gaps": [],
                  "limitations": ["The far side was not recorded."]}
    checks = {"current_fresh": True, "same_epoch": True, "observe_only": False,
              "readable_ids": gallery.readable_ids()}
    assert completion_refusal(completion, [key], coverage=gallery.coverage_assessment, **checks) == ""
    unseen = area | {"area": "adjoining room", "status": "needs_view",
                     "evidence_views": [], "gaps": ["room not yet observed"], "resolution": ""}
    assert gallery.apply_coverage([unseen], [key]) == ""
    assert "adjoining room" in completion_refusal(
        completion, [key], coverage=gallery.coverage_assessment, **checks)
