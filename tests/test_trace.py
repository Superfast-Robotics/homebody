"""Per-stage traces of skill calls, and what the coder gets from them."""
import json
from types import SimpleNamespace

import numpy as np

from homebody.harness.coder import coder_request
from homebody.session.trace import Trace, last_images, summary


def backend():
    rgb = np.zeros((8, 8, 3), np.uint8)
    return SimpleNamespace(time=1.0, ego_image=lambda: rgb, wrist_view=lambda side: SimpleNamespace(rgb=rgb))


def test_a_call_records_a_keyframe_per_stage_and_its_result(tmp_path):
    carried = {}
    trace = Trace(backend(), lambda: carried)
    trace.reset(tmp_path)
    trace.on_event({"type": "skill_started", "skill": "pick"}, "right")
    trace.on_event({"type": "skill_prepared", "plan": {"grasp": "top"}}, "right")
    trace.on_event({"type": "skill_stage", "stage": "Closing hand"}, "right")
    carried["right"] = {"state": "verified"}
    trace.on_event({"type": "skill_stage", "stage": "Lifting"}, "right")
    trace.on_event({"type": "skill_finished", "result": {"skill": "pick", "code": "HOLD_UNCERTAIN", "message": "m"}}, "right")
    folder = trace.latest("pick")
    record = json.loads((folder / "trace.json").read_text())
    assert [s["stage"] for s in record["stages"]] == ["start", "Closing hand", "Lifting", "end"]
    assert record["stages"][2]["carried"] == ["right"] and record["result"]["code"] == "HOLD_UNCERTAIN"
    assert (folder / "02-lifting-head.jpg").exists() and (folder / "02-lifting-wrist-right.jpg").exists()
    assert "Lifting: holding ['right']" in summary(folder) and "result: HOLD_UNCERTAIN" in summary(folder)
    assert [p.name for p in last_images(folder)] == ["03-end-wrist-right.jpg", "03-end-head.jpg"]
    assert trace.latest("place") is None


def test_no_trace_is_kept_unless_asked(tmp_path):
    trace = Trace(backend(), dict)
    trace.reset(None)
    trace.on_event({"type": "skill_started", "skill": "pick"}, "right")
    assert trace.latest() is None and not (tmp_path / "trace").exists()


def test_the_request_carries_the_trace_and_matching_fixes():
    text = coder_request(target="pick", origin="the built-in skill", request={"goal": "g", "success": "s"}, task="t",
                         recent=[], source_root="/src", python="py", limits={"seconds": 300, "checks": 6, "executes": 2},
                         library=[], trace="pick (right)\n- t=1.00 s Lifting", fixes=["HOLD_UNCERTAIN when m: firmer"])
    assert "Trace of the failed call" in text and "- t=1.00 s Lifting" in text
    assert "Fixes that worked before" in text and "- HOLD_UNCERTAIN when m: firmer" in text
