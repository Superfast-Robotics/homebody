"""Harvesting coder fixes that worked in train runs into the library."""
import json

import pytest

from evolve import harvest_fixes


def coder_call(root, status="success"):
    call = root / "train_00" / "r0" / "session" / "s1" / "tasks" / "task-000" / "coder" / "call-1"
    (call / "workspace" / "skills").mkdir(parents=True)
    (call / "REQUEST.md").write_text("# Request: skill `pick`\n...\nresult: NO_GRASP: no grasp passed the gates\n")
    (call / "last_message.json").write_text(json.dumps({"status": status, "summary": "Widened the aperture."}))
    (call / "start_pick.py").write_text("a = 1\nb = 2\n")
    (call / "workspace" / "skills" / "pick.py").write_text("a = 1\nb = 3\n")
    return call


def test_a_successful_call_becomes_an_entry_and_a_repeat_adds_a_validation(tmp_path):
    train = tmp_path / "train"
    coder_call(train)
    fixes = tmp_path / "fixes"
    harvest_fixes.main(str(train), str(fixes))
    harvest_fixes.main(str(train), str(fixes))
    (entry,) = [json.loads(p.read_text()) for p in fixes.glob("*.json")]
    assert entry["skill"] == "pick" and entry["code"] == "NO_GRASP" and entry["validated"] == 2
    assert "+b = 3" in entry["sketch"] and entry["repair"] == "Widened the aperture."


def test_failed_calls_and_held_out_splits_are_never_harvested(tmp_path):
    coder_call(tmp_path / "train", status="failed")
    harvest_fixes.main(str(tmp_path / "train"), str(tmp_path / "fixes"))
    assert not list((tmp_path / "fixes").glob("*.json"))
    with pytest.raises(SystemExit):
        harvest_fixes.main(str(tmp_path / "val"), str(tmp_path / "fixes"))
