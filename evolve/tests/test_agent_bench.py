import json

import numpy as np
from PIL import Image

from evolve import agent_bench as bench
from homebody.vlm.provider import Decision, ProviderError

OFFERED = ["navigate", "pick", "place", "done", "plan", "goal_done", "goal_failed"]


def session(root, steps):
    """A recorded session: one frame and one decision folder per (prompt, decision, result code)."""
    folder = root / "train" / "pp_x" / "r0" / "session" / "session-20261021T000000Z-abcd1234"
    labels = np.zeros((480, 640), np.int32)
    labels[200:300, 300:400] = 7  # the can
    events = []
    for step, (prompt, decision, code) in enumerate(steps):
        frame = folder / "frames" / f"001-{step + 1:08d}"
        frame.mkdir(parents=True)
        for name in bench.IMAGES:
            Image.new("RGB", (8, 8)).save(frame / name)
        np.savez(frame / "sensors.npz", labels=labels)
        decisions = folder / "tasks" / "task-000" / "decisions" / f"step-{step:03d}"
        decisions.mkdir(parents=True)
        (decisions / "prompt.txt").write_text(prompt)
        schema = {"properties": {"skill": {"type": "string", "enum": OFFERED}}}
        (decisions / "request.json").write_text(json.dumps({"text": {"format": {"schema": schema}}}))
        events += [{"kind": "decision", "step": step, "epoch": 1, "sequence": step + 1, "decision": decision},
                   {"kind": "result", "step": step, "result": {"skill": decision["skill"], "code": code}}]
    (folder / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    return root


def test_the_turns_allowed_decisions_follow_the_plan_note():
    offered = tuple(OFFERED)
    assert bench.allowed("task...\n\nThis turn: write the plan (skill plan).", offered) == ("plan",)
    assert bench.allowed("x\n\nThe plan is done; finish with done.\n{}", offered) == ("done",)
    goal = bench.allowed("x\n\nPlan:\n{}\nActive goal g1: pick the can ", offered)
    assert goal == ("navigate", "pick", "place", "goal_done", "goal_failed", "done")


def test_collect_run_and_score(tmp_path, monkeypatch, capsys):
    pick = {"text": "", "skill": "pick", "arguments": {"point_normalized_1000": [550, 500], "hand": "right"}}
    root = session(tmp_path / "runs", [("p\n\nThis turn: write the plan (skill plan).",
                                        {"text": "", "skill": "plan", "arguments": {"tree": {}}}, "OK"),
                                       ("p\n\nPlan:\n{}\nActive goal g1: pick ", pick, "OK")])
    bench.main(["collect", str(tmp_path / "cases"), str(root)])
    cases = sorted((tmp_path / "cases").glob("*/case.json"))
    meta = [json.loads(p.read_text()) for p in cases]
    assert [m["allowed"] for m in meta][0] == ["plan"] and meta[1]["target"] == 7 and meta[0]["target"] is None

    replies = iter([ProviderError("Invalid structured decision: Expecting ',' delimiter"),
                    Decision("", "pick", {"point_normalized_1000": [10, 10]})])

    class Fake:
        def __init__(self):
            self.allowed = None

        def turn(self, allowed):
            self.allowed = allowed

        def generate(self, prompt, *, images, names, attempt, tick, cancelled):
            assert len(images) == 4 and "plan" in names
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply

    monkeypatch.setattr(bench, "make", lambda settings: Fake())
    bench.main(["run", str(tmp_path / "cases"), str(tmp_path / "out"), "--model", "qwen3vl-8b-vllm", "--jobs", "1"])
    stats = bench.score_run(tmp_path / "out")
    assert stats["turns"] == 2 and stats["parsed"] == 1 and stats["allowed"] == 1 and stats["same_skill"] == 1
    assert (stats["picks_scored"], stats["pick_hits"], stats["pick_background"]) == (1, 0, 1)
    assert stats["errors"] == {"Invalid structured decision": 1}


def test_a_click_reads_the_label_under_the_pixel():
    labels = np.zeros((480, 640), np.int32)
    labels[200:300, 300:400] = 7
    assert bench.click({"point_normalized_1000": [550, 500]}, labels) == 7
    assert bench.click({"point_normalized_1000": [10, 10]}, labels) == "background"
    assert bench.click({"point_normalized_1000": [0.0, 1200]}, labels) == "invalid"
