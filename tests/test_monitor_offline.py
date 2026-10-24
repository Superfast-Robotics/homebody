"""The offline evaluator: the live monitor's rules over a recorded rollout's frames and events, writing beside (never
over) the live monitor's logs."""
import json
from dataclasses import asdict, replace

import numpy as np
from PIL import Image

from homebody.monitor.offline import FOLDER, estimate, evaluate
from homebody.monitor.targets import Label, NodeInfo, Reading, Vocabulary, VocabularyRequest
from homebody.session.config import Monitor, Rollouts
from homebody.session.settings import Settings

NODE = NodeInfo("g1", "pick the cup", None, "active")
PLAN = {"tree": {"type": "sequence", "children": [{"type": "goal", "id": "g1", "goal": "pick the cup", "check": None}]},
        "status": {"g1": "active"}, "reasons": {}, "flags": {}, "active": "g1", "used": 0, "results": []}
VOCAB = Vocabulary("g1", (Label("reaching", "progress", "reaches"), Label("missed", "failure", "misses the cup")))
SETTINGS = Settings(monitor=Monitor(enabled=True, window_frames=2, ticks=2, ema=1.0, grace_s=0.0, cooldown_s=0.0,
                                    refresh_on_skill=False),
                    rollouts=Rollouts(offline_period_s=0.5))


class Vocabularies:
    def __init__(self):
        self.requests = []

    def generate(self, request, attempt, cancelled):
        self.requests.append(request)
        return replace(VOCAB, goal_id=request.node.goal_id)


class Decisions:
    """The cup is missed from simulation time FAIL_AFTER on."""

    def __init__(self, fail_after=2.0):
        self.fail_after, self.requests = fail_after, []

    def decide(self, request, lag=0.0):
        self.requests.append(request)
        newest = max(shot.sim_time for shot in request.shots)
        activity = {"missed": 0.95} if newest >= self.fail_after else {"reaching": 0.9}
        return Reading(request.node.goal_id, newest, activity=activity, phase=request.phase, input_tokens=1000,
                       cost_usd=0.0001), {"recorded": True}


def rollout(tmp_path, *, turns=True, seconds=4.0, hz=2.0):
    """A recorded task: a turn on goal g1, a pick call with two stages, and frames at HZ over SECONDS."""
    root = tmp_path / "session-x"
    task = root / "tasks/task-000"
    (task / "monitor").mkdir(parents=True)
    (task / "monitor/readings.jsonl").write_text('{"live": true}\n')
    records = [{"kind": "task_started", "task": "pick the cup", "epoch": 1, "path": "tasks/task-000",
                "skills": ["pick: grasp an object"]}]
    first = None
    for index in range(int(seconds * hz)):
        t = index / hz
        folder = root / f"frames/001-{index:08d}"
        folder.mkdir(parents=True)
        Image.fromarray(np.full((8, 8, 3), index, np.uint8)).save(folder / "rgb.png")
        records.append({"kind": "frame", "epoch": 1, "simulation_time": t, "path": f"frames/001-{index:08d}"})
        if index == 0:
            first = folder / "rgb.png"
            if turns:
                request = VocabularyRequest("pick the cup", "g1 pick the cup [ACTIVE]", NODE, (), ("pick",),
                                            "both hands are free", (first,))
                data = asdict(request)
                data["images"] = [str(first.relative_to(root))]
                records.append({"kind": "turn", "step": 0, "simulation_time": t, "plan": PLAN,
                                "hands": "both hands are free", "carried": [], "request": data})
                skill = asdict(replace(request, skill="pick", arguments={"hand": "right"}, images=()))
                records.append({"kind": "skill_request", "simulation_time": t, "request": skill})
            records.append({"kind": "skill", "simulation_time": t, "event": {"type": "skill_started", "skill": "pick"}})
            records.append({"kind": "skill", "simulation_time": t,
                            "event": {"type": "skill_stage", "skill": "pick", "stage": "Approaching"}})
    records.append({"kind": "skill", "simulation_time": seconds,
                    "event": {"type": "skill_finished", "result": {"skill": "pick", "code": "NO_GRASP"}}})
    records.append({"kind": "task_finished", "status": "model_done", "steps": 2})
    (root / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return root


def test_the_offline_monitor_flags_what_the_live_one_would_have_and_leaves_the_live_logs_alone(tmp_path):
    root = rollout(tmp_path)
    vocabularies, decisions = Vocabularies(), Decisions(fail_after=2.0)
    result = evaluate(root, SETTINGS, vocabularies, decisions)
    report = result["tasks"][0]
    assert report["recorded_turns"] and report["frames"] == 8 and report["frame_hz"] == 2.0
    assert vocabularies.requests[0].node.goal_id == "g1" and vocabularies.requests[0].images[0].is_file()
    would = report["would"]
    assert would and would[0]["action"] == "would_abort_skill" and would[0]["goal_id"] == "g1"
    assert would[0]["label"] == "missed" and would[0]["sim_time"] >= 2.0
    folder = root / "tasks/task-000" / FOLDER
    assert (folder / "readings.jsonl").read_text().count("\n") == report["calls"] == len(decisions.requests)
    assert json.loads((folder / "report.json").read_text())["would"] == would
    assert (root / "tasks/task-000/monitor/readings.jsonl").read_text() == '{"live": true}\n'
    assert all(request.phase.skill == "pick" for request in decisions.requests[1:])


def test_a_dry_run_counts_the_calls_the_evaluation_makes():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as scratch:
        root = rollout(Path(scratch))
        planned = estimate(root, SETTINGS)
        decisions = Decisions(fail_after=99)
        evaluate(root, SETTINGS, Vocabularies(), decisions)
    assert planned["calls"] == len(decisions.requests) == 7  # the first frame comes before the first goal
    assert planned["tasks"][0]["frames"] == 8 and planned["usd"] > 0


def test_a_rollout_without_turns_is_watched_with_the_task_as_its_goal(tmp_path):
    root = rollout(tmp_path, turns=False)
    vocabularies, decisions = Vocabularies(), Decisions(fail_after=99)
    report = evaluate(root, SETTINGS, vocabularies, decisions)["tasks"][0]
    assert not report["recorded_turns"] and report["would"] == []
    assert vocabularies.requests[0].node.goal_id == "task" and decisions.requests


def test_the_offline_spend_stops_at_its_cap(tmp_path):
    root = rollout(tmp_path, seconds=10.0)
    capped = replace(SETTINGS, rollouts=Rollouts(offline_period_s=0.5, offline_max_cost_usd=0.0003))
    decisions = Decisions(fail_after=99)
    report = evaluate(root, capped, Vocabularies(), decisions)["tasks"][0]
    assert len(decisions.requests) == 3 and report["calls"] == 3
