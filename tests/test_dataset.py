"""Extract dataset over a real dashboard: the sample types in value order with what this session holds of each, and
an extraction that writes linked, time-stamped samples with the camera frames they point to."""
import json
from types import SimpleNamespace

from fakes import fake_dashboard

from homebody.ui.dashboard import dataset

ORDER = ["failure_recovery", "monitor_incidents", "human_interventions", "robot_state_actions", "task_language", "rgb"]


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def session(tmp_path):
    """A task in which a pick fails, the monitor flags it, the operator speaks and the second pick succeeds."""
    clock = Clock()
    dashboard = fake_dashboard(clock=clock)
    dataset.attach(dashboard, tmp_path / "datasets")
    store = dashboard.store
    store.update("Ready")
    store.update("Thinking", task="Pick the jam jar", robot={"hands": {"left": "free", "right": "free"}})
    for second in range(30):  # a camera frame every second
        dashboard.history.add(clock.t + second, b"jpeg-%d" % second, 100.0 + second)
    clock.t += 1
    store.decision(SimpleNamespace(skill="pick", text="Pick the jar", arguments={"hand": "right"}), goal="g1")
    store.update("Executing", "Grasping")
    store.skill({"type": "skill_started", "skill": "pick"})
    clock.t += 4
    dashboard.monitor({"level": "elevated", "risk": 0.9, "threshold": 0.8, "signal": "failure:dropped",
                       "reason": "dropped p=0.90", "kind": "observation", "activity": "dropping_jar", "activity_p": 0.9})
    store.skill({"type": "skill_finished", "result": SimpleNamespace(skill="pick", code="NO_GRASP", message="missed")})
    store.update("Thinking")
    clock.t += 2
    assert dashboard.command({"type": "submit", "text": "Use the right hand"})["ok"]
    store.decision(SimpleNamespace(skill="pick", text="Pick again", arguments={"hand": "right"}), goal="g1")
    store.update("Executing", "Grasping again", robot={"hands": {"left": "free", "right": "held"}})
    store.skill({"type": "skill_started", "skill": "pick"})
    clock.t += 5
    store.skill({"type": "skill_finished", "result": SimpleNamespace(skill="pick", code="OK", message="held")})
    store.update("model_done", "Picked")
    clock.t += 1
    return dashboard


def test_the_types_come_in_value_order_with_what_the_session_holds(tmp_path):
    dashboard = session(tmp_path)
    reply = dashboard.command({"type": "dataset_types"})
    assert reply["ok"] and reply["source"] == "current monitoring session"
    types = {row["id"]: row for row in reply["types"]}
    assert [row["id"] for row in reply["types"]] == ORDER  # highest value first; no depth on this dashboard
    assert types["failure_recovery"]["count"] == 1 and types["failure_recovery"]["badge"] == "Highest value"
    assert types["monitor_incidents"]["available"] and types["human_interventions"]["count"] == 1
    assert [row["id"] for row in reply["types"] if row["preselected"]] == ORDER[:3]


def test_an_extraction_writes_linked_samples_and_the_frames_they_point_to(tmp_path):
    dashboard = session(tmp_path)
    reply = dashboard.command({"type": "extract_dataset", "types": ["failure_recovery", "rgb", "human_interventions"]})
    assert reply["ok"] and "Extracted" in reply["message"]
    folder = tmp_path / "datasets" / reply["path"].rsplit("/", 1)[-1]
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["types"] == ["failure_recovery", "human_interventions", "rgb"]  # in value order
    failure, = [json.loads(line) for line in (folder / "failure_recovery.jsonl").read_text().splitlines()]
    assert failure["label"] == "NO_GRASP" and failure["recovered"] and failure["recovery"]["code"] == "OK"
    assert failure["context"]["readings"][0]["activity"] == "dropping_jar"
    assert failure["context"]["decisions"][0]["text"] == "Pick the jar" and failure["frames"]
    human, = [json.loads(line) for line in (folder / "human_interventions.jsonl").read_text().splitlines()]
    assert human["text"] == "Use the right hand" and human["followed_by"]["attempts"][0]["code"] == "OK"
    index = json.loads((folder / "rgb" / "index.json").read_text())
    assert index and (folder / index[0]["file"]).read_bytes().startswith(b"jpeg-") and index[0]["sim_time"] >= 100
    assert manifest["counts"]["rgb"] >= 2  # the episode and the clip around the failure


def test_an_empty_session_offers_nothing_and_a_bad_request_is_refused(tmp_path):
    dashboard = fake_dashboard()
    dataset.attach(dashboard, tmp_path / "datasets")
    types = dashboard.command({"type": "dataset_types"})["types"]
    assert not any(row["available"] for row in types) and types[0]["reason"] == "No failed attempts in this session"
    assert not dashboard.command({"type": "extract_dataset", "types": []})["ok"]
    assert not dashboard.command({"type": "extract_dataset", "types": ["everything"]})["ok"]


def test_each_type_previews_its_most_telling_sample_with_key_frames_and_what_it_includes(tmp_path):
    dashboard = session(tmp_path)
    types = {row["id"]: row for row in dashboard.command({"type": "dataset_types"})["types"]}
    failure = types["failure_recovery"]["preview"]
    assert failure["name"] == "failure_recovery_episode_0000"
    assert [f["caption"] for f in failure["frames"]] == ["1. Pick approach", "2. Pick failed", "3. Recovery & retry"]
    assert len({f["id"] for f in failure["frames"]}) == 3  # three distinct frames even when moments share one
    assert failure["record"]["failure_type"] == "NO_GRASP" and failure["record"]["outcome"] == "recovered"
    assert failure["record"]["task"] == "Pick the jam jar" and failure["thumbnail"] is not None
    assert {"camera frames", "actions", "monitor events", "outcome label"} <= set(failure["includes"])
    assert types["task_language"]["preview"]["text"] == "Pick the jam jar"
    assert "robot state" in types["robot_state_actions"]["preview"]["includes"]
    empty = fake_dashboard()
    dataset.attach(empty, tmp_path / "empty")
    assert all(row["preview"] is None for row in empty.command({"type": "dataset_types"})["types"])
