"""Session lifecycle tests use explicit fakes; they do not claim Astra/physics validation."""
import json
import re
import runpy
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import FakeHub

from homebody.primitives.observations import Frame, HandState, NavigationMap
from homebody.session.config import Agent, Coder
from homebody.session.recorder import Recorder
from homebody.session.session import Session
from homebody.session.settings import Settings
from homebody.skills.registry import SKILLS
from homebody.ui.controls import Controls
from homebody.ui.dashboard import Dashboard
from homebody.vlm.observations import describe, map_preview
from homebody.vlm.provider import Decision, ProviderBusy, ProviderError


def test_terminal_file_failure_does_not_mask_error_during_close(tmp_path):
    recorder = Recorder(tmp_path, Settings(), tmp_path / "runs")
    path = recorder.start_task("test", epoch=1)
    (path / "outcome.json").mkdir()
    with pytest.raises(IsADirectoryError):
        recorder.end_task({"status": "model_done"})
    recorder.close()
    assert (path / "outcome.json").is_dir()


class Backend:
    def __init__(self):
        self.time = 0.0
        self.epoch = 1
        self.sequence = 0
        self.sink = None
        self.stops = 0
        self.controls = None
        self.state_calls = 0
        self.fail_terminal = False

    def measure(self):
        return self.snapshot()

    def snapshot(self):
        self.sequence += 1
        return Frame(self.epoch, self.sequence, self.time,
                     np.zeros((16, 16, 3), dtype=np.uint8), np.ones((16, 16)),
                     np.array([[10, 0, 8], [0, 10, 8], [0, 0, 1]]), np.eye(4), np.eye(4),
                     np.zeros(3), np.zeros(29),
                     {"right": HandState(np.zeros(7), 0.0, np.zeros(3), np.zeros(3))},
                     np.ones((16, 16), dtype=np.int32),
                     NavigationMap(np.zeros((20, 20)),
                                   np.array([[0.1, 0, -1], [0, -0.1, 1], [0, 0, 1]]), 0.1))

    def base_velocity(self, epoch, forward, lateral, yaw):
        assert epoch == self.epoch
        assert (forward, lateral, yaw) == (0.0, 0.0, 0.0)

    def stop(self, epoch):
        assert epoch == self.epoch
        self.stops += 1

    def advance(self, epoch, seconds):
        assert epoch == self.epoch
        self.time += seconds
        if self.sink:
            self.sink(self.evaluation_state())

    def set_evaluation_sink(self, sink):
        self.sink = sink
        if sink:
            sink(self.evaluation_state())

    def evaluation_state(self):
        self.state_calls += 1
        if self.fail_terminal and self.state_calls == 5:
            raise RuntimeError("terminal sensor failure")
        return {"schema_version": 1, "time": self.time, "epoch": self.epoch,
                "physics_valid": True, "robot_upright": True, "warnings": [], "objects": {},
                "hidden_target": "SECRET_EVALUATOR"}

    def reset(self):
        self.epoch += 1
        self.time = 0
        self.sequence = 0
        self.controls.closing.set()


class Provider:
    def __init__(self, decisions, *, controls=None, error=None, ticks=3):
        self.decisions = iter(decisions)
        self.ticks = ticks
        self.controls = controls
        self.error = error
        self.prompts = []
        self.images = []
        self.names = []

    def generate(self, prompt, *, tick, **kwargs):
        self.prompts.append(prompt)
        self.images.append(kwargs["images"])
        self.names.append(kwargs["names"])
        for _ in range(self.ticks):
            tick()
        if self.controls:
            self.controls.stop()
        if self.error:
            raise self.error
        return next(self.decisions)


class ImageOnlyRecorder(Recorder):
    def _append_video(self, rgb, stream="observations", fps=4):
        pass


@pytest.fixture
def build(tmp_path):
    sessions = []

    def create(provider, *, settings=None, observer_frame=None, backend=None, controls=None, ui=None,
               scene_update=None, monitor=None):
        # these tests watch the heartbeat while the model thinks: physics runs on (pause_while_thinking off)
        settings = settings or Settings(agent=Agent(done_settle_seconds=0., pause_while_thinking=False))
        backend, controls = backend or Backend(), controls or Controls()
        recorder = ImageOnlyRecorder(tmp_path / "source", settings, tmp_path / "runs")
        session = Session(backend, provider, settings, recorder, controls,
                          arms={}, jaw_samples={}, observer_frame=observer_frame, ui=ui,
                          scene_update=scene_update, monitor=monitor)
        sessions.append(session)
        return session

    yield create
    for session in sessions:
        session.recorder.close()


def outcome(session):
    return json.loads(next(session.recorder.path.glob("tasks/*/outcome.json")).read_text())


def test_thinking_advances_physics_and_does_not_expose_scorer(build):
    provider = Provider([Decision("Observed completion", "done", {"summary": "claim"})])
    session = build(provider)
    result = session.run_task("Inspect the current view")
    assert result["status"] == "model_done"
    assert session.backend.time >= 0.06
    assert session.backend.stops == 0
    assert all("SECRET_EVALUATOR" not in text for text in provider.prompts)
    assert session.backend.sink is None
    assert outcome(session)["independently_scored"] is False
    times = [row["time"] for row in physics_rows(session)]
    assert len(times) == 4 and times[0] == 0.0 and times[-1] == session.backend.time
    assert all(a < b for a, b in pairwise(times))


def test_paused_thinking_holds_the_world_and_records_the_held_time(build):
    provider = Provider([Decision("Observed completion", "done", {"summary": "claim"})], ticks=5)
    session = build(provider, settings=Settings(agent=Agent(done_settle_seconds=0.)))
    result = session.run_task("Inspect the current view")
    assert result["status"] == "model_done"
    assert session.backend.time == 0.0
    assert result["pause_while_thinking"] is True and result["physics_paused_seconds"] >= 5 * 0.02
    events = [json.loads(line) for line in (session.recorder.path / "events.jsonl").read_text().splitlines()]
    paused = [event for event in events if event.get("kind") == "physics_paused"]
    assert len(paused) == 1 and paused[0]["simulation_advanced"] == 0.0 and paused[0]["wall_seconds"] >= 0.1


def test_cancel_between_decision_and_execution(build):
    provider = Provider([Decision("Move", "navigate", {"x": 10, "y": 10})])
    session = build(provider)
    provider.controls = session.controls
    result = session.run_task("Move")
    assert result["status"] == "cancelled"
    events = (session.recorder.path / "events.jsonl").read_text()
    assert "skill_started" not in events
    assert session.backend.stops > 0


def test_stop_before_task_is_not_lost(build):
    provider = Provider([])
    session = build(provider)
    session.controls.stop()
    assert session.run_task("Queued task")["status"] == "cancelled"
    assert not provider.prompts


@pytest.mark.parametrize("error, status", [
    (ProviderError("No account access"), "provider_error"),
    (RuntimeError("programming failure"), "error"), (KeyboardInterrupt(), "interrupted")])
def test_provider_failure_is_recorded_and_only_a_programming_error_propagates(build, error, status):
    session = build(Provider([], error=error))
    if isinstance(error, ProviderError):
        assert session.run_task("Observe")["status"] == status
        assert outcome(session)["reason"] == "No account access"
    else:
        with pytest.raises(type(error)):
            session.run_task("Observe")
    assert outcome(session)["status"] == status
    assert session.backend.sink is None


def test_provider_receives_only_the_current_views_and_the_scene_memory(build):
    provider = Provider([Decision("Refused", "navigate", {}), Decision("Finished", "done", {})])
    session = build(provider)
    session.semantic_context = {"entities": [{"label": "Island", "reference_anchor_ws_map_m": [1, 2, 3]}]}
    assert session.run_task("Observe")["status"] == "model_done"
    for images, prompt, names in zip(provider.images, provider.prompts, provider.names):
        assert [path.name for path in images] == ["rgb.png", "depth_preview.png", "map.png", "topdown.png"]
        assert len({path.parent for path in images}) == 1
        assert "\n- Island at [1, 2]\n" in prompt
        assert set(names) == {*SKILLS, "call_coder", "done"}
    assert provider.images[0][0].parent != provider.images[1][0].parent


@pytest.mark.parametrize("skill, arguments", [
    ("pick", {"point_normalized_1000": [625, 375], "hand": "left"}),
    ("navigate", {"goal_map_1000": [250, 600], "facing_map_1000": [350, 650]}),
    ("place", {"point_normalized_1000": [350, 700], "hand": "right"}),
    ("place", {"release_point_1000": [600, 400], "hand": "right"}),
])
def test_structured_visual_click_uses_normal_runner_and_captured_frame(
        build, monkeypatch, skill, arguments):
    from homebody.skills import navigate, place
    from homebody.skills.contract import Result, Runner, click_argument
    from homebody.skills.registry import REGISTRY

    decision = Decision.parse(json.dumps({"text": "Current visible target", "skill": skill,
                                         "arguments_json": json.dumps(arguments)}), tuple(REGISTRY))
    provider = Provider([decision, Decision("Finished", "done", {})])
    selections, prepared, executed = [], [], []
    ui = SimpleNamespace(update=lambda *args: None, show=lambda frame: None,
                         skill_event=lambda event: None,
                         select=lambda proposed, frame: selections.append((proposed, frame)))
    session = build(provider, ui=ui)

    def prepare(proposed, selected, current, planning):
        assert proposed == arguments
        assert selected.frame_id == selections[0][1].frame_id
        assert current.sequence > selected.sequence and current.time > selected.time
        assert provider.images[0][0].parent.name == f"{selected.epoch:03d}-{selected.sequence:08d}"
        if skill == "navigate":
            resolved = navigate.goal_arguments(proposed, selected.navigation)
        elif skill == "place":
            resolved = place.destination(proposed, selected, planning.settings)
        else:
            resolved = click_argument(proposed, selected)
        prepared.append(resolved)
        return selected.frame_id

    def execute(plan, context):
        executed.append(plan)
        return Result(skill, "OK", "Explicit test skill transport; no physical motion")

    monkeypatch.setitem(REGISTRY, skill, SimpleNamespace(
        NEEDS=(), FAILURES=(), PROMPT=REGISTRY[skill].PROMPT, prepare=prepare, execute=execute))
    assert isinstance(session.runner, Runner)
    assert session.run_task("Use the visible target")["status"] == "model_done"
    assert len(prepared) == len(executed) == 1
    assert executed[0] == selections[0][1].frame_id
    assert session.history[-1]["result"]["code"] == "OK"
    assert all("SECRET_EVALUATOR" not in prompt for prompt in provider.prompts)
    observation = json.loads(provider.prompts[0].split("Current observation:\n")[1].split(
        "\n\nRecent steps")[0])
    assert observation == {"base_pose": [0.0, 0.0, 0.0],
                           "map_click_from_world": [[500.0, 0.0, 500.0], [0.0, -500.0, 500.0]]}


class PrivilegedBackend(Backend):
    """Evaluation state as the simulator reports it: object poses, contacts and a file path."""
    def evaluation_state(self):
        state = super().evaluation_state()
        state["objects"] = {"carton_zz": {
            "position": [7.7731, -2.2185, 0.9113], "quaternion_wxyz": [1, 0, 0, 0],
            "corners": [[7.8431, -2.1485, 0.9813]] * 8, "linear_velocity": [0, 0, 0],
            "angular_velocity": [0, 0, 0], "hand_contacts": ["right"],
            "support_contacts": ["island_zz"], "object_contacts": []}}
        state["scene_file"] = "/home/someone/assets/physics.json"
        return state


def test_prompt_carries_no_evaluation_state_or_file_path(build, monkeypatch):
    """Only the model's decisions, the skills' public results, the robot's own sensors and
    the scan's annotations reach the prompt: never object poses, contacts, paths or tracebacks.
    The public result detail is kept, and the private state is still recorded for scoring."""
    from homebody.skills.contract import Result
    from homebody.skills.registry import REGISTRY

    provider = Provider([Decision("Pick it", "pick", {"point_normalized_1000": [500, 500], "hand": "right"}),
                         Decision("Finished", "done", {})])
    monkeypatch.setitem(REGISTRY, "pick", SimpleNamespace(
        NEEDS=(), FAILURES=(), PROMPT=REGISTRY["pick"].PROMPT,
        prepare=lambda proposed, selected, current, planning: selected.frame_id,
        execute=lambda plan, context: Result("pick", "OK", "Lift verified", {
            "palm_rise": 0.1234, "traceback": 'File "/home/someone/src/homebody/skills/pick.py", line 9'})))
    session = build(provider, backend=PrivilegedBackend())
    assert session.run_task("Pick the carton")["status"] == "model_done"
    assert len(provider.prompts) == 2
    assert '"palm_rise": 0.123}' in provider.prompts[1]
    assert '"frame_sequence"' not in provider.prompts[1] and '"epoch"' not in provider.prompts[1]
    private = ("7.7731", "-2.2185", "0.9113", "7.8431", "island_zz", "carton_zz", "SECRET_EVALUATOR",
               "scene_file", "/home/", "/tmp", "Traceback", "File \"", "pick.py", "runs/",
               str(session.recorder.path), str(provider.images[0][0].parent))
    for prompt in provider.prompts:
        assert not [word for word in private if word in prompt]
    rows = physics_rows(session)
    assert rows[-1]["objects"]["carton_zz"]["position"] == [7.7731, -2.2185, 0.9113]


@pytest.mark.parametrize("name", ["wait", "speak", "teleport"])
def test_unknown_skills_are_rejected_even_from_a_nonconforming_provider(build, name):
    session = build(Provider([Decision("Unsupported memory", name, {})]))
    result = session.run_task("Observe")
    assert result["status"] == "provider_error"
    assert result["reason"] == f"Unsupported skill: {name}"
    assert result["steps"] == 0


def test_reset_clears_task_history_and_carried_state(build):
    session = build(Provider([]))
    session.history.append({"operator_task": "Old task"})
    session.context.carried["right"] = {"observation": "old epoch"}
    session.controls.reset()
    session.backend.controls = session.controls
    session.run()
    assert session.backend.epoch == 2
    assert not session.history and not session.context.carried


@pytest.mark.parametrize("carried, line", [
    ({}, "hands now: the left hand is free; the right hand is free."),
    ({"left": {"state": "verified"}, "right": {"state": "pending"}},
     ("hands now: the left hand HOLDS the object from its last pick, not placed yet; only a "
      "place by that hand opens it; the right hand's grasp is uncertain; repeat pick on its "
      "target to recheck it.")),
])
def test_prompt_states_what_each_hand_holds(build, carried, line):
    session = build(Provider([]))
    session.context.carried.update(carried)
    status = next(part for part in session.prompt("Inspect", session.backend.snapshot()).split("\n\n")
                  if part.startswith("Status: "))
    assert line in status


def test_prompt_numbers_are_settings_limits_or_coordinate_conventions(build):
    """The skill lines carry no unrendered fields and the prompt renders the changed settings'
    limits (the harness's own instructions and primitive reference are free text)."""
    defaults = Settings()
    settings = replace(defaults, motion=replace(defaults.motion, stance_reach=0.52),
                       placement=replace(defaults.placement, release_height=0.02, max_release_height=0.27,
                                         forward_reach=0.43),
                       topdown=replace(defaults.topdown, ahead=(0.1, 1.2), left=(-0.6, 0.65)))
    session = build(Provider([]), settings=settings)
    prompt = session.prompt("Inspect", session.backend.snapshot())
    skills = prompt.split("\n\nSkills:\n")[1].split("\n\nInitial-scene memory:")[0]
    assert "{" not in skills and "}" not in skills
    numbers = set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?", prompt.split("\n\nInitial-scene memory:")[0]))
    assert {"0.52", "0.02", "0.27", "0.43", "0.1", "1.2", "0.6", "0.65"} <= numbers  # rendered from the settings


def test_each_click_argument_names_the_attachment_the_recorder_sends(build):
    session = build(Provider([]))
    text = session.prompt("Inspect", session.backend.snapshot())
    listed = re.findall(r"^([1-4])\. [^\n]*?(RGB|depth preview|map|from above)", text, re.MULTILINE)
    assert listed == [("1", "RGB"), ("2", "depth preview"), ("3", "map"), ("4", "from above")]
    for argument, index in (("point_normalized_1000", "1"), ("goal_map_1000", "3"),
                            ("facing_map_1000", "3"), ("release_point_1000", "4")):
        assert re.search(rf"{argument}[^.]*?image (\d)", text).group(1) == index


def test_map_click_from_world_inverts_the_drive_map_click():
    from homebody.skills.navigate import map_point
    from homebody.vlm.observations import model_observation
    navigation = NavigationMap(np.zeros((1024, 768)),
                               np.array([[0.0172, 0, -2.94], [0, -0.0172, 8.39], [0, 0, 1]]), 0.0172)
    observed = model_observation(SimpleNamespace(navigation=navigation, base_pose=np.array([3.2, 2.1, -.74])))
    assert observed["base_pose"] == [3.2, 2.1, -0.74]
    for world in ([3.2, 2.1], [-2.9, 8.3], [10.1, -9.0]):
        click = np.array(observed["map_click_from_world"]) @ [*world, 1.]
        np.testing.assert_allclose(map_point({"goal_map_1000": click.tolist()}, "goal", navigation), world,
                                   atol=2e-3)


def test_step_limit_is_not_success(build):
    settings = replace(Settings(), agent=replace(Settings().agent, max_steps=1))
    session = build(Provider([Decision("Refused", "navigate", {})]), settings=settings)
    assert session.run_task("Observe")["status"] == "step_limit"


def test_recorder_keeps_metric_depth_and_reuses_exact_frame(build):
    session = build(Provider([]))
    frame = session.backend.snapshot()
    paths = session.recorder.frame(frame)
    assert session.recorder.frame(frame) == paths
    session.recorder.flush()
    with np.load(paths[0].parent / "sensors.npz") as saved:
        np.testing.assert_array_equal(saved["depth"], frame.depth)
        for side in frame.hands:
            np.testing.assert_array_equal(saved[f"{side}_hand_joints"], frame.hands[side].joints)
    assert describe(frame)["map_world_xy_from_pixel"] == frame.navigation.T_map_px.tolist()
    picture = map_preview(frame)
    assert tuple(picture[10, 10]) == (10, 125, 180)
    assert len(list((session.recorder.path / "frames").iterdir())) == 1


def test_a_failed_frame_write_fails_loudly_after_closing_every_file(build, monkeypatch):
    from homebody.session import recorder as recording

    def full_disk(*_args):
        raise OSError("No space left on device")

    monkeypatch.setattr(recording, "write_frame", full_disk)
    session = build(Provider([]))
    frame = session.backend.snapshot()
    session.recorder.frame(frame)
    with pytest.raises(OSError, match="No space"):
        session.recorder.close()
    assert session.recorder._events.closed
    assert "frame_write_failed" in (session.recorder.path / "events.jsonl").read_text()


def test_after_done_the_robot_stands_still_before_the_final_sample(build):
    """The scripted fixture stands still after each release; a session stands still after
    the model's done, so an object its last action released is judged at rest."""
    session = build(Provider([Decision("Observed", "done", {"summary": "done"})]),
                    settings=Settings(agent=Agent(done_settle_seconds=1.)))
    session.run_task("Inspect")
    times = [row["time"] for row in physics_rows(session)]
    assert times[-1] - times[0] >= 1. - 1e-9  # the settle steps sum in floating point


def test_terminal_state_read_failure_still_records_outcome(build):
    provider = Provider([Decision("Observed", "done", {})])
    session = build(provider)
    session.backend.fail_terminal = True
    result = session.run_task("Inspect")
    assert result["status"] == "error"
    assert "terminal sensor failure" in outcome(session)["cleanup_error"]
    assert outcome(session)["status_before_cleanup"] == "model_done"
    assert session.backend.sink is None
    assert len(physics_rows(session)) == 4


def test_setup_failure_is_an_attempt(build):
    session = build(Provider([]))
    session.recorder.setup_failure("Move cartons", "No graphics context")
    assert outcome(session)["status"] == "setup_error"
    assert outcome(session)["steps"] == 0


def test_the_prior_carries_no_operator_alias_and_never_loads_live_physics(tmp_path):
    """The task's own name for an object (task_scene.json) is the operator's, not the scan's,
    so the model must ground "coffee bag" in the scan's description itself."""
    from homebody.vlm.observations import semantic_prior
    path = tmp_path / "semantics.json"
    path.write_text(json.dumps({"entities": [{"id": "source-item", "label": "Wrapped parcel",
                                             "reference_anchor_ws_map_m": [1, 2, 3],
                                             "portable_candidate": True,
                                             "internal_evidence": "OMIT"}]}))
    (tmp_path / "task_scene.json").write_text(json.dumps({
        "semantic_aliases": {"source-item": "Coffee bag"},
        "identity_provenance": "Operator-defined task identity",
        "robot_start": [9, 9, 9], "movable_entities": ["source-item", "private-list"]}))
    prior = semantic_prior(tmp_path)
    assert prior["entities"][0]["label"] == "Wrapped parcel"
    assert "Coffee bag" not in json.dumps(prior)
    assert prior["entities"][0]["reference_anchor_ws_map_m"] == [1, 2, 3]
    assert "private-list" not in json.dumps(prior)
    assert "OMIT" not in json.dumps(prior)
    assert "robot_start" not in prior
    assert "identity_provenance" not in json.dumps(prior) and "Operator-defined" not in json.dumps(prior)


def test_prior_places_every_entity_where_the_scan_saw_it(tmp_path):
    from homebody.vlm.observations import annotations, semantic_prior
    (tmp_path / "semantics.json").write_text(json.dumps({"entities": [
        {"id": "juice_carton", "label": "Sideways juice carton", "visual_description": "Orange print.",
         "location_description": "In front of the toaster on the counter.", "reference_anchor_ws_map_m": [1, 2, 3],
         "portable_candidate": True, "object_specific_limits": "Contents unknown."},
        {"id": "toaster", "label": "Toaster oven", "visual_description": "Chrome box; behind the juice carton.",
         "location_description": "On the counter. Behind juice carton.", "portable_candidate": False,
         "reference_anchor_ws_map_m": [1.3042, 2.0071, 3], "object_specific_limits": "No heat state."}]}))
    assert annotations(semantic_prior(tmp_path)) == (
        "- Sideways juice carton (portable) at [1, 2]: Orange print. In front of the toaster on the counter.\n"
        "- Toaster oven at [1.3, 2.01]: Chrome box; behind the juice carton. On the counter. Behind juice carton.")


def test_the_prompt_tells_the_model_what_the_scan_said_never_what_the_simulator_moves():
    """The simulator's movable_entities never reach the prompt: every entity keeps the scan's
    anchor, and what is marked portable is the scan's own judgement, so the four objects the
    tasks move look no different from the other portable things."""
    from homebody.vlm.observations import semantic_prior

    scene = Path(__file__).parents[1] / "assets/real2sim/src_kitchen"
    source = json.loads((scene / "semantics.json").read_text())
    movable = set(json.loads((scene / "task_scene.json").read_text())["movable_entities"])
    prior = semantic_prior(scene)["entities"]
    portable = {raw["id"] for raw, entry in zip(source["entities"], prior, strict=True) if entry.get("portable")}
    assert portable == {raw["id"] for raw in source["entities"] if raw.get("portable_candidate")}
    assert movable < portable
    for raw, entry in zip(source["entities"], prior, strict=True):
        assert entry.get("reference_anchor_ws_map_m") == raw.get("reference_anchor_ws_map_m")


def test_room_camera_is_recorded_during_tasks_but_never_passed_to_provider(build):
    provider = Provider([Decision("Observed", "done", {})])
    room = np.full((16, 16, 3), 127, dtype=np.uint8)
    session = build(provider, observer_frame=lambda: room)
    for _ in range(30):
        session.heartbeat()
    assert not any((session.recorder.path / "frames").iterdir())
    session.run_task("Observe")
    saved = list((session.recorder.path / "observer").glob("*.png"))
    assert len(saved) == 1
    assert len(provider.images[0]) == 4
    assert all("observer" not in path.parts for path in provider.images[0])
    events = [json.loads(line) for line in (session.recorder.path / "events.jsonl").read_text().splitlines()]
    assert next(event for event in events if event["kind"] == "observer_frame")["model_visible"] is False


def test_every_operator_video_frame_has_its_simulation_time(tmp_path):
    """The shoulder stream samples whenever a snapshot is a quarter second newer, so its
    frames are timed only by their events; the room stream also keeps PNGs."""
    recorder = ImageOnlyRecorder(tmp_path, Settings(), tmp_path / "runs")
    view = np.zeros((16, 16, 3), dtype=np.uint8)
    for time, stream in ((0., "room"), (.3, "shoulder"), (.7, "shoulder"), (2., "room")):
        recorder.observer(view, epoch=1, simulation_time=time, stream=stream, fps=4)
    recorder.close()
    events = [json.loads(line) for line in (recorder.path / "events.jsonl").read_text().splitlines()]
    frames = [(row["stream"], row["video_frame"], row["simulation_time"], row["path"])
              for row in events if row["kind"] == "observer_frame"]
    assert frames == [("room", 0, 0., "observer/00000000.png"), ("shoulder", 0, .3, None),
                      ("shoulder", 1, .7, None), ("room", 1, 2., "observer/00000001.png")]
    assert "events.jsonl" in json.loads((recorder.path / "provenance.json").read_text())["video"]


def physics_rows(session):
    path = next(session.recorder.path.glob("tasks/*/physics.jsonl"))
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("error, expected", [(None, "model_done"),
                                             (ProviderError("immediate failure"), "provider_error")])
def test_immediate_terminal_task_does_not_duplicate_initial_sample(build, error, expected):
    session = build(Provider([Decision("Finished", "done", {})], error=error, ticks=0))
    assert session.run_task("Inspect")["status"] == expected
    samples = physics_rows(session)
    assert len(samples) == 1
    assert samples[0]["time"] == 0.0


def test_equal_time_dedup_cannot_clear_failure_evidence(build):
    session = build(Provider([]))
    recorder = session.recorder
    recorder.start_task("Record flags", 1)
    state = {"epoch": 1, "time": 0.0, "physics_valid": False, "robot_upright": False,
             "warnings": ["first"], "objects": {"latest": False}}
    recorder.physics(state)
    recorder.physics({**state, "physics_valid": True, "robot_upright": True,
                      "warnings": ["second"], "objects": {"latest": True}})
    recorder.end_task({"status": "test"})
    samples = physics_rows(session)
    assert len(samples) == 1
    assert samples[0]["physics_valid"] is False and samples[0]["robot_upright"] is False
    assert samples[0]["warnings"] == ["first", "second"]
    assert samples[0]["objects"] == {"latest": True}


@pytest.mark.parametrize("last", [{"epoch": 1, "time": 0.5}, {"epoch": 2, "time": 1.0}])
def test_clock_regression_or_reset_is_never_deduplicated(build, last):
    session = build(Provider([]))
    session.recorder.start_task("Record discontinuity", 1)
    session.recorder.physics({"epoch": 1, "time": 1.0})
    session.recorder.physics(last)
    session.recorder.end_task({"status": "test"})
    assert physics_rows(session) == [{"epoch": 1, "time": 1.0}, last]


def test_done_summary_type_failure_is_recorded_without_poisoning_ui(build):
    session = build(Provider([Decision("Finished", "done", {"summary": ["not text"]})]))
    assert session.run_task("Inspect")["status"] == "provider_error"
    assert outcome(session)["reason"] == "done.summary must be text"


@pytest.mark.parametrize("terminal_warning", [False, True])
def test_recorded_terminal_dedup_is_compatible_with_independent_scorer(build, terminal_warning):
    score = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                              "evaluation/evaluate.py"))["score_attempt"]
    session = build(Provider([]))
    state = session.backend.evaluation_state()
    state["objects"] = {"carton": {
        "position": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0],
        "corners": [[x, y, z] for x in (-0.1, 0.1) for y in (-0.1, 0.1) for z in (-0.1, 0.1)],
        "linear_velocity": [0, 0, 0], "angular_velocity": [0, 0, 0],
        "hand_contacts": [], "robot_contacts": [], "support_contacts": ["table"]}}
    target = {"schema_version": 1, "frame": "ws_map", "objects": {
        "carton": {"bounds": [[-1, -1, -1], [1, 1, 1]], "support_contacts": ["table"]}}}
    session.recorder.start_task("Record settled object", 1)
    for index in range(21):
        state["time"] = index / 10
        session.recorder.physics(state)
    if terminal_warning:
        state.update(physics_valid=False, robot_upright=False, warnings=["terminal_warning"])
    session.recorder.physics(state)
    session.recorder.end_task({"status": "model_done"})
    result = score(physics_rows(session), target)
    assert result["sample_count"] == 21
    assert result["success"] is not terminal_warning
    assert "nonmonotonic_time" not in result["invalid_reasons"]
    if terminal_warning:
        assert result["invalid_reasons"] == ["invalid_physics", "physics_warning", "robot_fall"]


class InvalidPhysicsBackend(Backend):
    def __init__(self, reason="Robot fell"):
        super().__init__()
        self.reason = reason
        self.invalid = False
        self.fail_next_advance = True
        self.actuation = []
        self.advance_count = 0
        self.fail_reset = False

    def base_velocity(self, epoch, forward, lateral, yaw):
        assert not self.invalid, "Invalid physics received another base command"
        self.actuation.append(("base", epoch))
        super().base_velocity(epoch, forward, lateral, yaw)

    def advance(self, epoch, seconds):
        from homebody.primitives.actions import PhysicsInvalid
        assert not self.invalid, "Invalid physics was advanced again"
        self.advance_count += 1
        self.actuation.append(("advance", epoch))
        if self.fail_next_advance:
            self.time += seconds
            self.invalid = True
            self.fail_next_advance = False
            raise PhysicsInvalid(self.reason)
        super().advance(epoch, seconds)

    def stop(self, epoch):
        self.actuation.append(("stop", epoch))
        super().stop(epoch)

    def evaluation_state(self):
        state = super().evaluation_state()
        if self.invalid:
            state.update(physics_valid=False, robot_upright=False, warnings=[self.reason])
        return state

    def reset(self):
        from homebody.primitives.actions import PhysicsInvalid
        self.epoch += 1
        self.time, self.sequence = 0., 0
        self.actuation.append(("reset", self.epoch))
        if self.fail_reset:
            raise PhysicsInvalid("Reset warm-up failed")
        self.invalid = False


class ScriptedControls(Controls):
    def __init__(self, commands):
        super().__init__()
        self.commands = iter(commands)

    def next(self):
        from homebody.ui.controls import Command
        item = next(self.commands, "close")
        if item == "close":
            self.closing.set()
            return Command("close")
        return item() if callable(item) else item


class Display(Dashboard):
    """The real dashboard over a recording hub; decisions and skill events are recorded, frames dropped."""
    def __init__(self):
        super().__init__(Controls(), hub=FakeHub, render_in_background=False)
        self.selections, self.events = [], []

    def clear_selection(self):
        self.selections.clear()
        super().clear_selection()

    def select(self, decision, frame):
        self.selections.append((decision, frame.frame_id))

    def skill_event(self, event):
        self.events.append(event)

    def show(self, frame):
        pass


def test_idle_physics_failure_keeps_ui_resettable_and_blocks_queued_tasks(build):
    from homebody.ui.controls import Command
    backend, ui, provider = InvalidPhysicsBackend(), Display(), Provider([])
    stopped_commands = []
    def inspect_failure():
        """Returning no command exercises an idle heartbeat in the invalid generation."""
        assert session.requires_reset
        assert ui.store.status == "Reset required"
        assert not ui.store.accepting and not ui.store.reset_pending
        stopped_commands[:] = backend.actuation
    def inspect_blocked():
        assert backend.actuation == stopped_commands
        assert not provider.prompts
        assert outcome(session)["status"] == "physics_invalid"
        return Command("stop")
    def inspect_stopped():
        assert backend.actuation == stopped_commands
        assert session.requires_reset
        return Command("reset")
    def inspect_reset():
        """Returning no command resumes the normal heartbeat in the new generation."""
        assert not session.requires_reset and backend.epoch == 2
        assert ui.store.status == "Ready" and ui.store.accepting
        assert not ui.store.reset_pending
    controls = ScriptedControls([None, inspect_failure, Command("task", "Blocked task"),
                                 inspect_blocked, inspect_stopped, inspect_reset])
    session = build(provider, backend=backend, controls=controls, ui=ui)
    session.run()
    assert backend.advance_count == 2
    events = [json.loads(line) for line in (session.recorder.path / "events.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["kind"] == "physics_invalid"]) == 1
    assert len([event for event in events if event["kind"] == "scene_reset"]) == 1


def test_provider_heartbeat_failure_is_recorded_and_reset_resumes_tasks(build):
    from homebody.ui.controls import Command
    backend = InvalidPhysicsBackend("Robot fell")
    provider = Provider([Decision("Finished", "done", {})])
    controls = ScriptedControls([Command("reset")])
    session = build(provider, backend=backend, controls=controls)
    result = session.run_task("Inspect")
    assert result["status"] == "physics_invalid" and result["requires_reset"]
    assert outcome(session)["reason"] == "Robot fell"
    assert physics_rows(session)[-1]["physics_valid"] is False
    commands = backend.actuation.copy()
    assert session.run_task("Refused until reset")["status"] == "physics_invalid"
    assert backend.actuation == commands and len(provider.prompts) == 1
    session.run()
    assert not session.requires_reset and backend.epoch == 2
    assert session.run_task("Inspect after reset")["status"] == "model_done"
    assert len(provider.prompts) == 2


def test_skill_physics_failure_preserves_typed_terminal_and_reset_latch(build):
    backend = InvalidPhysicsBackend()
    provider = Provider([Decision("Hold current stance", "navigate", {"x": 0., "y": 0.})], ticks=0)
    session = build(provider, backend=backend)
    assert session.run_task("Verify stance")["status"] == "physics_invalid"
    assert session.requires_reset
    events = [json.loads(line) for line in (session.recorder.path / "events.jsonl").read_text().splitlines()]
    terminal = [row["event"]["result"] for row in events
                if row["kind"] == "skill" and row["event"]["type"] == "skill_finished"]
    assert len(terminal) == 1 and terminal[0]["code"] == "PHYSICS_INVALID"
    commands = backend.actuation.copy()
    session.heartbeat()
    assert backend.actuation == commands


def test_failed_reset_clears_the_view_keeps_the_ui_alive_and_requires_another_reset(build):
    """The human scene follows the new generation even when its reset fails."""
    from homebody.ui.controls import Command
    backend, display, generations = InvalidPhysicsBackend(), Display(), []
    backend.fail_reset = True
    display.selections.append("old target")
    def retry_reset():
        assert session.requires_reset and backend.epoch == 2
        assert session.status == "Reset required"
        assert generations == [(1, 0.), (1, .02), (2, 0.)]
        assert display.store.status == "Reset required" and display.store.plan is None
        assert not display.selections and {"decision", "topdown"} <= set(display.hub.forgotten)
        backend.fail_reset = False
        return Command("reset")
    controls = ScriptedControls([None, Command("reset"), None, retry_reset])
    session = build(Provider([]), backend=backend, controls=controls, ui=display,
                    scene_update=lambda: generations.append((backend.epoch, backend.time)))
    session.run()
    assert backend.epoch == 3 and not session.requires_reset
    assert backend.advance_count == 1


def test_unrelated_idle_programming_error_still_propagates(build):
    class BrokenBackend(Backend):
        def advance(self, epoch, seconds):
            raise RuntimeError("Idle programming defect")
    session = build(Provider([]), backend=BrokenBackend(), controls=ScriptedControls([None]))
    with pytest.raises(RuntimeError, match="Idle programming defect"):
        session.run()
    assert not session.requires_reset


def test_human_scene_updates_during_thinking_without_entering_provider_inputs(build):
    backend, times = Backend(), []
    def scene_update():
        times.append(backend.time)
        return {"hidden_geometry": "SECRET_SCENE_GRAPH"}
    provider = Provider([Decision("Done", "done", {})], ticks=4)
    session = build(provider, backend=backend, scene_update=scene_update)
    session.run_task("Observe")
    np.testing.assert_allclose(times, [0, .02, .04, .06, .08])
    assert all("SECRET_SCENE_GRAPH" not in text for text in provider.prompts)
    assert len(provider.images[0]) == 4
    assert [path.name for path in provider.images[0]] == ["rgb.png", "depth_preview.png", "map.png", "topdown.png"]


def test_session_forwards_selected_capture_and_skill_events(build):
    display = Display()
    provider = Provider([Decision("This route is invalid", "navigate", {"x": "bad", "y": 1}),
                         Decision("Finished", "done", {})], ticks=1)
    session = build(provider, ui=display)
    assert session.run_task("Check route")["status"] == "model_done"
    assert display.selections[0][0].text == "This route is invalid"
    assert display.selections[0][1] == display.events[0]["frame_id"]
    assert [event["type"] for event in display.events] == [
        "skill_started", "skill_stage", "skill_finished"]
    assert display.events[-1]["result"].code == "INVALID_ARGUMENT"


def test_a_watched_session_is_paced_to_the_wall_clock_and_restarts_when_behind(monkeypatch):
    from homebody.session import session as module
    clock = {"wall": 0.0}
    sleeps = []
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["wall"])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleeps.append(round(seconds, 6)))
    backend = SimpleNamespace(time=0.0, epoch=1)

    def advance(epoch, seconds):
        backend.time += seconds
    backend.advance = advance
    observed = module.ObservedBackend(backend, lambda frame: None, 1e9, real_time=True)
    observed._last = 0.0
    observed.advance(1, 0.1)               # anchors the pace
    observed.advance(1, 0.1)               # 0.1 s of physics in no wall time: wait it out
    assert sleeps == [0.1]
    clock["wall"] = 5.0                    # far behind (a long render): restart, no catch-up
    observed.advance(1, 0.1)
    assert sleeps == [0.1]
    unpaced = module.ObservedBackend(backend, lambda frame: None, 1e9)
    unpaced._last = backend.time
    unpaced.advance(1, 0.1)
    assert sleeps == [0.1]


def test_a_simulation_clock_that_jumps_ahead_restarts_the_pace_instead_of_sleeping_it_out(monkeypatch):
    """A free-running server (warehouse-sim keeps its own real-time clock) can be minutes ahead of the pace's anchor;
    sleeping that out froze a pick for 9 minutes per 0.04 s step (2026-10-24). A step waits at most its own length."""
    from homebody.session import session as module
    clock = {"wall": 0.0}
    sleeps = []
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["wall"])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleeps.append(round(seconds, 6)))
    backend = SimpleNamespace(time=0.0, epoch=1)

    def advance(epoch, seconds):
        backend.time += seconds
    backend.advance = advance
    observed = module.ObservedBackend(backend, lambda frame: None, 1e9, real_time=True)
    observed._last = 0.0
    observed.advance(1, 0.04)              # anchors the pace
    backend.time += 535.0                  # the server's clock ran on by itself
    observed.advance(1, 0.04)
    assert sleeps == []                    # re-anchored, no 535 s sleep
    observed.advance(1, 0.04)              # paced again from the new anchor
    assert sleeps == [0.04]


class BusyThenAnswers(Provider):
    """At capacity for the first BUSY requests, then answers."""
    def __init__(self, decisions, busy, error=None):
        super().__init__(decisions, ticks=1)
        self.busy, self.calls, self.attempts, self.failure = busy, 0, [], error

    def generate(self, prompt, *, attempt, **kwargs):
        self.calls += 1
        self.attempts.append(attempt.name)
        if self.calls <= self.busy:
            raise (self.failure or ProviderBusy("Codex exited 1: Selected model is at capacity."))
        return super().generate(prompt, **kwargs)


def test_a_busy_service_is_asked_again_and_the_task_goes_on(build):
    """A service at capacity is asked again; the task does not end on the refusal."""
    provider = BusyThenAnswers([Decision("Finished", "done", {"summary": "ok"})], busy=2)
    session = build(provider, settings=Settings(agent=Agent(done_settle_seconds=0., busy_wait=0.01)))
    assert session.run_task("Inspect")["status"] == "model_done"
    assert provider.attempts == ["step-000", "step-000-retry1", "step-000-retry2"]


def test_busy_retries_run_out_and_other_errors_are_not_retried(build):
    busy = BusyThenAnswers([], busy=9)
    session = build(busy, settings=Settings(agent=Agent(done_settle_seconds=0., busy_wait=0.01, busy_retries=2)))
    assert session.run_task("Inspect")["status"] == "provider_error" and busy.calls == 3
    broken = BusyThenAnswers([], busy=9, error=ProviderError("Codex exited 1: invalid schema"))
    session = build(broken)
    assert session.run_task("Inspect")["status"] == "provider_error" and broken.calls == 1


FAKE_CODEX = """#!{python}
# Stands in for `codex exec`: writes SKILL into the workspace, calls the robot tools over the socket
# the MCP server would use, and leaves STATUS as its final answer.
import json, re, socket, sys
args = sys.argv[1:]
workspace = args[args.index("-C") + 1]
socket_path = re.search(r'"-m", "homebody.session.coder_mcp", "([^"]+)"', " ".join(args)).group(1)
open(workspace + "/skills/lift.py", "w").write({skill!r})
def tool(name, arguments):
    with socket.socket(socket.AF_UNIX) as c:
        c.connect(socket_path)
        c.sendall((json.dumps({{"tool": name, "arguments": arguments}}) + "\\n").encode())
        data = b""
        while not data.endswith(b"\\n"):
            data += c.recv(65536)
    return json.loads(data)
log = [tool(name, {{"height": 0.1}}) for name in {tools!r}]
open(workspace + "/../tools.json", "w").write(json.dumps(log))
open(args[args.index("-o") + 1], "w").write(json.dumps({{"status": {status!r}, "summary": "s", "state": "st", "usage": "u"}}))
"""
LIFT = """from .contract import Result, SkillFailure
NEEDS = ("advance",)
FAILURES = ("TOO_HIGH",)
PROMPT = "lift: height in metres."
def prepare(arguments, selected, current, planning):
    if arguments.get("height", 0) > 1:
        raise SkillFailure("TOO_HIGH", "out of reach")
    return {"height": arguments["height"]}
def execute(plan, ctx):
    return Result("lift", "OK", f"lifted {plan['height']}")
"""


def coder_session(build, tmp_path, monkeypatch, decisions, *, skill=LIFT, tools=("check", "execute"), status="success"):
    import sys
    monkeypatch.delenv("HOMEBODY_CODER_CODEX", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    codex = tmp_path / "fake-codex"
    codex.write_text(FAKE_CODEX.format(python=sys.executable, skill=skill, tools=list(tools), status=status))
    codex.chmod(0o755)
    settings = Settings(agent=Agent(done_settle_seconds=0.), coder=Coder(codex=str(codex), seconds=60.0, calls=2))
    provider = Provider(decisions)
    return build(provider, settings=settings), provider


REQUEST = {"skill": "lift", "goal": "lift it", "tried": "pick refused", "success": "held up", "constraints": ""}


def test_call_coder_registers_the_version_that_executed_for_the_task(build, tmp_path, monkeypatch):
    session, provider = coder_session(build, tmp_path, monkeypatch, [
        Decision("Try", "navigate", {}), Decision("Ask", "call_coder", REQUEST), Decision("Use", "lift", {"height": 0.2}),
        Decision("Finished", "done", {}), Decision("Finished again", "done", {})])
    assert session.run_task("Lift")["status"] == "model_done"
    results = [entry["result"] for entry in session.history if "result" in entry][1:]
    assert [r["code"] for r in results] == ["OK", "OK"] and results[1]["message"] == "lifted 0.2"
    assert results[0]["details"]["skill_version"] == 1 and results[0]["details"]["executes"][0]["code"] == "OK"
    assert "lift: height in metres." in provider.prompts[2] and "1 call left" in provider.prompts[2]
    call = next(session.recorder.path.glob("tasks/*/coder/call-1"))
    assert (call / "workspace" / "AGENTS.md").is_file() and (call / "workspace" / "REQUEST.md").is_file()
    request = (call / "workspace" / "REQUEST.md").read_text()
    assert '"what_changed": ["base: ' in request  # what the earlier skill call did to the robot
    assert max(path.name for path in (call / "workspace" / "observation").iterdir()) == "state.json"
    checked = json.loads((call / "tools.json").read_text())[0]
    assert checked["prepare"]["code"] == "OK" and "0.1" in str(checked["prepare"]["plan"])  # any plan format
    assert 0 < checked["seconds_left"] <= 60
    assert session.run_task("Again")["status"] == "model_done"  # the next task starts without it
    assert "lift" not in provider.names[-1] and not session.coder.skills


def test_call_coder_refuses_forbidden_imports_and_unchecked_executes(build, tmp_path, monkeypatch):
    session, _ = coder_session(build, tmp_path, monkeypatch, [Decision("Try", "navigate", {}),
                                                              Decision("Ask", "call_coder", REQUEST),
                                                              Decision("Finished", "done", {})],
                               skill="import os\n" + LIFT, status="success")
    assert session.run_task("Lift")["status"] == "model_done"
    result = session.history[-1]["result"]
    assert result["code"] == "CODER_FAILED" and "without an execute that returned OK" in result["message"]
    call = next(session.recorder.path.glob("tasks/*/coder/call-1"))
    check, execute = json.loads((call / "tools.json").read_text())
    assert "not allowed" in check["error"] and "check" in execute["error"] and not session.coder.skills


def test_call_coder_validates_its_request_and_its_budget(build, tmp_path, monkeypatch):
    session, _ = coder_session(build, tmp_path, monkeypatch, [
        Decision("Too early", "call_coder", {**REQUEST, "skill": "pick"}), Decision("Try", "navigate", {}),
        Decision("Bad", "call_coder", {"skill": "Lift!", "goal": "g", "success": "s"}),
        Decision("Ask", "call_coder", REQUEST), Decision("Again", "call_coder", REQUEST),
        Decision("Third", "call_coder", REQUEST), Decision("Finished", "done", {})],
        tools=("check",), status="infeasible")
    assert session.run_task("Lift")["status"] == "model_done"
    codes = [entry["result"]["code"] for entry in session.history if "result" in entry]
    assert codes == ["CODER_NOT_YET", "INVALID_ARGUMENT", "INVALID_ARGUMENT", "CODER_INFEASIBLE",
                     "CODER_INFEASIBLE", "CODER_LIMIT"]


def test_a_new_skill_may_be_requested_before_any_skill_failed(build, tmp_path, monkeypatch):
    session, _ = coder_session(build, tmp_path, monkeypatch, [
        Decision("Create", "call_coder", REQUEST), Decision("Finished", "done", {})],
        tools=("check",), status="infeasible")
    assert session.run_task("Lift")["status"] == "model_done"
    codes = [entry["result"]["code"] for entry in session.history if "result" in entry]
    assert codes == ["CODER_INFEASIBLE"]  # "lift" is not a built-in skill, so there was nothing to try first


def test_older_steps_drop_result_details_but_keep_code_and_message():
    from homebody.session.session import history_line
    entry = {"base_pose": [0, 0, 0], "decision": {"text": "t", "skill": "pick", "arguments": {}},
             "result": {"skill": "pick", "code": "NO_GRASP", "message": "walls", "details": {"tried": 40}}}
    assert history_line(entry, full=True)["result"]["details"] == {"tried": 40}
    assert history_line(entry, full=False)["result"] == {"skill": "pick", "code": "NO_GRASP", "message": "walls"}


def test_auto_mode_hands_each_failure_to_the_coder_and_bounds_the_retries(build, monkeypatch):
    from homebody.skills.contract import Result

    settings = replace(Settings(agent=Agent(done_settle_seconds=0.)), coder=Coder(auto=True, calls=9, rescues=3))
    picks = [Decision("Try the cup.", "pick", {"hand": "right"}) for _ in range(5)]
    session = build(Provider([*picks, Decision("Give up.", "done", {})]), settings=settings)
    runs, asks = [], []
    monkeypatch.setattr(session.runner, "run", lambda skill, arguments, frame, **kw: (
        runs.append(skill), Result(skill, "NO_GRASP", "no grasp passed"))[1])

    def call(arguments, _session):
        asks.append(arguments)
        return {"skill": "call_coder", "code": "CODER_FAILED", "message": "no fix", "details": {}}
    monkeypatch.setattr(session.coder, "call", call)
    assert session.run_task("Put the cup away.")["status"] == "model_done"
    assert runs == ["pick"] * 4 and len(asks) == 3 and asks[0]["skill"] == "pick"
    codes = [entry["result"]["code"] for entry in session.history if "result" in entry]
    assert codes == ["NO_GRASP", "CODER_FAILED"] * 3 + ["NO_GRASP", "RECOVERY_LIMIT"]


def test_plan_mode_plans_then_works_goal_by_goal(build, monkeypatch):
    from homebody.session.config import PlanMode
    from homebody.skills.contract import Result

    tree = {"type": "sequence", "children": [{"type": "goal", "id": "g1", "goal": "pick the cup", "check": None}]}
    decisions = [Decision("Plan.", "plan", {"tree": tree}), Decision("Pick.", "pick", {"hand": "right"}),
                 Decision("Held.", "goal_done", {"reason": "held"}), Decision("Finished.", "done", {})]
    settings = replace(Settings(agent=Agent(done_settle_seconds=0.)), plan=PlanMode(enabled=True))
    provider = Provider(decisions)
    session = build(provider, settings=settings)
    monkeypatch.setattr(session.runner, "run", lambda skill, arguments, frame, **kw: Result(skill, "OK", "held"))
    assert session.run_task("Pick up the cup.")["status"] == "model_done"
    # the response schema offers every decision every turn (a stable prompt); the phase is checked in code
    assert len(set(provider.names)) == 1 and {"plan", "goal_done", "goal_failed", "done", "pick"} <= set(provider.names[0])
    assert "This turn: write the plan" in provider.prompts[0] and "Active goal g1" in provider.prompts[1]
    codes = [entry["result"]["code"] for entry in session.history if "result" in entry]
    assert codes == ["OK", "OK", "OK"] and session.planner.plan.status == {"g1": "done"}


def test_a_skill_prompt_with_literal_braces_keeps_them_and_fills_its_settings():
    from types import SimpleNamespace

    from homebody.session.session import prompt_line
    settings = {"grasp": SimpleNamespace(squeeze=0.12)}
    text = 'push: squeeze {grasp.squeeze}; example {"hand":"right","verify_released":{"label":21}}'
    assert prompt_line(text, settings) == 'push: squeeze 0.12; example {"hand":"right","verify_released":{"label":21}}'
    assert prompt_line("plain {grasp.squeeze}", settings) == "plain 0.12"


def test_a_decision_of_another_plan_phase_is_refused_and_the_agent_chooses_again(build, monkeypatch):
    from homebody.session.config import PlanMode
    from homebody.skills.contract import Result

    tree = {"type": "sequence", "children": [{"type": "goal", "id": "g1", "goal": "pick the cup", "check": None}]}
    decisions = [Decision("Pick first.", "pick", {"hand": "right"}), Decision("Plan.", "plan", {"tree": tree}),
                 Decision("Finished.", "done", {})]
    settings = replace(Settings(agent=Agent(done_settle_seconds=0.)), plan=PlanMode(enabled=True))
    session = build(Provider(decisions), settings=settings)
    monkeypatch.setattr(session.runner, "run", lambda skill, arguments, frame, **kw: Result(skill, "OK", "held"))
    assert session.run_task("Pick up the cup.")["status"] == "model_done"
    first = next(entry for entry in session.history if "result" in entry)["result"]
    assert first["code"] == "INVALID_ARGUMENT" and "not available now" in first["message"]


def test_the_prompt_opening_stays_the_same_across_turns(build):
    session = build(Provider([]))
    first = session.prompt("Inspect", session.backend.snapshot())
    session.context.carried["right"] = {"state": "verified"}
    session.history.append({"operator_task": "Inspect"})
    session.coder.calls += 1
    second = session.prompt("Inspect", session.backend.snapshot())
    opening = first[:first.index("Status: ")]
    assert second.startswith(opening) and "Current task:\nInspect" in opening and "Skills:" in opening


def test_zero_coder_calls_turn_the_coding_assistant_off_in_the_prompt():
    from homebody import harness
    on = vars(Settings())
    off = dict(on, coder=replace(on["coder"], calls=0, auto=False))
    assert "call_coder" in harness.agent_instructions(on) and "<coder>" not in harness.agent_instructions(on)
    assert "call_coder" not in harness.agent_instructions(off) and "coding assistant" not in harness.plan_instructions(off)
    assert "<coder>" not in harness.plan_instructions(on)


def test_the_settings_accept_zero_coder_calls():
    assert Settings(coder=Coder(calls=0, auto=False)).coder.calls == 0
    with pytest.raises(ValueError, match="coder.calls"):
        Settings(coder=Coder(calls=-1))


class FakeMonitor:
    """The live monitor's surface with scripted verdicts: one list per turn, oldest first."""
    def __init__(self, verdicts=()):
        self.pending, self.goals, self.phases, self.skills, self.stop = list(verdicts), [], [], [], None

    def start_task(self, task, task_dir):
        self.started = task

    def end_task(self):
        return {"calls": 7}

    def capture_due(self, sim_time):
        return False

    def goal(self, request, hands, plan=None):
        self.goals.append(request.node.goal_id)

    def set_phase(self, phase):
        self.phases.append(phase.name)

    def skill_started(self, request, sim_time):
        self.skills.append((request.skill, request.node.goal_id))

    def skill_finished(self, code=None, sim_time=None):
        self.phases.append("finished")

    def stop_requested(self):
        return self.stop

    def take_verdicts(self):
        return self.pending.pop(0) if self.pending else []

    def known_stages(self):
        return {}

    def recent_findings(self):
        return ()

    def decision(self, sim_time):
        pass


def test_a_monitor_stop_flags_the_goal_reaches_the_agent_and_skips_the_coder(build, monkeypatch):
    from homebody.monitor.targets import Verdict
    from homebody.session.config import PlanMode
    from homebody.skills.contract import Result

    tree = {"type": "sequence", "children": [{"type": "goal", "id": "g1", "goal": "pick the cup", "check": None}]}
    decisions = [Decision("Plan.", "plan", {"tree": tree}), Decision("Pick.", "pick", {"hand": "right"}),
                 Decision("Finished.", "done", {})]
    settings = replace(Settings(agent=Agent(done_settle_seconds=0.)), plan=PlanMode(enabled=True),
                       coder=Coder(auto=True, calls=3))
    stopped = Verdict("abort_skill", "g1", "missed", 0.95, "the fingers close beside the cup", 3.0)
    monitor = FakeMonitor([[], [], [stopped]])
    provider = Provider(decisions)
    session = build(provider, settings=settings, monitor=monitor)
    monkeypatch.setattr(session.runner, "run", lambda skill, arguments, frame, **kw: Result(
        skill, "MONITOR_STOPPED", "The monitor stopped the skill: missed p=0.95"))
    monkeypatch.setattr(session.coder, "call", lambda *a: pytest.fail("a monitor stop went to the coder"))
    outcome = session.run_task("Pick up the cup.")
    assert outcome["status"] == "model_done" and outcome["monitor"] == {"calls": 7}
    assert "MONITOR_STOPPED" in provider.prompts[0]
    assert "Monitor (" not in provider.prompts[1]
    assert "Monitor (a model watching the camera" in provider.prompts[2]
    assert "stopped the skill: missed p=0.95" in provider.prompts[2]
    assert "⚠ monitor: missed p=0.95" in provider.prompts[2]
    assert monitor.skills == [("pick", "g1")] and monitor.goals == ["task", "g1", "g1"]
    assert monitor.phases.count("planning") == 3 and "finished" in monitor.phases


def test_a_monitor_reopens_a_done_goal_and_the_agent_works_on_it_again(build, monkeypatch):
    from homebody.monitor.targets import Verdict
    from homebody.session.config import PlanMode
    from homebody.skills.contract import Result

    tree = {"type": "sequence", "children": [
        {"type": "goal", "id": "g1", "goal": "pick the cup", "check": None},
        {"type": "goal", "id": "g2", "goal": "walk to the table", "check": None}]}
    decisions = [Decision("Plan.", "plan", {"tree": tree}), Decision("Held.", "goal_done", {"reason": "held"}),
                 Decision("Walk.", "navigate", {}), Decision("Finished.", "done", {})]
    undone = Verdict("reopen_goal", "g1", "undone", 0.9, "no longer true: the cup is in the right hand", 5.0)
    monitor = FakeMonitor([[], [], [], [undone]])
    settings = replace(Settings(agent=Agent(done_settle_seconds=0.)), plan=PlanMode(enabled=True))
    provider = Provider(decisions)
    session = build(provider, settings=settings, monitor=monitor)
    monkeypatch.setattr(session.runner, "run", lambda skill, arguments, frame, **kw: Result(skill, "OK", "ok"))
    session.run_task("Carry the cup.")
    assert "Active goal g2" in provider.prompts[2]
    assert "Active goal g1" in provider.prompts[3] and "reopened goal g1" in provider.prompts[3]
    assert session.planner.plan.status == {"g1": "pending", "g2": "pending"}


def test_without_a_monitor_the_prompt_keeps_its_refusals(build):
    provider = Provider([Decision("Finished.", "done", {})])
    build(provider).run_task("Nothing.")
    assert "MONITOR_STOPPED" not in provider.prompts[0]


def test_a_human_needed_pause_holds_until_resume_or_stop(build):
    import threading

    from homebody.vlm.provider import Cancelled
    session = build(Provider([]), monitor=FakeMonitor())
    threading.Timer(0.1, session.controls.resume_task).start()
    session.await_operator("human_needed p=0.95")
    assert session.status == "Paused"
    threading.Timer(0.1, session.controls.cancel.set).start()
    with pytest.raises(Cancelled):
        session.await_operator("human_needed p=0.95")


def test_each_turn_records_the_plan_hands_and_goal_request_a_replay_and_an_offline_monitor_need(build, monkeypatch):
    from homebody.monitor.targets import request_from
    from homebody.session.config import PlanMode
    from homebody.skills.contract import Result

    tree = {"type": "sequence", "children": [{"type": "goal", "id": "g1", "goal": "pick the cup", "check": None}]}
    decisions = [Decision("Plan.", "plan", {"tree": tree}), Decision("Pick.", "pick", {"hand": "right"}),
                 Decision("Finished.", "done", {})]
    settings = replace(Settings(agent=Agent(done_settle_seconds=0.)), plan=PlanMode(enabled=True))
    session = build(Provider(decisions), settings=settings)
    monkeypatch.setattr(session.runner, "run", lambda skill, arguments, frame, **kw: Result(skill, "OK", "held"))
    session.run_task("Pick up the cup.")
    root = session.recorder.path
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    started = next(e for e in events if e["kind"] == "task_started")
    assert any(line.startswith("pick") for line in started["skills"])
    turns = [e for e in events if e["kind"] == "turn"]
    assert [t["step"] for t in turns] == [0, 1, 2] and turns[0]["plan"] is None
    assert turns[1]["plan"]["active"] == "g1" and "left" in turns[1]["hands"]
    request = request_from(turns[1]["request"], root)
    assert request.node.goal_id == "g1" and request.task == "Pick up the cup."
    assert request.images and all(image.is_file() for image in request.images)
    skill = next(e for e in events if e["kind"] == "skill_request")
    assert request_from(skill["request"], root).skill == "pick"


def test_a_session_behind_the_server_links_stops_its_skill_when_the_remote_monitor_says_so(build):
    from homebody.server.links import UiLink
    from homebody.server.monitor_link import MonitorLink
    from homebody.session.config import Monitor, PlanMode
    from homebody.skills.contract import Result

    sent = []
    hub = SimpleNamespace(publish=lambda message, **kw: sent.append(message))
    link = MonitorLink(hub, Monitor(enabled=True), quality=80, root=Path("."))
    link.client_connected(True)
    tree = {"type": "sequence", "children": [{"type": "goal", "id": "g1", "goal": "pick the cup", "check": None}]}
    decisions = [Decision("Plan.", "plan", {"tree": tree}), Decision("Pick.", "pick", {"hand": "right"}),
                 Decision("Finished.", "done", {})]
    settings = replace(Settings(agent=Agent(done_settle_seconds=0., pause_while_thinking=False)),
                       plan=PlanMode(enabled=True))
    provider = Provider(decisions)
    session = build(provider, settings=settings, monitor=link, ui=UiLink(hub, quality=80, view_hz=1000.0))

    def run(skill, arguments, frame, **kw):
        link.client_stop("missed p=0.95: the fingers close beside the cup", link.skill_call)
        link.client_verdicts([{"action": "abort_skill", "goal_id": "g1", "label": "missed", "p": 0.95,
                               "reason": "the fingers close beside the cup", "sim_time": 1.0}])
        code, reason = session.interruption()
        return Result(skill, code, reason)
    session.runner.run = run
    outcome = session.run_task("Pick up the cup.")
    assert outcome["status"] == "model_done" and outcome["monitor"]["remote"] is True
    assert "stopped the skill: missed p=0.95" in provider.prompts[2]
    assert any('"call": "skill_started"' in m for m in sent if isinstance(m, str))
    assert any('"type": "status"' in m for m in sent if isinstance(m, str))
    # the executor reported taking the stop, then the skill's end confirming it (the dashboard's lifecycle)
    stages = [m for m in sent if isinstance(m, str) and '"intervention"' in m]
    assert ['"stage": "accepted"' in m for m in stages] == [True, False]
    assert '"stage": "confirmed"' in stages[1] and '"initiator": "monitor"' in stages[1]
    link.client_stop("late", link.skill_call - 1)  # a stop for a call that already ended: refused, and said so
    assert '"stage": "rejected"' in sent[-1]
