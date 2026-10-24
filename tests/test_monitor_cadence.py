"""The monitor beside the session: one decision call at a time, spaced by the p95 latency, only on new frames;
vocabularies replace the fallback; a firing verdict stops the running skill and waits for the session."""
import threading
import time
from dataclasses import replace

import numpy as np

from homebody.monitor.decision import OutOfCredit
from homebody.monitor.monitor import Monitor
from homebody.monitor.targets import Label, NodeInfo, Phase, Reading, Vocabulary, VocabularyRequest
from homebody.session.config import Monitor as MonitorSettings

SETTINGS = MonitorSettings(enabled=True, min_period_s=0.01, p95_margin=1.0, window_frames=3, ticks=2, ema=1.0,
                           grace_s=0.0, cooldown_s=0.0, refresh_on_skill=False)
NODE = NodeInfo("g1", "pick the cup", {"name": "holding", "hand": "left"})
REQUEST = VocabularyRequest("pick the cup", "", NODE, (), (), "hands free")
VOCAB = Vocabulary("g1", (Label("reaching", "progress", "reaches"), Label("missed", "failure", "misses the cup")))


class Vocabularies:
    def __init__(self, gate=None):
        self.gate, self.requests = gate, []

    def generate(self, request, attempt, cancelled):
        self.requests.append(request)
        if self.gate is not None:
            self.gate.wait(2)
        return VOCAB


class Decisions:
    def __init__(self, latency=0.0, activity=None, error=None):
        self.latency, self.activity, self.error = latency, activity or {"reaching": 0.9}, error
        self.active = self.most = 0
        self.requests, self.lock = [], threading.Lock()

    def decide(self, request, lag=0.0):
        if self.error is not None:
            raise self.error
        with self.lock:
            self.active += 1
            self.most = max(self.most, self.active)
            self.requests.append(request)
        time.sleep(self.latency)
        with self.lock:
            self.active -= 1
        newest = max(shot.sim_time for shot in request.shots)
        return Reading(request.node.goal_id, newest, activity=self.activity, phase=request.phase,
                       latency_s=self.latency), {"recorded": True}


def started(tmp_path, decisions, vocabularies=None, settings=SETTINGS, events=None):
    monitor = Monitor(settings, vocabularies or Vocabularies(), decisions,
                      sink=(lambda kind, **f: events.append(f)) if events is not None else (lambda kind, **f: None))
    monitor.start_task("pick the cup", tmp_path)
    monitor.goal(REQUEST, "hands free")
    return monitor


def feed(monitor, seconds, start=0.0, hz=50):
    for index in range(int(seconds * hz)):
        monitor.capture(np.zeros((8, 8, 3), np.uint8), start + index / 10)
        time.sleep(1 / hz)


def wait_for(condition, timeout=3.0):
    until = time.monotonic() + timeout
    while time.monotonic() < until and not condition():
        time.sleep(0.01)
    return condition()


def test_one_call_at_a_time_spaced_by_p95(tmp_path):
    decisions = Decisions(latency=0.05)
    monitor = started(tmp_path, decisions)
    feed(monitor, 0.6)
    summary = monitor.end_task()
    assert decisions.most == 1
    assert 3 <= summary["calls"] <= 14  # back to back at most: 0.6 s at 0.05 s per call, never overlapping
    assert monitor.period() >= 0.05
    assert all(len(r.shots) <= 3 for r in decisions.requests)
    assert (tmp_path / "monitor" / "readings.jsonl").read_text().count("\n") == summary["calls"]


def test_no_new_frame_means_no_call(tmp_path):
    decisions = Decisions()
    monitor = started(tmp_path, decisions)
    monitor.capture(np.zeros((8, 8, 3), np.uint8), 1.0)
    assert wait_for(lambda: len(decisions.requests) == 1)
    time.sleep(0.2)
    assert len(decisions.requests) == 1  # the world paused: nothing new to judge
    monitor.end_task()


def test_capture_is_due_at_capture_hz():
    monitor = Monitor(SETTINGS, Vocabularies(), Decisions())
    assert not monitor.capture_due(0.0)  # not watching
    monitor.watch = object()
    assert monitor.capture_due(0.0)
    monitor.last_shot_time = 0.0
    assert not monitor.capture_due(0.05) and monitor.capture_due(0.1) and monitor.capture_due(-1.0)


def test_the_vocabulary_replaces_the_fallback(tmp_path):
    gate, events = threading.Event(), []
    vocabularies = Vocabularies(gate)
    decisions = Decisions()
    monitor = started(tmp_path, decisions, vocabularies, events=events)
    monitor.capture(np.zeros((8, 8, 3), np.uint8), 0.0)
    assert wait_for(lambda: decisions.requests)
    assert decisions.requests[0].vocabulary.source.startswith("fallback")
    gate.set()
    assert wait_for(lambda: any(e.get("type") == "vocabulary" for e in events))
    monitor.capture(np.zeros((8, 8, 3), np.uint8), 0.1)
    assert wait_for(lambda: decisions.requests[-1].vocabulary.labels == VOCAB.labels)
    monitor.goal(REQUEST, "hands free")  # the same goal asks no second vocabulary
    monitor.end_task()
    assert len(vocabularies.requests) == 1


def test_a_sustained_failure_stops_the_running_skill_and_waits_for_the_session(tmp_path):
    events = []
    decisions = Decisions(activity={"missed": 0.95})
    monitor = started(tmp_path, decisions, events=events)
    gate_vocab = wait_for(lambda: any(e.get("type") == "vocabulary" for e in events))
    assert gate_vocab
    monitor.skill_started(replace(REQUEST, skill="pick"), 0.0)
    feed(monitor, 0.3)
    assert wait_for(lambda: monitor.stop_requested() is not None)
    assert "missed" in monitor.stop_requested()
    verdicts = monitor.take_verdicts()
    assert verdicts and verdicts[0].action == "abort_skill" and monitor.take_verdicts() == []
    monitor.skill_finished()
    assert monitor.stop_requested() is None
    summary = monitor.end_task()
    assert summary["aborts"]["g1"] >= 1
    assert any(e.get("type") == "fired" for e in events)
    assert list((tmp_path / "monitor" / "decisions").glob("*.json"))


def test_a_planning_phase_failure_only_flags(tmp_path):
    decisions = Decisions(activity={"missed": 0.95})
    monitor = started(tmp_path, decisions)
    wait_for(lambda: monitor.vocabularies[("g1", "pick the cup")] is VOCAB)
    monitor.set_phase(Phase("planning"))
    feed(monitor, 0.3)
    monitor.end_task()
    verdicts = monitor.take_verdicts()
    assert verdicts and {v.action for v in verdicts} == {"flag"} and monitor.stop_requested() is None


def test_no_credit_stops_the_watch(tmp_path):
    events = []
    monitor = started(tmp_path, Decisions(error=OutOfCredit("credit_balance_exhausted")), events=events)
    monitor.capture(np.zeros((8, 8, 3), np.uint8), 0.0)
    assert wait_for(lambda: any(e.get("fatal") for e in events))
    assert wait_for(lambda: not monitor.worker.is_alive())
    assert "no credit" in monitor.view.status
    monitor.end_task()


def test_the_cost_cap_stops_asking(tmp_path):
    class Costly(Decisions):
        def decide(self, request, lag=0.0):
            reading, recorded = super().decide(request, lag)
            return replace(reading, cost_usd=1.0), recorded
    decisions = Costly()
    monitor = started(tmp_path, decisions, settings=replace(SETTINGS, max_cost_usd=1.0))
    feed(monitor, 0.3)
    monitor.end_task()
    assert len(decisions.requests) == 1


def test_a_reading_from_an_earlier_skill_call_never_stops_the_next(tmp_path):
    gate = threading.Event()

    class Slow(Decisions):
        def decide(self, request, lag=0.0):
            gate.wait(2)  # the reply arrives after the next skill started
            return super().decide(request, lag)
    decisions = Slow(activity={"missed": 0.99})
    settings = replace(SETTINGS, ticks=1)
    monitor = started(tmp_path, decisions, settings=settings)
    wait_for(lambda: monitor.vocabularies[("g1", "pick the cup")] is VOCAB)
    monitor.skill_started(replace(REQUEST, skill="pick"), 0.0)
    monitor.capture(np.zeros((8, 8, 3), np.uint8), 0.0)
    time.sleep(0.1)  # the call for skill call 1 is under way
    monitor.skill_finished()
    monitor.skill_started(replace(REQUEST, skill="pick"), 1.0)  # call 2
    gate.set()
    assert wait_for(lambda: monitor.calls >= 1)
    time.sleep(0.05)
    assert monitor.stop_requested() is None  # the verdict is kept, but it cannot stop call 2
    assert monitor.take_verdicts()
    monitor.end_task()


def test_a_fault_in_the_worker_is_reported_and_the_watch_goes_on(tmp_path):
    events = []

    class Flaky(Decisions):
        def decide(self, request, lag=0.0):
            if not hasattr(self, "failed"):
                self.failed = True
                raise RuntimeError("bad reply")
            return super().decide(request, lag)
    decisions = Flaky()
    monitor = started(tmp_path, decisions, events=events)
    monitor.capture(np.zeros((8, 8, 3), np.uint8), 0.0)
    assert wait_for(lambda: any("RuntimeError" in e.get("error", "") for e in events))
    monitor.capture(np.zeros((8, 8, 3), np.uint8), 0.1)
    assert wait_for(lambda: monitor.calls >= 1, 5)
    monitor.end_task()


def test_ending_a_task_does_not_wait_for_a_slow_call_and_drops_its_reading(tmp_path):
    gate = threading.Event()

    class Hanging(Decisions):
        def decide(self, request, lag=0.0):
            self.requests.append(request)
            gate.wait(5)
            return Reading(request.node.goal_id, 0.0, activity={"missed": 0.99}, phase=request.phase), {}
    decisions = Hanging()
    monitor = started(tmp_path, decisions)
    monitor.capture(np.zeros((8, 8, 3), np.uint8), 0.0)
    assert wait_for(lambda: decisions.requests)
    began = time.monotonic()
    summary = monitor.end_task()
    assert time.monotonic() - began < 2 and summary["calls"] == 0
    monitor.start_task("next task", tmp_path / "next")
    gate.set()
    time.sleep(0.2)
    assert monitor.calls == 0 and monitor.take_verdicts() == []  # the old call's reading went nowhere
    monitor.end_task()


def test_after_the_clock_goes_back_the_monitor_watches_again(tmp_path):
    decisions = Decisions()
    monitor = started(tmp_path, decisions)
    monitor.capture(np.zeros((8, 8, 3), np.uint8), 50.0)
    assert wait_for(lambda: len(decisions.requests) == 1)
    monitor.shots.clear()
    monitor.capture(np.zeros((8, 8, 3), np.uint8), 0.5)  # a reset restarted simulation time
    assert wait_for(lambda: len(decisions.requests) == 2)
    monitor.end_task()


def test_skill_stages_the_monitor_was_not_told_about_do_not_make_a_stoppable_skill(tmp_path):
    monitor = started(tmp_path, Decisions())
    monitor.set_phase(Phase("planning"))
    monitor.set_phase(Phase("tool_call", "pick", "Lifting"))  # a coder call running pick
    assert monitor.phase_now.name == "planning" and monitor.known_stages() == {"pick": ("Lifting",)}
    monitor.end_task()


def test_a_claim_of_a_goal_that_no_longer_holds_is_not_asked(tmp_path):
    from homebody.monitor.targets import Claim
    held = replace(VOCAB, standing=(Claim("g0", "the jar is in the right hand"),))

    class Held(Vocabularies):
        def generate(self, request, attempt, cancelled):
            return held
    decisions = Decisions()
    monitor = started(tmp_path, decisions, Held())
    wait_for(lambda: monitor.vocabularies[("g1", "pick the cup")] is held)
    monitor.capture(np.zeros((8, 8, 3), np.uint8), 0.0)
    assert wait_for(lambda: decisions.requests)
    assert decisions.requests[-1].vocabulary.standing == ()  # g0 is not among the goals that still hold
    monitor.end_task()
