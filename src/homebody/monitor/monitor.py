"""The live monitor: watches a rollout beside the session and never blocks it.

The session thread hands it camera shots (captured at `capture_hz` simulation hertz), the goal it works on, the
phase (planning or a skill at a stage) and skill starts. A vocabulary job (one at a time) writes the goal's labels
when the goal opens and, with `refresh_on_skill`, when a skill starts; until it answers, a fallback vocabulary is
used. The decision worker asks the decision model about the newest `window_frames` shots, one call at a time, the
next waiting max(`min_period_s`, rolling p95 latency x `p95_margin`), and only when a new shot has arrived. Each
reading goes through the rules (monitor/decide.py); a verdict that stops the running skill is raised through
`stop_requested`, and every verdict waits in `take_verdicts` for the session to apply to the plan.

Inline (monitor/offline.py replays a recorded rollout): no threads; the caller hands it shots and calls `poll()`,
time is the newest shot's simulation time and calls are `period_s` simulation seconds apart.

Files under the task's `monitor/` (or `folder`): readings.jsonl (every reading), vocab-NNN/ (each vocabulary call),
decisions/NNNNN.json (the request behind each verdict, images redacted) and, with `keep_frames`, frames/.
"""
import json
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import IO

import numpy as np
from PIL import Image

from homebody.monitor.decide import MonitorState, observe, silenced, skill_started
from homebody.monitor.decision import OutOfCredit
from homebody.monitor.targets import (
    DecisionRequest,
    NodeInfo,
    Phase,
    Reading,
    Shot,
    Verdict,
    Vocabulary,
)
from homebody.monitor.vocabulary import fallback
from homebody.vlm.provider import ProviderBusy, ProviderError

HISTORY = 4000  # readings the console's timeline keeps (a long task at a few readings a second)
FINDINGS = 6  # recent verdicts the vocabulary model reads
ERROR_WAIT = 2.0  # s before asking again after a failed decision call
JOIN_WAIT = 1.0  # s end_task waits for the worker; a call still running then is discarded when it returns


@dataclass(frozen=True)
class View:
    """What the console shows, replaced whole so it reads a consistent snapshot."""
    status: str = "off"
    goal_id: str | None = None
    tree: str = ""  # the rendered plan, with the monitor's marks
    plan: dict | None = None  # the plan as the console draws it (session/monitored.plan_snapshot)
    task: str = ""
    phase: Phase = field(default_factory=Phase)
    vocabulary: Vocabulary | None = None
    reading: Reading | None = None
    history: tuple = ()  # (sim_time, top label, p, action, kind of the top label)
    spans: tuple = ()  # skill calls: (start, end or None, skill, goal id, result code or None)
    marks: tuple = ()  # operator actions: (sim_time, kind, text)
    decisions: tuple = ()  # simulation times the agent decided (several at one time: the world held still)
    skill_call: int = 0  # the skill call running now, or the last one
    sim_start: float | None = None  # simulation time of the task's first camera shot
    sim_now: float | None = None  # and of its newest
    goals: tuple = ()  # (sim_time, goal id) whenever the goal changed
    verdicts: tuple = ()  # recent non-continue verdicts, newest last
    alert: Verdict | None = None
    period_s: float = 0.0
    p95_s: float = 0.0
    calls: int = 0
    cost_usd: float = 0.0


def p95(values) -> float:
    return float(np.percentile(list(values), 95)) if values else 0.0


class Watch:
    """One task's watch: its own stop event and readings file, so a worker that outlives its task (a decision call
    still running at end_task) changes nothing when it returns."""

    def __init__(self, folder: Path):
        self.folder, self.closing = folder, threading.Event()
        self.readings: IO = (folder / "readings.jsonl").open("a", buffering=1)


def history_entry(reading: Reading, vocabulary: Vocabulary | None, action: str) -> tuple:
    """A reading as the timeline draws it: (sim_time, top label, p, action, the top label's kind, risk)."""
    top, p = reading.top()
    label = vocabulary.label(top) if vocabulary and top else None
    risk = max(reading.unsafe or 0.0, reading.human_needed or 0.0,
               reading.progress.get("no_change", 0.0) if reading.phase.name == "tool_call" else 0.0)
    return reading.sim_time, top, p, action or "continue", label.kind if label else "static", risk


def vocabulary_key(node: NodeInfo) -> tuple[str, str]:
    """A vocabulary belongs to a goal id and its text: a rewritten plan may reuse the id for another goal."""
    return node.goal_id, node.goal


class Inline:
    """The vocabulary jobs of an inline monitor: each runs when submitted."""

    @staticmethod
    def submit(function, *args):
        function(*args)

    @staticmethod
    def shutdown(**_):
        pass


class Monitor:
    def __init__(self, settings, vocabulary_model, decision_model, *,
                 sink: Callable[..., None] = lambda kind, **fields: None,
                 on_update: Callable[[View], None] | None = None,
                 inline: bool = False, period_s: float | None = None, folder: str = "monitor"):
        self.settings, self.vocabulary_model, self.decision_model = settings, vocabulary_model, decision_model
        self.sink, self.on_update = sink, on_update
        self.inline, self.period_s, self.folder = inline, period_s, folder
        self.lock = threading.Lock()
        self.watch: Watch | None = None
        self.worker = None
        self.jobs = None
        self.task_dir = None
        self._reset()

    def _reset(self):
        self.task, self.node, self.hands, self.tree, self.plan = "", NodeInfo("task", ""), "", "", None
        self.phase_now = Phase()
        self.shots = deque(maxlen=4 * self.settings.window_frames)
        self.captured_at = None  # wall time of the newest shot
        self.last_shot_time = float("-inf")
        self.vocabularies: dict[tuple[str, str], Vocabulary] = {}
        self.state = MonitorState()
        self.latencies = deque(maxlen=self.settings.p95_window)
        self.verdicts: list[Verdict] = []
        self.stop_reason = None
        self.skill_running, self.skill_call = False, 0
        self.view = View()
        self.calls = self.vocab_calls = self.fired = 0
        self.cost = 0.0
        self.stages: dict[str, list[str]] = {}
        self.findings = deque(maxlen=FINDINGS)
        self.spans: list[tuple] = []
        self.goal_marks: list[tuple] = []
        self.marks: list[tuple] = []
        self.decision_times: list[float] = []
        self.done_ids: set | None = None  # done goals whose result still holds (their claims are checked)
        self.first_shot_time = None
        self.asked_through, self.started = float("-inf"), float("-inf")  # an inline monitor's worker state
        self.halted = False  # an inline monitor met a fault that ends its watch (no decision credit)

    # --- task lifecycle ---
    def start_task(self, task: str, task_dir: Path):
        self.end_task()
        self._reset()
        self.task, self.task_dir = task, Path(task_dir) / self.folder
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.watch = Watch(self.task_dir)
        if self.inline:
            self.jobs = Inline()
        else:
            self.jobs = ThreadPoolExecutor(max_workers=1, thread_name_prefix="monitor-vocabulary")
            self.worker = threading.Thread(target=self._watch, args=(self.watch,), name="monitor-decisions",
                                           daemon=True)
            self.worker.start()
        self._publish(status="watching")

    def end_task(self) -> dict:
        """Stop watching; the task's monitor summary. Waits at most JOIN_WAIT for the worker."""
        watch = self.watch
        if watch is None:
            return {}
        with self.lock:
            self.watch, self.stop_reason, self.skill_running = None, None, False
            summary = {"calls": self.calls, "vocabulary_calls": self.vocab_calls, "verdicts": self.fired,
                       "aborts": dict(self.state.aborts), "cost_usd": round(self.cost, 6),
                       "p95_latency_s": round(p95(self.latencies), 3)}
        watch.closing.set()
        if self.worker is not None:
            self.worker.join(timeout=JOIN_WAIT)
        else:  # inline: no worker owns the readings file
            watch.readings.close()
        self.jobs.shutdown(wait=False, cancel_futures=True)
        self.worker = self.jobs = None
        self._publish(status="stopped")
        return summary

    # --- inputs from the session thread ---
    def capture_due(self, sim_time: float) -> bool:
        return self.watch is not None and (sim_time - self.last_shot_time >= 1 / self.settings.capture_hz
                                           or sim_time < self.last_shot_time)

    def capture(self, rgb: np.ndarray, sim_time: float, view: str = "head"):
        shot = Shot(np.array(rgb, dtype=np.uint8, copy=True), sim_time, view)
        with self.lock:
            self.shots.append(shot)
            if view == "head":
                self.last_shot_time = sim_time
                self.captured_at = self.now()
                if self.first_shot_time is None:
                    self.first_shot_time = sim_time
        if self.settings.keep_frames and self.task_dir is not None:
            folder = self.task_dir / "frames"
            folder.mkdir(exist_ok=True)
            Image.fromarray(shot.rgb).save(folder / f"{sim_time:010.3f}-{view}.jpg", quality=90)

    def goal(self, request, hands: str, plan: dict | None = None):
        """The goal the session works on now (a VocabularyRequest): a new goal gets a vocabulary job. A reopened
        goal that is done again may be watched for coming undone again."""
        done = {node.goal_id for node in request.done}
        with self.lock:
            self.done_ids = done
            if not self.goal_marks or self.goal_marks[-1][1] != request.node.goal_id:
                # before the first camera shot the goal starts with the task (its time is filled in then)
                started = self.last_shot_time if self.last_shot_time > float("-inf") else None
                self.goal_marks.append((started, request.node.goal_id))
            self.node, self.hands, self.tree, self.plan = request.node, hands, request.tree, plan
            self.state = replace(self.state, reopened=self.state.reopened - done)
            key = vocabulary_key(request.node)
            known = key in self.vocabularies
            if not known:
                self.vocabularies[key] = fallback(request)
        if not known:
            self._vocabulary(request)
        self._publish()

    def set_phase(self, phase: Phase):
        """A stage of the running skill (the session's skill events). Stages of skills the monitor was not told
        about (a coder call running them) do not count as a skill it may stop."""
        with self.lock:
            if phase.skill and phase.stage and phase.stage not in self.stages.setdefault(phase.skill, []):
                self.stages[phase.skill].append(phase.stage)
            if phase.name == "tool_call" and not self.skill_running:
                return
            self.phase_now = replace(phase, call=self.skill_call)

    def skill_started(self, request, sim_time: float):
        """A skill starts in the current goal (REQUEST names it): its grace period begins, and only readings
        taken from now on may stop it."""
        with self.lock:
            self.state = skill_started(self.state, request.node.goal_id, sim_time)
            self.skill_call += 1
            self.stop_reason, self.skill_running = None, True
            self.spans.append((sim_time, None, request.skill, request.node.goal_id, None))
            self.phase_now = Phase("tool_call", request.skill, call=self.skill_call)
        if self.settings.refresh_on_skill:
            self._vocabulary(request)

    def skill_finished(self, code: str | None = None, sim_time: float | None = None):
        """The running skill ended with result CODE at SIM_TIME (else the newest shot's time)."""
        with self.lock:
            self.stop_reason, self.skill_running = None, False
            self.phase_now = Phase("planning", call=self.skill_call)
            if self.spans and self.spans[-1][1] is None:
                start, _, skill, goal, _ = self.spans[-1]
                end = sim_time if sim_time is not None else max(self.last_shot_time, start)
                self.spans[-1] = (start, end, skill, goal, code)
        self._publish()

    def operator_resumed(self):
        """The operator ended a human_needed pause: the scene is theirs to judge, so this goal is not paused for
        it again."""
        with self.lock:
            self.state = silenced(self.state, self.node.goal_id, "human_needed")

    def dismiss(self) -> str | None:
        """The operator dismissed the latest alert: it leaves the view; the goal it was about, if any."""
        with self.lock:
            alert, self.view = self.view.alert, replace(self.view, alert=None)
        if alert is not None:
            self.mark("dismiss", f"dismissed {alert.label}", self.last_shot_time)
        self._publish()
        return alert.goal_id if alert is not None else None

    def mark(self, kind: str, text: str, sim_time: float):
        """An operator action, for the timeline."""
        with self.lock:
            self.marks.append((sim_time, kind, text))
        self._publish()

    def decision(self, sim_time: float):
        """The agent is deciding its next step at SIM_TIME, for the timeline."""
        with self.lock:
            self.decision_times.append(sim_time)
        self._publish()

    def stop_requested(self) -> str | None:
        """Why the running skill must stop, or None."""
        return self.stop_reason

    def take_verdicts(self) -> list[Verdict]:
        with self.lock:
            taken, self.verdicts = self.verdicts, []
        return taken

    def known_stages(self) -> dict:
        with self.lock:
            return {name: tuple(stages) for name, stages in self.stages.items()}

    def recent_findings(self) -> tuple[str, ...]:
        with self.lock:
            return tuple(self.findings)

    # --- vocabulary jobs ---
    def _vocabulary(self, request):
        watch = self.watch
        if self.jobs is None or watch is None:
            return
        self.vocab_calls += 1
        attempt = self.task_dir / f"vocab-{self.vocab_calls:03d}"
        self.jobs.submit(self._write_vocabulary, watch, request, attempt)

    def _write_vocabulary(self, watch, request, attempt):
        try:
            vocabulary = self.vocabulary_model.generate(request, attempt, watch.closing.is_set)
        except Exception as error:  # noqa: BLE001 -- a vocabulary failure must not end the watch
            vocabulary = fallback(request, f"{type(error).__name__}: {error}"[:200])
        with self.lock:
            if watch is not self.watch:  # the task ended meanwhile
                return
            self.vocabularies[vocabulary_key(request.node)] = vocabulary
        try:
            self.sink("monitor", type="vocabulary", goal_id=vocabulary.goal_id, source=vocabulary.source,
                      skill=request.skill, labels=[asdict(label) for label in vocabulary.labels],
                      standing=[asdict(claim) for claim in vocabulary.standing], attempt=attempt.name)
            self._publish()
        except Exception as error:  # noqa: BLE001 -- a recording failure must not hide the vocabulary
            self._publish(status=f"vocabulary event failed: {type(error).__name__}: {error}"[:160])

    # --- the decision worker ---
    def now(self) -> float:
        """The worker's clock: wall time, or for an inline monitor the newest shot's simulation time."""
        return self.last_shot_time if self.inline else time.monotonic()

    def period(self) -> float:
        if self.period_s is not None:
            return self.period_s
        return max(self.settings.min_period_s, p95(self.latencies) * self.settings.p95_margin)

    def poll(self) -> bool:
        """An inline monitor's turn: ask the decision model if a call is due; whether one was made."""
        watch = self.watch
        if not self.inline or watch is None:
            return False
        calls = self.calls
        try:
            self.asked_through, self.started, done = self._step(watch, self.asked_through, self.started)
            self.halted = self.halted or done
        except Exception as error:  # noqa: BLE001 -- reported like the live worker's faults
            self.sink("monitor", type="error", error=f"{type(error).__name__}: {error}"[:500])
        return self.calls > calls

    def _wait(self, watch, seconds):
        if not self.inline:
            watch.closing.wait(seconds)

    def _request(self) -> tuple[DecisionRequest | None, float]:
        with self.lock:
            heads = [shot for shot in self.shots if shot.view == "head"][-self.settings.window_frames:]
            vocabulary = self.vocabularies.get(vocabulary_key(self.node))
            if not heads or vocabulary is None:
                return None, 0.0
            others = [shot for shot in self.shots if shot.view != "head"][-self.settings.window_frames:]
            if self.done_ids is not None and vocabulary.standing:  # a claim whose goal no longer holds is not asked
                vocabulary = replace(vocabulary, standing=tuple(c for c in vocabulary.standing
                                                                if c.goal_id in self.done_ids))
            lag = self.now() - self.captured_at
            shots = tuple(sorted((*heads, *others), key=lambda shot: (shot.view != "head", shot.sim_time)))
            return DecisionRequest(self.task, self.node, self.phase_now, self.hands, vocabulary, shots), lag

    def _watch(self, watch: Watch):
        asked_through, started = float("-inf"), float("-inf")
        try:
            while not watch.closing.is_set():
                try:
                    asked_through, started, done = self._step(watch, asked_through, started)
                except Exception as error:  # noqa: BLE001 -- a fault is reported, never a silent end of the watch
                    self.sink("monitor", type="error", error=f"{type(error).__name__}: {error}"[:500])
                    self._publish(status=f"monitor fault: {type(error).__name__}: {error}"[:160])
                    watch.closing.wait(ERROR_WAIT)
                    continue
                if done:
                    return
        finally:
            watch.readings.close()

    def _step(self, watch, asked_through, started):
        """One pass of the worker: (asked_through, started, whether the watch ends)."""
        wait = started + self.period() - self.now()
        if wait > 0:
            self._wait(watch, wait)
            return asked_through, started, False
        if self.settings.max_cost_usd and self.cost >= self.settings.max_cost_usd:
            self._publish(status=f"budget spent (${self.cost:.2f})")
            self._wait(watch, 0.5)
            return asked_through, started, False
        request, lag = self._request()
        newest = max((shot.sim_time for shot in request.shots), default=None) if request else None
        if newest is not None and newest < asked_through:
            asked_through = float("-inf")  # the simulation clock went back (a reset): watch from its new time
        if newest is None or newest <= asked_through:
            self._wait(watch, 0.02)  # no new shot: the world is paused or the robot not yet seen
            return asked_through, started, False
        started, asked_through = self.now(), newest
        try:
            reading, recorded = self.decision_model.decide(request, lag)
        except OutOfCredit as error:
            self.sink("monitor", type="error", error=str(error)[:500], fatal=True)
            self._publish(status="stopped: the decision account has no credit")
            return asked_through, started, True
        except (ProviderBusy, ProviderError) as error:
            self.sink("monitor", type="error", error=str(error)[:500])
            self._publish(status=f"decision call failed: {str(error)[:120]}")
            self._wait(watch, ERROR_WAIT)
            return asked_through, started, False
        self._read(watch, request, reading, recorded)
        return asked_through, started, False

    def _read(self, watch, request, reading, recorded):
        verdict = None
        with self.lock:
            if watch.closing.is_set():  # the task ended while the call ran
                return
            self.calls += 1
            self.cost += reading.cost_usd
            self.latencies.append(reading.latency_s)
            if reading.goal_id == self.node.goal_id:
                self.state, verdict = observe(self.state, reading, request.vocabulary, self.settings)
            if verdict is not None and verdict.action != "continue":
                self.fired += 1
                self.verdicts.append(verdict)
                self.findings.append(f"{verdict.action} {verdict.text()}")
                this_skill = (self.skill_running and reading.phase.name == "tool_call"
                              and reading.phase.call == self.skill_call)
                if verdict.action in ("abort_skill", "reopen_goal") and this_skill:
                    self.stop_reason = verdict.text()
            calls = self.calls
        watch.readings.write(json.dumps({**asdict(reading), "action": verdict.action if verdict else None}) + "\n")
        if verdict is not None and verdict.action != "continue":
            path = watch.folder / "decisions" / f"{calls:05d}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(recorded, indent=1))
            self.sink("monitor", type="fired", verdict=asdict(verdict), request=f"decisions/{path.name}",
                      simulation_time=verdict.sim_time)
        self._publish(reading=reading, verdict=verdict)

    def _publish(self, *, status=None, reading=None, verdict=None):
        with self.lock:
            view = self.view
            history = view.history
            if reading is not None:
                entry = history_entry(reading, self.vocabularies.get(vocabulary_key(self.node)),
                                      verdict.action if verdict else "continue")
                history = (*history, entry)[-HISTORY:]
            fired = verdict is not None and verdict.action != "continue"
            self.view = replace(
                view, status=status or view.status, goal_id=self.node.goal_id, tree=self.tree, phase=self.phase_now,
                plan=self.plan, task=self.task,
                vocabulary=self.vocabularies.get(vocabulary_key(self.node)), reading=reading or view.reading,
                history=history, verdicts=(*view.verdicts, verdict)[-8:] if fired else view.verdicts,
                alert=verdict if fired and verdict.action in ("abort_skill", "reopen_goal") else view.alert,
                period_s=self.period(), p95_s=p95(self.latencies), calls=self.calls, cost_usd=self.cost,
                spans=tuple(self.spans), marks=tuple(self.marks), decisions=tuple(self.decision_times),
                skill_call=self.skill_call,
                goals=tuple((self.first_shot_time if t is None else t, goal) for t, goal in self.goal_marks
                            if t is not None or self.first_shot_time is not None),
                sim_start=self.first_shot_time,
                sim_now=self.last_shot_time if self.first_shot_time is not None else None)
            view = self.view
        if self.on_update is not None:
            try:
                self.on_update(view)
            except Exception as error:  # noqa: BLE001 -- the console must not stop the watch
                self.sink("monitor", type="ui_error", error=f"{type(error).__name__}: {error}")
