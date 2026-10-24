"""How the store follows interventions: a request to stop one skill attempt, from the live monitor or the operator,
through the stages the components that handle it report.

Each stage comes from the component that owns it: the monitor's detection and decision (one reading's verdict,
naming the attempt it watched by its call number), the request, the skill executor's acceptance (the session's
"accepted" event, when the skill's cancellation check took the request), the skill's result (its stop code confirms,
another code fails or cancels the request), and the harness's refusal of a stale request. Everything matches by
attempt (the task's skill call number), never by which attempt happens to run when an event arrives; an event that
names no attempt matches none. "Timed out" and "executing" are only derived for display, and the robot physically
stopping is never claimed. When the stopped attempt's goal makes its next attempt, that is the retry (the plan's goal
is its retry boundary), recorded with its outcome.
"""
from dataclasses import asdict, replace

from homebody.ui.dashboard.records import Intervention, call_number, clip

FINAL = ("confirmed", "rejected", "failed", "cancelled")
INITIATORS = {"monitor": "Live monitor", "operator": "Human operator"}
DECIDERS = {"monitor": "Monitor rules (abort on a failure label, stall or unsafe reading)",
            "operator": "Human operator"}
STOP_CODES = {"monitor": "MONITOR_STOPPED", "operator": "OPERATOR_STOPPED"}
INTERVENTION_LIMIT = 100


def final_stage(intervention: Intervention):
    return next((stage for stage, *_ in reversed(intervention.stages) if stage in FINAL), None)


def state_of(intervention: Intervention, now: float, ack_timeout: float) -> str:
    """The intervention's state for the page: its final stage, else executing (accepted, the skill still ending),
    timed out (requested, unanswered for ACK_TIMEOUT seconds) or requested."""
    final = final_stage(intervention)
    if final is not None:
        return final
    stages = [stage for stage, *_ in intervention.stages]
    if "accepted" in stages:
        return "executing"
    requested = next((t for stage, t, *_ in intervention.stages if stage == "requested" and t is not None), None)
    if requested is not None and now - requested > ack_timeout:
        return "timed_out"
    return "requested"


class InterventionTracking:
    """Mixed into Store; every method runs under the store's lock."""

    def _task_id(self):
        return self.task["id"] if self.task else None

    def _intervention_on(self, initiator, tool_id, *, open_only=True):
        """The newest request from INITIATOR on attempt TOOL_ID in this task (unfinished ones only, by default)."""
        return next((i for i in reversed(self.interventions.values())
                     if i.initiator == initiator and i.task == self._task_id() and i.tool == tool_id
                     and not (open_only and final_stage(i))), None)

    def _stage(self, intervention, stage, source, note="", t=None):
        """Add STAGE once (a repeated event is the same stage, not a second one)."""
        if any(existing == stage for existing, *_ in intervention.stages):
            return intervention
        updated = replace(intervention, stages=(*intervention.stages, (
            stage, self.clock() if t is None else t, source, clip(note, 300))))
        self.interventions[updated.id] = updated
        if stage in FINAL and updated.incident in self.incidents:
            self._outcome(updated.incident, self._outcome_text(updated, stage))
        return updated

    @staticmethod
    def _outcome_text(intervention, stage):
        attempt = f" ({intervention.skill} attempt {intervention.attempt})" if intervention.attempt else ""
        return {"confirmed": f"{intervention.initiator}'s stop{attempt}: the skill executor stopped the skill",
                "failed": f"{intervention.initiator}'s stop{attempt}: the skill did not stop as asked",
                "rejected": f"{intervention.initiator}'s stop{attempt}: refused",
                "cancelled": f"{intervention.initiator}'s stop{attempt}: withdrawn before it was taken"}[stage]

    def _new_intervention(self, who, tool, *, reason="", stages=(), incident=None, detection=""):
        intervention = Intervention(
            self._id("v"), self.clock(), "stop_skill", INITIATORS[who], decided_by=DECIDERS[who], detection=detection,
            tool=tool and tool.id, skill=tool and tool.skill, node=tool and tool.goal,
            incident=incident and incident.id, task=self._task_id(), reason=clip(reason, 600), stages=tuple(stages),
            execution=tool.execution if tool else "", attempt=tool.number if tool else None)
        while len(self.interventions) >= INTERVENTION_LIMIT:
            self.interventions.popitem(last=False)
        self.interventions[intervention.id] = intervention
        return intervention

    def request_intervention(self, who, reason, *, incident=None, detection="", tool=None):
        """WHO ("monitor" or "operator") asked the executor to stop an attempt: TOOL, the attempt the monitor's
        reading watched (None when it named none: the request is kept, its attempt Unknown); the operator's stop is
        for the attempt running now. A second request on the same attempt while one is open is the same request."""
        with self.lock:
            initiator, now = INITIATORS[who], self.clock()
            if who == "operator":
                tool = self._running_tool()
            note = reason if tool is not None else f"{reason} (the attempt it was for is unknown)"
            requested = ("requested", now, initiator, clip(note, 300))
            before = ()
            if incident is not None:
                before = (("detected", incident.t, incident.detection or detection, clip(incident.title, 300)),
                          ("decided", incident.t, DECIDERS[who], "Abort the skill"))
            open_ = self._intervention_on(initiator, tool and tool.id)
            if open_ is not None:
                if any(stage == "requested" and t is None for stage, t, *_ in open_.stages):
                    # the executor's acceptance reached the dashboard first (another thread): the request fills in
                    rest = tuple(st for st in open_.stages if st[0] != "requested")
                    self.interventions[open_.id] = replace(open_, stages=(*before, requested, *rest),
                                                           incident=incident and incident.id, detection=detection,
                                                           reason=clip(reason, 600))
                    self._touch()
                return None
            if tool is not None and tool.end is not None:  # a late reading of an attempt that has ended
                if self._intervention_on(initiator, tool.id, open_only=False) is not None:
                    return None  # that attempt's own request already holds it
                ended = self._new_intervention(who, tool, reason=reason, stages=(*before, requested),
                                               incident=incident, detection=detection)
                self._stage(ended, "cancelled", f"Skill executor (result {tool.code})",
                            f"Attempt {tool.number} had already ended with {tool.code} when the request arrived")
                return ended
            intervention = self._new_intervention(who, tool, reason=reason, stages=(*before, requested),
                                                  incident=incident, detection=detection)
            self.log("warning", "operator" if who == "operator" else "monitor",
                     f"Stop requested by the {initiator.lower()}"
                     f"{f' for {tool.skill} attempt {tool.number}' if tool else ''}: {reason}", intervention.node)
            self._touch()
            return intervention

    def _intervention_event(self, event):
        """A stage another component reported for the attempt its "call" names: the executor's "accepted",
        "confirmed" or "failed", the harness's "rejected". Without a request on that attempt (one the dashboard did
        not see), the stage starts a record of its own."""
        who = event.get("initiator") if event.get("initiator") in INITIATORS else "monitor"
        initiator, stage = INITIATORS[who], event.get("stage")
        if stage not in ("accepted", "confirmed", "failed", "rejected"):
            return
        call = call_number(event.get("call"))
        tool = self._tool_for_call(call) if call is not None else (None if stage == "rejected" else self._running_tool())
        intervention = self._intervention_on(initiator, tool and tool.id)
        if intervention is None:
            closed = self._intervention_on(initiator, tool and tool.id, open_only=False) if tool else next(
                (i for i in reversed(self.interventions.values()) if i.initiator == initiator and i.task == self._task_id()),
                None)
            if stage in ("confirmed", "failed") and closed is not None and final_stage(closed):
                return  # the skill's result already closed it; this is the executor saying so again
            intervention = self._new_intervention(
                who, tool, reason=event.get("reason", ""),
                stages=(("requested", None, initiator, "The request itself did not reach the dashboard"),))
        source = {"rejected": "Harness (monitor link)"}.get(stage, "Skill executor (session)")
        self._stage(intervention, stage, source, event.get("note") or event.get("reason", ""))
        self.log("error" if stage in ("failed", "rejected") else "info", "session",
                 f"{initiator}'s stop {stage}: {event.get('note', '')}", intervention.node)

    def _skill_ended(self, tool, code):
        """The attempt's own result settles the requests on it: its stop code confirms one the executor took, any
        other code fails it, and one never taken was withdrawn when the skill ended first. An attempt that was a
        retry after a stop records how the retry ended."""
        for intervention in list(self.interventions.values()):
            if tool is None:
                break
            if intervention.tool == tool.id and not final_stage(intervention):
                who = next(key for key, name in INITIATORS.items() if name == intervention.initiator)
                taken = any(stage == "accepted" for stage, *_ in intervention.stages)
                if code == STOP_CODES[who]:
                    self._stage(intervention, "confirmed", f"Skill executor (result {code})", "The skill ended as asked")
                elif taken:
                    self._stage(intervention, "failed", f"Skill executor (result {code})", f"The skill ended with {code}")
                else:
                    self._stage(intervention, "cancelled", f"Skill executor (result {code})",
                                f"The skill ended with {code} before taking the request")
            retry = next((st for st in intervention.stages if st[0] == "retry_started"), None)
            if retry is not None and retry[3].endswith(f"[{tool.id}]"):
                self._stage(intervention, "retry_finished", f"Skill executor (result {code})",
                            f"Attempt {tool.number} ended with {code} [{tool.id}]")

    def _retry_started(self, tool):
        """TOOL started in its goal: for a stop confirmed on the goal's previous attempt, the goal trying again."""
        for intervention in reversed(self.interventions.values()):
            if intervention.task != self._task_id() or intervention.node != tool.goal or tool.goal is None:
                continue
            if final_stage(intervention) == "confirmed" and intervention.tool != tool.id and \
                    not any(stage == "retry_started" for stage, *_ in intervention.stages):
                self._stage(intervention, "retry_started", "Plan goal (its next skill call, chosen by the agent)",
                            f"{tool.skill} attempt {tool.number} [{tool.id}]")
            return

    def _task_ended_interventions(self):
        for intervention in list(self.interventions.values()):
            if intervention.task == self._task_id() and not final_stage(intervention):
                self._stage(intervention, "cancelled", "Task execution service", "The task ended first")

    def intervention_rows(self, now, ack_timeout):
        return [{**asdict(i), "state": state_of(i, now, ack_timeout), "ack_timeout": ack_timeout}
                for i in self.interventions.values()]
