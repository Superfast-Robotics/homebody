"""How the store follows interventions: a request to stop the running skill, from the live monitor or the operator,
through the stages the components that handle it report.

Each stage comes from the component that owns it: the monitor's request (its reading's verdict), the skill
executor's acceptance (the session's "accepted" event, when the skill's cancellation check took the request), its
result (the skill's own result code: the stop code confirms, another code fails or cancels the request), and the
harness's refusal of a stale request. "Timed out" and "executing" are only derived for display: a request no
component answered within ACK_TIMEOUT_S, and an accepted request whose skill has not ended yet. The robot physically
stopping is never claimed: no motion or safety signal reaches the dashboard.
"""
from dataclasses import asdict, replace

from homebody.ui.dashboard.records import Intervention, clip

ACK_TIMEOUT_S = 10.  # s a request may wait for the executor to take it before it shows as timed out
FINAL = ("confirmed", "rejected", "failed", "cancelled")
INITIATORS = {"monitor": "Live monitor", "operator": "Human operator"}
DECIDERS = {"monitor": "Monitor rules (abort on a failure label, stall or unsafe reading)",
            "operator": "Human operator"}
STOP_CODES = {"monitor": "MONITOR_STOPPED", "operator": "OPERATOR_STOPPED"}
INTERVENTION_LIMIT = 100


def state_of(intervention: Intervention, now: float) -> str:
    """The intervention's state for the page: its newest final stage, else executing (accepted, the skill still
    ending), timed out (requested, unanswered for ACK_TIMEOUT_S) or requested."""
    stages = [stage for stage, *_ in intervention.stages]
    final = next((stage for stage in reversed(stages) if stage in FINAL), None)
    if final is not None:
        return final
    if "accepted" in stages:
        return "executing"
    requested = next((t for stage, t, *_ in intervention.stages if stage == "requested" and t is not None), None)
    if requested is not None and now - requested > ACK_TIMEOUT_S:
        return "timed_out"
    return "requested"


class InterventionTracking:
    """Mixed into Store; every method runs under the store's lock."""

    def _open_intervention(self, initiator):
        """The newest unfinished request from INITIATOR in this task, if any."""
        task = self.task["id"] if self.task else None
        return next((i for i in reversed(self.interventions.values()) if i.initiator == initiator and i.task == task
                     and not any(stage in FINAL for stage, *_ in i.stages)), None)

    def _stage(self, intervention, stage, source, note=""):
        """Add STAGE once (a repeated event is the same stage, not a second one)."""
        if any(existing == stage for existing, *_ in intervention.stages):
            return intervention
        updated = replace(intervention, stages=(*intervention.stages, (stage, self.clock(), source, clip(note, 300))))
        self.interventions[updated.id] = updated
        if stage in FINAL and updated.incident in self.incidents:
            self._outcome(updated.incident, self._outcome_text(updated, stage))
        return updated

    @staticmethod
    def _outcome_text(intervention, stage):
        return {"confirmed": f"{intervention.initiator}'s stop: the skill executor stopped the skill",
                "failed": f"{intervention.initiator}'s stop: the skill did not stop as asked",
                "rejected": f"{intervention.initiator}'s stop: refused",
                "cancelled": f"{intervention.initiator}'s stop: withdrawn before it was taken"}[stage]

    def request_intervention(self, who, reason, *, incident=None, detection=""):
        """WHO ("monitor" or "operator") asked the executor to stop the running skill. A second request while one is
        open is the same request."""
        with self.lock:
            initiator = INITIATORS[who]
            now = self.clock()
            tool = self._running_tool()
            note = reason if tool is not None else f"{reason} (no skill was running on the dashboard's record)"
            requested = ("requested", now, initiator, clip(note, 300))
            detected = (("detected", incident.t, incident.detection or detection, clip(incident.title, 300)),) \
                if incident is not None else ()
            open_ = self._open_intervention(initiator)
            if open_ is not None:
                if any(stage == "requested" and t is None for stage, t, *_ in open_.stages):
                    # the executor's acceptance reached the dashboard first (another thread): the request fills in
                    rest = tuple(st for st in open_.stages if st[0] != "requested")
                    filled = replace(open_, stages=(*detected, requested, *rest), incident=incident and incident.id,
                                     detection=detection, reason=clip(reason, 600))
                    self.interventions[filled.id] = filled
                    self._touch()
                return None
            stages = [*detected, requested]
            intervention = Intervention(
                self._id("v"), now, "stop_skill", initiator, decided_by=DECIDERS[who], detection=detection,
                tool=tool and tool.id, skill=tool and tool.skill, node=tool and tool.goal,
                incident=incident and incident.id, task=self.task["id"] if self.task else None,
                reason=clip(reason, 600), stages=tuple(stages))
            self._bounded_interventions()[intervention.id] = intervention
            self.log("warning", "operator" if who == "operator" else "monitor",
                     f"Stop requested by the {initiator.lower()}: {reason}", intervention.node)
            self._touch()
            return intervention

    def _bounded_interventions(self):
        while len(self.interventions) >= INTERVENTION_LIMIT:
            self.interventions.popitem(last=False)
        return self.interventions

    def _intervention_event(self, event):
        """A stage another component reported: the executor's "accepted", "confirmed" or "failed", the harness's
        "rejected". Without an open request (one the dashboard did not see), the stage starts a record of its own."""
        who = event.get("initiator") if event.get("initiator") in INITIATORS else "monitor"
        initiator, stage = INITIATORS[who], event.get("stage")
        if stage not in ("accepted", "confirmed", "failed", "rejected"):
            return
        intervention = self._open_intervention(initiator)
        if intervention is None:
            if stage in ("confirmed", "failed") and self._closed_by_result(initiator):
                return  # the skill's result already closed it; this is the executor saying so again
            tool = self._running_tool()
            intervention = Intervention(
                self._id("v"), self.clock(), str(event.get("action", "stop_skill")), initiator,
                decided_by=DECIDERS[who], tool=tool and tool.id, skill=event.get("skill") or (tool and tool.skill),
                node=tool and tool.goal, task=self.task["id"] if self.task else None,
                reason=clip(event.get("reason", ""), 600),
                stages=(("requested", None, initiator, "The request itself did not reach the dashboard"),))
            self._bounded_interventions()[intervention.id] = intervention
        source = {"rejected": "Harness (monitor link)"}.get(stage, "Skill executor (session)")
        self._stage(intervention, stage, source, event.get("note") or event.get("reason", ""))
        self.log("error" if stage in ("failed", "rejected") else "info", "session",
                 f"{initiator}'s stop {stage}: {event.get('note', '')}", intervention.node)

    def _closed_by_result(self, initiator):
        task = self.task["id"] if self.task else None
        newest = next((i for i in reversed(self.interventions.values())
                       if i.initiator == initiator and i.task == task), None)
        return newest is not None and any(stage in FINAL for stage, *_ in newest.stages)

    def _skill_ended(self, tool, code):
        """The skill's own result settles the requests on it: its stop code confirms one the executor took, any
        other code fails it, and one never taken was withdrawn when the skill ended first."""
        for intervention in list(self.interventions.values()):
            if tool is None or intervention.tool != tool.id or any(s in FINAL for s, *_ in intervention.stages):
                continue
            who = next(key for key, name in INITIATORS.items() if name == intervention.initiator)
            taken = any(stage == "accepted" for stage, *_ in intervention.stages)
            if code == STOP_CODES[who]:
                self._stage(intervention, "confirmed", f"Skill executor (result {code})", "The skill ended as asked")
            elif taken:
                self._stage(intervention, "failed", f"Skill executor (result {code})", f"The skill ended with {code}")
            else:
                self._stage(intervention, "cancelled", f"Skill executor (result {code})",
                            f"The skill ended with {code} before taking the request")

    def _task_ended_interventions(self):
        task = self.task["id"] if self.task else None
        for intervention in list(self.interventions.values()):
            if intervention.task == task and not any(s in FINAL for s, *_ in intervention.stages):
                self._stage(intervention, "cancelled", "Task execution service", "The task ended first")

    def intervention_rows(self, now):
        return [{**asdict(i), "state": state_of(i, now), "ack_timeout": ACK_TIMEOUT_S}
                for i in self.interventions.values()]
