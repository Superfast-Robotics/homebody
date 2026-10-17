"""Exploration evidence and bounded continuations; the ExplorationSession owns decisions and actions."""
import json
import time

from homebody.primitives.navigation.errors import Abort, DecisionInterrupted, Stopped

from .decision_context import build_context
from .decision_interpretation import interpret
from .decision_types import Proposal, Recovery


class ExplorationPolicy:
    """Freeze one context and interpret only evidence-validated completion as terminal."""

    def __init__(self, runtime=None):
        self.runtime = runtime
        self.prepared = None

    def prepare_context(self, session, task):
        runtime = self.runtime if self.runtime is not None else session.exploration
        evidence = build_context(runtime, task.x, task.y, task.yaw,
                                 rejection=task.rejection, continuation=task.continuation,
                                 clock=time)
        self.prepared = evidence.request
        return evidence

    def offered_actions(self, session):
        if self.prepared is None:
            raise RuntimeError("Exploration actions require a prepared visual context")
        return tuple(tool.name for tool in self.prepared.tools)

    @staticmethod
    def terminal(result):
        if getattr(result, "skill", None) == "explored" and result.ok:
            return {"status": "done", "reason": result.message}
        return None

    def heartbeat(self, session):
        runtime = self.runtime if self.runtime is not None else session.exploration
        runtime.check()

    def timing(self, session):
        runtime = self.runtime if self.runtime is not None else session.exploration
        return {"wall_seconds": time.monotonic(), "active_seconds": runtime.active_elapsed(),
                "paused_seconds": runtime.paused_elapsed()}


def proposals(explorer, x: float, y: float, yaw: float):
    """Allow one stationary reassessment after two rejected walking proposals."""
    rejection, rejected, attempt = "", [], 0
    done_refusals, correcting_done = 0, False
    while True:
        x, y, yaw = explorer.decision_pose()
        explorer.refresh_planning_radius()
        decision_live = explorer.live
        try:
            continuation = (explorer._proposal_continuation(x, y, rejected)
                            if attempt >= 2 and not correcting_done else None)
            if attempt >= 2 and not correcting_done:
                verdict = continuation or {
                    "available": False,
                    "reason": ("bounded reassessment already used" if attempt >= 3 else
                               "prior refusals are not eligible walking proposals"),
                }
                rejected[-1]["proposal_continuation"] = verdict
                path = explorer.out_dir / "vlm" / f"leg_{rejected[-1]['n']}.json"
                if path.exists():
                    saved = json.loads(path.read_text())
                    saved["proposal_continuation"] = verdict
                    path.write_text(json.dumps(saved, indent=1))
                if continuation is None:
                    break
            if continuation:
                if explorer.live is not decision_live:
                    raise Abort("map frame changed during proposal admission; holding position")
                x, y, yaw = explorer.decision_pose()
                with explorer.lock:
                    if explorer.live is not decision_live:
                        raise Abort(
                            "map frame changed during proposal admission; holding position")
                    if not explorer._proposal_budget_available():
                        break
        except DecisionInterrupted:
            explorer.check()
            continue
        count = len(explorer.vlm_picks)
        try:
            leg = yield Proposal(x, y, yaw, rejection, continuation)
        except DecisionInterrupted:
            x, y, yaw = explorer.decision_pose()
            continue
        explorer.check()
        rec = explorer.vlm_picks[-1] if len(explorer.vlm_picks) > count else {}
        if not rec.get("rejection"):
            return leg
        if rec.get("kind") == "explored":
            done_refusals += 1
            rejection = f"Completion refused: {rec['rejection']}"
            if done_refusals >= 2:
                raise Stopped(f"incomplete: completion was still refused after one corrective "
                              f"decision; holding position: {rejection}")
            explorer.message = f"{rejection}. Holding for one corrective decision."
            correcting_done = True
            continue
        correcting_done = False
        code = f" [{rec['rejection_code']}]" if rec.get("rejection_code") else ""
        subject = (f"Requested {rec.get('xy')}" if rec.get("kind") == "point" else
                   f"{rec.get('kind', 'action')} proposal")
        rejection = f"{subject}{code}: {rec['rejection']}"
        explorer.message = f"Action rejected without execution. {rejection}"
        rejected.append(rec)
        attempt += 1
        if rec.get("rejection_code") == "start_blocked" and (yield Recovery(rec)):
            rejection += "; local recovery moved to a new position; reassess this fresh view"
            rejected, attempt = [], 0
        elif rec.get("departure_recovery"):
            rejection += f"; local recovery: {rec['departure_recovery']['why']}"
    attempts = {2: "twice", 3: "three times"}.get(len(rejected), f"{len(rejected)} times")
    subject = ("model destination" if all(r.get("kind") == "point" for r in rejected)
               else "model action proposals")
    raise Stopped(
        f"incomplete: {subject} rejected {attempts}; "
        f"holding position: {rejection}")



def decision(explorer, x: float, y: float, yaw: float, *, rejection: str = "",
             continuation=None):
    """Yield one frozen provider request, then bounded prepared/action requests."""
    session = getattr(explorer, "session", None)
    policy = getattr(session, "policy", None)
    if not isinstance(policy, ExplorationPolicy):
        policy = ExplorationPolicy(explorer)
    evidence = policy.prepare_context(session, Proposal(x, y, yaw, rejection, continuation))
    answer = yield evidence.request
    return (yield from interpret(explorer, evidence, answer, clock=time, terminal=policy.terminal))
