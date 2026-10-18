"""Per-stage traces of skill calls: at every stage a skill enters, a keyframe from the head camera and from the
wrist camera of the hand in use, with what each hand holds; the skill's plan outline and its result. The coder
gets the trace of the call it is asked to fix, and of each of its own executes.

    <task_dir>/trace/call-007/trace.json      {skill, side, plan, stages: [{stage, time, head, wrist, carried}], result}
    <task_dir>/trace/call-007/03-lifting-head.jpg
"""
import json
import re
from pathlib import Path

from PIL import Image

from homebody.session.recorder import json_value

PLAN_CHARS = 1200  # the plan outline kept in the trace
JPEG_QUALITY = 85


class Trace:
    """Collects one skill call at a time from the session's skill events."""

    def __init__(self, backend, carried):
        self.backend, self.carried = backend, carried  # carried: () -> {side: record}
        self.root: Path | None = None
        self.calls, self.current, self.folder = 0, None, None

    def reset(self, task_dir: Path | None):
        self.root = None if task_dir is None else task_dir / "trace"
        self.calls, self.current, self.folder = 0, None, None

    def on_event(self, event, side):
        """Feed one skill event; SIDE is the hand the call uses, or None."""
        if self.root is None:
            return
        kind = event.get("type")
        if kind == "skill_started":
            self.calls += 1
            self.folder = self.root / f"call-{self.calls:03d}"
            self.folder.mkdir(parents=True, exist_ok=True)
            self.current = {"skill": event.get("skill"), "side": side, "plan": None, "stages": [], "result": None}
            self._keyframe("start")
        elif self.current is None:
            return
        elif kind == "skill_prepared":
            self.current["plan"] = json.dumps(json_value(event.get("plan")), default=str)[:PLAN_CHARS]
        elif kind == "skill_stage":
            self._keyframe(event.get("stage") or "stage")
        elif kind == "skill_finished":
            self._keyframe("end")
            result = event.get("result")
            self.current["result"] = json_value(result.as_dict() if hasattr(result, "as_dict") else result)
            (self.folder / "trace.json").write_text(json.dumps(self.current, indent=1, default=str))
            self.current = None

    def latest(self, skill=None):
        """The newest finished call's folder (of SKILL when given), or None."""
        if self.root is None or not self.root.exists():
            return None
        for folder in sorted(self.root.glob("call-*"), reverse=True):
            path = folder / "trace.json"
            if path.exists() and (skill is None or json.loads(path.read_text()).get("skill") == skill):
                return folder
        return None

    def _keyframe(self, stage):
        index = len(self.current["stages"])
        name = f"{index:02d}-{re.sub(r'[^a-z0-9]+', '-', stage.lower()).strip('-')[:40] or 'stage'}"
        entry = {"stage": stage, "time": round(float(self.backend.time), 2),
                 "carried": sorted(self.carried().keys()), "head": None, "wrist": None}
        entry["head"] = self._save(self.backend.ego_image(), f"{name}-head.jpg")
        side = self.current["side"]
        if side in ("left", "right"):
            entry["wrist"] = self._save(self.backend.wrist_view(side).rgb, f"{name}-wrist-{side}.jpg")
        self.current["stages"].append(entry)

    def _save(self, rgb, filename):
        Image.fromarray(rgb).save(self.folder / filename, quality=JPEG_QUALITY)
        return filename


def summary(folder: Path) -> str:
    """The trace as text for a request: one line per stage, then the result."""
    trace = json.loads((folder / "trace.json").read_text())
    lines = [f"{trace['skill']} ({trace.get('side') or 'no hand'}), plan: {trace.get('plan') or 'none recorded'}"]
    for stage in trace["stages"]:
        images = ", ".join(x for x in (stage["head"], stage["wrist"]) if x)
        lines.append(f"- t={stage['time']:.2f} s {stage['stage']}: holding {stage['carried'] or 'nothing'}; {images}")
    result = trace.get("result") or {}
    lines.append(f"result: {result.get('code')}: {str(result.get('message'))[:300]}")
    return "\n".join(lines)


def last_images(folder: Path, count=2):
    """The newest COUNT keyframe files of a trace (wrist first: closest to the failure)."""
    trace = json.loads((folder / "trace.json").read_text())
    files = [s[k] for s in trace["stages"][::-1] for k in ("wrist", "head") if s[k]]
    return [folder / f for f in files[:count]]
