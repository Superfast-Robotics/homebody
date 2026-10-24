"""Record data for the monitor benchmark: the scripted manipulation run of tools/check_manipulation.py (no model
calls) with the head camera saved at --hz simulation hertz, each frame labelled with the skill, stage and holding
hands at that time, and an optional scripted failure with a known onset.

    python tools/monitor_record.py --hz 10 [--drop-after 2.0] -- --sequence configs/continuous_four.json --no-visuals

Everything after `--` goes to check_manipulation. --drop-after S opens the carrying hand S seconds into the first
navigate that runs while a hand holds an object, for --drop-seconds, so the object falls. The attempt directory
gains monitor/frames/*.jpg, monitor/labels.jsonl ({sim_time, file, skill, stage, carried, disturbed}),
monitor/requests.jsonl (each scripted skill call with its simulation time) and monitor/disturbances.jsonl. The
labels are offline ground truth from the harness's own events; they never reach a model.
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_manipulation as manipulation


class Recording:
    """The frame and label writer and the scripted drop, driven from the backend's advance."""

    def __init__(self, folder: Path, hz: float, drop_after: float | None, drop_seconds: float):
        self.folder, self.period = folder, 1 / hz
        (folder / "frames").mkdir(parents=True, exist_ok=True)
        self.labels = (folder / "labels.jsonl").open("a", buffering=1)
        self.requests = (folder / "requests.jsonl").open("a", buffering=1)
        self.disturbances = (folder / "disturbances.jsonl").open("a", buffering=1)
        self.drop_after, self.drop_seconds = drop_after, drop_seconds
        self.last, self.skill, self.stage = float("-inf"), None, None
        self.loaded_since = self.drop = None
        self.session = None

    def on_event(self, event, time):
        if event["type"] == "skill_started":
            self.skill, self.stage = event.get("skill"), None
        elif event["type"] == "skill_stage":
            self.skill, self.stage = event.get("skill") or self.skill, event.get("stage")
            carried = self.session.context.carried if self.session else {}
            if self.skill == "navigate" and carried and self.loaded_since is None and self.drop_after is not None:
                self.loaded_since = time
        elif event["type"] == "skill_finished":
            result = event["result"]
            self.requests.write(json.dumps({"event": "finished", "sim_time": time, "skill": result.skill,
                                            "code": result.code}) + "\n")
            self.skill, self.stage = None, None

    def after_advance(self, backend):
        now = backend.time
        if self.loaded_since is not None and self.drop is None and now >= self.loaded_since + self.drop_after:
            carried = self.session.context.carried if self.session else {}
            if carried:
                self.drop = {"kind": "drop", "side": next(iter(carried)), "sim_time": now}
                self.disturbances.write(json.dumps(self.drop) + "\n")
        if self.drop is not None and now < self.drop["sim_time"] + self.drop_seconds:
            backend.grip(backend.epoch, self.drop["side"], 0.0)
        if now - self.last >= self.period or now < self.last:
            self.last = now
            name = f"{now:010.3f}.jpg"
            Image.fromarray(backend.ego_image()).save(self.folder / "frames" / name, quality=90)
            carried = sorted(self.session.context.carried) if self.session else []
            self.labels.write(json.dumps({"sim_time": now, "file": name, "skill": self.skill, "stage": self.stage,
                                          "carried": carried, "disturbed": self.drop is not None}) + "\n")

    def close(self):
        for stream in (self.labels, self.requests, self.disturbances):
            stream.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    own = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    own.add_argument("--hz", type=float, default=10.0, help="Frames per simulation second (default 10)")
    own.add_argument("--drop-after", type=float, help="Open the carrying hand this long into the first loaded walk")
    own.add_argument("--drop-seconds", type=float, default=1.0, help="How long the hand stays open (default 1 s)")
    mine = own.parse_args(argv[:split])
    args, steps = manipulation.parse_args(argv[split + 1:])
    root, attempt, settings, scene = manipulation.setup(args)
    from homebody.backends.mujoco import MujocoBackend  # the attempt's source snapshot, after setup

    recording = Recording(attempt / "monitor", mine.hz, mine.drop_after, mine.drop_seconds)
    advance, emit, scripted_init = MujocoBackend.advance, manipulation.ScriptedSession.emit, \
        manipulation.ScriptedSession.__init__
    execute = manipulation.ScriptedSession.execute

    def advancing(self, epoch, seconds):
        advance(self, epoch, seconds)
        recording.after_advance(self)

    def emitting(self, event):
        recording.on_event(event, self.backend.time)
        emit(self, event)

    def creating(self, *a, **kw):
        scripted_init(self, *a, **kw)
        recording.session = self

    def executing(self, name, arguments, frame, **kw):
        recording.requests.write(json.dumps({"event": "request", "sim_time": self.backend.time, "skill": name,
                                             "arguments": arguments}, default=str) + "\n")
        return execute(self, name, arguments, frame, **kw)

    MujocoBackend.advance = advancing
    manipulation.ScriptedSession.emit = emitting
    manipulation.ScriptedSession.__init__ = creating
    manipulation.ScriptedSession.execute = executing
    try:
        outcome = manipulation.run_attempt(args, root, attempt, settings, steps, scene)
    finally:
        recording.close()
    print(f"MONITOR DATA {attempt / 'monitor'} drop={json.dumps(recording.drop)}")
    return 0 if outcome["status"] in ("skills_completed", "skill_failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
