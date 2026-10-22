"""Profile one headless session, the robot replaying a recorded session's decisions (no model calls).
usage: python profile_session.py EVENTS_JSONL OUT_PSTATS -- <homebody CLI args, --model replay>"""
import cProfile
import json
import sys

from homebody.session import config
from homebody.vlm import catalog
from homebody.vlm.provider import Decision

events, out, argv = sys.argv[1], sys.argv[2], sys.argv[sys.argv.index("--") + 1:]
recorded = [Decision(e["decision"]["text"], e["decision"]["skill"], e["decision"]["arguments"])
            for e in map(json.loads, open(events)) if e["kind"] == "decision"]


class Replay:
    LABEL, KEY = "Replay", "HOME"

    def __init__(self, settings):
        self.settings, self.queue = settings, list(recorded)

    def generate(self, prompt, *, images, names, attempt, tick, cancelled):
        attempt.mkdir(parents=True, exist_ok=True)
        if not self.queue:
            return Decision("replay finished", "done", {"summary": "replay finished"})
        return self.queue.pop(0)


catalog.PROVIDERS["replay"] = Replay
config.MODELS["replay"] = ("replay", "replay", None)
from homebody.__main__ import main  # noqa: E402 -- after the provider is registered

sys.argv = ["homebody", *argv]
profiler = cProfile.Profile()
try:
    profiler.runcall(main)
except SystemExit:
    pass
finally:
    profiler.dump_stats(out)
