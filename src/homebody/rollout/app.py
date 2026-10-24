"""The dashboard's Rollouts: list this machine's recorded and imported rollouts, export one as a bundle, import a
bundle, run the offline monitor on one (its cost shown first) and, in `homebody rollout view`, replay one.

A replay takes over the dashboard it is shown on, so only a dashboard without a live session offers it.
"""
import threading
import time
from pathlib import Path

from homebody.monitor import offline
from homebody.monitor.choice import ActivityChoice
from homebody.rollout import bundle
from homebody.rollout.reader import Rollout
from homebody.rollout.replay import replay

LIST_TTL_S = 2.0  # the rollout list is read from disk at most this often (the page asks on every change)


def listed(runs: Path, imported: Path) -> dict[str, Path]:
    """Rollouts by the name the page shows: imported ones first, then this machine's recorded sessions."""
    found = {}
    for prefix, root in (("imported/", imported), ("", runs)):
        if root.is_dir():
            for folder in sorted(root.iterdir(), reverse=True):
                if (folder / "events.jsonl").is_file() and any((folder / "tasks").glob("task-*")):
                    found[prefix + folder.name] = folder
    return found


def summary(manifest: dict, task: dict | None) -> str:
    """One line about a rollout and one of its tasks."""
    scene = manifest.get("scene")
    parts = [f"{manifest.get('source', 'sim')} rollout" + (f" · {scene}" if scene else ""),
             f"{manifest.get('frames', 0)} frames"]
    if task:
        parts.append(f"{task.get('id')}: {task.get('status') or 'unfinished'}")
    parts.append("monitor logs" if manifest.get("has_monitor_logs") else "no monitor logs")
    return " · ".join(str(part) for part in parts)


class RolloutsApp:
    def __init__(self, dashboard, *, runs: Path, imported: Path, settings, models=None, replays=False):
        """MODELS() returns the (vocabulary model, decision model) an evaluation uses; without it, no evaluation.
        REPLAYS: this dashboard has no live session, so it may show a replay."""
        self.dashboard, self.runs, self.imported = dashboard, Path(runs), Path(imported)
        self.settings, self.models, self.replays = settings, models, replays
        self.exports = self.runs / "exports"
        self.message, self.opened = "", None
        self._rows, self._listed_at = [], float("-inf")
        self.busy = threading.Lock()  # one export, import, evaluation or replay at a time
        dashboard.extras["rollouts"] = self.state
        dashboard.handlers.update(rollout_open=self.open, rollout_export=self.export, rollout_import=self.import_,
                                  rollout_evaluate=self.evaluate, rollout_describe=self.describe_command)

    # --- what the page shows ---
    def rows(self) -> list[dict]:
        if time.monotonic() - self._listed_at >= LIST_TTL_S:
            self._listed_at = time.monotonic()
            self._rows = [{"name": name, "tasks": sorted(path.name for path in (folder / "tasks").glob("task-*"))}
                          for name, folder in listed(self.runs, self.imported).items()]
        return self._rows

    def state(self) -> dict:
        return {"list": self.rows(), "opened": self.opened, "message": self.message, "replays": self.replays,
                "evaluates": self.models is not None, "busy": self.busy.locked()}

    def found(self, request) -> tuple[Path, str | None]:
        folder = listed(self.runs, self.imported).get(request.get("name"))
        if folder is None:
            raise ValueError("choose a rollout from the list")
        task = request.get("task") if isinstance(request.get("task"), str) else None
        if task is not None and not (folder / "tasks" / task).is_dir():
            raise ValueError(f"{folder.name} has no {task}")
        return folder, task

    def describe(self, folder: Path, task: str | None) -> str:
        manifest = bundle.describe(folder)
        row = next((t for t in manifest["tasks"] if t["id"] == task), None)
        evaluated = task is not None and (folder / "tasks" / task / offline.FOLDER / "report.json").is_file()
        return summary(manifest, row) + (" · evaluated offline" if evaluated else "")

    def in_background(self, label: str, work) -> dict:
        if not self.busy.acquire(blocking=False):
            return {"ok": False, "message": "Another rollout job is running; wait for it to finish."}

        def run():
            try:
                self.message = work()
            except (bundle.BundleError, ValueError, OSError) as error:
                self.message = f"{label} failed: {error}"
            except Exception as error:  # noqa: BLE001 -- every failure is shown, the dashboard keeps running
                self.message = f"{label} failed: {type(error).__name__}: {error}"
            finally:
                self.busy.release()
                self._listed_at = float("-inf")  # an import or export may have changed the list
                self.dashboard.hub.changed()
        self.message = f"{label}…"
        threading.Thread(target=run, name="rollout-job", daemon=True).start()
        return {"ok": True, "message": f"{label}…"}

    # --- the page's commands ---
    def describe_command(self, request: dict) -> dict:
        folder, task = self.found(request)
        return {"ok": True, "message": self.describe(folder, task)}

    def open(self, request: dict) -> dict:
        if not self.replays:
            return {"ok": False, "message": "Replays open in their own dashboard: homebody rollout view."}
        folder, task = self.found(request)
        source = request.get("source") if request.get("source") in ("live", "offline") else "auto"

        def work():
            rollout = Rollout(folder, task, source)
            replay(self.dashboard, rollout, self.settings.monitor.tau,
                   choice=ActivityChoice(self.settings.monitor.choice_window, self.settings.monitor.choice_votes))
            self.opened = {"name": request["name"], "task": rollout.task_name, "source": rollout.source}
            return (f"Showing {folder.name} {rollout.task_name} with the {rollout.source} monitor's readings. "
                    "Click the timeline to inspect a moment.")
        return self.in_background(f"Opening {folder.name}", work)

    def export(self, request: dict) -> dict:
        folder, _ = self.found(request)

        def work():  # the whole session, as `homebody rollout export` writes it without --task
            archive = bundle.export(folder, self.exports, light=self.settings.rollouts.light)
            return f"Exported {archive} ({archive.stat().st_size / bundle.MB:.1f} MB)."
        return self.in_background(f"Exporting {folder.name}", work)

    def import_(self, request: dict) -> dict:
        path = Path(str(request.get("path", ""))).expanduser()
        if not path.is_file():
            return {"ok": False, "message": f"No bundle at {path}."}
        if path.stat().st_size > self.settings.rollouts.max_import_mb * bundle.MB:
            return {"ok": False, "message": "That bundle is larger than rollouts.max_import_mb allows."}

        def work():
            folder = bundle.import_(path, self.imported, max_mb=self.settings.rollouts.max_import_mb)
            return f"Imported imported/{folder.name}."
        return self.in_background(f"Importing {path.name}", work)

    def evaluate(self, request: dict) -> dict:
        """Without CONFIRM: the cost estimate. With it: the evaluation, in the background."""
        if self.models is None:
            return {"ok": False, "message": "No monitor models: set [monitor] and OPENAI_DECISIONS_API_KEY."}
        folder, task = self.found(request)
        planned = offline.estimate(folder, self.settings, task)
        row = planned["tasks"][0] if planned["tasks"] else {"calls": 0, "usd": 0.0, "frames": 0}
        estimate = (f"{row['frames']} recorded frames → about {row['calls']} decision calls, ${row['usd']:.2f} "
                    f"(stops at ${planned['cap_usd']:.2f}).")
        if not request.get("confirm") or not row["calls"]:
            return {"ok": True, "message": estimate, "calls": row["calls"]}

        def work():
            vocabulary, decisions = self.models()
            result = offline.evaluate(folder, self.settings, vocabulary, decisions, task=task)
            report = result["tasks"][0]
            would = len(report["would"])
            return (f"Evaluated {report['task']}: {report['calls']} calls, ${report['cost_usd']:.3f}, {would} "
                    f"intervention{'s' if would != 1 else ''} it would have made. Open it to see them.")
        return self.in_background(f"Evaluating {folder.name}", work)
