"""The console's Rollouts folder: open a recorded rollout in the main views, export one, import one, and run the
offline monitor on one that has no monitor logs.

A replay fills the strip, plan, timeline, monitor panel, camera and agent bubble from `rollout.reader.Rollout` at a
time chosen on a slider (or played); the live run keeps going underneath and is drawn again on "Back to live".
"""
import tempfile
import threading
import time
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image

from homebody.monitor import offline
from homebody.rollout import bundle
from homebody.rollout.reader import Rollout
from homebody.ui.console import thought_bubble

SPEEDS = {"1x": 1.0, "2x": 2.0, "4x": 4.0, "8x": 8.0}
PLAY_STEP = 0.2  # wall seconds between replay frames while playing
NONE = "(no rollouts yet)"


def listed(runs: Path, imported: Path) -> dict[str, Path]:
    """Rollouts by the name the dropdown shows: imported ones first, then this machine's recorded sessions."""
    found = {}
    for prefix, root in (("imported/", imported), ("", runs)):
        if root.is_dir():
            for folder in sorted(root.iterdir(), reverse=True):
                if (folder / "events.jsonl").is_file() and any((folder / "tasks").glob("task-*")):
                    found[prefix + folder.name] = folder
    return found


def summary(manifest: dict, task: dict | None) -> str:
    """One line about a rollout; its manifest's words may come from another machine, so they are escaped."""
    scene = manifest.get("scene")
    parts = [f"{manifest.get('source', 'sim')} rollout" + (f" · {scene}" if scene else ""),
             f"{manifest.get('frames', 0)} frames"]
    if task:
        parts.append(f"{task.get('id')}: {task.get('status') or 'unfinished'}")
    parts.append("monitor logs" if manifest.get("has_monitor_logs") else "no monitor logs")
    return escape(" · ".join(str(part) for part in parts))


class RolloutsPanel:
    def __init__(self, console, *, runs: Path, imported: Path, settings, models=None):
        """MODELS() returns the (vocabulary model, decision model) an evaluation uses; without it, Evaluate is off."""
        self.console, self.runs, self.imported, self.settings, self.models = console, runs, imported, settings, models
        self.rollout: Rollout | None = None
        self.playing = threading.Event()
        self.busy = threading.Lock()  # one export, import or evaluation at a time
        self.planned = None
        gui = console.server.gui
        with gui.add_folder("Rollouts", expand_by_default=False):
            self.choice = gui.add_dropdown("Rollout", (NONE,), initial_value=NONE)
            self.task_choice = gui.add_dropdown("Task", ("task-000",), initial_value="task-000")
            self.about = gui.add_markdown("")
            self.open_button = gui.add_button("Open replay", icon="player-play", color="gray")
            self.export_button = gui.add_button("Export", icon="download", color="gray",
                                                hint="Download this rollout as a .tar.gz bundle")
            self.evaluate_button = gui.add_button("Evaluate with the monitor", icon="eye", color="gray",
                                                  disabled=models is None,
                                                  hint="Run the monitor over the recorded frames; shows the cost first")
            self.confirm_button = gui.add_button("Run evaluation", icon="eye", color="blue", visible=False)
            self.upload = gui.add_upload_button("Import a bundle", icon="upload", color="gray",
                                                mime_type=".gz,application/gzip")
            self.message = gui.add_markdown("")
            self.slider = gui.add_slider("Time", min=0.0, max=1.0, step=0.1, initial_value=0.0, visible=False)
            self.play_button = gui.add_button("Play", icon="player-play", color="gray", visible=False)
            self.speed = gui.add_dropdown("Speed", tuple(SPEEDS), initial_value="1x", visible=False)
            self.live_button = gui.add_button("Back to live", icon="arrow-back-up", color="gray", visible=False)
        self.choice.on_update(lambda _: self.chose())
        self.task_choice.on_update(lambda _: self.describe())
        self.open_button.on_click(lambda _: self.open())
        self.export_button.on_click(lambda event: self.in_background(self.export, event.client))
        self.evaluate_button.on_click(lambda _: self.plan_evaluation())
        self.confirm_button.on_click(lambda _: self.in_background(self.evaluate))
        self.upload.on_upload(lambda _: self.in_background(self.import_))
        self.slider.on_update(lambda _: self.draw(self.slider.value))
        self.play_button.on_click(lambda _: self.toggle_play())
        self.live_button.on_click(lambda _: self.close())
        self.refresh()

    # --- the list ---
    def refresh(self, select: str | None = None):
        self.found = listed(self.runs, self.imported)
        names = tuple(self.found) or (NONE,)
        self.choice.options = names
        self.choice.value = select if select in self.found else names[0]
        self.chose()

    def folder(self) -> Path | None:
        return self.found.get(self.choice.value)

    def chose(self):
        folder = self.folder()
        tasks = tuple(sorted(path.name for path in (folder / "tasks").glob("task-*"))) if folder else ("task-000",)
        self.task_choice.options = tasks or ("task-000",)
        self.task_choice.value = self.task_choice.options[-1]
        self.describe()

    def describe(self):
        folder = self.folder()
        enabled = folder is not None
        for button in (self.open_button, self.export_button):
            button.disabled = not enabled
        self.evaluate_button.disabled = not enabled or self.models is None
        self.confirm_button.visible, self.planned = False, None
        if folder is None:
            self.about.content = "Recorded and imported rollouts appear here."
            return
        if self.task_choice.value not in self.task_choice.options:
            return
        manifest = bundle.describe(folder)
        task = next((t for t in manifest["tasks"] if t["id"] == self.task_choice.value), None)
        offline_report = folder / "tasks" / self.task_choice.value / offline.FOLDER / "report.json"
        evaluated = " · evaluated offline" if offline_report.is_file() else ""
        self.about.content = summary(manifest, task) + evaluated

    def in_background(self, work, *args):
        def run():
            if not self.busy.acquire(blocking=False):
                self.message.content = "Another rollout job is running; wait for it to finish."
                return
            try:
                work(*args)
            except Exception as error:  # noqa: BLE001 -- every failure is shown, the console keeps running
                self.message.content = f"**Failed:** {type(error).__name__}: {error}"
            finally:
                self.busy.release()
        threading.Thread(target=run, name="rollout-job", daemon=True).start()

    # --- export, import, evaluate ---
    def export(self, client):
        folder = self.folder()
        if folder is None or client is None:
            return
        self.message.content = f"Packing {folder.name}…"
        with tempfile.TemporaryDirectory() as scratch:
            archive = bundle.export(folder, Path(scratch), light=self.settings.rollouts.light)
            client.send_file_download(archive.name, archive.read_bytes())
            self.message.content = f"Exported {archive.name} ({archive.stat().st_size / bundle.MB:.1f} MB)."

    def import_(self):
        upload = self.upload.value
        if upload is None:
            return
        if len(upload.content) > self.settings.rollouts.max_import_mb * bundle.MB:
            self.message.content = "That bundle is larger than rollouts.max_import_mb allows."
            return
        self.message.content = f"Checking {escape(upload.name)}…"
        with tempfile.TemporaryDirectory() as scratch:
            archive = Path(scratch) / "upload.tar.gz"
            archive.write_bytes(upload.content)
            folder = bundle.import_(archive, self.imported, max_mb=self.settings.rollouts.max_import_mb)
        self.message.content = f"Imported {folder.name}."
        self.refresh(select=f"imported/{folder.name}")

    def plan_evaluation(self):
        folder = self.folder()
        if folder is None or self.models is None:
            return
        self.planned = offline.estimate(folder, self.settings, self.task_choice.value)
        row = self.planned["tasks"][0] if self.planned["tasks"] else {"calls": 0, "usd": 0.0, "frames": 0}
        self.message.content = (f"{row['frames']} recorded frames → about {row['calls']} decision calls, "
                                f"${row['usd']:.2f} (stops at ${self.planned['cap_usd']:.2f}).")
        self.confirm_button.visible = row["calls"] > 0

    def evaluate(self):
        folder = self.folder()
        if folder is None or self.models is None:
            return
        self.confirm_button.visible = False
        vocabulary, decisions = self.models()

        def progress(task, fraction):
            self.message.content = f"Evaluating {task}: {fraction:.0%}"
        result = offline.evaluate(folder, self.settings, vocabulary, decisions, task=self.task_choice.value,
                                  progress=progress)
        report = result["tasks"][0]
        would = report["would"]
        self.message.content = (f"Evaluated {report['task']}: {report['calls']} calls, ${report['cost_usd']:.3f}, "
                                f"{len(would)} intervention{'s' if len(would) != 1 else ''} it would have made. "
                                "Open the replay to see them.")
        self.describe()

    # --- replay ---
    def open(self):
        folder = self.folder()
        if folder is None:
            return
        rollout = Rollout(folder, self.task_choice.value)
        self.rollout = rollout
        self.slider.min, self.slider.max = rollout.t0, max(rollout.t1, rollout.t0 + 0.1)
        self.slider.value = rollout.t0
        for widget in (self.slider, self.play_button, self.speed, self.live_button):
            widget.visible = True
        self.console.replaying = True
        self.draw(rollout.t0)

    def draw(self, t: float):
        rollout, console = self.rollout, self.console
        if rollout is None or not console.replaying:
            return
        view = rollout.view(t)
        console.draw_view(view, "Replay", len(view.decisions), f"{self.choice.value} · {rollout.doing_at(t)}")
        frame = rollout.frame_at(t)
        if frame is not None:
            with Image.open(frame) as image:
                console.ego.image = np.asarray(image.convert("RGB"))
        console.ego.label = f"Replay camera · {t - rollout.t0:.1f} s"
        console.thought.content = thought_bubble(rollout.thought_at(t), model="recorded agent")

    def toggle_play(self):
        if self.playing.is_set():
            self.playing.clear()
            return
        self.playing.set()
        self.play_button.label, self.play_button.icon = "Pause", "player-pause"
        threading.Thread(target=self.play, name="rollout-replay", daemon=True).start()

    def play(self):
        rollout = self.rollout
        while self.playing.is_set() and rollout is self.rollout and rollout is not None:
            t = self.slider.value + PLAY_STEP * SPEEDS.get(self.speed.value, 1.0)
            if t >= rollout.t1:
                self.slider.value = rollout.t1
                break
            self.slider.value = t
            self.draw(t)
            time.sleep(PLAY_STEP)
        self.playing.clear()
        self.play_button.label, self.play_button.icon = "Play", "player-play"

    def close(self):
        self.playing.clear()
        self.rollout = None
        for widget in (self.slider, self.play_button, self.speed, self.live_button):
            widget.visible = False
        self.console.end_replay()

