"""Operator commands for the session thread; UI callbacks only enqueue them."""
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event


@dataclass(frozen=True)
class Command:
    name: str
    text: str = ""


class Controls:
    def __init__(self):
        self.queue = Queue()
        self.cancel = Event()
        self.closing = Event()
        self.resumed = Event()  # the operator ends a monitor pause (session.await_operator)
        self.paused = Event()  # the operator froze the simulation and the agent
        self.skill_stop = Event()  # the operator stopped the running skill
        self.dismissed = Event()  # the operator dismissed the monitor's alert
        self.notes = Queue()  # the operator's notes for the agent's next prompt

    def submit(self, text: str):
        if text.strip():
            self.queue.put(Command("task", text.strip()))

    def stop(self):
        """Cancel the running task and drop tasks that have not started."""
        self.cancel.set()
        with self.queue.mutex:
            kept = [command for command in self.queue.queue if command.name != "task"]
            self.queue.queue.clear()
            self.queue.queue.extend(kept)
        self.queue.put(Command("stop"))

    def reset(self):
        self.cancel.set()
        self.queue.put(Command("reset"))

    def pause(self):
        self.paused.set()

    def resume(self):
        """Continue: from the operator's pause, and from a pause the monitor asked for."""
        self.paused.clear()
        self.resumed.set()

    def unpause(self):
        self.paused.clear()

    def resume_task(self):
        self.resumed.set()

    def stop_skill(self):
        self.skill_stop.set()

    def dismiss(self):
        self.dismissed.set()

    def tell(self, text: str):
        if text.strip():
            self.notes.put(text.strip())

    def take_notes(self) -> list[str]:
        notes = []
        while True:
            try:
                notes.append(self.notes.get_nowait())
            except Empty:
                return notes

    def next(self):
        try:
            return self.queue.get_nowait()
        except Empty:
            return None


def capabilities(controls) -> dict:
    """Which execution controls CONTROLS implements; a control it lacks is shown disabled, never simulated."""
    def has(name):
        return callable(getattr(controls, name, None))
    return {"pause": has("pause") and has("resume"), "stop_skill": has("stop_skill"),
            "stop_task": has("stop"), "reset": has("reset"), "tell": has("tell"), "dismiss": has("dismiss")}
