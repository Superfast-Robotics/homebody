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
            "stop_task": has("stop"), "reset": has("reset")}
