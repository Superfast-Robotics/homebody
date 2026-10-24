"""What the harness server's session talks to in place of a console in its own process: `UiLink` takes the
session's console calls and sends them to the clients (server/monitor_link.py does the same for the monitor)."""
import time

from homebody.server import protocol


class UiLink:
    """The session's console, sent to the clients. VIEW_HZ caps the camera images sent (wall hertz)."""

    def __init__(self, hub, *, quality: int, view_hz: float):
        self.hub, self.quality, self.period = hub, quality, 1 / view_hz
        self.camera = "robot"  # the view a client chose: "robot", "observer" or "hand"
        self.send_views = True  # False when the clients see a simulation server's camera stream instead
        self.sent = 0.0
        self.status = ("Loading", "")

    def update(self, status: str, text: str = ""):
        self.status = (status, text)
        self.hub.publish(protocol.text("status", status=status, text=text))

    def _view(self, rgb, label: str, t=None):
        if not self.send_views:
            return
        now = time.monotonic()
        if now - self.sent < self.period:
            return
        self.sent = now
        self.hub.publish(protocol.image("view", rgb, self.quality, label=label, t=t), droppable=True)

    def show(self, frame):
        if self.camera == "robot":
            self._view(frame.rgb, "Live camera", frame.time)

    def show_camera(self, rgb, label):
        if self.camera != "robot":
            self._view(rgb, label)

    def skill_event(self, event):
        self.hub.publish(protocol.text("skill", event=event))

    def select(self, decision, frame):
        self.hub.publish(protocol.text("decision", text=decision.text, skill=decision.skill,
                                       arguments=decision.arguments))

    def clear_view(self):
        self.hub.publish(protocol.text("clear_view"))

    def clear_selection(self):
        pass
