"""The camera views the dashboard streams, JPEG-encoded, and a short history of the robot camera."""
import io
from collections import OrderedDict
from itertools import count
from threading import Lock

import numpy as np
from PIL import Image

from homebody.vlm.observations import depth_preview, label_preview

# name -> (label, whether it is redrawn from every frame)
VIEWS = {
    "robot": ("Robot camera (head)", True),
    "depth": ("Depth", True),
    "labels": ("Segmentation", True),
    "map": ("Navigation map", True),
    "observer": ("Observer (behind the robot)", False),  # rendered by the session while watched
    "hand": ("Hand camera (the working hand)", False),
    "decision": ("Decision click", False),
    "topdown": ("Place target (top-down)", False),
}
LIVE = tuple(name for name, (_, live) in VIEWS.items() if live)
# Auto camera: the views each activity prefers, best first; the page takes the first this robot has. Extend it as a
# robot gains cameras (a "wrist_right" view, say): names that are not served are skipped, never assumed.
AUTO_CAMERAS = {
    "navigate": ("robot",),
    "pick": ("wrist_{hand}", "robot"),
    "place": ("wrist_{hand}", "robot"),
    "finished": ("scene3d", "robot"),
    "idle": ("robot",),
}
QUALITY, HISTORY_QUALITY, HISTORY_WIDTH = 82, 70, 480
HISTORY_PERIOD, HISTORY_SECONDS = 1.0, 600.  # one stored robot frame per second, kept as long as the timeline


def jpeg(rgb, quality=QUALITY, width=None) -> bytes:
    image = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8))
    if width is not None and image.width > width:
        image = image.resize((width, round(image.height * width / image.width)))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


def render(view, overlays, frame):
    """VIEW's RGB from the newest FRAME, or None when it has nothing to show."""
    if view == "robot":
        return overlays.ego_rgb(frame)
    if view == "depth":
        return depth_preview(frame.depth)
    if view == "labels":
        return label_preview(frame.labels)
    if view == "map":
        return overlays.map(frame)
    raise ValueError(f"{view} is not a live view")


class History:
    """Robot-camera JPEGs at HISTORY_PERIOD, for inspecting a past moment and an incident's evidence."""
    def __init__(self, period=HISTORY_PERIOD, seconds=HISTORY_SECONDS):
        self.period, self.seconds = period, seconds
        self.frames, self.lock, self._ids = OrderedDict(), Lock(), count(1)
        self.last = float("-inf")

    def due(self, t) -> bool:
        return t - self.last >= self.period

    def add(self, t, data) -> int:
        with self.lock:
            frame_id = next(self._ids)
            self.frames[frame_id] = (t, data)
            self.last = t
            while self.frames and next(iter(self.frames.values()))[0] < t - self.seconds:
                self.frames.popitem(last=False)
            return frame_id

    def clear(self):
        with self.lock:
            self.frames.clear()
            self.last = float("-inf")

    def get(self, frame_id):
        with self.lock:
            return self.frames.get(frame_id, (None, None))[1]

    def index(self):
        with self.lock:
            return [[frame_id, t] for frame_id, (t, _) in self.frames.items()]
