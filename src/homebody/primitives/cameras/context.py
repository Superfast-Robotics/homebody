"""Uncalibrated phone RGB with a bounded receipt-time buffer."""
from __future__ import annotations

import math
import threading
import time
from copy import deepcopy

import numpy as np
from PIL import Image

from homebody.primitives.cameras.color_stream import LatestColorClient


class ContextCamera:
    def __init__(self, endpoint: str, max_age_s: float = 3.0):
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("context camera max age must be positive and finite")
        self.client = LatestColorClient(endpoint)
        self.max_age_s = max_age_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame = None
        self._received = None
        self._retention = None
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self):
        while not self._stop.is_set():
            frame = self.client.wait(0.2)
            if frame is not None:
                received, sampled_at = time.monotonic(), time.time()
                with self._lock:
                    self._frame, self._received = frame, received
                self._retain(frame, received, sampled_at)

    def begin_retention(self, context: dict, *, sample_context=None):
        """Begin one action's bounded receipt-time sample; no metric exposure pose is implied."""
        context = deepcopy(context)
        started = context.get("started_wall_s", time.time())
        with self._lock:
            self._retention = {"context": context, "samples": [], "interval_s": 1.0,
                               "drops": 0, "compactions": 0, "started_wall_s": started,
                               "ended_wall_s": None, "tail_s": 0.0, "next_sample": None,
                               "sample_context": sample_context, "pending": None}

    def finish_retention(self, ended_wall_s: float, *, tail_s: float = 3.0):
        if not math.isfinite(ended_wall_s) or not math.isfinite(tail_s) or tail_s < 0:
            raise ValueError("retention end and nonnegative tail must be finite")
        with self._lock:
            if self._retention is not None:
                self._retention.update(ended_wall_s=ended_wall_s, tail_s=tail_s)

    def retained_samples(self):
        """Include the already-received pending frame without waiting for the camera worker."""
        with self._lock:
            retention = self._retention
            if retention is None:
                return {"context": {}, "samples": [], "interval_s": 1.0,
                        "drops": 0, "compactions": 0, "started_wall_s": None,
                        "ended_wall_s": None, "tail_s": 0.0}
            result = {k: deepcopy(v) for k, v in retention.items()
                      if k not in ("samples", "next_sample", "sample_context", "pending")}
            result["samples"] = [{"rgb": s["rgb"], **deepcopy({
                k: v for k, v in s.items() if k != "rgb"})} for s in retention["samples"]]
            pending = None if self._stop.is_set() else retention["pending"]
            if pending is not None:
                frame, sample = pending[0], deepcopy(pending[1])
        # Thumbnail the pending frame outside the lock, with its sampled context.
        if pending is not None:
            rgb = self._thumbnail(frame)
            with self._lock:
                if self._retention is retention and not self._stop.is_set():
                    samples = result["samples"]
                    if len(samples) >= 64:
                        reduced = samples[::2]
                        if reduced[-1] is not samples[-1]:
                            reduced.append(samples[-1])
                        result["drops"] += len(samples) - len(reduced)
                        result["samples"] = samples = reduced
                        result["compactions"] += 1
                        result["interval_s"] *= 2
                    samples.append(sample | {"rgb": rgb})
        return result

    @staticmethod
    def _thumbnail(frame):
        rgb = Image.fromarray(frame.image[:, :, ::-1])
        rgb.thumbnail((960, 960))
        result = np.asarray(rgb).copy()
        result.flags.writeable = False
        return result

    def _retain(self, frame, received, sampled_at):
        with self._lock:
            retention = self._retention
            if retention is None or self._stop.is_set():
                return
            end = retention["ended_wall_s"]
            if ((end is not None and sampled_at > end + retention["tail_s"])
                    or sampled_at < retention["started_wall_s"]
                    or (retention["next_sample"] is not None
                        and received < retention["next_sample"])):
                return
            retention["next_sample"] = received + retention["interval_s"]
        # Image work runs outside the lock used by motion's fresh() guard.
        sampler = retention["sample_context"]
        context = sampler() if sampler is not None else {}
        sample = {"sampled_at": sampled_at,
                  "metadata": {"status": "fresh", "receive_age_s": 0.0,
                               "sequence": frame.generation,
                               "receiver_decode_time_s": frame.captured_at,
                               "timestamp_kind": "receiver_decode_time",
                               "metric_calibration": False}, **context}
        with self._lock:
            if self._retention is not retention or self._stop.is_set():
                return
            retention["pending"] = (frame, sample)
        rgb = self._thumbnail(frame)
        with self._lock:
            retention["pending"] = None
            if self._retention is not retention or self._stop.is_set():
                return  # A new action or shutdown invalidates an in-flight resize.
            samples = retention["samples"]
            if len(samples) >= 64:
                reduced = samples[::2]
                if reduced[-1] is not samples[-1]:
                    reduced.append(samples[-1])
                retention["drops"] += len(samples) - len(reduced)
                retention["samples"] = samples = reduced
                retention["compactions"] += 1
                retention["interval_s"] *= 2
                retention["next_sample"] = received + retention["interval_s"]
            samples.append(sample | {"rgb": rgb})

    def fresh(self):
        """Liveness without copying pixels on the motion guard's fast path."""
        with self._lock:
            return (self._frame is not None and self._received is not None
                    and time.monotonic() - self._received <= self.max_age_s)

    def snapshot(self):
        with self._lock:
            frame, received = self._frame, self._received
        if frame is None:
            return None, {"status": "unavailable"}
        age = time.monotonic() - received
        meta = {"status": "fresh" if age <= self.max_age_s else "stale",
                "receive_age_s": round(age, 3), "sequence": frame.generation,
                "receiver_decode_time_s": frame.captured_at,
                "timestamp_kind": "receiver_decode_time", "metric_calibration": False}
        rgb = np.ascontiguousarray(frame.image[:, :, ::-1])
        return (rgb if age <= self.max_age_s else None), meta

    def close(self):
        self._stop.set()
        self._thread.join()
        self.client.close()
