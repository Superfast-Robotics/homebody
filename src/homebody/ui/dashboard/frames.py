"""Whether the robot camera's frames are usable, from an explicit check of each frame (not from how it looks).

A frame is *invalid* when it has no image (no dimensions), could not be decoded, or is blank (every pixel one value);
*suspected invalid* when it is nearly uniform (its spread under `frame_uniform_std`: a plain wall can be that too, so
only suspected) or when the same image keeps arriving while simulated time moves on for `frame_repeat_s` (a renderer
that stopped). Otherwise *valid*; *unknown* before any frame. The thresholds are `[dashboard]` settings, so a camera or
simulator with other conditions sets its own. The check reads a subsample of each frame, so it costs little per frame.

Each checked frame with a simulation time is kept in runs of one state, broken where frames are more than RUN_GAP_S
apart (`at`), so a monitor reading can carry the quality of the frames at its own observation time, not whatever the
camera shows when the reading arrives. An undecodable frame has no simulation time: it breaks the run and marks the
gap it arrived in, so a reading in that gap is not reported as valid.
"""
import hashlib
from dataclasses import asdict, dataclass

import numpy as np

STEP = 8  # px between the sampled pixels
MATCH_S = 1.0  # s of simulated time between an observation and the nearest checked frame that still counts
KEEP_RUNS = 400  # runs kept for matching (a run lasts while the state does and frames keep coming)
RUN_GAP_S = 0.5  # s of simulated time between two checked frames that still belong to one run


@dataclass(frozen=True)
class FrameQuality:
    state: str = "unknown"  # valid, suspected_invalid, invalid, unknown
    reason: str = ""
    since: float | None = None  # when the current state began (dashboard clock)
    checked: float | None = None  # the newest frame checked
    frames: int = 0
    problems: int = 0  # frames that were not valid since the dashboard started


@dataclass(frozen=True)
class FrameAt:
    """The checked state of the robot camera near an asked time, and how it was established (HOW): "covered" (a run of
    frames of one state, at most RUN_GAP_S apart, spans the time; SIM_TIME is the asked time), "nearest" (the nearest
    checked frame, at SIM_TIME) or "undecodable" (an undecodable frame arrived in the gap that holds the time)."""
    state: str
    reason: str
    sim_time: float
    how: str = "covered"


class FrameCheck:
    def __init__(self, timing):
        self.uniform_std, self.repeat_s = timing.frame_uniform_std, timing.frame_repeat_s
        self.quality = FrameQuality()
        self._digest, self._same_since_sim = None, None
        self._runs = []  # [state, reason, first sim time, last sim time, undecodable frame before it], oldest first
        self._sim = None  # the simulation time of the frame being checked
        self._broken = None  # an undecodable frame's reason, until the next checked frame starts a run

    def _set(self, state, reason, now):
        q = self.quality
        since = q.since if q.state == state and q.since is not None else now
        self.quality = FrameQuality(state, reason, since, now, q.frames + 1, q.problems + (state != "valid"))
        self._keep(state, reason)

    def _keep(self, state, reason):
        sim = self._sim
        if sim is None:
            return
        if self._runs and sim < self._runs[-1][3] - MATCH_S:  # simulated time went back: a reset scene
            self._runs = []
        last = self._runs[-1] if self._runs else None
        if last is not None and last[0] == state and not self._broken and sim - last[3] <= RUN_GAP_S:
            last[3] = max(last[3], sim)
        else:
            after = self._broken  # the undecodable frame that arrived before this run, if any
            self._runs = [*self._runs[-(KEEP_RUNS - 1):], [state, reason, sim, sim, after]]
        self._broken = None

    def at(self, sim_time) -> FrameAt | None:
        """The checked state of the frames at SIM_TIME (see FrameAt); None without a checked frame within MATCH_S (or
        without a time)."""
        if sim_time is None:
            return None
        best = None
        previous = None
        for state, reason, first, last, after in self._runs:
            if first <= sim_time <= last:
                return FrameAt(state, reason, sim_time, "covered")
            if after and previous is not None and previous < sim_time < first:
                return FrameAt("invalid", f"{after} (between sim {previous:.2f} s and {first:.2f} s)", sim_time, "undecodable")
            near = first if abs(first - sim_time) < abs(last - sim_time) else last
            if abs(near - sim_time) <= MATCH_S and (best is None or abs(near - sim_time) < abs(best.sim_time - sim_time)):
                best = FrameAt(state, reason, near, "nearest")
            previous = last
        return best

    def undecodable(self, now, error: str):
        """A frame that arrived but could not be decoded (the stream's own decoder said so): it has no simulation time,
        so it breaks the run and marks the gap until the next checked frame."""
        self._sim = None
        self._set("invalid", f"Undecodable frame: {error}"[:200], now)
        self._broken = self.quality.reason

    def check(self, rgb, now, sim_time=None) -> FrameQuality:
        """RGB, received at NOW (simulation time SIM_TIME when known): the quality it gives the camera."""
        self._sim = sim_time
        image = np.asarray(rgb) if rgb is not None else None
        if image is None or image.ndim < 2 or image.shape[0] == 0 or image.shape[1] == 0:
            self._set("invalid", "Frame without dimensions", now)
            return self.quality
        sample = image[::STEP, ::STEP]
        if sample.min() == sample.max():
            self._set("invalid", f"Blank frame (every sampled pixel is {int(sample.min())})", now)
            return self.quality
        spread = float(sample.astype(np.float32).std())
        digest = hashlib.blake2b(np.ascontiguousarray(sample).tobytes(), digest_size=8).digest()
        if digest == self._digest and sim_time is not None and self._same_since_sim is not None:
            if sim_time - self._same_since_sim >= self.repeat_s:
                self._set("suspected_invalid", f"The same image for {sim_time - self._same_since_sim:.0f} s of simulated "
                                               "time (the renderer may have stopped)", now)
                return self.quality
        elif digest != self._digest:
            self._digest, self._same_since_sim = digest, sim_time
        if spread < self.uniform_std:
            self._set("suspected_invalid", f"Nearly uniform frame (spread {spread:.1f} < {self.uniform_std:g})", now)
            return self.quality
        self._set("valid", "", now)
        return self.quality

    def snapshot(self) -> dict:
        return asdict(self.quality)
