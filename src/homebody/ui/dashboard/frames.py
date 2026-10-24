"""Whether the robot camera's frames are usable, from an explicit check of each frame (not from how it looks).

A frame is *invalid* when it has no image (no dimensions), could not be decoded, or is blank (every pixel one value);
*suspected invalid* when it is nearly uniform (its spread under `frame_uniform_std`: a plain wall can be that too, so
only suspected) or when the same image keeps arriving while simulated time moves on for `frame_repeat_s` (a renderer
that stopped). Otherwise *valid*; *unknown* before any frame. The thresholds are `[dashboard]` settings, so a camera or
simulator with other conditions sets its own. The check reads a subsample of each frame, so it costs little per frame.
"""
import hashlib
from dataclasses import asdict, dataclass

import numpy as np

STEP = 8  # px between the sampled pixels


@dataclass(frozen=True)
class FrameQuality:
    state: str = "unknown"  # valid, suspected_invalid, invalid, unknown
    reason: str = ""
    since: float | None = None  # when the current state began (dashboard clock)
    checked: float | None = None  # the newest frame checked
    frames: int = 0
    problems: int = 0  # frames that were not valid since the dashboard started


class FrameCheck:
    def __init__(self, timing):
        self.uniform_std, self.repeat_s = timing.frame_uniform_std, timing.frame_repeat_s
        self.quality = FrameQuality()
        self._digest, self._same_since_sim = None, None

    def _set(self, state, reason, now):
        q = self.quality
        since = q.since if q.state == state and q.since is not None else now
        self.quality = FrameQuality(state, reason, since, now, q.frames + 1, q.problems + (state != "valid"))

    def undecodable(self, now, error: str):
        """A frame that arrived but could not be decoded (the stream's own decoder said so)."""
        self._set("invalid", f"Undecodable frame: {error}"[:200], now)

    def check(self, rgb, now, sim_time=None) -> FrameQuality:
        """RGB, received at NOW (simulation time SIM_TIME when known): the quality it gives the camera."""
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
