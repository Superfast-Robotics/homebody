"""Base-pose reads behind `gait.pose`: waiting out a frozen or stale localizer, and an innovation
gate (`PoseGate`) that compares each read with the dead reckoning of the twists `gait.hold`
streams (a unicycle in the map frame, metres and radians). An unexplained read is coasted over,
taken when it persists, and stood out, bounded, when the fit keeps flipping.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable

from homebody.helpers.geometry import wrap
from homebody.primitives.navigation.contracts import HardwareUnavailable, StaleObservation

RETRY_S = 1.0  # the base stands this long between retries
FRESH_WAIT_S, FRESH_POLL_S = 0.6, 0.05  # a stale sample is re-polled this long, commanding nothing
# xy allowance: sway + slack x commanded travel + orbit + coast + drift
SWAY_M = 0.08
TRAVEL_SLACK = 1.3
ORBIT_M_PER_RAD = 0.25
COAST_M, COAST_S = 0.25, 1.5
DRIFT_M_S = 0.05
STILL_DRIFT_M_S = 0.002  # drift allowed while nothing is commanded
YAW_SLACK_RAD = 0.35  # a yaw step under this is a turn, over it a re-pin
YAW_TURN_SLACK = 1.5          # x the commanded turn since the last accepted read
YAW_PER_M = 0.5               # heading error while walking, per metre
WALK_REALIZED = 0.72  # fraction of commanded walking the dead reckoning integrates
COAST_MAX_S = 2.0  # longest coast on the dead reckoning
SETTLE_MAX_S = 10.0  # longest stand waiting for a consistent read


def _clock(ctx):
    now = getattr(ctx, "now", None)
    return float(now()) if callable(now) else None


class PoseGate:
    """One robot's base_pose gate: commanded dead reckoning, the last accepted and rejected reads;
    `waits` counts every wait and recovery."""

    def __init__(self) -> None:
        self.waits = 0
        self._odo = {"x": 0.0, "y": 0.0, "yaw": 0.0, "travel": 0.0, "turned": 0.0, "moving_at": None}
        self._last = {"pose": None, "t": None, "odo": None}
        self._pending = {"pose": None, "t": None, "odo": None, "since": None}

    def forget(self) -> None:
        """Drop the accepted and rejected reads, for a new start frame."""
        self._last.update(pose=None, t=None, odo=None)
        self._pending.update(pose=None, t=None, odo=None, since=None)

    def commanded(self, vx: float, vy: float, wz: float, seconds: float, now) -> None:
        """One streamed twist from `gait.hold`: the dead reckoning between two map reads."""
        o = self._odo
        yaw = o["yaw"] + 0.5 * wz * seconds        # midpoint heading over the twist
        o["x"] += (vx * math.cos(yaw) - vy * math.sin(yaw)) * seconds * WALK_REALIZED
        o["y"] += (vx * math.sin(yaw) + vy * math.cos(yaw)) * seconds * WALK_REALIZED
        o["yaw"] = wrap(o["yaw"] + wz * seconds)
        o["travel"] += math.hypot(vx, vy) * seconds
        o["turned"] += abs(wz) * seconds
        if now is not None and (vx != 0.0 or vy != 0.0):
            o["moving_at"] = float(now)

    def _snapshot(self) -> dict:
        return dict(self._odo)

    def _reset(self, live, now) -> None:
        """A new clock: no history."""
        self._odo.update(x=0.0, y=0.0, yaw=float(live[2]), travel=0.0, turned=0.0, moving_at=None)
        self._last.update(pose=live, t=now, odo=self._snapshot())
        self._pending.update(pose=None, t=None, odo=None, since=None)

    def _still(self, ref: dict, now) -> bool:
        """Whether nothing was commanded since `ref` and the stopping tail is over."""
        o, r = self._odo, ref["odo"]
        return (o["travel"] == r["travel"] and o["turned"] == r["turned"]
                and (o["moving_at"] is None or now - o["moving_at"] >= COAST_S))

    def _accept(self, live, now, anchor: bool = True) -> None:
        if anchor:                          # moved: the heading and the reference follow the read
            self._odo["yaw"] = float(live[2])
            self._last.update(pose=live, t=now, odo=self._snapshot())
        self._pending.update(pose=None, t=None, odo=None, since=None)

    def _predicted(self, ref: dict):
        """Where the commanded motion since `ref` puts the base: (x, y, yaw, travel, turned)."""
        o, r = self._odo, ref["odo"]
        return (ref["pose"][0] + o["x"] - r["x"], ref["pose"][1] + o["y"] - r["y"],
                wrap(ref["pose"][2] + o["yaw"] - r["yaw"]),
                o["travel"] - r["travel"], o["turned"] - r["turned"])

    def _explained(self, ref: dict, live, now) -> bool:
        """Whether `live` is within the allowance of the commanded motion since `ref`."""
        px, py, pyaw, travel, turned = self._predicted(ref)
        dt = max(0.0, now - ref["t"])
        moving_at = self._odo["moving_at"]
        coast = COAST_M if moving_at is not None and now - moving_at < COAST_S else 0.0
        drift = STILL_DRIFT_M_S if self._still(ref, now) else DRIFT_M_S
        allow_m = SWAY_M + TRAVEL_SLACK * travel + ORBIT_M_PER_RAD * turned + coast + drift * dt
        allow_rad = YAW_SLACK_RAD + YAW_TURN_SLACK * turned + YAW_PER_M * travel
        return (math.hypot(live[0] - px, live[1] - py) <= allow_m
                and abs(wrap(live[2] - pyaw)) <= allow_rad)

    def consistent(self, ctx, value, observe: Callable[[], object], stand: Callable[[], object],
                   *, accepted_only: bool = False):
        """Judge a base_pose read: coast an unexplained one, take it when it persists, stand
        (bounded) while the fit keeps flipping."""
        now = _clock(ctx)
        live = (float(value[0]), float(value[1]), float(value[2]))
        last, pending = self._last, self._pending
        if now is None or last["t"] is None or now < last["t"]:
            self._reset(live, now)
            return value
        if self._explained(last, live, now):
            self._accept(live, now, anchor=not self._still(last, now))  # standing: reference stays
            return value
        if pending["pose"] is not None and self._explained(pending, live, now):
            self._accept(live, now)          # the same unexplained place twice: re-registration
            self.waits += 1
            ctx.log("base.localization_reregistered", {
                "value": [round(v, 4) for v in live], "then": "re-planning from it"})
            return value
        px, py, pyaw, travel, turned = self._predicted(last)
        if pending["pose"] is None:
            ctx.log("base.localization_jump", {
                "from": [round(v, 4) for v in last["pose"]], "to": [round(v, 4) for v in live],
                "predicted": [round(px, 4), round(py, 4), round(pyaw, 4)],
                "commanded_m": round(travel, 3), "commanded_rad": round(turned, 3),
                "then": "coasting on the commanded motion until the map agrees with itself"})
            pending["since"] = now
        pending.update(pose=live, t=now, odo=self._snapshot())
        coasted = (px, py, pyaw)
        if not accepted_only and (now - pending["since"] < COAST_MAX_S or not _stand(stand)):
            return coasted
        if accepted_only:
            stand()  # stand, raising on loss of command authority
        # stand until a read is explained; past SETTLE_MAX_S take the nearest one
        waited, reads = 1, [live]
        while True:
            p = None
            try:
                p = observe()
            except (HardwareUnavailable, StaleObservation):
                pass  # one more bounded retry
            now = _clock(ctx)
            deadline = getattr(ctx, "deadline", None)
            if accepted_only and deadline is not None and deadline.expired():
                raise StaleObservation("base_pose correction did not settle before the deadline")
            if p is not None:
                cur = (float(p[0]), float(p[1]), float(p[2]))
                reads.append(cur)
                if self._explained(last, cur, now) or self._explained(pending, cur, now):
                    self._accept(cur, now)
                    self.waits += 1
                    ctx.log("base.localization_settled", {
                        "waited_s": round(waited * RETRY_S, 1),
                        "value": [round(v, 4) for v in cur], "then": "walking on"})
                    return p
                pending.update(pose=cur, t=now, odo=self._snapshot())
            if deadline is not None and deadline.expired():
                raise HardwareUnavailable(
                    f"base_pose kept jumping for {waited * RETRY_S:.0f} s: {last['pose']} -> {live}")
            if waited * RETRY_S >= SETTLE_MAX_S:
                if accepted_only:
                    raise StaleObservation("base_pose correction remains inconsistent after settling")
                lx, ly, _ = last["pose"]
                cur = min(reads, key=lambda r: math.hypot(r[0] - lx, r[1] - ly))
                self._accept(cur, now)
                self.waits += 1
                ctx.log("base.localization_unsettled", {
                    "waited_s": round(waited * RETRY_S, 1), "value": [round(v, 4) for v in cur],
                    "then": "steering by the read nearest the last accepted one"})
                return cur
            stand()
            waited += 1


def _stand(stand: Callable[[], object]) -> bool:
    """Stand once; False when the caller may not command."""
    try:
        stand()
    except RuntimeError:
        return False
    return True


def read(ctx, term: str, observe: Callable[[], object], stand: Callable[[], object],
         *, accepted_only: bool = False):
    """`observe()`, standing out a frozen or stale localizer until the deadline, then a base_pose
    judged by `ctx.gate`; `accepted_only` returns only reads the gate accepted."""
    first = last = None
    waited = polled = 0
    while True:
        deadline = getattr(ctx, "deadline", None)
        if accepted_only and deadline is not None and deadline.expired():
            raise StaleObservation("base_pose verification deadline expired")
        try:
            value = observe()
        except (HardwareUnavailable, StaleObservation) as exc:
            last = exc
            if first is None:
                first = exc
                ctx.log("base.localization_wait", {
                    "term": term, "why": f"{type(exc).__name__}: {exc}",
                    "then": "standing until it returns"})
            deadline = getattr(ctx, "deadline", None)
            if deadline is not None and deadline.expired():
                raise HardwareUnavailable(
                    f"{term}: localizer down for {waited * RETRY_S:.0f} s: {last}") from last
            if isinstance(exc, StaleObservation) and polled * FRESH_POLL_S < FRESH_WAIT_S:
                polled += 1
                getattr(ctx, "wait", time.sleep)(FRESH_POLL_S)
                continue
            if not _stand(stand):
                raise first from None
            waited += 1
            continue
        if first is not None:
            ctx.gate.waits += 1
            ctx.log("base.localization_recovered", {
                "term": term, "waited_s": round(waited * RETRY_S + polled * FRESH_POLL_S, 2),
                "value": [round(float(v), 4) for v in value]})
        return (ctx.gate.consistent(ctx, value, observe, stand, accepted_only=accepted_only)
                if term == "base_pose" else value)
