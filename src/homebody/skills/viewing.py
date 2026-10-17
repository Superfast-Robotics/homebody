"""Settled viewing turns and the all-round survey shared by the live viewing skills."""
import math
import time

from homebody.primitives.navigation.errors import ActionRefused
from homebody.real2sim.exploration.settings import current

HOLD_S = 1.0
DRIFT_BACK_M = 0.15
SPIN_HEADINGS = 6  # settled views in one all-round survey


def face(ctx, yaw: float, *, recover_blocked: bool = False) -> None:
    """Turn in place to `yaw` for a settled view; a model look may ask for checked recovery
    when the turn is blocked. A refusal or stop propagates to the caller."""
    world = ctx.world
    with world.action_capture("look"):
        ctx.actions.turn_hold(yaw, HOLD_S, **({"recover_blocked": True} if recover_blocked else {}))
    world.check()


def survey(ctx, label: str = "start spin") -> None:
    with ctx.world.action_capture("initial_survey" if label == "start spin" else label):
        survey_motion(ctx, label)


def survey_motion(ctx, label):
    """A slow full turn: SPIN_HEADINGS turns in place with a pause at each, so the cameras see
    all round from this spot. A refused heading ends the survey; the next move plans from wherever
    the turns left the base, after one step back if they drifted it."""
    world = ctx.world
    _, _, yaw0 = world.pose()
    ax, ay = world.base_xy()
    for i in range(SPIN_HEADINGS):
        try:
            ctx.actions.turn_hold(yaw0 + 2 * math.pi * i / SPIN_HEADINGS, HOLD_S)
        except ActionRefused as why:
            world.emit("action", action="look", outcome="refused", error=why)
            world.emit("status", message=f"The look-around stopped: {why}. Asking for another view.")
            return
        world.capture_survey_heading()
        world.check()
    x, y = world.base_xy()
    off = math.hypot(x - ax, y - ay)
    if off > DRIFT_BACK_M and world.live.scope_ready:
        world.emit("status", message=f"The turn moved the robot {off * 100:.0f} cm off its spot; stepping back.")
        try:
            ctx.actions.step_to((ax, ay))
        except ActionRefused as why:
            world.emit("action", action="survey recenter", target=(ax, ay), outcome="refused", error=why)
            return
        world.check()
    world.stop_here(label)
    world.emit("gallery_after", time=time.monotonic() + current().iphone.post_action_buffer_s)
