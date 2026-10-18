"""A HomeBody skill in LeRobot's policy form (skills.runtime = "stepwise") gives the robot the same calls at the
same simulated times, and the same result and events, as the skill driving the robot directly."""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("lerobot_skills")

from homebody.primitives.actions import PhysicsInvalid
from homebody.session.config import Skills
from homebody.session.settings import Settings
from homebody.skills.contract import Context, Runner
from tests.test_skills import FakeBackend


class Recording(FakeBackend):
    """FakeBackend that logs every robot call with the simulated time it came at."""

    def __init__(self, fail_after=None):
        super().__init__()
        self.calls, self.fail_after = [], fail_after

    def _log(self, *call):
        self.calls.append((round(self.time, 6), *call))

    def base_velocity(self, epoch, forward, lateral, yaw_rate):
        self._log("base", epoch, round(forward, 9), round(lateral, 9), round(yaw_rate, 9))
        super().base_velocity(epoch, forward, lateral, yaw_rate)

    def arm_target(self, epoch, side, joints):
        self._log("arm", epoch, side, tuple(np.round(joints, 9)))

    def grip(self, epoch, side, closure):
        self._log("grip", epoch, side, closure)

    def stop(self, epoch):
        self._log("stop", epoch)
        super().stop(epoch)

    def advance(self, epoch, seconds):
        self._log("advance", epoch, round(seconds, 9))
        if self.fail_after is not None and self.time >= self.fail_after:
            raise PhysicsInvalid("the robot fell")
        super().advance(epoch, seconds)


def navigate(backend, runtime):
    settings = replace(Settings(), skills=Skills(runtime=runtime))
    events = []
    result = Runner(Context(backend, backend, {}, settings=settings, emit=events.append)).run(
        "navigate", {"goal_xy_m": [1.6, 1.2], "facing_xy_m": [2.4, 1.2]}, backend.snapshot())
    return result, [event["type"] for event in events]


@pytest.mark.parametrize("fail_after", [None, 0.3])
def test_stepwise_gives_the_robot_the_same_calls_result_and_events(fail_after):
    direct, stepwise = Recording(fail_after), Recording(fail_after)
    expected = navigate(direct, "direct")
    assert navigate(stepwise, "stepwise") == expected
    assert len(direct.calls) > 5 and stepwise.calls == direct.calls


def test_the_runtime_setting_is_checked():
    with pytest.raises(ValueError):
        Skills(runtime="threaded")


def test_every_robot_access_happens_on_the_callers_thread():
    import threading

    class ThreadBound(Recording):
        """A robot that, like a GL renderer, only works on the thread that made it."""

        def __init__(self):
            super().__init__()
            self.owner, self.strays = threading.get_ident(), []

        def __getattribute__(self, name):
            state = object.__getattribute__(self, "__dict__")
            if not name.startswith("_") and name not in state and threading.get_ident() != state.get(
                    "owner", threading.get_ident()):
                state["strays"].append(name)
            return object.__getattribute__(self, name)

    robot = ThreadBound()
    navigate(robot, "stepwise")
    assert len(robot.calls) > 5 and robot.strays == []
