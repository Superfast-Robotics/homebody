"""The coder's perception primitives read a hand's wrist camera only when asked."""
from types import SimpleNamespace

import numpy as np
import pytest

from homebody.harness.env import Env, Target
from homebody.primitives.observations import WristView


def view(label, side="right"):
    labels = np.full((8, 8), label)
    return WristView(1, 0.0, side, np.zeros((8, 8, 3), np.uint8), np.full((8, 8), 0.5), labels,
                     np.array([[8.0, 0, 3.5], [0, 8.0, 3.5], [0, 0, 1]]), np.eye(4))


def env_with(head_label, wrist_label):
    calls = []
    ctx = SimpleNamespace(observations=SimpleNamespace(epoch=1),
                          observe=lambda epoch: SimpleNamespace(**vars(view(head_label)), world_torso=np.eye(4)),
                          wrist=lambda epoch, side: calls.append(side) or view(wrist_label, side),
                          measure=lambda epoch: SimpleNamespace(world_torso=np.eye(4)))
    return Env(ctx, None), calls


def test_the_head_camera_is_the_default_and_a_wrist_camera_is_read_on_request():
    env, calls = env_with(head_label=0, wrist_label=7)
    assert env.locate(Target(7)) is None and calls == []
    found = env.locate(Target(7), camera="left")
    assert calls == ["left"] and found["n_points"] == 64
    assert found["center"][2] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        env.locate(Target(7), camera="elbow")
