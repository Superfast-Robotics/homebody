"""lerobot_skills: registered configs, the step loop and a policy as a skill."""
from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("draccus")

from lerobot_skills import (
    ADVANCE,
    EPOCH,
    FAILED,
    SUCCEEDED,
    PolicySkill,
    PolicySkillConfig,
    Skill,
    SkillConfig,
    Status,
    run_skill,
)


@SkillConfig.register_subclass("count_to")
@dataclass
class CountConfig(SkillConfig):
    target: int = 3


class Count(Skill):
    """Moves `x` up by one per step until it reaches the target."""

    config_class, name = CountConfig, "count_to"

    def reset(self, observation=None, arguments=None):
        self.status = Status()

    def select_action(self, observation):
        if observation["x"] >= self.config.target:
            self.status = Status(SUCCEEDED, "OK")
            return {}
        return {"x.vel": 1.0, EPOCH: 1, ADVANCE: 0.5}


class Robot:
    def __init__(self, refuse=False):
        self.x, self.velocity, self.epoch, self.refuse, self.advanced = 0, 0.0, 1, refuse, []

    def get_observation(self):
        return {"x": self.x}

    def send_action(self, action):
        if self.refuse:
            raise RuntimeError("refused")
        self.velocity = action["x.vel"]
        return action

    def advance(self, epoch, seconds):
        self.advanced.append((epoch, seconds))
        self.x += int(self.velocity)


def test_a_config_is_a_registered_choice():
    assert SkillConfig.get_choice_class("count_to") is CountConfig
    assert CountConfig().type == "count_to"
    assert SkillConfig.get_choice_class("policy") is PolicySkillConfig


def test_the_runner_steps_a_skill_until_it_finishes():
    robot, skill = Robot(), Count(CountConfig(target=3))
    skill.reset()
    status = run_skill(skill, robot)
    assert status.state == SUCCEEDED and robot.x == 3
    assert robot.advanced == [(1, 0.5)] * 3


def test_a_refused_action_ends_the_run_by_default_and_max_steps_bounds_it():
    with pytest.raises(RuntimeError, match="refused"):
        run_skill(Count(CountConfig()), Robot(refuse=True))
    skill = Count(CountConfig(target=10))
    assert run_skill(skill, Robot(), max_steps=2).state == "running"


class Policy:
    def __init__(self, vector):
        self.vector, self.resets, self.batches = np.asarray(vector), 0, []

    def reset(self):
        self.resets += 1

    def select_action(self, batch):
        self.batches.append(batch)
        return self.vector


def test_a_policy_runs_as_a_skill_until_success_or_its_step_budget():
    policy = Policy([1.0])
    skill = PolicySkill(PolicySkillConfig(max_steps=10, step_seconds=0.5), policy,
                        lambda observation, arguments: {"state": observation["x"], **arguments}, ["x.vel"],
                        succeeded=lambda observation: observation["x"] >= 2)
    skill.reset(arguments={"task": "go"})
    robot = Robot()
    assert run_skill(skill, robot).state == SUCCEEDED and robot.x == 2
    assert policy.resets == 1 and policy.batches[0] == {"state": 0, "task": "go"}
    timed = PolicySkill(PolicySkillConfig(max_steps=1), Policy([0.0]), lambda o, a: o, ["x.vel"])
    timed.reset()
    status = run_skill(timed, Robot())
    assert (status.state, status.code) == (FAILED, "TIMEOUT")


def test_a_policy_must_name_every_value_it_outputs():
    skill = PolicySkill(PolicySkillConfig(), Policy([1.0, 2.0]), lambda o, a: o, ["x.vel"])
    skill.reset()
    with pytest.raises(ValueError):
        skill.select_action({"x": 0})
