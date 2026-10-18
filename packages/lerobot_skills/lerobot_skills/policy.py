"""A LeRobot policy as a skill: each observation goes through the policy's `select_action`, its action vector is
named by the robot's action features, and the skill succeeds after a fixed number of steps or when a success
check on the observation holds."""
from dataclasses import dataclass

import numpy as np

from .skill import ADVANCE, FAILED, SUCCEEDED, Skill, SkillConfig, Status


@SkillConfig.register_subclass("policy")
@dataclass
class PolicySkillConfig(SkillConfig):
    max_steps: int = 300
    step_seconds: float = 1 / 30


class PolicySkill(Skill):
    """POLICY has `reset()` and `select_action(batch)` (a LeRobot PreTrainedPolicy); TO_BATCH turns an
    observation and the task's arguments into its input batch; ACTION_NAMES names its output vector; SUCCEEDED,
    if given, ends the skill when it holds for an observation."""

    config_class = PolicySkillConfig
    name = "policy"

    def __init__(self, config, policy, to_batch, action_names, succeeded=None):
        super().__init__(config)
        self.policy, self.to_batch, self.action_names, self.succeeded = policy, to_batch, tuple(action_names), succeeded
        self.arguments, self.steps = {}, 0

    def reset(self, observation=None, arguments=None):
        self.policy.reset()
        self.arguments, self.steps, self.status = dict(arguments or {}), 0, Status()

    def select_action(self, observation):
        if self.succeeded is not None and self.succeeded(observation):
            self.status = Status(SUCCEEDED, "OK", f"succeeded after {self.steps} steps")
            return {}
        if self.steps >= self.config.max_steps:
            self.status = Status(FAILED, "TIMEOUT", f"no success within {self.config.max_steps} steps")
            return {}
        vector = np.asarray(_numpy(self.policy.select_action(self.to_batch(observation, self.arguments)))).reshape(-1)
        if vector.size != len(self.action_names):
            raise ValueError(f"the policy gave {vector.size} values for {len(self.action_names)} action names")
        self.steps += 1
        return {**dict(zip(self.action_names, vector.tolist())), ADVANCE: self.config.step_seconds}


def _numpy(value):
    """A torch tensor's values (detached, on the CPU), or VALUE itself."""
    return value.detach().cpu().numpy() if hasattr(value, "detach") else value
