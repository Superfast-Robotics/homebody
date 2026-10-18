"""A skill in LeRobot's policy form. Its config is a registered choice (`--skill.type=...`, as LeRobot chooses
robots and policies); the skill is reset with an observation and its arguments, then `select_action` turns each
observation into the next action dict over the robot's action features until its status leaves "running".

Two keys an action may carry besides the robot's features: ADVANCE, the seconds the robot runs that action
before the next observation (a simulated robot advances its clock; a real one waits), and EPOCH, the command
generation the action belongs to."""
import abc
from dataclasses import dataclass, field

import draccus

ADVANCE = "advance_s"
EPOCH = "epoch"
RUNNING, SUCCEEDED, FAILED = "running", "succeeded", "failed"


@dataclass
class SkillConfig(draccus.ChoiceRegistry, abc.ABC):
    """A skill's settings; subclasses register with `@SkillConfig.register_subclass("name")`."""

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@dataclass(frozen=True)
class Status:
    state: str = RUNNING
    code: str = ""          # the outcome's code ("OK", or a declared failure)
    message: str = ""
    details: dict = field(default_factory=dict)

    @property
    def done(self):
        return self.state != RUNNING


class Skill(abc.ABC):
    """Set `config_class` and `name` in subclasses. `observes` False means select_action ignores its observation
    (the skill reads its robot itself), so a runner need not render one."""

    config_class: type[SkillConfig]
    name: str
    observes: bool = True

    def __init__(self, config: SkillConfig):
        self.config = config
        self.status = Status()

    @property
    def done(self) -> bool:
        return self.status.done

    @abc.abstractmethod
    def reset(self, observation: dict | None = None, arguments: dict | None = None) -> None:
        """Start a new attempt from OBSERVATION with ARGUMENTS."""

    @abc.abstractmethod
    def select_action(self, observation: dict) -> dict:
        """The next action for OBSERVATION; an empty dict once the skill has finished."""

    def on_error(self, error: BaseException) -> None:
        """The robot refused this skill's last action or could not run it; by default the error ends the run."""
        raise error
