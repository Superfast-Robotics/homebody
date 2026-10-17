"""The context exploration skills run in: the explorer as their `LiveWorld`, the live task frame
and only the source commands a skill declares."""
from dataclasses import dataclass

from homebody.primitives.navigation.errors import Abort, ActionRefused
from homebody.primitives.navigation.live import LiveFrame
from homebody.real2sim.exploration import planner
from homebody.skills.contract import SkillFailure
from homebody.skills.live import LiveWorld


@dataclass(frozen=True)
class LivePlanning:
    """What a live skill's preparation reads."""
    world: LiveWorld
    routes: object
    fail: object


class AllowedActions:
    """The source's commands that the running skill declares in NEEDS or OPTIONAL, and no others."""

    def __init__(self, source):
        self._source, self.allowed = source, frozenset()

    def __getattr__(self, name):
        if name not in self.allowed:
            raise AttributeError(f"Undeclared action capability: {name}")
        return getattr(self._source, name)


class LiveContext:
    """A live skill's handle on exploration. EXPLORATION is the explorer, which skills read as
    `world`; ROUTES defaults to the planner module. A typed live interruption that ends a skill
    stays in `failure`."""

    def __init__(self, exploration, *, routes=None, emit=lambda event: None):
        self.world: LiveWorld = exploration
        self.routes = planner if routes is None else routes
        self.actions, self.observations = AllowedActions(exploration.source), self
        self.emit, self.failure = emit, None
        self._skill_name = self._stage_name = None

    def begin(self, name, skill):
        """Start SKILL's attempt under NAME: no failure yet, and only its declared commands."""
        self.failure = None
        self._skill_name, self._stage_name = name, None
        commands = () if skill is None else (*skill.NEEDS, *getattr(skill, "OPTIONAL", ()))
        self.actions.allowed = frozenset(commands)

    def settle(self, epoch, result):
        """Live motion holds the base itself when it returns or unwinds: no second command."""
        return result

    @property
    def epoch(self):
        """The map's generation (set_boundary replaces the map) and its task-region revision."""
        world = self.world
        return world.task_region_generation, getattr(world.live, "task_region_revision", -1)

    def frame(self):
        epoch = self.epoch
        return LiveFrame(epoch, f"live-{epoch[1]}")

    def check(self, epoch):
        if epoch != self.epoch:
            self.fail(Abort("map frame or task boundary changed before action execution"))

    def observe(self, epoch):
        self.check(epoch)
        return self.frame()

    def planning(self):
        return LivePlanning(self.world, self.routes, self.fail)

    def stage(self, name):
        if name != self._stage_name:
            self._stage_name = name
            self.emit({"type": "skill_stage", "skill": self._skill_name, "stage": name})

    def fail(self, error):
        """End the skill with ERROR's code, keeping ERROR in `failure`."""
        self.failure = error
        code = (error.code if isinstance(error, ActionRefused) else
                "ABORTED" if isinstance(error, Abort) else "STOPPED")
        raise SkillFailure(code, str(error), getattr(error, "details", {})) from error

    def unwrap(self, result):
        """RESULT as the explorer's motion outcome: the interruption that ended it, or a refusal,
        raises; otherwise whether the motion reached its goal."""
        if self.failure is not None:
            raise self.failure
        if not result.ok:
            raise ActionRefused(result.message, code=result.code, details=result.details)
        return result.details.get("reached", True)

    def ask(self, record):
        """Hold for the operator's answer; only a skill that declares hold_position may ask."""
        if "hold_position" not in self.actions.allowed:
            raise SkillFailure("ACTION_REJECTED", "Operator questions require declared hold_position")
        return self.world.ask_question(record)


def run_live_result(owner, name, arguments):
    """NAME's result through OWNER's session; the live interruption that ended it raises instead."""
    session = owner.session
    result = session.runner.run(name, arguments, session.context.frame())
    if session.context.failure is not None:
        raise session.context.failure
    return result


def run_live(owner, name, arguments):
    """Whether NAME reached its goal through OWNER's session; an interruption or a refusal raises."""
    return owner.session.context.unwrap(run_live_result(owner, name, arguments))
