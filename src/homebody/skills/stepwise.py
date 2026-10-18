"""A HomeBody skill in LeRobot's policy form (lerobot_skills). The skill's own `execute` loop runs on a worker
thread against a proxy robot: its commands are held until the loop next reads the robot or lets time pass, and
that batch, with the seconds to run it, is the step's action dict (`homebody.robots.g1` keys). Its reads run on
the caller's thread too (a renderer's GL context belongs to the thread that made it), so only the caller's
thread ever touches the robot. The runner sends
it to the session's robot and advances, so the robot gets the same commands, in the same order, at the same
simulated times as when the loop drives it directly; an error sending or running a step is raised in the loop
where it read or advanced."""
import copy
import queue
import threading
from dataclasses import dataclass

from lerobot_skills import ADVANCE, EPOCH, FAILED, SUCCEEDED, Skill, SkillConfig, Status, run_skill

from homebody.robots import g1


@SkillConfig.register_subclass("homebody")
@dataclass
class HarnessSkillConfig(SkillConfig):
    skill: str = ""  # the registered HomeBody skill's name


class _Proxy:
    """The worker's Robot: reads pass through once held commands are sent; commands are held for the step."""

    def __init__(self, robot, request, read):
        self._robot, self._request, self._read = robot, request, read
        self._held, self._epoch = {}, None

    @property
    def epoch(self):
        return self._read(lambda: self._robot.epoch)

    @property
    def time(self):
        return self._read(lambda: self._robot.time)

    def snapshot(self):
        self._flush()
        return self._read(lambda: self._robot.snapshot())

    def measure(self):
        self._flush()
        return self._read(lambda: self._robot.measure())

    def wrist_view(self, side):
        self._flush()
        return self._read(lambda: self._robot.wrist_view(side))

    def wrist_views(self):
        self._flush()
        return self._read(lambda: self._robot.wrist_views())

    def base_velocity(self, epoch, forward, lateral, yaw_rate):
        self._hold(epoch, dict(zip(g1.BASE, (forward, lateral, yaw_rate))))

    def arm_target(self, epoch, side, joints):
        keys = [f"{joint}.q" for joint in g1.ARM_JOINTS[side]]
        if len(joints) != len(keys):
            raise ValueError(f"an arm command takes {len(keys)} joints")
        self._hold(epoch, dict(zip(keys, (float(value) for value in joints))))

    def grip(self, epoch, side, closure):
        self._hold(epoch, {f"{side}_hand.closure": closure})

    def stop(self, epoch):
        self._flush()
        self._request({EPOCH: epoch, g1.STOP: True})

    def advance(self, epoch, seconds):
        if self._held and epoch != self._epoch:
            self._flush()
        held, self._held = self._held, {}
        self._request({EPOCH: epoch, **held, ADVANCE: seconds})  # the held commands run for these seconds

    def _hold(self, epoch, commands):
        if self._held and epoch != self._epoch:
            self._flush()
        self._held.update(commands)
        self._epoch = epoch

    def _flush(self):
        if self._held:
            held, self._held = self._held, {}
            self._request({EPOCH: self._epoch, **held})


class HarnessSkill(Skill):
    """MODULE's `execute(PLAN, ctx)` (a registered HomeBody skill, prepared) stepped by select_action. It reads
    the robot itself, so it ignores its observation; `result` is the skill's Result once it is done."""

    config_class = HarnessSkillConfig
    name = "homebody"
    observes = False

    def __init__(self, config, module, plan, ctx):
        super().__init__(config)
        self.module, self.plan, self.ctx = module, plan, ctx
        self.result, self._worker, self._error = None, None, None
        self._steps, self._replies = queue.Queue(), queue.Queue()

    def reset(self, observation=None, arguments=None):
        self.status, self.result, self._worker, self._error = Status(), None, None, None

    def select_action(self, observation):
        if self._worker is None:
            self._worker = threading.Thread(target=self._work, name=f"skill-{self.config.skill}", daemon=True)
            self._worker.start()
        else:
            self._replies.put((None, self._error))
            self._error = None
        kind, value = self._steps.get()
        while kind == "read":  # the worker's robot reads, run here
            try:
                self._replies.put((value(), None))
            except BaseException as error:  # noqa: BLE001 -- raised in the worker, where it read
                self._replies.put((None, error))
            kind, value = self._steps.get()
        if kind == "step":
            return value
        if kind == "error":
            self.status = Status(FAILED, "ERROR", f"{type(value).__name__}: {value}")
            raise value
        self.result = value
        self.status = Status(SUCCEEDED if value.ok else FAILED, value.code, value.message, dict(value.details or {}))
        return {}

    def on_error(self, error):
        self._error = error

    def _request(self, action):
        self._steps.put(("step", action))
        _, error = self._replies.get()
        if error is not None:
            raise error

    def _read(self, read):
        self._steps.put(("read", read))
        value, error = self._replies.get()
        if error is not None:
            raise error
        return value

    def _work(self):
        ctx = copy.copy(self.ctx)
        proxy = _Proxy(self.ctx.actions, self._request, self._read)
        ctx.observations = ctx.actions = proxy
        try:
            result = self.module.execute(self.plan, ctx)
            proxy._flush()  # commands given after the last read or advance, as a direct run had sent them
            self._steps.put(("result", result))
        except BaseException as error:  # noqa: BLE001 -- the caller's thread re-raises it
            try:
                proxy._flush()
            except BaseException:  # noqa: BLE001, S110 -- the skill's own error is the one to report
                pass
            self._steps.put(("error", error))


class _Port:
    """The session's robot as the runner's: actions as g1 dicts, advance by epoch."""

    def __init__(self, robot):
        self.robot = robot

    @property
    def epoch(self):
        return self.robot.epoch

    def send_action(self, action):
        return g1.apply_action(self.robot, action)

    def advance(self, epoch, seconds):
        self.robot.advance(epoch, seconds)


def execute_stepwise(module, plan, ctx, name):
    """MODULE's execute run as a lerobot_skills skill on CTX's robot; its Result."""
    skill = HarnessSkill(HarnessSkillConfig(skill=name), module, plan, ctx)
    skill.reset()
    run_skill(skill, _Port(ctx.actions))
    return skill.result
