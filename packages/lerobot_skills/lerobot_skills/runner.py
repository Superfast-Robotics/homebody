"""Run a skill on a robot: observe, select an action, send it, let it run, until the skill finishes."""
from .skill import ADVANCE, EPOCH


def run_skill(skill, robot, max_steps=None):
    """Step SKILL on ROBOT (LeRobot's `get_observation` / `send_action`, plus `advance(epoch, seconds)` for a
    robot with a clock) until the skill's status is done or MAX_STEPS actions have run; returns the status. An
    error sending or running an action goes to the skill's `on_error`."""
    steps = 0
    while not skill.done and (max_steps is None or steps < max_steps):
        observation = robot.get_observation() if skill.observes else {}
        action = dict(skill.select_action(observation))
        if skill.done:
            break
        seconds = action.pop(ADVANCE, None)
        try:
            if set(action) - {EPOCH}:
                robot.send_action(action)
            if seconds is not None:
                robot.advance(action.get(EPOCH, robot.epoch), float(seconds))
        except Exception as error:  # noqa: BLE001 -- the skill decides what a refused action means
            skill.on_error(error)
        steps += 1
    return skill.status
