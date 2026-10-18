"""HomeBody's Robot interface over a LeRobot-style robot. The skills and the session keep their epoch-stamped
commands and Frame/Measurement records; underneath, every observation and command goes through the robot's
LeRobot dicts (`get_observation`, `send_action`), decoded with the robot's feature module (`robots.g1` for
the G1)."""
from homebody.robots import g1


class LeRobotSurface:
    """ROBOT is connected and offers LeRobot's `get_observation` / `send_action` plus what a simulated robot adds:
    `epoch`, `time`, `get_state` (the observation without the camera), `get_wrist(side)`, `advance(epoch,
    seconds)`, `stop(epoch)`, `navigation` (the scene's map) and `simulator` (simulator-only hooks: the scorer's
    physics record, observer views, reset)."""

    def __init__(self, robot, features=g1):
        self.robot, self.features = robot, features

    # --- Robot ---
    @property
    def epoch(self):
        return self.robot.epoch

    @property
    def time(self):
        return self.robot.time

    def snapshot(self):
        return self.features.decode_frame(self.robot.get_observation(), self.robot.navigation)

    def measure(self):
        return self.features.decode_measurement(self.robot.get_state())

    def wrist_view(self, side):
        return self.features.decode_wrist(self.robot.get_wrist(side), side)

    def base_velocity(self, epoch, forward, lateral, yaw_rate):
        self.robot.send_action({self.features.EPOCH: epoch, **dict(zip(self.features.BASE,
                                                                        (forward, lateral, yaw_rate)))})

    def arm_target(self, epoch, side, joints):
        keys = [f"{joint}.q" for joint in self.features.ARM_JOINTS[side]]
        if len(joints) != len(keys):
            raise ValueError(f"an arm command takes {len(keys)} joints")
        self.robot.send_action({self.features.EPOCH: epoch, **dict(zip(keys, joints))})

    def grip(self, epoch, side, closure):
        self.robot.send_action({self.features.EPOCH: epoch, f"{side}_hand.closure": closure})

    def advance(self, epoch, seconds):
        self.robot.advance(epoch, seconds)

    def stop(self, epoch):
        self.robot.stop(epoch)

    # --- simulator-only, as the MuJoCo backend offers them ---
    @property
    def navigation(self):
        return self.robot.navigation

    @property
    def simulator(self):
        return self.robot.simulator

    @property
    def label_ids(self):
        return self.robot.simulator.label_ids

    def reset(self):
        return self.robot.simulator.reset()

    def close(self):
        self.robot.disconnect()

    def ego_image(self):
        return self.robot.simulator.ego_image()

    def wrist_views(self):
        return self.robot.simulator.wrist_views()

    def set_evaluation_sink(self, callback):
        return self.robot.simulator.set_evaluation_sink(callback)

    def evaluation_state(self):
        return self.robot.simulator.evaluation_state()

    def observer_frame(self):
        return self.robot.simulator.observer_frame()

    def shoulder_frame(self, side):
        return self.robot.simulator.shoulder_frame(side)
