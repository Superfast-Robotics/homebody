"""The G1 as LeRobot features: flat observation and action dicts keyed after LeRobot's `unitree_g1` (its 29
joints as `k<Joint>.q`, in controller order), plus the hand, base, pose and camera keys HomeBody adds, and the
conversions between those dicts and HomeBody's Frame, Measurement and WristView. Nothing here imports LeRobot:
the harness reads any robot through these dicts, whether or not LeRobot is installed."""
import numpy as np

from homebody.primitives.observations import Frame, HandState, Measurement, WristView

JOINTS = ("kLeftHipPitch", "kLeftHipRoll", "kLeftHipYaw", "kLeftKnee", "kLeftAnklePitch", "kLeftAnkleRoll",
          "kRightHipPitch", "kRightHipRoll", "kRightHipYaw", "kRightKnee", "kRightAnklePitch", "kRightAnkleRoll",
          "kWaistYaw", "kWaistRoll", "kWaistPitch",
          "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
          "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
          "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
          "kRightWristRoll", "kRightWristPitch", "kRightWristYaw")
SIDES = ("left", "right")
ARM_JOINTS = {"left": JOINTS[15:22], "right": JOINTS[22:29]}
BASE = ("base.vx", "base.vy", "base.yaw_rate")  # body-frame m/s and rad/s
EPOCH = "epoch"  # an action's command generation; the robot's current one when absent
STOP = "stop"    # true: zero the base and keep both arms where they are (the Robot's stop)

ACTION_FEATURES = {**{f"{joint}.q": float for side in SIDES for joint in ARM_JOINTS[side]},
                   **{f"{side}_hand.closure": float for side in SIDES},
                   **{key: float for key in BASE}}


def state_features():
    """The measured state: joints, hands, base pose and the torso and head-camera poses."""
    features: dict[str, object] = {f"{joint}.q": float for joint in JOINTS}
    for side in SIDES:
        features.update({f"{side}_hand.joints": (7,), f"{side}_hand.closure": float,
                         f"{side}_hand.finger_effort": (3,), f"{side}_hand.finger_gap": (3,)})
    features.update({"base.x": float, "base.y": float, "base.yaw": float, "torso.pose": (4, 4),
                     "head.pose": (4, 4), EPOCH: int, "sequence": int, "time": float})
    return features


def observation_features(height, width):
    """The state plus the head camera: RGB, depth (metres, NaN invalid), object labels and intrinsics."""
    return {**state_features(), "head": (height, width, 3), "head.depth": (height, width),
            "head.labels": (height, width), "head.intrinsics": (3, 3)}


def wrist_features(side, height, width):
    """SIDE's on-request wrist camera."""
    camera = f"{side}_wrist"
    return {camera: (height, width, 3), f"{camera}.depth": (height, width), f"{camera}.labels": (height, width),
            f"{camera}.intrinsics": (3, 3), f"{camera}.pose": (4, 4), EPOCH: int, "time": float}


def encode_state(state):
    """A Measurement (or a Frame's measured part) as state keys."""
    observation: dict[str, object] = {f"{joint}.q": float(value)
                                      for joint, value in zip(JOINTS, state.joints, strict=True)}
    for side in SIDES:
        hand = state.hands[side]
        observation.update({f"{side}_hand.joints": np.array(hand.joints), f"{side}_hand.closure": float(hand.closure),
                            f"{side}_hand.finger_effort": np.array(hand.finger_effort_fraction),
                            f"{side}_hand.finger_gap": np.array(hand.finger_closure_gap)})
    x, y, yaw = state.base_pose
    observation.update({"base.x": float(x), "base.y": float(y), "base.yaw": float(yaw),
                        "torso.pose": np.array(state.world_torso), "head.pose": np.array(state.world_camera),
                        EPOCH: int(state.epoch), "sequence": int(state.sequence), "time": float(state.time)})
    return observation


def encode_frame(frame):
    return {**encode_state(frame), "head": np.array(frame.rgb), "head.depth": np.array(frame.depth),
            "head.labels": np.array(frame.labels), "head.intrinsics": np.array(frame.intrinsics)}


def encode_wrist(view):
    camera = f"{view.side}_wrist"
    return {camera: np.array(view.rgb), f"{camera}.depth": np.array(view.depth),
            f"{camera}.labels": np.array(view.labels), f"{camera}.intrinsics": np.array(view.intrinsics),
            f"{camera}.pose": np.array(view.world_camera), EPOCH: int(view.epoch), "time": float(view.time)}


def _hands(observation):
    return {side: HandState(observation[f"{side}_hand.joints"], observation[f"{side}_hand.closure"],
                            observation[f"{side}_hand.finger_effort"], observation[f"{side}_hand.finger_gap"])
            for side in SIDES}


def _measured(observation):
    return (observation["head.pose"], observation["torso.pose"],
            np.array([observation["base.x"], observation["base.y"], observation["base.yaw"]]),
            np.array([observation[f"{joint}.q"] for joint in JOINTS]), _hands(observation))


def decode_measurement(observation):
    camera, torso, base, joints, hands = _measured(observation)
    return Measurement(observation[EPOCH], observation["sequence"], observation["time"], camera, torso, base,
                       joints, hands)


def decode_frame(observation, navigation):
    camera, torso, base, joints, hands = _measured(observation)
    return Frame(observation[EPOCH], observation["sequence"], observation["time"], observation["head"],
                 observation["head.depth"], observation["head.intrinsics"], camera, torso, base, joints, hands,
                 observation["head.labels"], navigation)


def decode_wrist(observation, side):
    camera = f"{side}_wrist"
    return WristView(observation[EPOCH], observation["time"], side, observation[camera],
                     observation[f"{camera}.depth"], observation[f"{camera}.labels"],
                     observation[f"{camera}.intrinsics"], observation[f"{camera}.pose"])


def apply_action(robot, action):
    """Send ACTION's commands to ROBOT (HomeBody's Robot commands): STOP first, then the base takes all three
    velocities, an arm all seven joints, a hand its closure; keys it leaves out keep their current command."""
    unknown = set(action) - set(ACTION_FEATURES) - {EPOCH, STOP}
    if unknown:
        raise ValueError(f"unknown action keys: {', '.join(sorted(unknown))}")
    epoch = int(action.get(EPOCH, robot.epoch))
    if action.get(STOP):
        robot.stop(epoch)
    given = [key in action for key in BASE]
    if any(given):
        if not all(given):
            raise ValueError("a base command takes base.vx, base.vy and base.yaw_rate together")
        robot.base_velocity(epoch, *(float(action[key]) for key in BASE))
    for side in SIDES:
        keys = [f"{joint}.q" for joint in ARM_JOINTS[side]]
        present = [key in action for key in keys]
        if any(present):
            if not all(present):
                raise ValueError(f"an arm command takes all seven {side} arm joints")
            robot.arm_target(epoch, side, np.array([float(action[key]) for key in keys]))
        if f"{side}_hand.closure" in action:
            robot.grip(epoch, side, float(action[f"{side}_hand.closure"]))
    return action
