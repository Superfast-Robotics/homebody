"""A simulated arm as a LeRobot robot on MuJoCo: the arm (arms.ARMS) bolted to a table top at the origin, objects
on the table, a front camera and the arm's wrist camera. Observations and actions are the LeRobot follower's:
`<motor>.pos`, joints in degrees and the gripper 0-100 over its actuator range (0 closed), plus the camera
images. A simulated robot also has a clock (`epoch`, `time`, `advance`, `stop`) and `simulator`, the scene's
truth for scripted checks (object poses in the arm's base frame), never for skills."""
from dataclasses import dataclass, field

import mujoco
import numpy as np
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot

from .arms import ARMS, MOTORS

CUBE = 0.03  # m, the cube's side


@dataclass(kw_only=True)
class SimArmConfig(RobotConfig):
    arm: str = "so101"
    fps: int = 30
    width: int = 320
    height: int = 240
    # {name: [x, y]} cubes on the table in the base frame (m); the arm's pick spot when empty
    cubes: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if self.arm not in ARMS:
            raise ValueError(f"robot.arm must be one of {', '.join(ARMS)}")


@RobotConfig.register_subclass("so101_follower_sim")
@dataclass(kw_only=True)
class So101FollowerSimConfig(SimArmConfig):
    arm: str = "so101"


@RobotConfig.register_subclass("koch_follower_sim")
@dataclass(kw_only=True)
class KochFollowerSimConfig(SimArmConfig):
    arm: str = "koch"


@RobotConfig.register_subclass("omx_follower_sim")
@dataclass(kw_only=True)
class OmxFollowerSimConfig(SimArmConfig):
    arm: str = "omx"


def build_scene(embodiment, cubes):
    """The table, the cubes and the cameras, with the arm attached at the origin."""
    spec = mujoco.MjSpec()
    arm = mujoco.MjSpec.from_file(str(embodiment.xml_path()))
    for name in dir(arm.option):  # attaching drops the arm's own physics options
        if not name.startswith("_"):
            setattr(spec.option, name, getattr(arm.option, name))
    # grasping contacts for every arm (Menagerie's SO-101 settings): pyramidal cones let a held object creep out
    spec.option.cone, spec.option.impratio = mujoco.mjtCone.mjCONE_ELLIPTIC, max(10.0, spec.option.impratio)
    spec.worldbody.add_light(pos=[0, 0, 2.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.worldbody.add_light(pos=[0.4, -0.5, 1.0], dir=[-0.3, 0.5, -1], diffuse=[0.3, 0.3, 0.3], castshadow=True)
    spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.05],
                            rgba=[0.55, 0.45, 0.35, 1], friction=[1.0, 0.005, 0.0001])
    colours = ([0.85, 0.2, 0.2, 1], [0.2, 0.7, 0.3, 1], [0.2, 0.35, 0.85, 1], [0.9, 0.8, 0.2, 1])
    for index, (name, (x, y)) in enumerate(cubes.items()):
        body = spec.worldbody.add_body(name=name, pos=[x, y, CUBE / 2])
        body.add_freejoint()
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE / 2] * 3, mass=0.03,
                      rgba=colours[index % len(colours)], friction=[1.5, 0.01, 0.0001], condim=4)
    spec.worldbody.add_camera(name="front", pos=[0.55, 0.0, 0.3], xyaxes=[0, 1, 0, -0.45, 0, 0.89], fovy=60)
    spec.worldbody.add_camera(name="top", pos=[0.22, 0.0, 0.55], xyaxes=[0, -1, 0, 1, 0, 0], fovy=60)
    spec.attach(arm, frame=spec.worldbody.add_frame(), prefix="")
    return spec.compile()


class SimArm(Robot):
    config_class = SimArmConfig
    name = "sim_arm"

    def __init__(self, config):
        super().__init__(config)
        self.config, self.embodiment = config, ARMS[config.arm]
        self.motors = (*MOTORS, "gripper")  # the follower's motor names; the model's joints are the embodiment's
        self._model_joints = (*self.embodiment.joints, self.embodiment.gripper)
        self.model = self.data = self._renderer = None
        self.epoch = 0

    # --- LeRobot Robot ---
    @property
    def observation_features(self):
        image = (self.config.height, self.config.width, 3)
        cameras = {"front": image, **({"wrist": image} if self.embodiment.wrist_camera else {})}
        return {**{f"{motor}.pos": float for motor in self.motors}, **cameras}

    @property
    def action_features(self):
        return {f"{motor}.pos": float for motor in self.motors}

    @property
    def is_connected(self):
        return self.model is not None

    def connect(self, calibrate=True):
        if self.model is not None:
            return
        cubes = self.config.cubes or {"cube": list(self.embodiment.pick_xy)}
        self.model = build_scene(self.embodiment, cubes)
        self.data = mujoco.MjData(self.model)
        joints = [self.model.joint(name).id for name in self._model_joints]
        self._qpos = self.model.jnt_qposadr[joints]
        self._actuators = np.array([next(a for a in range(self.model.nu) if self.model.actuator_trnid[a, 0] == j)
                                    for j in joints])
        self._low, self._high = self.model.actuator_ctrlrange[self._actuators].T
        self.reset()

    @property
    def is_calibrated(self):
        return True

    def calibrate(self):
        pass

    def configure(self):
        pass

    def get_observation(self):
        observation = self.get_state()
        observation["front"] = self._render("front")
        if self.embodiment.wrist_camera:
            observation["wrist"] = self._render(self.embodiment.wrist_camera)
        return observation

    def send_action(self, action):
        unknown = set(action) - set(self.action_features) - {"epoch"}
        if unknown:
            raise ValueError(f"unknown action keys: {', '.join(sorted(unknown))}")
        if action.get("epoch", self.epoch) != self.epoch:
            raise RuntimeError("stale epoch: the scene was reset")
        for index, motor in enumerate(self.motors):
            if f"{motor}.pos" in action:
                self.data.ctrl[self._actuators[index]] = self._to_ctrl(index, action[f"{motor}.pos"])
        return action

    def disconnect(self):
        if self._renderer is not None:
            self._renderer.close()
        self.model = self.data = self._renderer = None

    # --- what a simulated robot adds ---
    @property
    def time(self):
        return float(self.data.time)

    def get_state(self):
        """The motors' positions, without rendering."""
        q = self.data.qpos[self._qpos]
        return {f"{motor}.pos": float(self._from_q(index, q[index])) for index, motor in enumerate(self.motors)}

    def advance(self, epoch, seconds):
        if epoch != self.epoch:
            raise RuntimeError("stale epoch: the scene was reset")
        mujoco.mj_step(self.model, self.data, nstep=max(1, round(seconds / self.model.opt.timestep)))

    def stop(self, epoch):
        """Hold every motor where it is."""
        self.send_action({"epoch": epoch, **self.get_state()})

    def reset(self):
        """The arm at home, the cubes where the config puts them, a new epoch."""
        mujoco.mj_resetData(self.model, self.data)
        home = self.embodiment.home
        for index in range(len(self.motors)):
            value = self._to_ctrl(index, home[index])
            self.data.qpos[self._qpos[index]] = value
            self.data.ctrl[self._actuators[index]] = value
        mujoco.mj_forward(self.model, self.data)
        self.epoch += 1

    @property
    def simulator(self):
        return self

    def object_position(self, name):
        """NAME's centre in the arm's base frame (the world, here): scripted checks only."""
        return self.data.xpos[self.model.body(name).id].copy()

    # --- units ---
    def _to_ctrl(self, index, value):
        if index == len(self.motors) - 1:  # the gripper: 0 closed - 100 open over its actuator range
            fraction = float(value) / 100
            fraction = 1 - fraction if self.embodiment.gripper_closes_high else fraction
            return self._low[index] + fraction * (self._high[index] - self._low[index])
        return np.deg2rad(float(value))

    def _from_q(self, index, q):
        if index == len(self.motors) - 1:
            fraction = (q - self._low[index]) / (self._high[index] - self._low[index])
            return (1 - fraction if self.embodiment.gripper_closes_high else fraction) * 100
        return np.rad2deg(q)

    def _render(self, camera):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.config.height, self.config.width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()


class So101FollowerSim(SimArm):
    config_class = So101FollowerSimConfig
    name = "so101_follower_sim"


class KochFollowerSim(SimArm):
    config_class = KochFollowerSimConfig
    name = "koch_follower_sim"


class OmxFollowerSim(SimArm):
    config_class = OmxFollowerSimConfig
    name = "omx_follower_sim"
