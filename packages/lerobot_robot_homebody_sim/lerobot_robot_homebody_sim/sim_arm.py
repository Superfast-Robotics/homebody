"""Simulated arms as LeRobot robots on MuJoCo: a rig (rigs.RIGS: one arm, two side by side, or an arm on a mobile
base) with cubes on the table or floor, front and top cameras on the rig's workspace and each arm's wrist camera. Observations and
actions are the LeRobot followers': `<prefix><motor>.pos`, joints in degrees and the gripper in the follower's
units over its actuator range, plus the camera images. A simulated robot also has a clock (`epoch`, `time`,
`advance`, `stop`) and the scene's truth for scripted checks (`object_position`, `arm_frame`), never for skills."""
from dataclasses import dataclass, field

import mujoco
import numpy as np
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot

from .arms import ARMS
from .rigs import RIGS

CUBE = 0.03  # m, the cube's side


@dataclass(kw_only=True)
class SimArmConfig(RobotConfig):
    rig: str = "so101"
    fps: int = 30
    width: int = 320
    height: int = 240
    # {name: [x, y]} cubes on the table in the world (m); a cube at each arm's pick spot when empty
    cubes: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if self.rig not in RIGS:
            raise ValueError(f"robot.rig must be one of {', '.join(RIGS)}")


def _config(type_name, rig):
    """A registered config class for RIG under LeRobot's TYPE_NAME."""
    cls = dataclass(kw_only=True)(type(f"{type_name.title().replace('_', '')}Config", (SimArmConfig,),
                                      {"__annotations__": {"rig": str}, "rig": rig}))
    return RobotConfig.register_subclass(type_name)(cls)


So101FollowerSimConfig = _config("so101_follower_sim", "so101")
KochFollowerSimConfig = _config("koch_follower_sim", "koch")
OmxFollowerSimConfig = _config("omx_follower_sim", "omx")
OpenarmFollowerSimConfig = _config("openarm_follower_sim", "openarm")
BiSo101FollowerSimConfig = _config("bi_so101_follower_sim", "bi_so101")
LekiwiSimConfig = _config("lekiwi_sim", "lekiwi")
BASE = ("x.vel", "y.vel", "theta.vel")  # LeRobot's lekiwi: m/s, m/s and deg/s in the base's frame


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """A camera's xyaxes at EYE looking at TARGET with UP toward the image top (MuJoCo looks along -z)."""
    forward = np.asarray(target, float) - np.asarray(eye, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return [*right, *np.cross(right, forward)]


def build_scene(rig, cubes):
    """The floor or table, the cubes and the cameras, with each arm attached at its mount (on the table, or on
    the mobile base) under its joints' prefix, and a site "<prefix>mount" at each mount."""
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    arms = [(mount, mujoco.MjSpec.from_file(str(ARMS[mount.arm].xml_path()))) for mount in rig.mounts]
    for name in dir(arms[0][1].option):  # attaching drops the arm's own physics options
        if not name.startswith("_"):
            setattr(spec.option, name, getattr(arms[0][1].option, name))
    # grasping contacts for every arm (Menagerie's SO-101 settings): pyramidal cones let a held object creep out
    spec.option.cone, spec.option.impratio = mujoco.mjtCone.mjCONE_ELLIPTIC, max(10.0, spec.option.impratio)
    spec.worldbody.add_light(pos=[0, 0, 2.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.worldbody.add_light(pos=[0.4, -0.5, 1.0], dir=[-0.3, 0.5, -1], diffuse=[0.3, 0.3, 0.3], castshadow=True)
    ground = 0.0 if rig.base is None else rig.base.floor  # a mobile base's own contact pairs name the "floor"
    spec.worldbody.add_geom(name="table" if rig.base is None else "floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            pos=[0, 0, ground], size=[3, 3, 0.05], rgba=[0.55, 0.45, 0.35, 1],
                            friction=[1.0, 0.005, 0.0001])
    colours = ([0.85, 0.2, 0.2, 1], [0.2, 0.7, 0.3, 1], [0.2, 0.35, 0.85, 1], [0.9, 0.8, 0.2, 1])
    for index, (name, (x, y)) in enumerate(cubes.items()):
        body = spec.worldbody.add_body(name=name, pos=[x, y, ground + CUBE / 2])
        body.add_freejoint()
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE / 2] * 3, mass=0.03,
                      rgba=colours[index % len(colours)], friction=[1.5, 0.01, 0.0001], condim=4)
    # the scene's cameras frame the cubes, seen from beyond them and from above; a mobile robot brings its own
    # "front" camera, so the scene's beyond-view is "room" there
    spot = np.array([*np.mean(list(cubes.values()), axis=0), ground])
    out = np.array([*spot[:2], 0.0]) / np.linalg.norm(spot[:2])
    reach = max(np.linalg.norm(np.array(xy) - spot[:2]) for xy in cubes.values())
    beyond = spot + out * (0.32 + reach) + np.array([0, 0, 0.26 + reach * 0.6])
    spec.worldbody.add_camera(name="front" if rig.base is None else "room", pos=beyond.tolist(),
                              xyaxes=look_at(beyond, spot * [0.6, 0.6, 1]), fovy=58)
    top = spot * [0.75, 0.75, 1] + np.array([0, 0, 0.5 + reach])
    spec.worldbody.add_camera(name="top", pos=top.tolist(), xyaxes=look_at(top, spot * [0.75, 0.75, 1], up=out),
                              fovy=58)
    parent = spec.worldbody
    if rig.base is not None:
        spec.attach(mujoco.MjSpec.from_file(str(rig.base.xml_path())), frame=spec.worldbody.add_frame(), prefix="")
        parent = spec.body(rig.base.mount_body)
    for mount, arm in arms:
        pose = {"pos": list(mount.pos), "euler": [0, 0, np.radians(mount.yaw)]}
        parent.add_site(name=f"{mount.prefix}mount", **pose)
        spec.attach(arm, frame=parent.add_frame(**pose), prefix=mount.joint_prefix)
    return spec.compile()


@dataclass(frozen=True)
class _Motor:
    key: str           # the action and observation key, without ".pos"
    joint: str         # its joint in the compiled scene
    gripper: bool
    embodiment: object


class SimArm(Robot):
    config_class = SimArmConfig
    name = "sim_arm"

    def __init__(self, config):
        super().__init__(config)
        self.config, self.rig = config, RIGS[config.rig]
        self._motors = [
            _Motor(mount.prefix + motor, mount.joint_prefix + joint, motor == "gripper", ARMS[mount.arm])
            for mount in self.rig.mounts
            for motor, joint in zip((*ARMS[mount.arm].motors, "gripper"),
                                    (*ARMS[mount.arm].joints, ARMS[mount.arm].gripper), strict=True)]
        many = len(self.rig.mounts) > 1  # one arm's wrist camera is "wrist", as LeRobot's robots name it
        self._wrists = {(f"{mount.prefix}wrist" if many else "wrist"): mount.joint_prefix + ARMS[mount.arm].wrist_camera
                        for mount in self.rig.mounts if ARMS[mount.arm].wrist_camera}
        self._base_keys = BASE if self.rig.base is not None else ()
        self.model = self.data = self._renderer = None
        self.epoch = 0

    # --- LeRobot Robot ---
    @property
    def observation_features(self):
        image = (self.config.height, self.config.width, 3)
        return {**self.action_features, "front": image, **dict.fromkeys(self._wrists, image)}

    @property
    def action_features(self):
        return {**{f"{m.key}.pos": float for m in self._motors}, **dict.fromkeys(self._base_keys, float)}

    @property
    def is_connected(self):
        return self.model is not None

    def connect(self, calibrate=True):
        if self.model is not None:
            return
        self.model = build_scene(self.rig, self.config.cubes or self.rig.cubes())
        self.data = mujoco.MjData(self.model)
        joints = [self.model.joint(m.joint).id for m in self._motors]
        self._qpos = self.model.jnt_qposadr[joints]
        self._actuators = np.array([next(a for a in range(self.model.nu) if self.model.actuator_trnid[a, 0] == j)
                                    for j in joints])
        self._low, self._high = self.model.actuator_ctrlrange[self._actuators].T
        if self.rig.base is not None:
            base = self.rig.base
            self._wheels = np.array([self.model.actuator(name).id for name in base.wheels])
            self._wheel_from_body = np.linalg.inv(base.wheel_matrix())
            cubes = set(self.config.cubes or self.rig.cubes())
            free = next(j for j in range(self.model.njnt) if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                        and self.model.body(self.model.jnt_bodyid[j]).name not in cubes)
            self._base_body = self.model.jnt_bodyid[free]
            self._base_qvel = self.model.jnt_dofadr[free]
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
        for key, camera in self._wrists.items():
            observation[key] = self._render(camera)
        return observation

    def send_action(self, action):
        unknown = set(action) - set(self.action_features) - {"epoch"}
        if unknown:
            raise ValueError(f"unknown action keys: {', '.join(sorted(unknown))}")
        if action.get("epoch", self.epoch) != self.epoch:
            raise RuntimeError("stale epoch: the scene was reset")
        given = [key in action for key in self._base_keys]
        if any(given):
            if not all(given):
                raise ValueError("a base command takes x.vel, y.vel and theta.vel together")
            body = [float(action["x.vel"]), float(action["y.vel"]), np.radians(float(action["theta.vel"]))]
            wheels = self._wheel_from_body @ body
            self.data.ctrl[self._wheels] = np.clip(wheels, *self.model.actuator_ctrlrange[self._wheels].T)
        for index, motor in enumerate(self._motors):
            if f"{motor.key}.pos" in action:
                self.data.ctrl[self._actuators[index]] = self._to_ctrl(index, action[f"{motor.key}.pos"])
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
        state = {f"{m.key}.pos": float(self._from_q(index, q[index])) for index, m in enumerate(self._motors)}
        if self.rig.base is not None:  # the body's own velocity in its frame (a slipping wheel shows)
            rotation = self.data.xmat[self._base_body].reshape(3, 3)
            linear = rotation.T @ self.data.qvel[self._base_qvel:self._base_qvel + 3]
            omega = self.data.qvel[self._base_qvel + 5]  # a free joint's angular velocity is in its body frame
            state.update({"x.vel": float(linear[0]), "y.vel": float(linear[1]), "theta.vel": float(np.degrees(omega))})
        return state

    def advance(self, epoch, seconds):
        if epoch != self.epoch:
            raise RuntimeError("stale epoch: the scene was reset")
        mujoco.mj_step(self.model, self.data, nstep=max(1, round(seconds / self.model.opt.timestep)))

    def stop(self, epoch):
        """Hold every motor where it is and stop the base."""
        hold = {key: value for key, value in self.get_state().items() if key not in self._base_keys}
        self.send_action({"epoch": epoch, **hold, **dict.fromkeys(self._base_keys, 0.0)})

    def reset(self):
        """Every arm at home, the cubes where the config puts them, a new epoch."""
        mujoco.mj_resetData(self.model, self.data)
        homes = [value for mount in self.rig.mounts for value in ARMS[mount.arm].home]
        for index, home in enumerate(homes):
            value = self._to_ctrl(index, home)
            self.data.qpos[self._qpos[index]] = value
            self.data.ctrl[self._actuators[index]] = value
        mujoco.mj_forward(self.model, self.data)
        if self.rig.base is not None:  # let the base settle on its wheels before anything is commanded
            mujoco.mj_step(self.model, self.data, nstep=round(0.5 / self.model.opt.timestep))
        self.epoch += 1

    @property
    def simulator(self):
        return self

    def arm_spec(self, prefix=""):
        """The ArmSpec of the arm under PREFIX, for arm skills."""
        mount = next(m for m in self.rig.mounts if m.prefix == prefix)
        return ARMS[mount.arm].spec(prefix)

    def arm_frame(self, prefix=""):
        """The world pose (4x4) of the arm under PREFIX now (a mobile base carries it): its skills' targets are in
        this frame."""
        site = self.model.site(f"{prefix}mount").id
        frame = np.eye(4)
        frame[:3, :3] = self.data.site_xmat[site].reshape(3, 3)
        frame[:3, 3] = self.data.site_xpos[site]
        return frame

    def object_position(self, name):
        """NAME's centre in the world: scripted checks only."""
        return self.data.xpos[self.model.body(name).id].copy()

    # --- units ---
    def _to_ctrl(self, index, value):
        motor = self._motors[index]
        if motor.gripper:  # the follower's closed..open values over the actuator range
            closed, opened = motor.embodiment.gripper_units
            fraction = (float(value) - closed) / (opened - closed)
            fraction = 1 - fraction if motor.embodiment.gripper_closes_high else fraction
            return self._low[index] + fraction * (self._high[index] - self._low[index])
        return np.deg2rad(float(value))

    def _from_q(self, index, q):
        motor = self._motors[index]
        if motor.gripper:
            closed, opened = motor.embodiment.gripper_units
            fraction = (q - self._low[index]) / (self._high[index] - self._low[index])
            fraction = 1 - fraction if motor.embodiment.gripper_closes_high else fraction
            return closed + fraction * (opened - closed)
        return np.rad2deg(q)

    def _render(self, camera):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.config.height, self.config.width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()


def _robot(config_class, type_name):
    return type(f"{config_class.__name__.removesuffix('Config')}", (SimArm,),
                {"config_class": config_class, "name": type_name})


So101FollowerSim = _robot(So101FollowerSimConfig, "so101_follower_sim")
KochFollowerSim = _robot(KochFollowerSimConfig, "koch_follower_sim")
OmxFollowerSim = _robot(OmxFollowerSimConfig, "omx_follower_sim")
OpenarmFollowerSim = _robot(OpenarmFollowerSimConfig, "openarm_follower_sim")
BiSo101FollowerSim = _robot(BiSo101FollowerSimConfig, "bi_so101_follower_sim")
LekiwiSim = _robot(LekiwiSimConfig, "lekiwi_sim")
