"""What a simulated robot is made of: one or more arms (arms.ARMS), each mounted on the table at a pose under a key
prefix, so a bimanual rig's actions are `left_shoulder_pan.pos`, `right_gripper.pos`, ... as LeRobot's bimanual
followers name them."""
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .arms import ARMS, PREPARED, prepare_lekiwi
from .models import model_dir


@dataclass(frozen=True)
class Mount:
    arm: str                                  # an ARMS key
    prefix: str = ""                          # its keys' prefix ("left_")
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0                          # degrees about the vertical
    model_prefix: str | None = None           # its joints' prefix in the scene; PREFIX when None

    @property
    def joint_prefix(self):
        return self.prefix if self.model_prefix is None else self.model_prefix

    def frame(self):
        """The world pose of the arm's base: a 4x4 transform."""
        c, s = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        frame = np.eye(4)
        frame[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        frame[:3, 3] = self.pos
        return frame

    def pick_spot(self):
        """The arm's comfortable cube position, in the world."""
        return (self.frame() @ [*ARMS[self.arm].pick_xy, 0.0, 1.0])[:3]


@dataclass(frozen=True)
class Base:
    """A mobile robot's model, carrying the arms (mounted on MOUNT_BODY) and driven by three omni wheels: LeRobot's
    x.vel, y.vel (m/s) and theta.vel (deg/s) become wheel speeds through the base's kinematics."""
    model: str                       # a models.SOURCES key
    xml: str
    mount_body: str
    wheels: tuple[str, str, str]     # its wheel actuators, in the order of the kinematics' columns
    wheel_radius: float
    base_radius: float
    floor: float                     # the floor's height below the model's origin
    cubes: dict                      # {name: [x, y]} on the floor, in the world
    prepare: Callable[[Path, Path], None] | None = None

    def xml_path(self):
        source = model_dir(self.model) / self.xml
        prepared = source.with_name(f"{source.stem}.{PREPARED}.xml")
        if self.prepare is not None and not prepared.exists():
            self.prepare(source, prepared)
        return prepared if self.prepare is not None else source

    def wheel_matrix(self):
        """Wheel speeds (rad/s) to body velocities [vx, vy, omega]: the three-wheel omni drive's forward kinematics."""
        r, L = self.wheel_radius, self.base_radius
        return r * np.array([[np.sqrt(3) / 2, -np.sqrt(3) / 2, 0], [-1 / 2, -1 / 2, 1], [-1 / (3 * L)] * 3])


@dataclass(frozen=True)
class Rig:
    mounts: tuple[Mount, ...]
    base: Base | None = None

    def cubes(self):
        """A cube at each arm's pick spot: "cube" for one arm, "<prefix>cube" for several."""
        if self.base is not None:
            return dict(self.base.cubes)
        if len(self.mounts) == 1:
            return {"cube": self.mounts[0].pick_spot()[:2].tolist()}
        return {f"{mount.prefix}cube": mount.pick_spot()[:2].tolist() for mount in self.mounts}

    def workspace(self):
        """The centre of the arms' pick spots, where the cameras look."""
        return np.mean([mount.pick_spot() for mount in self.mounts], axis=0)


RIGS = {
    "so101": Rig((Mount("so101"),)),
    "koch": Rig((Mount("koch"),)),
    "omx": Rig((Mount("omx"),)),
    "openarm": Rig((Mount("openarm"),)),
    # LeRobot's bi_so_follower: two SO-101s side by side, facing the same way
    "bi_so101": Rig((Mount("so101", "left_", (0.0, 0.18, 0.0)), Mount("so101", "right_", (0.0, -0.18, 0.0)))),
    # LeRobot's lekiwi: Ekumen's omni-wheel base carrying an SO-101 (keys arm_*, joints unprefixed), facing forward
    "lekiwi": Rig((Mount("so101", "arm_", yaw=-90.0, model_prefix=""),),
                  Base("lekiwi", "assets/lekiwi/lekiwi.xml", "so_arm100_link",
                       ("base_left_wheel", "base_right_wheel", "base_back_wheel"), 0.05, 0.125, -0.1,
                       {"cube": [0.25, 0.0]}, prepare=prepare_lekiwi)),
}
