"""The arms HomeBody simulates: each one's model, its LeRobot follower's motor names mapped onto the model's
joints (so a simulated arm's actions are `so101_follower`'s, `koch_follower`'s, `omx_follower`'s), and what its
skills need. A model that is not ready to simulate as fetched is prepared once into an MJCF beside it."""
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from lerobot_skills.kinematics import ArmSpec

from .models import model_dir

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
PREPARED = "homebody-3"  # the prepared models' version: a change here rebuilds them


@dataclass(frozen=True)
class ArmEmbodiment:
    model: str                    # a models.SOURCES key
    xml: str                      # the arm's model file in that folder (MJCF, or a URDF to prepare)
    joints: tuple[str, ...]       # the model's arm joints for MOTORS, in kinematic order
    gripper: str                  # the model's gripper joint
    ee_site: str
    approach_axis: tuple[float, float, float]
    home: tuple[float, ...]       # MOTORS (degrees) then the gripper (0 closed - 100 open)
    wrist_camera: str | None
    pick_xy: tuple[float, float]  # a comfortable cube position on the table, in the base frame (m)
    gripper_closes_high: bool = False  # the gripper closes toward its actuator range's upper end
    prepare: Callable[[Path, Path], None] | None = None

    def xml_path(self):
        source = model_dir(self.model) / self.xml
        if self.prepare is None:
            return source
        prepared = source.with_name(f"{source.stem}.{PREPARED}.xml")
        if not prepared.exists():
            self.prepare(source, prepared)
        return prepared

    def spec(self):
        return ArmSpec(MOTORS, "gripper", str(self.xml_path()), self.ee_site, self.approach_axis, home=self.home[:-1],
                       model_joints=self.joints)


def prepare_omx(source, prepared):
    """ROBOTIS's OMX MJCF with a site at its end-effector link (x out between the jaws)."""
    spec = mujoco.MjSpec.from_file(str(source))
    spec.body("end_effector_link").add_site(name="ee", pos=[0, 0, 0])
    prepared.write_text(spec.to_xml())


def prepare_koch(source, prepared):
    """ManiSkill's Koch v1.1 URDF as MJCF: its .glb meshes as STL, visuals kept, a position servo on each joint
    (XL430/XL330 torques), its SRDF's disabled collisions as contact exclusions, and a site where the jaws close
    (-z out between them)."""
    import trimesh
    meshes = source.parent / "meshes"
    for glb in meshes.glob("*.glb"):
        stl = glb.with_suffix(".stl")
        if not stl.exists():
            trimesh.load(glb, force="mesh").export(stl)
    urdf = re.sub(r'package://koch/meshes/([^"]+)\.(glb|stl)', r'meshes/\1.stl', source.read_text())
    root = ET.fromstring(urdf)
    extension = ET.SubElement(root, "mujoco")
    ET.SubElement(extension, "compiler", discardvisual="false", fusestatic="false", strippath="false")
    staged = source.with_name(f"{source.stem}.staged.urdf")
    staged.write_text(ET.tostring(root, encoding="unicode"))
    spec = mujoco.MjSpec.from_file(str(staged))
    staged.unlink()
    for index, name in enumerate(("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")):
        joint = spec.joint(name)
        joint.damping, joint.armature = [0.1, 0.0, 0.0], 0.005  # a hinge reads the first damping value
        actuator = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        actuator.set_to_position(kp=8.0 if index < 2 else 4.0)
        actuator.ctrllimited, actuator.ctrlrange = True, joint.range
        actuator.forcelimited, actuator.forcerange = True, [-1.4, 1.4] if index < 2 else [-0.5, 0.5]
    srdf = ET.parse(source.with_suffix(".srdf")).getroot()  # the link pairs the model's authors keep apart
    for pair in srdf.findall("disable_collisions"):
        spec.add_exclude(bodyname1=pair.get("link1"), bodyname2=pair.get("link2"))
    spec.body("gripper_tcp").add_site(name="ee", pos=grasp_centre(spec.compile()))
    prepared.write_text(spec.to_xml())


def grasp_centre(model, width=0.03):
    """Koch's jaws: gripper_tcp is the fixed jaw's tip and gripper_tcp2 the swinging jaw's. The point they close
    on (midway between the tips when they are WIDTH apart) in gripper_tcp's frame, where its site goes."""
    data = mujoco.MjData(model)
    joint = model.joint("joint_6")
    fixed, swinging = model.body("gripper_tcp").id, model.body("gripper_tcp2").id
    best = None
    for angle in np.linspace(*model.jnt_range[joint.id], 200):
        data.qpos[model.jnt_qposadr[joint.id]] = angle
        mujoco.mj_kinematics(model, data)
        gap = abs(np.linalg.norm(data.xpos[swinging] - data.xpos[fixed]) - width)
        if best is None or gap < best[0]:
            midpoint = (data.xpos[swinging] + data.xpos[fixed]) / 2
            best = (gap, data.xmat[fixed].reshape(3, 3).T @ (midpoint - data.xpos[fixed]))
    return best[1].tolist()


ARMS = {
    "so101": ArmEmbodiment("so101", "so101.xml", MOTORS, "gripper", "gripperframe", (1.0, 0.0, 0.0),
                           (0.0, -60.0, 60.0, 35.0, 0.0, 30.0), "wrist_cam", (0.22, 0.0)),
    "omx": ArmEmbodiment("omx", "mujoco/omx/omx.xml", ("joint1", "joint2", "joint3", "joint4", "joint5"),
                         "gripper_joint_1", "ee", (1.0, 0.0, 0.0), (0.0, -45.0, 30.0, 60.0, 0.0, 30.0), None,
                         (0.20, 0.0), prepare=prepare_omx),
    "koch": ArmEmbodiment("koch", "follower_arm_v1.1.urdf", ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5"),
                          "joint_6", "ee", (0.0, 0.0, -1.0), (0.0, 90.0, 66.0, 114.0, 0.0, 30.0), None,
                          (0.0, -0.20), prepare=prepare_koch),
}
