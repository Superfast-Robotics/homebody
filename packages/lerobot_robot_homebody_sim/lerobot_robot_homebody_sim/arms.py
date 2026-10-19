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
PREPARED = "homebody-4"  # the prepared models' version: a change here rebuilds them


@dataclass(frozen=True)
class ArmEmbodiment:
    model: str                    # a models.SOURCES key
    xml: str                      # the arm's model file in that folder (MJCF, or a URDF to prepare)
    joints: tuple[str, ...]       # the model's arm joints for MOTORS, in kinematic order
    gripper: str                  # the model's gripper joint
    ee_site: str
    approach_axis: tuple[float, float, float]
    home: tuple[float, ...]       # the arm joints (degrees) then the gripper (gripper_units)
    wrist_camera: str | None
    pick_xy: tuple[float, float]  # a comfortable cube position on the table, in the base frame (m)
    gripper_closes_high: bool = False  # the gripper closes toward its actuator range's upper end
    prepare: Callable[[Path, Path], None] | None = None
    motors: tuple[str, ...] = MOTORS      # the follower's names for JOINTS
    gripper_units: tuple[float, float] = (0.0, 100.0)  # the follower's gripper value closed and open

    def xml_path(self):
        source = model_dir(self.model) / self.xml
        if self.prepare is None:
            return source
        prepared = source.with_name(f"{source.stem}.{PREPARED}.xml")
        if not prepared.exists():
            self.prepare(source, prepared)
        return prepared

    def spec(self, prefix=""):
        """The arm's ArmSpec, its keys under PREFIX ("left_" for a bimanual rig's left arm)."""
        closed, opened = self.gripper_units
        return ArmSpec(tuple(prefix + motor for motor in self.motors), prefix + "gripper", str(self.xml_path()),
                       self.ee_site, self.approach_axis, gripper_open=opened, gripper_closed=closed,
                       home=self.home[:-1], model_joints=self.joints)


def jaw_tip(model, data, body, along):
    """The point of BODY's collision geoms furthest along ALONG (world): a jaw's tip."""
    best = None
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        rotation, centre = data.geom_xmat[geom].reshape(3, 3), data.geom_xpos[geom]
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            points = centre + model.mesh_vert[start:start + count] @ rotation.T
        else:  # a primitive: its bounding box's corners
            corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
            points = centre + (corners * model.geom_size[geom]) @ rotation.T
        candidate = points[np.argmax(points @ along)]
        if best is None or candidate @ along > best @ along:
            best = candidate
    return best


def prepare_omx(source, prepared):
    """ROBOTIS's OMX MJCF with a site at its end-effector link (x out between the jaws)."""
    spec = mujoco.MjSpec.from_file(str(source))
    spec.body("end_effector_link").add_site(name="ee", pos=[0, 0, 0])
    prepared.write_text(spec.to_xml())


def prepare_openarm(source, prepared):
    """Enactic's OpenARM v1 MJCF with a site at the finger pads: its hand's tool frame, +z out between the fingers,
    set back to where they close."""
    spec = mujoco.MjSpec.from_file(str(source))
    spec.body("openarm_hand_tcp").add_site(name="ee", pos=[0, 0, -0.02])
    prepared.write_text(spec.to_xml())


def prepare_lekiwi(source, prepared):
    """Ekumen's LeKiwi without its SO-100: the robot's SO-101 is attached at so_arm100_link when the scene is
    built (the arm LeRobot's LeKiwi kits now carry, and the one the other rigs share)."""
    text = source.read_text()
    text = re.sub(r'\s*<model name="so_arm100"[^>]*/>', "", text)
    text = re.sub(r'\s*<attach model="so_arm100"[^>]*/>', "", text)
    prepared.write_text(text)


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
    # LeRobot's openarm_follower: joint_1..joint_7 in degrees and the gripper in degrees, 0 closed to -65 open
    "openarm": ArmEmbodiment("openarm", "openarm.xml", tuple(f"openarm_joint{i}" for i in range(1, 8)),
                             "openarm_finger_joint1", "ee", (0.0, 0.0, 1.0),
                             (0.0, 0.0, 0.0, 90.0, 0.0, 0.0, 0.0, -30.0), None, (0.35, 0.0), prepare=prepare_openarm,
                             motors=tuple(f"joint_{i}" for i in range(1, 8)), gripper_units=(0.0, -65.0)),
}
