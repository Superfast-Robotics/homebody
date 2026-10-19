"""Assemble the robot model and a scene package into one MuJoCo model."""
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from homebody.helpers.kinematics import camera_mount
from homebody.session.config import DEFAULT_SCENE

SPAWN_HEIGHT = 0.76
# Wrist cameras above the wrist yaw link, aimed halfway between along the hand (+x of the link) and the
# open hand's jaw (calibration palm_to_jaw): both what the grasp closes on and the scene beyond it in view.
WRIST_CAMERA_POS = (0.02, 0.0, 0.06)
WRIST_CAMERA_XYAXES = {"left": (-0.2146, -0.9754, 0.0509, 0.2307, 0.0, 0.9730),
                       "right": (0.2146, -0.9754, -0.0509, 0.2307, 0.0, 0.9730)}
WRIST_CAMERA_FOVY = 70.0


def numbers(values):
    return " ".join(str(float(value)) for value in values)


def load(root: Path, settings, scene: Path | None = DEFAULT_SCENE):
    """Compile the robot into SCENE, a package directory relative to ROOT or absolute, or
    alone on a flat floor when SCENE is None."""
    profile = settings.robot.directory(root)
    robot = profile / "g1_grasp"
    xml = ET.parse(robot / "g1_closd_grasp_deploy.xml").getroot()
    xml.find("compiler").set("meshdir", str(robot / "meshes"))
    for keyframe in xml.findall("keyframe"):
        xml.remove(keyframe)
    option = xml.find("option")
    option.attrib.update(timestep=str(settings.physics.timestep), integrator=settings.physics.integrator,
                         cone=settings.physics.cone, impratio=str(settings.physics.impratio),
                         noslip_iterations=str(settings.physics.noslip_iterations),
                         solver=settings.physics.solver)
    for world in xml.findall("worldbody"):
        for child in list(world):
            if (child.tag == "body" and child.get("name") != "pelvis") or (
                    scene is not None and child.tag == "geom" and child.get("type") == "plane"):
                world.remove(child)
    spawn = [0., 0., 0.]
    if scene is not None:
        directory = root / scene
        package = ET.parse(directory / "scene.xml").getroot()
        for asset in package.find("asset"):
            if "file" in asset.attrib:
                asset.set("file", str(directory / asset.get("file")))
        for child in package:
            xml.append(child)
        spawn = json.loads((directory / "task_scene.json").read_text())["robot_start"]
    torso = next(b for b in xml.iter("body") if b.get("name") == "torso_link")
    camera = camera_mount(profile / "g1_29dof_with_hand.urdf")
    rotation = camera[:3, :3] @ np.diag([1, -1, -1])
    quat = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
    c = settings.camera
    fovy = math.degrees(2 * math.atan(math.tan(math.radians(c.horizontal_fov) / 2) * c.height / c.width))
    ET.SubElement(torso, "camera", name="ego", pos=numbers(camera[:3, 3]),
                  quat=numbers(quat), fovy=str(fovy))
    # a hand whose calibration places its own wrist cameras (a gripper body in the way) uses them
    mounts = json.loads((profile / "calibration.json").read_text()).get("wrist_camera", {})
    for side in ("left", "right"):
        wrist = next(b for b in xml.iter("body") if b.get("name") == f"{side}_wrist_yaw_link")
        mount = mounts.get(side, {"pos": WRIST_CAMERA_POS, "xyaxes": WRIST_CAMERA_XYAXES[side]})
        ET.SubElement(wrist, "camera", name=f"{side}_wrist", pos=numbers(mount["pos"]),
                      xyaxes=numbers(mount["xyaxes"]), fovy=str(WRIST_CAMERA_FOVY))
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    data = mujoco.MjData(model)
    from holosoma_inference.g1_control.config import amo_profile
    profile = amo_profile()
    for name, value in zip(profile.robot.joint_names, profile.robot.idle_positions):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    data.qpos[:2] = spawn[:2]
    data.qpos[3:7] = [math.cos(spawn[2] / 2), 0, 0, math.sin(spawn[2] / 2)]
    mujoco.mj_forward(model, data)
    floor = 0.
    if scene is not None:
        start, geomid = np.array([*spawn[:2], 2.]), np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(model, data, start, np.array([0., 0., -1.]),
                                 np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8), 1, -1, geomid)
        if distance < 0 or not model.geom(int(geomid[0])).name.startswith("floor__"):
            raise ValueError("Robot spawn must be on the reconstructed floor")
        floor = start[2] - distance
    data.qpos[2] = floor + SPAWN_HEIGHT
    mujoco.mj_forward(model, data)
    return model, data
