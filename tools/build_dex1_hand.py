"""build_dex1_hand.py HANDOFF_ROBOT [OUT]: HomeBody's G1 with Unitree's Dex1-1 parallel grippers in place of its Dex3
hands, as the robot profile folder assets/robot/dex1 (selected by `[robot] hands = "dex1"`).

HANDOFF_ROBOT is the warehouse handoff's robot/ folder (Unitree's official g1_29dof_mode_15_with_dex1_1, its
MuJoCo conversion and meshes). The folder mirrors assets/robot: g1_grasp/g1_closd_grasp_deploy.xml is HomeBody's
model with each wrist's Dex3 subtree and hand actuators replaced by the Dex1 base and its two slide fingers
(named as HomeBody's hands are: <side>_hand_palm_link, <side>_hand_finger_N_link/_joint, collision geoms
<side>_hand_*_col), one position servo per finger driven to the same target, so each jaw reports its own effort;
g1_29dof_with_hand.urdf is the same swap in the URDF the arm kinematics read; calibration.json keeps the arm's
entries and gives the gripper's: its joints, open and closed positions, servo gains, finger groups (finger 1
opposes finger 2) and jaw samples measured from the fingers' inner collision faces. The G1 body meshes are
symlinks to assets/robot/g1_grasp/meshes; the Dex1 meshes are copied (unitree_ros, BSD-3-Clause)."""
import argparse
import copy
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SIDES = ("left", "right")
OPEN, CLOSED = 0.0245, -0.02     # finger slide (m): fully open, and the range end (a 6 mm gap)
SAMPLES = 18
# Wrist camera on top of the gripper body (wrist-yaw frame, m): its top is 3.1 cm up, so the view of the pads clears it
WRIST_CAMERA_POS = (0.08, 0.0, 0.065)
HAND_RATE = 0.1  # finger closing speed (m/s): fully open to closed in about half a second
SERVO = {"kp": 400.0, "kd": 8.0, "force": 20.0}  # the handoff's gripper servo: N/m, N s/m, N
FINGER = {"contype": "1", "conaffinity": "2", "friction": "1.2 0.005 0.0001"}  # HomeBody's hand collision bits


def dex1_bodies(handoff, side):
    """The handoff's <side>_dex1_base_link subtree, renamed to HomeBody's hand names."""
    root = ET.parse(handoff / "g1_dex1.xml").getroot()
    base = copy.deepcopy(next(b for b in root.iter("body") if b.get("name") == f"{side}_dex1_base_link"))
    for child in list(base):
        if child.tag in ("site", "camera"):
            base.remove(child)
    base.set("name", f"{side}_hand_palm_link")
    collision = [g for g in base.findall("geom") if g.get("contype") != "0"]
    collision[0].attrib.update({"name": f"{side}_hand_palm_col", **FINGER})
    for index in (1, 2):
        finger = base.find(f"body[@name='{side}_dex1_finger_link_{index}']")
        finger.set("name", f"{side}_hand_finger_{index}_link")
        joint = finger.find("joint")
        joint.attrib.update({"name": f"{side}_hand_finger_{index}_joint", "armature": "0.01", "damping": "0"})
        geom = next(g for g in finger.findall("geom") if g.get("contype") != "0")
        geom.attrib.update({"name": f"{side}_hand_finger_{index}_col", **FINGER})
    return base


def build_model(handoff, out):
    tree = ET.parse(REPO / "assets/robot/g1_grasp/g1_closd_grasp_deploy.xml")
    root = tree.getroot()
    for side in SIDES:
        wrist = next(b for b in root.iter("body") if b.get("name") == f"{side}_wrist_yaw_link")
        wrist.remove(wrist.find(f"body[@name='{side}_hand_palm_link']"))
        wrist.append(dex1_bodies(handoff, side))
    meshes = {m.get("name"): m for m in root.iter("mesh")}
    asset = next(a for a in root.findall("asset") if a.find("mesh") is not None)
    for name in [n for n in meshes if "_hand_" in n or "rubber_hand" in n]:
        for parent in root.findall("asset"):
            if meshes[name] in list(parent):
                parent.remove(meshes[name])
    handoff_meshes = {m.get("name"): m.get("file") for m in ET.parse(handoff / "g1_dex1.xml").getroot().iter("mesh")}
    for name in ("Dex1_base_link", "Dex1_finger_link_1", "Dex1_finger_link_2", "dex1_col_1", "dex1_col_2"):
        ET.SubElement(asset, "mesh", name=name, file=Path(handoff_meshes[name]).name)
    actuators = root.find("actuator")
    for actuator in [a for a in actuators if "_hand_" in (a.get("joint") or "")]:
        actuators.remove(actuator)
    for side in SIDES:
        for index in (1, 2):
            joint = f"{side}_hand_finger_{index}_joint"
            ET.SubElement(actuators, "position", name=joint, joint=joint, kp=str(SERVO["kp"]), kv=str(SERVO["kd"]),
                          ctrlrange=f"{CLOSED} {OPEN}", forcerange=f"-{SERVO['force']} {SERVO['force']}")
    mesh_dir = out / "g1_grasp/meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted((REPO / "assets/robot/g1_grasp/meshes").iterdir()):
        target = mesh_dir / source.name
        if not ("_hand_" in source.name or "rubber_hand" in source.name) and not target.exists():
            target.symlink_to(Path("../../../g1_grasp/meshes") / source.name)
    for name in ("Dex1_base_link", "Dex1_finger_link_1", "Dex1_finger_link_2", "dex1_col_1", "dex1_col_2"):
        shutil.copy(handoff / handoff_meshes[name] if (handoff / handoff_meshes[name]).exists()
                    else handoff / "meshes" / Path(handoff_meshes[name]).name, mesh_dir / Path(handoff_meshes[name]).name)
    # the collision helper reads each hand link's mesh by its link name
    for side in SIDES:
        for link, mesh in ((f"{side}_hand_palm_link", "Dex1_base_link.STL"), (f"{side}_hand_finger_1_link", "dex1_col_1.stl"),
                           (f"{side}_hand_finger_2_link", "dex1_col_2.stl")):
            (mesh_dir / f"{link}.STL").unlink(missing_ok=True)
            (mesh_dir / f"{link}.STL").symlink_to(mesh)
    for keyframe in root.findall("keyframe"):  # Dex3's joint count; HomeBody's scene drops keyframes anyway
        root.remove(keyframe)
    tree.write(out / "g1_grasp/g1_closd_grasp_deploy.xml", encoding="unicode")
    return out / "g1_grasp/g1_closd_grasp_deploy.xml"


def build_urdf(handoff, out):
    """HomeBody's URDF with each hand's links and joints replaced by the Dex1 base and fingers (same names as the
    model)."""
    tree = ET.parse(REPO / "assets/robot/g1_29dof_with_hand.urdf")
    root = tree.getroot()
    hand = [e for e in root if e.tag in ("link", "joint") and "_hand_" in e.get("name", "")]
    for element in hand:
        root.remove(element)
    source = ET.parse(handoff / "g1_dex1_portable.urdf").getroot()
    renames = {}
    for side in SIDES:
        renames[f"{side}_dex1_base_link"] = f"{side}_hand_palm_link"
        for index in (1, 2):
            renames[f"{side}_dex1_finger_link_{index}"] = f"{side}_hand_finger_{index}_link"
            renames[f"{side}_dex1_finger_joint_{index}"] = f"{side}_hand_finger_{index}_joint"
    for element in source:
        name = element.get("name", "")
        if "dex1" not in name and not (element.tag == "joint" and "dex1" in ET.tostring(element, encoding="unicode")):
            continue
        element = copy.deepcopy(element)
        element.set("name", renames.get(name, name.replace("dex1_base_joint", "hand_palm_joint")))
        for ref in element.iter():
            if ref.get("link") in renames:
                ref.set("link", renames[ref.get("link")])
            if ref.tag == "mesh":
                ref.set("filename", "g1_grasp/meshes/" + Path(ref.get("filename")).name)
        root.append(element)
    tree.write(out / "g1_29dof_with_hand.urdf", encoding="unicode")


def jaw_samples(model_path):
    """Jaw samples from the fingers' inner collision faces: at each closure the gap between the faces (the
    aperture) and the jaw frame in the palm (x across the gap, z out of the gripper, at the pads' centre)."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    samples = {}
    for side in SIDES:
        palm = model.body(f"{side}_hand_palm_link").id
        joints = [model.joint(f"{side}_hand_finger_{i}_joint") for i in (1, 2)]
        bodies = [model.body(f"{side}_hand_finger_{i}_link").id for i in (1, 2)]
        out = []
        for closure in np.linspace(0, 1, SAMPLES):
            q = OPEN + (CLOSED - OPEN) * closure
            for joint in joints:
                data.qpos[model.jnt_qposadr[joint.id]] = q
            mujoco.mj_forward(model, data)
            rotation, origin = data.xmat[palm].reshape(3, 3), data.xpos[palm]
            faces = []
            for body in bodies:
                points = np.vstack([data.geom_xpos[g] + model.mesh_vert[model.mesh_vertadr[model.geom_dataid[g]]:
                                    model.mesh_vertadr[model.geom_dataid[g]] + model.mesh_vertnum[model.geom_dataid[g]]]
                                    @ data.geom_xmat[g].reshape(3, 3).T
                                    for g in range(model.ngeom) if model.geom_bodyid[g] == body
                                    and model.geom_contype[g] and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH])
                faces.append((points - origin) @ rotation)
            inner = faces[0][:, 1].max(), faces[1][:, 1].min()  # finger 1 on -y, finger 2 on +y
            pads = np.vstack([faces[0][faces[0][:, 1] > inner[0] - 0.003], faces[1][faces[1][:, 1] < inner[1] + 0.003]])
            centre = np.array([(pads[:, 0].min() + pads[:, 0].max()) / 2, (inner[0] + inner[1]) / 2, 0.0])
            frame = np.eye(4)
            frame[:3, :3] = np.column_stack(([0, 1, 0], [0, 0, 1], [1, 0, 0]))  # closing, across, approach
            frame[:3, 3] = centre
            gap = float(inner[1] - inner[0])
            out.append({"closure": float(closure), "aperture": gap, "palm_to_jaw": frame.tolist(),
                        "inner_contact": {"valid": gap > 0, "mean_surface_axis_gap": gap, "palm_to_contact_frame": frame.tolist()}})
        samples[side] = out
    return samples


def wrist_camera(model_path, samples):
    """The wrist cameras: at WRIST_CAMERA_POS, looking halfway between straight out of the wrist and the open
    jaw's centre, horizontal image rows; as MuJoCo camera xyaxes in each wrist-yaw frame."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    cameras = {}
    for side in SIDES:
        mount = model.body_pos[model.body(f"{side}_hand_palm_link").id]  # palm in the wrist-yaw frame, unrotated
        jaw = mount + np.array(samples[side][0]["palm_to_jaw"])[:3, 3] - np.array(WRIST_CAMERA_POS)
        look = np.array([1.0, 0.0, 0.0]) + jaw / np.linalg.norm(jaw)
        look /= np.linalg.norm(look)
        x = np.cross(look, [0.0, 0.0, 1.0])
        x /= np.linalg.norm(x)
        cameras[side] = {"pos": list(WRIST_CAMERA_POS), "xyaxes": np.r_[x, np.cross(-look, x)].round(4).tolist()}
    return cameras


def build_calibration(model_path, out):
    base = json.loads((REPO / "assets/robot/calibration.json").read_text())
    calibration = {key: base[key] for key in ("arm_limits", "arm_effort", "gains", "limit_margin")}
    calibration.update({
        "hand_joints": {side: [f"{side}_hand_finger_{i}_joint" for i in (1, 2)] for side in SIDES},
        "hand_open": {side: [OPEN, OPEN] for side in SIDES},
        "hand_closed": {side: [CLOSED - OPEN, CLOSED - OPEN] for side in SIDES},  # the closing ray from open
        # closure 1 stops short of the range end so the squeeze (closure up to 1.12) still fits in it
        "hand_close_fraction": 0.88, "thumb_close_fraction": 1.0,
        "hand_rate": HAND_RATE,
        "finger_groups": {"thumb": [0], "index": [1], "middle": [1]},  # the two jaws oppose each other
        "hand_servo": SERVO,
        "jaw_samples": (samples := jaw_samples(model_path)),
        "wrist_camera": wrist_camera(model_path, samples),
        "provenance": "Arm entries of HomeBody's G1 calibration; gripper of Unitree's Dex1-1 (unitree_ros "
                      "g1_29dof_mode_15_with_dex1_1), servo from the warehouse handoff's MuJoCo conversion.",
        "jaw_provenance": "Jaw samples from the Dex1 fingers' inner collision faces across their slide travel.",
    })
    (out / "calibration.json").write_text(json.dumps(calibration, indent=1) + "\n")


def curobo_spheres(model_path, out):
    """cuRobo's view of the gripper (tools/curobo_server.py --hand): spheres covering each hand link's collision
    geometry (its box in the link frame cut into cells half as wide as its thinnest side, one sphere per cell
    reaching the cell's corners, so the box is covered and slightly inflated), the
    finger joints with their open value, and the link pairs that may touch."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    links, joints = {}, {}
    for side in SIDES:
        names = [f"{side}_hand_palm_link"] + [f"{side}_hand_finger_{i}_link" for i in (1, 2)]
        for name in names:
            body = model.body(name).id
            points = []
            for g in range(model.ngeom):
                if model.geom_bodyid[g] != body or not model.geom_contype[g] or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                mesh = model.geom_dataid[g]
                vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
                rotation = np.zeros(9)
                mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
                points.append(vertices @ rotation.reshape(3, 3).T + model.geom_pos[g])
            low, high = np.vstack(points).min(axis=0), np.vstack(points).max(axis=0)
            step = float(np.clip((high - low).min() / 2, 0.006, 0.012))  # cells half as wide as the thinnest side, at most 12 mm
            counts = np.maximum(1, np.ceil((high - low) / step)).astype(int)
            cell = (high - low) / counts
            axes = [lo + cell_i * (np.arange(n) + 0.5) for lo, cell_i, n in zip(low, cell, counts)]
            grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            radius = float(np.linalg.norm(cell) / 2)  # each sphere covers its whole cell, corners included
            links[name] = [{"center": np.round(c, 4).tolist(), "radius": round(radius, 4)} for c in grid]
        joints[side] = [f"{side}_hand_finger_{i}_joint" for i in (1, 2)]
    ignore = {}
    for side in SIDES:
        hand = [f"{side}_hand_palm_link", f"{side}_hand_finger_1_link", f"{side}_hand_finger_2_link"]
        wrist = [f"{side}_wrist_yaw_link", f"{side}_wrist_pitch_link", f"{side}_wrist_roll_link"]
        for link in hand:
            ignore[link] = [other for other in hand + wrist if other != link]
    (out / "curobo.json").write_text(json.dumps({"links": links, "joints": joints, "open": OPEN, "ignore": ignore},
                                                indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handoff_robot", type=Path)
    parser.add_argument("out", type=Path, nargs="?", default=REPO / "assets/robot/dex1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = build_model(args.handoff_robot, args.out)
    build_urdf(args.handoff_robot, args.out)
    build_calibration(model, args.out)
    curobo_spheres(model, args.out)
    compiled = mujoco.MjModel.from_xml_path(str(model))
    print(f"built {args.out}: {compiled.nu} actuators, {compiled.njnt} joints; jaw apertures "
          f"{json.loads((args.out / 'calibration.json').read_text())['jaw_samples']['right'][0]['aperture']:.4f} to "
          f"{json.loads((args.out / 'calibration.json').read_text())['jaw_samples']['right'][-1]['aperture']:.4f} m")


if __name__ == "__main__":
    main()
