"""cuRobo planning server for HomeBody's G1 arms. Runs in the cuRobo environment, not HomeBody's.

    curobo_server.py [--endpoint tcp://127.0.0.1:5560]

ZeroMQ REQ/REP with msgpack + msgpack_numpy (the same wire as homebody.remote.wire). Requests:
  {"endpoint": "ping"}
  {"endpoint": "configure", "data": {"locks": {joint: rad}}}  per-side planners, other joints locked
  {"endpoint": "plan_grasp", "data": {side, start (7), pre (N,4,4), palm (N,4,4), depth (H,W) m with
      NaN where invalid or excluded, intrinsics (3,3), camera (4,4) optical frame in the pelvis}}
  -> {success, index, approach (T,7), descent (T,7), joint_names, status, seconds}
All poses are in the pelvis frame. The world is an ESDF built from that one depth image only.
"""
import argparse
import copy
import io
import json
import tempfile
import time
from pathlib import Path

import msgpack
import msgpack_numpy
import numpy as np
import torch
import yaml
import zmq
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Scene, VoxelGrid
from curobo.types import GoalToolPose, JointState, Pose
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation

ROBOT = Path(__file__).resolve().parents[0]
SIDES = ("left", "right")
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
WORKSPACE_CENTER = (0.45, 0.0, 0.1)  # pelvis frame, metres
WORKSPACE_EXTENT = (1.4, 1.6, 1.6)
VOXEL = 0.01  # occupancy resolution of the depth points (divides WORKSPACE_EXTENT exactly); finger gaps beside a basket wall are 3-16 mm
MAX_FREE_M = 0.5  # signed distance cap
BEHIND_SURFACE_M = 0.05  # space this far BELOW each seen surface counts as occupied: a counter seen from above becomes a slab, while gaps beside walls stay free
PIXEL_STRIDE = 2
START_CLEARANCE_M = 0.03  # occupancy this close to the robot's start spheres is cleared (it starts there)
LEG_LINKS = [f"{side}_{part}_link" for side in SIDES for part in
             ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
# Clearance kept from the depth surface: robot spheres grown by SPHERE_BUFFER_M, and the cost
# active within ACTIVATION_M, so planned paths pass HomeBody's own finger-vs-depth live check.
SPHERE_BUFFER_M = 0.0
ACTIVATION_M = 0.005
MAX_GOALSET = 48
MAX_CANDIDATE_TRIES = 6  # candidates tried (approach then descent) before giving up
IK_SEEDS = 128  # random restarts per IK solve; obstacle-dense scenes (a basket) need many  # grasp candidates planned per request (the client sends its best-ranked)


def pack(message):
    return msgpack.packb(message, default=msgpack_numpy.encode)


def unpack(raw):
    def decode(value):
        if isinstance(value, dict):
            if value.get("__ndarray_class__", value.get(b"__ndarray_class__")):
                return np.load(io.BytesIO(value.get("as_npy", value.get(b"as_npy"))), allow_pickle=False)
            if value.get("nd", value.get(b"nd")) and value.get("kind", value.get(b"kind")) in ("O", b"O"):
                raise ValueError("Refusing an object-dtype array payload")
        return msgpack_numpy.decode(value)
    return msgpack.unpackb(raw, object_hook=decode, raw=False)


def with_hand(config: dict, hand: Path) -> dict:
    """CONFIG (cuRobo's G1 with Dex3) with another hand: HAND is a robot profile's folder (assets/robot/<hands>)
    holding its URDF and curobo.json (tools/build_dex1_hand.py: per-link spheres, finger joints and their open
    value, link pairs that may touch). Dex3's hand links and joints are dropped, and so is anything the
    profile's URDF lacks."""
    import xml.etree.ElementTree as ET

    profile = json.loads((hand / "curobo.json").read_text())
    tree = ET.parse(hand / "g1_29dof_with_hand.urdf").getroot()
    if tree.find("link[@name='base_link']") is None:  # cuRobo's virtual base attaches to pelvis through base_link
        tree.insert(0, ET.Element("link", name="base_link"))
        joint = ET.SubElement(tree, "joint", name="base_fixed", type="fixed")
        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(joint, "parent", link="base_link")
        ET.SubElement(joint, "child", link="pelvis")
    urdf = Path(tempfile.gettempdir()) / f"homebody_curobo_{hand.name}.urdf"
    urdf.write_text(ET.tostring(tree, encoding="unicode"))
    links = {link.get("name") for link in tree.findall("link")}
    joint_names = {joint.get("name") for joint in tree.findall("joint")}
    dex3 = ("_hand_thumb_", "_hand_index_", "_hand_middle_")

    def keep(name):
        return name in links and not any(part in name for part in dex3)

    k = config["kinematics"]
    k["urdf_path"], k["asset_root_path"], k["load_meshes"] = str(urdf), str(hand), False
    k["collision_spheres"] = {**{n: v for n, v in k["collision_spheres"].items() if keep(n)}, **profile["links"]}
    k["collision_link_names"] = [n for n in k["collision_link_names"] if keep(n)] + [
        n for n in profile["links"] if n not in k["collision_link_names"]]
    k["mesh_link_names"] = [n for n in k.get("mesh_link_names", []) if keep(n)]
    ignore = {n: [m for m in v if keep(m)] for n, v in k.get("self_collision_ignore", {}).items() if keep(n)}
    for name, others in profile["ignore"].items():
        ignore[name] = sorted(set(ignore.get(name, [])) | set(others))
    k["self_collision_ignore"] = ignore
    cspace = k["cspace"]
    fingers = [j for side in SIDES for j in profile["joints"][side]]
    kept = [i for i, n in enumerate(cspace["joint_names"]) if n in joint_names and not any(part in n for part in dex3)
            or n.startswith("base_j_")]
    count = len(cspace["joint_names"])
    for key, value in list(cspace.items()):
        if isinstance(value, list) and len(value) == count:
            extra = fingers if key == "joint_names" else [profile["open"] if key == "default_joint_position" else
                                                          value[kept[-1]]] * len(fingers)
            cspace[key] = [value[i] for i in kept] + extra
    k["tool_frames"] = [f"{side}_hand_palm_link" for side in SIDES]
    return config


def robot_config(curobo_root: Path, side: str, locks: dict, hand: Path | None = None) -> dict:
    """cuRobo's bundled G1 (with Dex3, or the HAND profile's) planning only SIDE's arm to its palm, the rest locked."""
    path = curobo_root / "curobo/content/configs/robot/unitree_g1.yml"
    config = copy.deepcopy(yaml.safe_load(path.read_text()))
    if hand is not None:
        config = with_hand(config, hand)
    kinematics = config["kinematics"]
    moving = {f"{side}_{name}_joint" for name in ARM_JOINTS}
    kinematics["tool_frames"] = [f"{side}_hand_palm_link"]
    kinematics["collision_sphere_buffer"] = SPHERE_BUFFER_M
    defaults = dict(zip(kinematics["cspace"]["joint_names"], kinematics["cspace"]["default_joint_position"]))
    # Locked joints default to the config's own collision-free posture; the virtual base stays
    # at zero so the planning frame is the pelvis.
    kinematics["lock_joints"] = {
        name: float(locks.get(name, 0.0 if name.startswith("base_j_") else defaults[name]))
        for name in kinematics["cspace"]["joint_names"] if name not in moving}
    return config


def pose_tensor(matrices):
    matrices = np.asarray(matrices, dtype=np.float64).reshape(-1, 4, 4)
    position = torch.tensor(matrices[:, :3, 3], dtype=torch.float32, device="cuda")
    xyzw = Rotation.from_matrix(matrices[:, :3, :3]).as_quat()
    quaternion = torch.tensor(xyzw[:, [3, 0, 1, 2]], dtype=torch.float32, device="cuda")
    return Pose(position=position, quaternion=quaternion)


class Planners:
    def __init__(self, curobo_root: Path, hand: Path | None = None):
        self.root, self.planners, self.locks, self.dump, self.hand = curobo_root, {}, None, None, hand

    def configure(self, locks):
        self.locks = dict(locks)
        self.planners = {}
        for side in SIDES:
            config = MotionPlannerCfg.create(robot=robot_config(self.root, side, self.locks, self.hand),
                                             scene_model=self.world(None),
                                             collision_cache={"voxel": 1}, max_goalset=MAX_GOALSET, num_ik_seeds=IK_SEEDS,
                                             optimizer_collision_activation_distance=ACTIVATION_M)
            planner = MotionPlanner(config)
            # The legs are locked straight here but bent on the robot (which is controlled by its
            # walking policy); their spheres would sit in the floor, so they take no collisions.
            planner.disable_link_collision(LEG_LINKS)
            planner.warmup(enable_graph=True, num_warmup_iterations=2)
            self.planners[side] = planner
        return {"joint_names": {side: list(p.joint_names) for side, p in self.planners.items()}}

    def world(self, request, planner=None):
        """The request's depth as a signed distance grid in the pelvis frame: voxels from each seen
        surface down to BEHIND_SURFACE_M below it are occupied (one view sees only tops, e.g. a
        counter and not the overhang under it); Euclidean distance transforms outside and inside give
        the SDF, positive in free space (cuRobo's convention)."""
        center, extent = np.asarray(WORKSPACE_CENTER), np.asarray(WORKSPACE_EXTENT)
        shape = np.round(extent / VOXEL).astype(int)
        occupied = np.zeros(shape, dtype=bool)
        if request is not None:
            depth, K = np.asarray(request["depth"], dtype=np.float64), np.asarray(request["intrinsics"])
            valid = np.isfinite(depth) & (depth > 0.05)
            v, u = np.nonzero(valid)
            keep = (v % PIXEL_STRIDE == 0) & (u % PIXEL_STRIDE == 0)
            v, u = v[keep], u[keep]
            z = depth[v, u]
            camera = np.asarray(request["camera"], dtype=np.float64)
            surface = np.c_[(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z] @ camera[:3, :3].T + camera[:3, 3]
            below = np.arange(0.0, BEHIND_SURFACE_M + 1e-9, VOXEL / 2)
            points = (surface[:, None, :] - below[None, :, None] * np.array([0.0, 0.0, 1.0])).reshape(-1, 3)
            cells = np.floor((points - (center - extent / 2)) / VOXEL).astype(int)
            inside = np.all((cells >= 0) & (cells < shape), axis=1)
            occupied[tuple(cells[inside].T)] = True
        if planner is not None and request is not None and occupied.any():
            occupied &= ~self.near_start(planner, request, center - extent / 2, shape)
        if occupied.any():
            sdf = (distance_transform_edt(~occupied) - distance_transform_edt(occupied)) * VOXEL
        else:
            sdf = np.full(shape, MAX_FREE_M)
        sdf = np.clip(sdf - VOXEL / 2, -MAX_FREE_M, MAX_FREE_M).astype(np.float32)
        grid = VoxelGrid(name="depth", pose=[*center.tolist(), 1, 0, 0, 0], dims=extent.tolist(), voxel_size=VOXEL,
                         feature_tensor=torch.tensor(sdf, device="cuda", dtype=torch.float16).reshape(-1, 1),
                         feature_dtype=torch.float16)
        return Scene(voxel=[grid])

    def near_start(self, planner, request, low, shape):
        """Voxels within START_CLEARANCE_M of the robot's spheres at the request's start state: what
        the resting body already touches (as HomeBody's own start exemption)."""
        state = self.state(planner, request["start"], request["side"])
        spheres = planner.compute_kinematics(state).robot_spheres.reshape(-1, 4).detach().cpu().numpy()
        spheres = spheres[spheres[:, 3] > 0]
        axes = [low[i] + VOXEL * (np.arange(shape[i]) + 0.5) for i in range(3)]
        mask = np.zeros(shape, dtype=bool)
        for x, y, z, radius in spheres:
            reach = radius + START_CLEARANCE_M
            box = [slice(max(0, int((c - reach - low[i]) / VOXEL)), min(shape[i], int((c + reach - low[i]) / VOXEL) + 1))
                   for i, c in enumerate((x, y, z))]
            gx, gy, gz = np.meshgrid(axes[0][box[0]], axes[1][box[1]], axes[2][box[2]], indexing="ij")
            mask[tuple(box)] |= (gx - x) ** 2 + (gy - y) ** 2 + (gz - z) ** 2 <= reach ** 2
        return mask

    def arm_rows(self, planner, state, side):
        """SIDE's seven arm joints per row of STATE, indexed by the state's own joint names (an
        interpolated plan reports every joint, locked ones included)."""
        names = list(state.joint_names) if state.joint_names is not None else list(planner.joint_names)
        index = [names.index(f"{side}_{name}_joint") for name in ARM_JOINTS]
        return state.position.reshape(-1, len(names))[:, index].cpu().numpy()

    def state(self, planner, start, side):
        names = list(planner.joint_names)
        full = torch.zeros((1, len(names)), dtype=torch.float32, device="cuda")
        for value, name in zip(start, ARM_JOINTS):
            full[0, names.index(f"{side}_{name}_joint")] = float(value)
        return JointState.from_position(full, joint_names=names)

    def goal(self, planner, matrices):
        poses = pose_tensor(np.asarray(matrices)[:MAX_GOALSET])
        return GoalToolPose.from_poses({planner.tool_frames[0]: poses}, num_goalset=len(poses.position))

    def choose(self, planner, start, goals, side):
        """Index of the goal-set pose collision-free IK reaches (batched, one returned seed), or None."""
        ik = planner.ik_solver.solve_pose(self.goal(planner, goals), return_seeds=1,
                                          current_state=self.state(planner, start, side))
        if not bool(ik.success.any()):
            return None
        return int(ik.goalset_index.reshape(-1)[0])

    def plan(self, planner, start, goal, side):
        """Joint rows of a collision-free plan from START to the single pose GOAL, or None."""
        result = planner.plan_pose(self.goal(planner, goal[None]), self.state(planner, start, side))
        if result is None or not bool(result.success.any()):
            return None
        return self.arm_rows(planner, result.get_interpolated_plan(), side)

    def plan_grasp(self, request):
        started = time.monotonic()
        side = request["side"]
        if self.dump is not None:
            np.savez_compressed(self.dump, **{k: np.asarray(v) for k, v in request.items()})
        planner = self.planners[side]
        planner.update_world(self.world(request, planner))
        pre, palm = np.asarray(request["pre"])[:MAX_GOALSET], np.asarray(request["palm"])[:MAX_GOALSET]
        reply = {"success": False, "index": -1, "approach": np.zeros((0, 7)), "descent": np.zeros((0, 7))}
        remaining, failures = list(range(len(pre))), []
        for _ in range(MAX_CANDIDATE_TRIES):
            if not remaining:
                break
            choice = self.choose(planner, request["start"], pre[remaining], side)
            if choice is None:
                failures.append("no collision-free IK for the remaining pre-grasp poses")
                break
            index = remaining.pop(choice)
            approach = self.plan(planner, request["start"], pre[index], side)
            if approach is None:
                failures.append(f"{index}: no approach plan")
                continue
            descent = self.plan(planner, approach[-1], palm[index], side)
            if descent is None:
                failures.append(f"{index}: no descent plan")
                continue
            reply.update(success=True, index=index, approach=approach, descent=descent,
                         status="ok" + (f" after {len(failures)} rejected" if failures else ""))
            return self.finish(reply, started)
        reply["status"] = "; ".join(failures) or "no candidates"
        return self.finish(reply, started)

    @staticmethod
    def finish(reply, started):
        reply["seconds"] = time.monotonic() - started
        return reply


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--curobo-root", type=Path, required=True)
    parser.add_argument("--dump", type=Path, help="Save each plan_grasp request here (debugging)")
    parser.add_argument("--hand", type=Path, help="A robot profile folder with curobo.json (e.g. assets/robot/dex1) "
                                                  "to plan with in place of the bundled Dex3 hands")
    args = parser.parse_args()
    planners = Planners(args.curobo_root, args.hand)
    planners.dump = args.dump
    socket = zmq.Context().socket(zmq.REP)
    socket.bind(args.endpoint)
    print(f"cuRobo planning server on {args.endpoint}", flush=True)
    while True:
        request = unpack(socket.recv())
        try:
            name = request["endpoint"]
            if name == "ping":
                reply = {"ok": True, "configured": bool(planners.planners)}
            elif name == "configure":
                print(f"configure locks {request['data']['locks']}", flush=True)
                reply = planners.configure(request["data"]["locks"])
            elif name == "plan_grasp":
                reply = planners.plan_grasp(request["data"])
                print(f"plan_grasp {request['data']['side']}: success {reply['success']} index {reply['index']} "
                      f"status {reply['status']} in {reply['seconds']:.2f}s", flush=True)
            else:
                reply = {"error": f"unknown endpoint {name}"}
        except Exception as error:  # noqa: BLE001 -- report every failure to the client
            import traceback
            traceback.print_exc()
            reply = {"error": f"{type(error).__name__}: {error}"}
        socket.send(pack(reply))


if __name__ == "__main__":
    main()
