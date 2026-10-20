"""Client for the cuRobo planning server (tools/curobo_server.py): collision-free goal-set
planning of an arm to one of several pre-poses, then down to the matching pose, against an ESDF
of the current depth image only (no simulator state).

Poses go to the server in the pelvis frame; the robot's own pixels and the EXCLUDE labels (the
target, or the carried object) are removed from the depth so the planner may reach them.
"""
import fcntl
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from homebody.remote.g1 import ARM_JOINT_SUFFIXES, G1Kinematics
from homebody.remote.wire import PolicyClient, PolicyWireError

ROOT = Path(__file__).resolve().parents[3]
URDF = ROOT / "assets/robot/g1_29dof_with_hand.urdf"
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
TARGET_MASK_GROWTH_PX = 4
ADMISSION_POLL = 0.2  # s between tries for a free planning server
PING_SECONDS = 15.0  # an admitted server is idle, so it answers a ping at once; silence means it is dead


class PlannerUnavailable(RuntimeError):
    """The planning server cannot be reached or configured."""


class CuroboPlanner:
    """SETTINGS is the full Settings (its `curobo` group, and the arms' rest pose)."""

    def __init__(self, settings):
        self.all_settings, self.settings = settings, settings.curobo
        profile = settings.robot.directory(ROOT)  # the robot's hands (settings.robot.hands): its URDF and calibration
        self.kinematics = G1Kinematics(profile / "g1_29dof_with_hand.urdf")
        calibration = json.loads((profile / "calibration.json").read_text())
        self.hand_open = {side: dict(zip(calibration["hand_joints"][side],
                                         calibration.get("hand_open", {}).get(side, [0.0] * len(calibration["hand_joints"][side]))))
                          for side in ("left", "right")}
        self.client = self.address = None  # the server this planner holds while it plans (admitted)

    def locks(self, frame, side):
        """The measured posture the non-planning joints are fixed at: the waist and the other arm where they are,
        the other hand as measured, and SIDE's hand open (it plans the approach with the jaws open, as pick opens
        before it moves). SIDE's arm starts where it is (the request's start)."""
        locks = {name: float(value) for name, value in zip(self.kinematics.names, frame.joints) if name in WAIST_JOINTS}
        for arm in ("left", "right"):
            names = [f"{arm}_{suffix}_joint" for suffix in ARM_JOINT_SUFFIXES]
            if arm != side:
                locks.update(zip(names, map(float, frame.arm_joints(arm))))
            else:
                locks.update(zip(names, map(float, self.all_settings.placement.rest_arm(arm))))
            hand = self.hand_open[arm]
            if arm == side:
                locks.update(hand)
            else:  # HandState joints count from the open pose
                measured = np.asarray(frame.hands[arm].joints, dtype=float)
                locks.update({name: opened + float(q) for (name, opened), q in zip(hand.items(), measured)})
        return locks

    def ensure_configured(self, frame, side):
        """Build the admitted server's per-arm planners once (a plan whose posture differs rebuilds them,
        tools/curobo_server.py). A server that does not answer a short ping is dead (it exits on a CUDA error and
        its launcher restarts it): PlannerUnavailable at once, not after the whole timeout."""
        probe = PolicyClient(self.address, PING_SECONDS)
        try:
            status = probe.call("ping")
            if not status.get("configured"):
                self.client.call("configure", {"locks": self.locks(frame, side)})
        except PolicyWireError as error:
            raise PlannerUnavailable(str(error)) from error
        finally:
            probe.close()

    @contextmanager
    def admitted(self):
        """Hold one of the host's planning servers for one plan: a free one (this process's own first), else wait
        for whichever frees first, up to the timeout. Each server serves one request at a time (a REP socket), so
        sessions queue here, each for its own turn, instead of piling onto one server while others idle."""
        with admission(self.settings) as address:
            self.client, self.address = PolicyClient(address, self.settings.timeout), address
            try:
                yield self.client
            finally:
                self.client.close()
                self.client = self.address = None

    def world_pelvis(self, frame):
        return frame.world_torso @ np.linalg.inv(self.kinematics.torso_in_pelvis(frame.joints))

    def plan(self, frame, side, pre_world, pose_world, exclude_labels=()):
        """The server's reply for the arm's goal set of (pre, pose) world poses, best first."""
        pelvis_world = np.linalg.inv(self.world_pelvis(frame))
        depth = np.array(frame.depth, dtype=np.float32)
        excluded = frame.labels == -1
        for label in exclude_labels:
            # Grow the target's mask: its silhouette pixels read as background and would leave a
            # shell of phantom obstacle exactly where the fingers close.
            excluded |= binary_dilation(frame.labels == label, iterations=TARGET_MASK_GROWTH_PX)
        depth[excluded] = np.nan
        count = self.settings.max_candidates
        request = {"side": side, "start": np.asarray(frame.arm_joints(side), dtype=np.float64),
                   "pre": pelvis_world @ np.asarray(pre_world)[:count],
                   "palm": pelvis_world @ np.asarray(pose_world)[:count],
                   "depth": depth, "intrinsics": np.asarray(frame.intrinsics, dtype=np.float64),
                   "camera": pelvis_world @ frame.world_camera, "locks": self.locks(frame, side)}
        with self.admitted():
            self.ensure_configured(frame, side)
            try:
                return self.client.call("plan_grasp", request)
            except PolicyWireError as error:
                raise PlannerUnavailable(str(error)) from error

    def close(self):
        if self.client is not None:
            self.client.close()


def endpoints(settings):
    """The host's SERVERS planning servers (ports endpoint, endpoint + 1, ...), this process's own first."""
    base, port = settings.endpoint.rsplit(":", 1)
    count = max(1, settings.servers)
    own = os.getpid() % count
    return [f"{base}:{int(port) + (own + i) % count}" for i in range(count)]


@contextmanager
def admission(settings):
    """The address of a planning server this process holds (an exclusive lock file per server port on this
    host) until the block ends; PlannerUnavailable when none frees within the settings' timeout."""
    deadline = time.monotonic() + settings.timeout
    while True:
        for address in endpoints(settings):
            with open(Path(tempfile.gettempdir()) / f"homebody-curobo-{address.rsplit(':', 1)[1]}.lock", "w") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                try:
                    yield address
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
            return
        if time.monotonic() > deadline:
            raise PlannerUnavailable(f"no planning server free within {settings.timeout:.0f} s")
        time.sleep(ADMISSION_POLL)


def thinned(rows, limit):
    """At most LIMIT rows of a joint path, endpoints kept."""
    rows = np.asarray(rows)
    if len(rows) <= limit:
        return rows
    return rows[np.unique(np.linspace(0, len(rows) - 1, limit).round().astype(int))]
