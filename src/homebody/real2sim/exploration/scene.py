"""Read-only G1 pose and joint display for the exploration console.

Poses one visual robot; the app draws the map, scans, route and trail.
"""
from __future__ import annotations

import contextlib
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FOOT_SOLE_M = 0.03
GROUND_DISC_COLOR = (38, 45, 58)


def _disc_mesh(radius: float = 0.82, segments: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """A flat disc of SEGMENTS triangles around the origin: the ground under the robot model."""
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    rim = np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.zeros(segments)))
    vertices = np.vstack((np.zeros((1, 3)), rim)).astype(np.float32)
    faces = np.array([[0, 1 + index, 1 + ((index + 1) % segments)] for index in range(segments)],
                     dtype=np.int32)
    return vertices, faces


@dataclass(frozen=True)
class Feeds:
    """Read-only callables; each answers None when its source is not there right now."""

    pose: Callable[[], Any]                      # (x, y, yaw) in the map frame
    qpos: Callable[[], Any]                      # joint positions in `joint_names` order
    joint_names: tuple[str, ...]



def pose_transform(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = (float(x), float(y), float(z))
    return T


def wxyz_of(T: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation of a 4x4 as a unit quaternion (w, x, y, z); viser's frame convention."""
    R = np.asarray(T, dtype=float)[:3, :3]
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        k = math.sqrt(trace + 1.0) * 2.0
        return (0.25 * k, (R[2, 1] - R[1, 2]) / k, (R[0, 2] - R[2, 0]) / k,
                (R[1, 0] - R[0, 1]) / k)
    i = int(np.argmax(np.diag(R)))
    j, m = (i + 1) % 3, (i + 2) % 3
    k = math.sqrt(1.0 + R[i, i] - R[j, j] - R[m, m]) * 2.0
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (R[m, j] - R[j, m]) / k
    q[1 + i] = 0.25 * k
    q[1 + j] = (R[j, i] + R[i, j]) / k
    q[1 + m] = (R[m, i] + R[i, m]) / k
    return tuple(q)


class _RobotModel:
    """The visual G1 under /env/robot, posed by the live joints; FK for the camera and feet."""

    def __init__(self, server, urdf_path: Path, body_joints: tuple[str, ...]) -> None:
        import yourdfpy
        from viser.extras import ViserUrdf

        self.urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=True,
                                       build_scene_graph=True, load_collision_meshes=False)
        self.frame = server.scene.add_frame("/env/robot", show_axes=False)
        self.viz = ViserUrdf(server, self.urdf, root_node_name="/env/robot")
        self.names = tuple(self.viz.get_actuated_joint_names())
        self.index = {name: i for i, name in enumerate(self.names)}
        self.body = body_joints
        self.cfg = np.zeros(len(self.names))
        vertices, faces = _disc_mesh()
        self.ground = server.scene.add_mesh_simple(
            "/env/robot/ground", vertices, faces, color=GROUND_DISC_COLOR, opacity=0.48,
            material="standard", flat_shading=True, cast_shadow=False, receive_shadow=True,
            side="double", position=(0.0, 0.0, -0.80))
        self.last_qpos: np.ndarray | None = None

    def update(self, qpos, pose) -> dict[str, np.ndarray]:
        cfg = self.cfg.copy()
        if qpos is not None:
            self.last_qpos = np.asarray(qpos, dtype=float).ravel()
            for name, q in zip(self.body, self.last_qpos):
                if name in self.index:
                    cfg[self.index[name]] = q
        self.cfg = cfg
        self.viz.update_cfg(cfg)
        feet = [self.urdf.get_transform(link, "pelvis")[2, 3]
                for link in ("left_ankle_roll_link", "right_ankle_roll_link")]
        z = FOOT_SOLE_M - min(feet)
        self.ground.position = (0.0, 0.0, -z + 0.003)
        x, y, yaw = (0.0, 0.0, 0.0) if pose is None else tuple(float(v) for v in pose[:3])
        # the pose is the pelvis pose
        T_world_pelvis = pose_transform(x, y, yaw, z)
        self.frame.position = tuple(T_world_pelvis[:3, 3])
        self.frame.wxyz = wxyz_of(T_world_pelvis)
        return {
            "T_world_pelvis": T_world_pelvis,
            "T_world_torso": T_world_pelvis @ self.urdf.get_transform("torso_link", "pelvis"),
            "T_world_lidar": T_world_pelvis @ self.urdf.get_transform("mid360_link", "pelvis"),
        }


class EnvironmentView:
    """Refresh the robot display independently of sensor and command owners."""

    def __init__(self, server, feeds: Feeds, *, urdf_path: Path, hz: float = 4.0) -> None:
        self.feeds = feeds
        self._period = 1.0 / max(float(hz), 0.2)
        self._stop = threading.Event()
        self._robot = _RobotModel(server, urdf_path, feeds.joint_names)
        self._thread = threading.Thread(target=self._loop, name="exploration-robot-view", daemon=True)
        self._thread.start()

    def _tick(self) -> None:
        self._robot.update(self.feeds.qpos(), self.feeds.pose())

    def _loop(self) -> None:
        while not self._stop.wait(self._period):
            with contextlib.suppress(Exception):  # display failures cannot stop sensor ingestion
                self._tick()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError("robot view did not stop")
