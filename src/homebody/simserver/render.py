"""The sim server's renders, in their own process: a MujocoBackend of the same scene mirrors the physics state it is
sent (positions, mocap, time) and renders it, so rendering never holds up the physics thread. Requests are served one
at a time, in order."""
import multiprocessing as mp
import threading
from pathlib import Path

import mujoco


def state_of(backend) -> dict:
    """The state a render needs, copied from BACKEND's data (call with the physics lock held)."""
    data = backend.data
    return {"qpos": data.qpos.copy(), "mocap_pos": data.mocap_pos.copy(), "mocap_quat": data.mocap_quat.copy(),
            "time": float(data.time), "epoch": backend.epoch}


class Mirror:
    """A render-only MujocoBackend that takes on a sent state before each render."""

    def __init__(self, backend):
        self.backend = backend

    def render(self, kind: str, state: dict, side: str | None = None):
        b, data = self.backend, self.backend.data
        data.qpos[:], data.mocap_pos[:], data.mocap_quat[:] = state["qpos"], state["mocap_pos"], state["mocap_quat"]
        data.time, b.epoch = state["time"], state["epoch"]
        mujoco.mj_forward(b.model, data)
        if kind == "frame":
            return b._render("ego")
        if kind == "wrist_view":
            view = b.wrist_view(side)
            return {"rgb": view.rgb, "depth": view.depth, "labels": view.labels, "intrinsics": view.intrinsics,
                    "world_camera": view.world_camera}
        if kind == "wrist_views":
            return b.wrist_views()
        if kind == "observer":
            return b.overview_frame()
        if kind == "ego_rgb":
            return b.ego_image()
        if kind == "stream":
            wrists = b.wrist_views()
            return {"ego": b.ego_image(), "left_wrist": wrists["left"], "right_wrist": wrists["right"],
                    "observer": b.overview_frame()}
        raise ValueError(f"unknown render {kind}")


def _worker(root: Path, settings, scene: Path, connection):
    from homebody.backends.mujoco import MujocoBackend
    try:
        mirror = Mirror(MujocoBackend(root, settings, scene=scene))
        connection.send(("ready", None))
    except Exception as error:  # noqa: BLE001 -- the parent reports why the renderer did not start
        connection.send(("error", f"{type(error).__name__}: {error}"))
        return
    while True:
        request = connection.recv()
        if request is None:
            return
        kind, state, side = request
        try:
            connection.send(("ok", mirror.render(kind, state, side)))
        except Exception as error:  # noqa: BLE001 -- one failed render is that request's failure, not the worker's
            connection.send(("error", f"{type(error).__name__}: {error}"))


class RenderProcess:
    """Renders on a child process; `render(kind, state, side)` blocks for the result (one request at a time)."""

    def __init__(self, root: Path, settings, scene: Path, start_timeout_s: float = 300.0):
        context = mp.get_context("spawn")  # a fresh interpreter: no forked GL or thread state
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_worker, args=(root, settings, scene, child), name="sim-render",
                                        daemon=True)
        self._process.start()
        self._lock = threading.Lock()
        if not self._connection.poll(start_timeout_s):
            self.close()
            raise RuntimeError(f"the render process did not start within {start_timeout_s:.0f} s")
        status, detail = self._connection.recv()
        if status != "ready":
            self.close()
            raise RuntimeError(f"the render process failed to start: {detail}")

    def render(self, kind: str, state: dict, side: str | None = None):
        with self._lock:
            if not self._process.is_alive():
                raise RuntimeError("the render process stopped")
            self._connection.send((kind, state, side))
            status, value = self._connection.recv()
        if status != "ok":
            raise RuntimeError(f"render {kind} failed: {value}")
        return value

    def close(self):
        try:
            self._connection.send(None)
        except (OSError, ValueError):
            pass
        self._process.join(5)
        if self._process.is_alive():
            self._process.terminate()


class InlineRenderer:
    """Renders on the calling thread (tests, and scenes too small to need a process)."""

    def __init__(self, mirror):
        self.mirror = mirror

    def render(self, kind, state, side=None):
        return self.mirror.render(kind, state, side)

    def close(self):
        pass
