"""homebody sim-serve: one scene package's physics behind the Robot calls, at real time.

The physics thread steps the backend (MujocoBackend: the walking policy and the arm and hand servos) one controller
period at a time against the wall clock, whether or not a client is attached. Commands (base_velocity, arm_target,
grip, stop) take effect at once; `advance(epoch, seconds)` waits until simulated time has moved on by SECONDS, as on
the real robot. A velocity command expires after physics.command_timeout without a new one, so a client that goes
away leaves the robot standing. Renders (the head frame, the wrists, the observer) run in a render process on a copy
of the state, and the cameras are published on the stream in warehouse-sim's format, so the monitor app shows this
server like a warehouse-sim one. `describe` says what the monitor app reads of a warehouse-sim server, plus the engine
and the scene's identity.
"""
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import zmq

from homebody.primitives.actions import PhysicsInvalid
from homebody.primitives.observations import Frame, WristView
from homebody.simserver import codec, protocol
from homebody.simserver.render import state_of
from homebody.simserver.stream import encode

POLL_MS = 20
WORKERS = 4  # concurrent calls: an advance waits while describe, measure or a render are answered
SAMPLES = 4096  # evaluation samples kept for the client to collect


class SimServer:
    """BACKEND (a MujocoBackend or a stand-in) stepped at real time, served on SETTINGS.sim_server's addresses.
    RENDERER renders sent states (render.RenderProcess). CLOCK and SLEEP are the wall clock (tests pass fakes)."""

    def __init__(self, backend, renderer, settings, scene: Path, *, clock=time.monotonic, sleep=time.sleep):
        self.backend, self.renderer, self.settings, self.scene = backend, renderer, settings, Path(scene)
        self.config = settings.sim_server
        self.clock, self.sleep = clock, sleep
        self.identity = protocol.scene_identity(self.scene) if self.scene.is_dir() else ""
        self.episode = uuid.uuid4().hex
        self.lock = threading.Condition()  # held for every touch of the backend; notified after each step
        self.fault: str | None = None  # why physics stopped (PhysicsInvalid) until a reset
        self.samples: deque = deque(maxlen=SAMPLES)
        self.steps, self.dropped_s, self.started = 0, 0.0, clock()
        self.stream_errors, self.stream_error = 0, ""  # stream frames not rendered, and why the last one was not
        self.stream_endpoint = self.config.stream
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # --- physics ---
    def step_once(self):
        """One controller period (with the lock held by the caller)."""
        if self.fault is not None:
            return
        try:
            self.backend.advance(self.backend.epoch, 1 / self.settings.physics.control_hz)
            self.steps += 1
        except PhysicsInvalid as error:
            self.fault = str(error)

    def _physics(self):
        period = 1 / self.settings.physics.control_hz
        due = self.clock()
        while not self._stop.is_set():
            with self.lock:
                self.step_once()
                self.lock.notify_all()
            due += period
            lag = self.clock() - due
            if lag > self.config.max_lag_s:  # too slow to catch up: run slower than real time, not in bursts
                self.dropped_s += lag
                due = self.clock()
            elif lag < 0:
                self.sleep(-lag)

    def real_time_factor(self) -> float:
        wall = self.clock() - self.started
        return round(self.steps / self.settings.physics.control_hz / wall, 3) if wall > 0 else 0.0

    # --- the Robot calls and what the session uses beyond them ---
    def call(self, method: str, args: dict):
        handler = getattr(self, f"rpc_{method}", None)
        if handler is None:
            raise ValueError(f"unknown method {method!r}")
        return handler(**args)

    def rpc_describe(self):
        with self.lock:
            sim_time, epoch = self.backend.time, self.backend.epoch
        return {"schema_version": protocol.SCHEMA, "engine": protocol.ENGINE, "episode_id": self.episode,
                "profile": {"id": f"homebody-{self.scene.name}"}, "mode": "continuous",
                "scene": str(self.scene), "scene_identity": self.identity, "sim_time": sim_time, "epoch": epoch,
                "capabilities": ["controller", "robot"], "camera_stream": self.stream_endpoint,
                "cameras": ["ego", "left_wrist", "right_wrist", "observer"],
                "controller": {"running": self.fault is None, "error": self.fault,
                               "real_time_factor": self.real_time_factor(), "dropped_s": round(self.dropped_s, 3)},
                "stream_errors": self.stream_errors, "stream_error": self.stream_error}

    def rpc_clock(self):
        with self.lock:
            return {"epoch": self.backend.epoch, "time": self.backend.time}

    def rpc_measure(self):
        with self.lock:
            return self.backend.measure()

    def rpc_snapshot(self):
        with self.lock:
            self._check()
            state, measured = state_of(self.backend), self.backend.measure()
        rgb, depth, labels = self.renderer.render("frame", state)
        b = self.backend
        return Frame(measured.epoch, measured.sequence, measured.time, rgb, depth, b._intrinsics, measured.world_camera,
                     measured.world_torso, measured.base_pose, measured.joints, measured.hands, labels, b.navigation)

    def rpc_wrist_view(self, side: str):
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        with self.lock:
            state = state_of(self.backend)
        view = self.renderer.render("wrist_view", state, side)
        return WristView(state["epoch"], state["time"], side, **view)

    def _render(self, kind):
        with self.lock:
            state = state_of(self.backend)
        return self.renderer.render(kind, state)

    def rpc_wrist_views(self):
        return self._render("wrist_views")

    def rpc_ego_image(self):
        return self._render("ego_rgb")

    def rpc_overview_frame(self):
        return self._render("observer")

    def _check(self):
        if self.fault is not None:
            raise PhysicsInvalid(self.fault)

    def rpc_base_velocity(self, epoch, forward, lateral, yaw_rate):
        with self.lock:
            self._check()
            self.backend.base_velocity(epoch, forward, lateral, yaw_rate)

    def rpc_arm_target(self, epoch, side, joints):
        with self.lock:
            self._check()
            self.backend.arm_target(epoch, side, joints)

    def rpc_grip(self, epoch, side, closure):
        with self.lock:
            self._check()
            self.backend.grip(epoch, side, closure)

    def rpc_stop(self, epoch):
        with self.lock:
            self.backend.stop(epoch)

    def rpc_advance(self, epoch, seconds):
        """Wait until simulated time has moved on by SECONDS (checked as the in-process backend checks them)."""
        with self.lock:
            self._check()
            self.backend.limits.epoch(epoch, self.backend.epoch)
            seconds = self.backend.limits.duration(seconds)
            until = self.backend.time + seconds - 0.5 / self.settings.physics.control_hz
            while self.backend.time < until and self.fault is None and not self._stop.is_set():
                self.lock.wait(1.0)
            self._check()
            self.backend.limits.epoch(epoch, self.backend.epoch)  # a reset while waiting ends this generation

    def rpc_reset(self):
        with self.lock:
            self.backend.reset()
            self.fault = None
            self.samples.clear()
            return {"epoch": self.backend.epoch, "time": self.backend.time}

    def rpc_evaluation_state(self):
        with self.lock:
            return self.backend.evaluation_state()

    def rpc_evaluation_sink(self, on: bool):
        """Keep the physics samples the backend takes (its evaluation sink) for the client to collect, or stop."""
        with self.lock:
            self.samples.clear()
            self.backend.set_evaluation_sink(self.samples.append if on else None)

    def rpc_samples(self):
        """The physics samples kept since the last collection."""
        with self.lock:
            taken = list(self.samples)
            self.samples.clear()
        return taken

    def rpc_status(self):
        described = self.rpc_describe()
        return {"profile": described["profile"]["id"], "controller": described["controller"],
                "sim_time": described["sim_time"]}

    # --- serving ---
    def start(self):
        """Start physics, the RPC socket and the stream; returns once all three run."""
        context = zmq.Context.instance()
        self._rpc = context.socket(zmq.ROUTER)
        self._rpc.setsockopt(zmq.LINGER, 0)
        self._rpc.bind(self.config.endpoint)
        self._pub = context.socket(zmq.PUB)
        self._pub.setsockopt(zmq.LINGER, 0)
        self._pub.setsockopt(zmq.SNDHWM, 8)
        self._pub.bind(self.config.stream)
        self._replies: deque = deque()
        self._pool = ThreadPoolExecutor(WORKERS, thread_name_prefix="sim-call")
        for target, name in ((self._physics, "sim-physics"), (self._serve, "sim-rpc"), (self._publish, "sim-stream")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        return self

    def _answer(self, envelope, payload):
        try:
            request = codec.unpack(payload)
            reply = {"ok": True, "value": self.call(str(request["method"]), dict(request.get("args") or {}))}
        except Exception as error:  # noqa: BLE001 -- every failure goes back to the caller as its reply
            reply = protocol.failure(error)
        try:
            packed = codec.pack(reply)
        except TypeError as error:
            packed = codec.pack(protocol.failure(error))
        self._replies.append((envelope, packed))

    def _serve(self):
        while not self._stop.is_set():
            if self._rpc.poll(POLL_MS):
                *envelope, payload = self._rpc.recv_multipart()
                self._pool.submit(self._answer, envelope, payload)
            while self._replies:
                envelope, packed = self._replies.popleft()
                self._rpc.send_multipart([*envelope, packed])

    def _publish(self):
        period = 1 / self.config.stream_hz
        while not self._stop.wait(period):
            try:
                with self.lock:
                    state = state_of(self.backend)
                views = self.renderer.render("stream", state)
            except Exception as error:  # noqa: BLE001 -- a missed stream frame is only a missed frame; the next follows
                self.stream_errors, self.stream_error = self.stream_errors + 1, f"{type(error).__name__}: {error}"[:200]
                continue
            stamp = {"sim_time_s": state["time"], "epoch": state["epoch"], "episode_id": self.episode}
            for camera, rgb in views.items():
                self._pub.send_multipart([camera.encode(), encode(camera, stamp, rgb)])

    def close(self):
        self._stop.set()
        with self.lock:
            self.lock.notify_all()
        for thread in self._threads:
            thread.join(5)
        if self._threads:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._rpc.close(0)
            self._pub.close(0)
        self.renderer.close()
