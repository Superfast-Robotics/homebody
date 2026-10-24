"""A Robot on a HomeBody sim server (homebody sim-serve): the session's backend when the harness attaches with
--sim-engine homebody. Every call is one request on the server's socket; a call the server does not answer within
LINK_TIMEOUT_S, or a closed socket, is SimLinkLost (the task stops as it does when physics fails). The server's scene
must be the scene package this harness reads (scene_identity), or the attach is refused."""
import threading
from pathlib import Path

import zmq

from homebody.backends.warehouse import SimLinkLost
from homebody.simserver import codec, protocol

LINK_TIMEOUT_S = 10.0  # one call; an advance waits at most a second of simulated time on a server near real time


class Link:
    """One request at a time to ENDPOINT, each with a reply or SimLinkLost."""

    def __init__(self, endpoint: str, timeout_s: float = LINK_TIMEOUT_S):
        self.endpoint, self.timeout_s = endpoint, timeout_s
        self._lock = threading.Lock()
        self._socket = None

    def _connect(self):
        socket = zmq.Context.instance().socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.REQ_RELAXED, 1)  # after a timeout the next request may still be sent
        socket.setsockopt(zmq.REQ_CORRELATE, 1)  # and a late reply to the old one is dropped
        socket.setsockopt(zmq.RCVTIMEO, int(self.timeout_s * 1000))
        socket.connect(self.endpoint)
        return socket

    def call(self, method: str, **args):
        with self._lock:
            if self._socket is None:
                self._socket = self._connect()
            try:
                self._socket.send(codec.pack({"method": method, "args": args}))
                reply = codec.unpack(self._socket.recv())
            except zmq.Again as error:
                raise SimLinkLost(f"sim server {self.endpoint}: no answer to {method} within "
                                  f"{self.timeout_s:.0f} s") from error
            except zmq.ZMQError as error:
                raise SimLinkLost(f"sim server {self.endpoint}: {error}") from error
        if not reply.get("ok"):
            raise protocol.raised(reply)
        return reply.get("value")

    def close(self):
        with self._lock:
            if self._socket is not None:
                self._socket.close(0)
                self._socket = None


def describe(endpoint: str, timeout_s: float = 2.0) -> dict:
    """The server's describe, on a link of its own (the monitor app's status check)."""
    link = Link(endpoint, timeout_s)
    try:
        return link.call("describe")
    finally:
        link.close()


class RemoteRobot:
    """The session's backend on a sim server: the Robot calls, the extras the session uses, and the server status."""

    def __init__(self, target, link: Link, description: dict):
        self.target, self.link, self.description = target, link, description
        self.cameras = tuple(description.get("cameras", ()))
        self.camera_stream = description.get("camera_stream")
        self._sink = None

    # --- Robot ---
    @property
    def epoch(self):
        return self.link.call("clock")["epoch"]

    @property
    def time(self):
        return self.link.call("clock")["time"]

    def snapshot(self):
        return self.link.call("snapshot")

    def measure(self):
        return self.link.call("measure")

    def wrist_view(self, side):
        return self.link.call("wrist_view", side=side)

    def base_velocity(self, epoch, forward, lateral, yaw_rate):
        self.link.call("base_velocity", epoch=epoch, forward=float(forward), lateral=float(lateral),
                       yaw_rate=float(yaw_rate))

    def arm_target(self, epoch, side, joints):
        self.link.call("arm_target", epoch=epoch, side=side, joints=joints)

    def grip(self, epoch, side, closure):
        self.link.call("grip", epoch=epoch, side=side, closure=float(closure))

    def advance(self, epoch, seconds):
        try:
            self.link.call("advance", epoch=epoch, seconds=float(seconds))
        finally:
            self._collect()

    def stop(self, epoch):
        self.link.call("stop", epoch=epoch)

    # --- what the session uses beyond Robot ---
    def ego_image(self):
        return self.link.call("ego_image")

    def wrist_views(self):
        return self.link.call("wrist_views")

    def overview_frame(self):
        return self.link.call("overview_frame")

    def evaluation_state(self):
        return self.link.call("evaluation_state")

    def set_evaluation_sink(self, callback):
        """Physics samples taken on the server reach CALLBACK after each advance (None: stop taking them)."""
        if self._sink is not None:
            self._collect()
        self._sink = callback
        self.link.call("evaluation_sink", on=callback is not None)
        self._collect()

    def _collect(self):
        if self._sink is None:
            return
        for sample in self.link.call("samples"):
            self._sink(sample)

    def reset(self):
        self.link.call("reset")

    def status(self) -> dict:
        return {"endpoint": self.target.endpoint, **self.link.call("status")}

    def close(self):
        """Stop using the server: the robot is stopped and samples are no longer taken; the server keeps running."""
        for method, args in (("evaluation_sink", {"on": False}), ("stop", {"epoch": self.link.call("clock")["epoch"]})):
            try:
                self.link.call(method, **args)
            except Exception:  # noqa: BLE001 -- closing a lost link must not hide the failure that ended the session
                break
        self.link.close()


def connect(target, root: Path, settings) -> RemoteRobot:
    """Attach to TARGET (a SimTarget with engine "homebody"); refuses a server serving another scene package."""
    link = Link(target.endpoint)
    try:
        description = link.call("describe")
    except SimLinkLost as error:
        link.close()
        raise SimLinkLost(f"simulation server {target.endpoint} does not answer: {error}") from error
    if description.get("engine") != protocol.ENGINE:
        link.close()
        raise ValueError(f"{target.endpoint} is not a HomeBody sim server")
    expected = protocol.scene_identity(target.scene)
    if description.get("scene_identity") != expected:
        link.close()
        raise ValueError(f"{target.endpoint} serves another scene than {target.scene} "
                         f"(it serves {description.get('scene')}); attach the matching scene package")
    return RemoteRobot(target, link, description)
