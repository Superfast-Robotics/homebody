"""A ZeroMQ REQ client with msgpack framing and msgpack_numpy arrays (pickle refused in both
directions), as the cuRobo server and GR00T-style policy servers speak."""
import io

import msgpack
import msgpack_numpy
import numpy as np
import zmq


class PolicyWireError(RuntimeError):
    """The server could not be reached, timed out, or answered with an error."""


def encode(value):
    if isinstance(value, np.ndarray) and value.dtype.kind == "O":
        raise TypeError(f"Refusing to send an object-dtype array of shape {value.shape}")
    return msgpack_numpy.encode(value)


def decode(value):
    if isinstance(value, dict):
        if value.get("__ndarray_class__", value.get(b"__ndarray_class__")):
            payload = value.get("as_npy", value.get(b"as_npy"))
            if payload is None:
                raise ValueError("Malformed ndarray payload")
            return np.load(io.BytesIO(payload), allow_pickle=False)
        if value.get("nd", value.get(b"nd")) and value.get("kind", value.get(b"kind")) in ("O", b"O"):
            raise ValueError("Refusing an object-dtype array payload")
    return msgpack_numpy.decode(value)


def pack(message) -> bytes:
    return msgpack.packb(message, default=encode)


def unpack(raw: bytes):
    return msgpack.unpackb(raw, object_hook=decode, raw=False)


class PolicyClient:
    """One REQ socket to a GR00T policy server at ENDPOINT (e.g. tcp://127.0.0.1:5555)."""

    def __init__(self, endpoint: str, timeout_s: float):
        self.endpoint, self.timeout_ms = endpoint, int(timeout_s * 1000)
        self.context = zmq.Context()
        self.socket = None
        self._connect()

    def _connect(self):
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(self.endpoint)

    def call(self, endpoint: str, data=None):
        request = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self.socket.send(pack(request))
            raw = self.socket.recv()
        except zmq.error.ZMQError as error:
            self._connect()
            raise PolicyWireError(f"Policy server {self.endpoint} did not answer {endpoint}: {error}") from error
        if raw == b"ERROR":
            raise PolicyWireError(f"Policy server {self.endpoint} failed {endpoint}")
        response = unpack(raw)
        if isinstance(response, dict) and "error" in response:
            raise PolicyWireError(f"Policy server error on {endpoint}: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self.call("ping")
            return True
        except PolicyWireError:
            return False

    def get_action(self, observation: dict) -> dict:
        """The action chunk dict for OBSERVATION (GR00T returns [action, info])."""
        response = self.call("get_action", {"observation": observation, "options": None})
        return response[0] if isinstance(response, (list, tuple)) else response

    def reset(self):
        return self.call("reset", {"options": None})

    def close(self):
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        self.context.term()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
