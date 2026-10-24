"""A Robot on a warehouse-sim server (github.com/Superfast-Robotics/warehouse-sim): physics and the balance/arm
controller run in the server (its controller plugin); this side reads frames and sends targets.

Built from warehouse-sim's HomeBody integration (warehouse_homebody: ObservationDecoder, HomebodyRobotAdapter,
PluginControl), so it needs warehouse_sim and warehouse_homebody importable and the same MuJoCo as the server.
The harness and the server share the scene files (a shared volume): the harness loads the server's profile to
decode its frames, and refuses a server whose model differs.
"""
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from homebody.primitives.actions import PhysicsInvalid

LINK_TIMEOUT_S = 10.0  # a request to the simulation server that takes longer means the link is down
STALE_WAIT_S = 3.0  # the server's camera process can fall behind under load: a fresh frame is waited for this long
STALE_POLL_S = 0.05


class SimLinkLost(PhysicsInvalid):
    """The simulation server stopped answering; the task stops as it does when physics fails."""


ENGINES = ("warehouse", "homebody")  # warehouse-sim's server, or HomeBody's own (homebody sim-serve, simserver/)


@dataclass(frozen=True)
class SimTarget:
    """A simulation server to attach to: its control ENDPOINT, the Homebody SCENE package it serves and its ENGINE.
    A warehouse-sim server also names the PROFILE it serves (a profile.toml on the shared volume, with ROOTS naming its
    source roots), exported from SCENE; a HomeBody sim server serves SCENE itself (no profile)."""
    endpoint: str
    profile: Path | None
    scene: Path
    roots: dict = field(default_factory=dict)
    engine: str = "warehouse"

    @classmethod
    def from_json(cls, data: dict) -> "SimTarget":
        engine = str(data.get("engine") or "warehouse")
        if engine not in ENGINES:
            raise ValueError(f"a simulation target's engine is one of {ENGINES}")
        missing = {"endpoint", "scene", *(("profile",) if engine == "warehouse" else ())} - set(data)
        if missing:
            raise ValueError(f"a simulation target needs {sorted(missing)}")
        profile = Path(data["profile"]) if data.get("profile") else None
        return cls(str(data["endpoint"]), profile, Path(data["scene"]),
                   {str(k): str(v) for k, v in (data.get("roots") or {}).items()}, engine)


def linked(call):
    """CALL with the link's failures (timeouts, a closed socket) as SimLinkLost."""
    def guarded(self, *args, **kwargs):
        try:
            return call(self, *args, **kwargs)
        except PhysicsInvalid:
            raise
        except (TimeoutError, ConnectionError, OSError) as error:
            raise SimLinkLost(f"simulation server {self.target.endpoint}: {type(error).__name__}: {error}") from error
        except RuntimeError as error:
            if "zmq" in type(error).__module__ or "control channel" in str(error):
                raise SimLinkLost(f"simulation server {self.target.endpoint}: {error}") from error
            raise
        except Exception as error:
            if type(error).__name__ != "ObservationUnavailable":  # warehouse_sim.contracts: no fresh frame
                raise
            raise SimLinkLost(f"simulation server {self.target.endpoint}: no fresh camera frame for "
                              f"{STALE_WAIT_S:.0f} s ({error})") from error
    guarded.__name__, guarded.__doc__ = call.__name__, call.__doc__
    return guarded


class WarehouseRobot:
    """The session's backend: the Robot calls, the extras the session uses, and the server's status."""

    def __init__(self, target: SimTarget, adapter, client, description: dict):
        self.target, self.adapter, self.client, self.description = target, adapter, client, description
        self.cameras = tuple(description.get("cameras", ()))
        self.camera_stream = description.get("camera_stream")

    # --- Robot ---
    @property
    @linked
    def epoch(self):
        return self.adapter.epoch

    @property
    @linked
    def time(self):
        return self.adapter.time

    @linked
    def snapshot(self):
        """The newest frame, waiting up to STALE_WAIT_S while the server has none fresh enough."""
        deadline = time.monotonic() + STALE_WAIT_S
        while True:
            try:
                return self.adapter.snapshot()
            except Exception as error:
                if type(error).__name__ != "ObservationUnavailable" or time.monotonic() >= deadline:
                    raise
                time.sleep(STALE_POLL_S)

    @linked
    def measure(self):
        return self.adapter.measure()

    @linked
    def wrist_view(self, side):
        return self.adapter.wrist_view(side)

    @linked
    def base_velocity(self, epoch, forward, lateral, yaw_rate):
        self.adapter.base_velocity(epoch, forward, lateral, yaw_rate)

    @linked
    def arm_target(self, epoch, side, joints):
        self.adapter.arm_target(epoch, side, joints)

    @linked
    def grip(self, epoch, side, closure):
        self.adapter.grip(epoch, side, closure)

    @linked
    def advance(self, epoch, seconds):
        self.adapter.advance(epoch, seconds)

    @linked
    def stop(self, epoch):
        self.adapter.stop(epoch)

    # --- what the session uses beyond Robot ---
    @linked
    def ego_image(self):
        return self.adapter.ego_image()

    @linked
    def wrist_views(self):
        return self.adapter.wrist_views()

    @linked
    def overview_frame(self):
        """The server's observer view (the console's observer camera), or None when its profile has none."""
        if "observer" not in self.cameras:
            return None
        return self.client.snapshot(self.adapter.clock.episode_id, images=True, max_age_s=1)["cameras"]["observer"]["rgb"]

    @linked
    def evaluation_state(self):
        return self.adapter.evaluation_state()

    @linked
    def set_evaluation_sink(self, callback):
        self.adapter.set_evaluation_sink(callback)

    @linked
    def reset(self):
        self.adapter.reset()

    @linked
    def status(self) -> dict:
        """For the harness's state: the server's profile, simulated time and controller."""
        description = self.client.describe()
        return {"endpoint": self.target.endpoint, "profile": description["profile"]["id"],
                "controller": description.get("controller"), "sim_time": self.adapter.time}

    def close(self):
        """Stop using the server and hand its commands back (warehouse-sim's release), so the next harness can take
        it over without a restart."""
        for step, call in (("closing", self.adapter.close), ("releasing", self.client.release)):
            try:
                call()
            except Exception as error:  # noqa: BLE001 -- closing a lost link must not hide the original failure
                print(f"warehouse backend: {step} {self.target.endpoint} failed: {error}", file=sys.stderr)


def connect(target: SimTarget, root: Path, settings) -> WarehouseRobot:
    """Attach to TARGET's server with ROOT's robot calibration. Refuses a server without the controller plugin or
    whose model differs from TARGET's profile."""
    from holosoma_inference.g1_control.config import amo_profile
    from warehouse_homebody.adapter import HomebodyRobotAdapter, ObservationDecoder
    from warehouse_homebody.plugin_control import PluginControl
    from warehouse_sim.client import SimulatorClient
    from warehouse_sim.profiles import load_profile

    client = SimulatorClient(target.endpoint, timeout_s=LINK_TIMEOUT_S)
    try:
        description = client.describe()
    except (TimeoutError, OSError, RuntimeError) as error:
        raise SimLinkLost(f"simulation server {target.endpoint} does not answer: {error}") from error
    if "controller" not in description.get("capabilities", ()):
        raise ValueError(f"{target.endpoint} runs no controller plugin; start it with "
                         "--controller warehouse_homebody.plugin:make_plugin")
    profile = load_profile(target.profile, {key: Path(value) for key, value in target.roots.items()})
    calibration = json.loads((settings.robot.directory(root) / "calibration.json").read_text())
    decoder = ObservationDecoder(profile, calibration, amo_profile().robot.joint_names, target.scene / "map.npz")
    if decoder.manifest["profile_hash"] != description["profile_hash"]:
        raise ValueError(f"{target.endpoint} serves another model than {target.profile} "
                         "(re-export the scene or attach the matching profile)")
    control = PluginControl(client)
    adapter = HomebodyRobotAdapter(client, decoder, control, settings, control_factory=lambda: control)
    return WarehouseRobot(target, adapter, client, description)
