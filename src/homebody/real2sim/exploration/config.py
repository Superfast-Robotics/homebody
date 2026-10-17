"""Exploration settings: one frozen dataclass of defaults per group, overridden by group and key
from `configs/exploration.toml` and then `--settings`. The agent group reuses the simulation's
`Agent`."""
from dataclasses import dataclass

from homebody.session.config import Agent

AGENT = Agent(model="astra", reasoning_effort="medium", service_tier="fast", timeout=120.0)


@dataclass(frozen=True)
class Exploration:
    """The task, its time budget, the console and the output directory."""
    task: str = "Explore the accessible parts of this space."
    budget_s: float = 420.0
    host: str = "127.0.0.1"
    port: int = 8099
    output: str = ""  # output root; empty means runs/explore under the repository
    record: bool = True
    observe_only: bool = False
    ask_operator: bool = False  # the model may pause to ask; set by --enable-ask-operator
    zone_size_m: float = 0.0  # side of a square start zone drawn on the map; 0 draws none
    zone_start_xy_m: str = ""  # "x,y" start inside that zone; empty centres it 0.5 m from the back
    start_spin: bool = True  # one slow turn in place before the first decision
    done_fraction: float = 0.90  # visited share of the task region that counts as covered
    decision_recovery_wait_s: float = 60.0  # longest wait for fresh map and image evidence
    turn_drift_m: float = 0.50  # position drift tolerated while turning in place


@dataclass(frozen=True)
class Robot:
    """How the workstation reaches the G1. `HOMEBODY_G1_SSH_PASSWORD`, when set, overrides
    `ssh_password`."""
    iface: str = ""  # network interface facing the robot; required to launch
    ssh_target: str = "unitree@192.168.123.164"
    ssh_password: str = "123"  # the G1 factory default


@dataclass(frozen=True)
class Body:
    """The body host process that keeps the G1 balancing."""
    address: str = "127.0.0.1:29400"
    cores: str = ""  # CPU set for the body host; empty lets the scheduler choose
    use_sudo: bool = True
    threads: int = 2
    writer_lock_path: str = "/tmp/homebody-g1-writer.lock"


@dataclass(frozen=True)
class Iphone:
    """The phone camera stream."""
    input_url: str = ""  # RTSP or USB source; required to launch
    endpoint: str = "tcp://127.0.0.1:55568"  # local relay endpoint the explorer reads
    relay_service: str = ""
    max_age_s: float = 3.0  # a frame older than this is not evidence
    hwaccel: str = "none"
    post_action_buffer_s: float = 3.0  # wait after a motion for delayed RGB to catch up


@dataclass(frozen=True)
class Slam:
    """The LiDAR odometry container."""
    image: str = "homebody-superodom:57a6e233-serial-imu"
    adopt: bool = False  # keep an already healthy engine at launch instead of restarting it
    hold: bool = True  # delay each cloud until the IMU stream passes its last point
    hold_margin_s: float = 0.010  # margin past the last point
    hold_max_s: float = 0.080  # longest delay


@dataclass(frozen=True)
class Recording:
    """The sensor bag recorder container and its exports."""
    image: str = "homebody-recorder:humble-57a6e233"
    min_free_gb: float = 40.0
    color_fps: int = 30
    export_fps: int = 30
    optional_topics: str = ""  # comma-separated topics the post-check may find absent


@dataclass(frozen=True)
class Hands:
    """Dex3 finger servo gains (N m/rad, N m s/rad) and limits."""
    kp: float = 8.0
    kd: float = 0.6
    lead_rad: float = 1.04
    lead_min_rad: float = 0.12
    torque_fraction: float = 1.0
    temp_ease_c: float = 60.0
    temp_rest_c: float = 75.0
    wrist_effort_nm: float = 5.0


@dataclass(frozen=True)
class Lidar:
    """Points ignored around the robot's own body."""
    self_clear_m: float = 1.0
    body_column_m: float = 0.36


@dataclass(frozen=True)
class Map:
    """The live occupancy map and its pictures."""
    extent_m: float = 8.0
    picture_cell_px: int = 3
    zone_corners: int = 4


@dataclass(frozen=True)
class Gait:
    """Base velocity commands realised by the walking policy."""
    turn_rad_s: float = 1.00
    side_mps: float = 0.45
    final_turn_gain_hz: float = 1.5
    max_walk_s: float = 2.0


@dataclass(frozen=True)
class Drive:
    """The checked navigation controller: route following, turning and arrival."""
    cruise_mps: float = 0.30
    floor_mps: float = 0.35
    close_mps: float = 0.45
    close_step_m: float = 0.60
    close_gain_hz: float = 1.1
    gain_hz: float = 0.63
    turn_gain_hz: float = 1.8
    turn_floor_rad_s: float = 0.35
    tolerance_m: float = 0.25
    waypoint_tol_m: float = 0.40
    waypoint_spacing_m: float = 0.5
    leave_deg: float = 15.0
    align_deg: float = 20.0
    overshoot_m: float = 0.20
    run_in_m: float = 0.0
    final_turn_drift_m: float = 0.25
