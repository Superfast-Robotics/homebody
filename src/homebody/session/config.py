"""The simulation's settings. Each group is a frozen dataclass of defaults.
`configs/simulation.toml` overrides them by group and key, and a scene package's
`settings.toml` overlays that."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_ENV = "src_kitchen"  # the environment a run uses unless --env names another
DEFAULT_SCENE = Path("assets/real2sim") / DEFAULT_ENV


@dataclass(frozen=True)
class Physics:
    """MuJoCo stepping and the simulated hand (`hand_kp` in N m/rad)."""
    timestep: float = 0.001
    control_hz: int = 250
    policy_hz: int = 50
    solver: str = "Newton"
    integrator: str = "Euler"
    # pyramidal cone at impratio 1 lets resting objects creep
    cone: str = "elliptic"
    impratio: float = 10.0
    noslip_iterations: int = 2
    hand_kp: float = 2.0
    hand_kd: float = 0.10
    hand_torque_fraction: float = 0.80
    hand_friction: tuple[float, float, float] = (4.0, 0.20, 0.020)
    hand_contact: tuple[float, float] = (0.004, 1.0)
    arm_rate: float = 0.525
    arm_authority_seconds: float = 2.0
    repeated_state_stop_ticks: int = 25
    hand_rate: float = 2.0
    command_timeout: float = 0.20
    min_base_height: float = 0.45
    max_base_height: float = 1.05
    max_tilt_degrees: float = 45.0

@dataclass(frozen=True)
class Camera:
    width: int = 640
    height: int = 480
    horizontal_fov: float = 69.0
    near: float = 0.02
    far: float = 15.0
    observer_width: int = 960
    observer_height: int = 540

@dataclass(frozen=True)
class Recording:
    """Sampling periods of the recorded streams, in simulation seconds. Each video plays at its sampling rate."""
    frame_period: float = 0.5
    room_period: float = 1.0
    shoulder_period: float = 0.25
    physics_period: float = 0.1
    publish_period: float = 0.5
    keyframes: bool = False  # per-stage head and wrist keyframes of every skill call (session/trace.py)

@dataclass(frozen=True)
class Servo:
    tick: float = 0.04
    perception_period: float = 0.20
    position_gain: float = 2.0
    rotation_gain: float = 2.0
    max_linear_speed: float = 0.08
    max_angular_speed: float = 0.4
    damping_squared: float = 0.002
    max_joint_lag: float = 0.35
    rotation_tolerance: float = 0.12
    stable_frames: int = 3

@dataclass(frozen=True)
class Collision:
    cloud_margin: float = 0.008
    voxel_size: float = 0.01
    min_hit_voxels: int = 1
    joint_step: float = 0.04

@dataclass(frozen=True)
class Topdown:
    ahead: tuple[float, float] = (0.0, 1.05)
    left: tuple[float, float] = (-0.70, 0.70)
    height: tuple[float, float] = (-0.90, 0.60)
    pixels_per_metre: float = 760.0
    grid_metres: float = 0.10
    label_metres: float = 0.20
    chest_size: tuple[float, float] = (0.22, 0.26)

# name -> (provider, model id, CLI effort). "-api" names use a key, bare names a CLI login
MODELS = {
    "astra": ("codex", "gpt-6-astra", "high"),
    "astra-api": ("openai-api", "gpt-6-astra", None),
    "astra-litellm": ("litellm", "gpt-6-astra", "medium"),
    "sol61-litellm": ("litellm", "gpt-6.1-sol", "medium"),
    "sol": ("codex", "gpt-5.6-sol", "high"),
    "sol61": ("codex", "gpt-6.1-sol", "high"),
    "sol61-api": ("openai-api", "gpt-6.1-sol", None),
    "sol-api": ("openai-api", "gpt-5.6-sol", None),
    "luna": ("codex", "gpt-5.6-luna", "high"),
    "luna-api": ("openai-api", "gpt-5.6-luna", None),
    "terra": ("codex", "gpt-5.6-terra", "high"),
    "terra-api": ("openai-api", "gpt-5.6-terra", None),
    "opus": ("claude-code", "claude-opus-5-5", "high"),
    "opus-api": ("claude-api", "claude-opus-5-5", None),
    "fable": ("claude-code", "claude-fable-5-1", "high"),
    "fable-api": ("claude-api", "claude-fable-5-1", None),
    "sonnet": ("claude-code", "claude-sonnet-5-5", "high"),
    "sonnet-api": ("claude-api", "claude-sonnet-5-5", None),
    "haiku": ("claude-code", "claude-haiku-4-5-20251001", "high"),
    "haiku-api": ("claude-api", "claude-haiku-4-5-20251001", None),
}


@dataclass(frozen=True)
class Agent:
    """The model, by one name from MODELS, and its limits. The robot stands `done_settle_seconds` after `done` before the final physics sample."""
    model: str = "astra"
    # "model" takes the model's own effort, max means Codex's xhigh
    reasoning_effort: str = "model"
    service_tier: str = "default"
    timeout: float = 180.0
    # retries of a busy service, after busy_wait, twice that, three times that seconds
    busy_retries: int = 3
    busy_wait: float = 20.0
    max_steps: int = 80
    done_settle_seconds: float = 3.0
    # the world holds still while a decision is outstanding, as for the coder; false: physics runs on, as on a robot
    pause_while_thinking: bool = True

    def __post_init__(self):
        if self.service_tier not in ("default", "fast", "flex"):
            raise ValueError("agent.service_tier must be default, fast or flex")
        if self.model not in MODELS:
            raise ValueError(f"agent.model must be one of {', '.join(MODELS)}")
        if self.reasoning_effort not in ("model", "low", "medium", "high", "xhigh", "max"):
            raise ValueError("agent.reasoning_effort must be model, low, medium, high, xhigh or max")

    @property
    def provider(self) -> str:
        return MODELS[self.model][0]

    @property
    def model_id(self) -> str:
        return MODELS[self.model][1]

    @property
    def effort(self) -> str:
        """The CLI effort level: the model's own unless reasoning_effort names a level."""
        return MODELS[self.model][2] if self.reasoning_effort == "model" else self.reasoning_effort


@dataclass(frozen=True)
class Curobo:
    """The cuRobo planning server a skill may call (helpers/curobo.py): collision-free arm paths
    against the head camera's depth. The first request builds the server's planners, hence the
    long `timeout`."""
    endpoint: str = "tcp://127.0.0.1:5560"
    servers: int = 1  # planning servers per host, on consecutive ports from `endpoint` (helpers/curobo.py)
    timeout: float = 300.0
    max_candidates: int = 48
    max_rows: int = 40


@dataclass(frozen=True)
class Coder:
    """The coding sub-agent the agent calls with call_coder (session/coder.py): Codex at `codex`
    (HOMEBODY_CODER_CODEX overrides it) running `model` through the provider at `base_url` (the key in
    LITELLM_API_KEY). One call may take `seconds` of wall time, motion included, with at most `checks`
    dry runs and `executes` real runs; a task allows `calls` calls. With `auto`, every failed skill call is
    handed to the coder without the agent asking (session/recovery.py), at most `rescues` times at a stuck
    point, where the agent may call the failing skill at most `tries` more times."""
    codex: str = "codex"
    model: str = "gpt-6.1-sol"
    effort: str = "medium"
    base_url: str = "http://127.0.0.1:14000/v1"
    seconds: float = 300.0
    checks: int = 6
    executes: int = 2
    calls: int = 3
    auto: bool = False
    tries: int = 3
    rescues: int = 1


@dataclass(frozen=True)
class PlanMode:
    """Plan mode (session/planned.py): the agent first writes the task as a behavior tree of goals and works through
    it goal by goal; a failed tree may be rewritten at most `replans` times."""
    enabled: bool = False
    replans: int = 2


def private_http(url: str) -> bool:
    """Whether URL is plain http to a private-network address (a GPU host on the cluster's network)."""
    import ipaddress
    from urllib.parse import urlparse
    parts = urlparse(url)
    try:
        return parts.scheme == "http" and ipaddress.ip_address(parts.hostname or "").is_private
    except ValueError:
        return False


@dataclass(frozen=True)
class Monitor:
    """The live monitor (homebody/monitor): `vocab_model` (chat completions at `vocab_base_url`) writes each goal's
    action labels; the OpenAI Decisions API (`decision_model` at `decision_base_url`, with the key in
    OPENAI_DECISIONS_API_KEY, which nothing else reads) reads the newest `window_frames` head images, captured at
    `capture_hz` simulation hertz, and returns label probabilities. Calls never overlap: the next waits
    max(`min_period_s`, rolling p95 latency x `p95_margin`). A signal fires when its average (weight `ema` on the
    newest reading) crosses `tau` for `ticks` readings in a row (`tau_unsafe`, `ticks_unsafe` for unsafe;
    `stall_s` simulation seconds for no progress); a firing aborts the running skill at most `max_aborts_per_goal` times per goal
    (0: flag only), never within `grace_s` of a skill's start nor `cooldown_s` after an abort. A stall is only flagged:
    labelled on the dashboard, never a stop. `max_cost_usd` caps
    one task's decision spend (0: no cap). The dashboard shows the most probable activity label, switching only when
    another is the top one in `choice_votes` of the last `choice_window` readings (monitor/choice.py)."""
    enabled: bool = False
    vocab_model: str = "qwen3.8-27b"
    vocab_base_url: str = "http://127.0.0.1:4000/v1"
    vocab_key_file: str = "~/vlm/api_key"  # the vLLM servers' key; without the file, LITELLM_API_KEY (the proxy)
    vocab_thinking: bool = False
    vocab_temperature: float = 0.7
    vocab_top_p: float = 0.8
    vocab_top_k: int = 20
    vocab_presence_penalty: float = 1.5
    vocab_timeout: float = 120.0
    max_labels: int = 8
    refresh_on_skill: bool = True
    decision_model: str = "gpt-6-luna"
    decision_base_url: str = "https://api.openai.com/v1"
    decision_timeout: float = 15.0
    decision_usd_per_mtok: float = 0.10
    capture_hz: float = 10.0
    window_frames: int = 8
    image_short_side: int = 384
    wrist: bool = False
    min_period_s: float = 0.25
    p95_margin: float = 1.2
    p95_window: int = 20
    tau: float = 0.8
    tau_unsafe: float = 0.9
    ticks: int = 3
    ticks_unsafe: int = 2
    stall_s: float = 8.0  # skills stand still for seconds while they plan, verify and squeeze
    ema: float = 0.5
    grace_s: float = 2.0
    cooldown_s: float = 3.0
    max_aborts_per_goal: int = 2
    max_cost_usd: float = 0.0
    choice_window: int = 5  # the dashboard's activity: a new label must be the top one in choice_votes of these
    choice_votes: int = 3
    keep_frames: bool = False  # write every captured frame under the task's monitor/frames (tools/monitor_bench.py)

    def __post_init__(self):
        if not 1 <= self.window_frames <= 16:
            raise ValueError("monitor.window_frames must be from 1 to 16")
        if not 0.5 < self.tau < 1 or not 0.5 < self.tau_unsafe < 1:
            raise ValueError("monitor.tau and monitor.tau_unsafe must be between 0.5 and 1")
        if not 0 < self.ema <= 1:
            raise ValueError("monitor.ema must be above 0 and at most 1")
        if not 1 <= self.choice_votes <= self.choice_window:
            raise ValueError("monitor.choice_votes must be from 1 to monitor.choice_window")
        for name in ("decision_base_url", "vocab_base_url"):  # keys and camera frames go there
            url = getattr(self, name)
            if not url.startswith(("https://", "http://127.0.0.1", "http://localhost", "http://[::1]")) \
                    and not private_http(url):
                raise ValueError(f"monitor.{name} must be https, or http on this machine or a private network")


@dataclass(frozen=True)
class Rollouts:
    """Rollout bundles (homebody/rollout): an import larger than `max_import_mb` unpacked is refused; `light` leaves
    sensors.npz and depth previews out of an export. The offline monitor (monitor/offline.py) asks the decision model
    every `offline_period_s` simulation seconds of a recorded rollout and stops at `offline_max_cost_usd` (0: no cap)."""
    max_import_mb: int = 4096
    light: bool = True
    offline_period_s: float = 0.5
    offline_max_cost_usd: float = 2.0


@dataclass(frozen=True)
class Server:
    """The harness server (homebody serve): it listens on `host`:`port` (this machine only unless changed; reach it
    over an SSH tunnel) and admits a client presenting the token in `token_file`. `on_disconnect` is what a running
    task does when the commanding client goes away: "pause" (hold until a client resumes or stops it) or "continue"."""
    host: str = "127.0.0.1"
    port: int = 8765
    token_file: str = "~/.homebody/server_token"
    on_disconnect: str = "pause"
    jpeg_quality: int = 85
    view_hz: float = 5.0  # camera images sent for the console's view, in wall hertz

    def __post_init__(self):
        if self.on_disconnect not in ("pause", "continue"):
            raise ValueError('server.on_disconnect must be "pause" or "continue"')
        if not 1 <= self.port <= 65535 or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("server.port must be 1-65535 and server.jpeg_quality 1-100")


@dataclass(frozen=True)
class Dashboard:
    """How long the operator dashboard lets sources disagree before saying so. `transition_s`: a running attempt and
    the executor's status may disagree this long while one catches up (a decision starting its skill, a skill
    ending); longer is a suspected mismatch. `stale_s`: a controller or server status older than this is stale, so
    the dashboard says Unknown rather than claim a mismatch. `ack_timeout_s`: a stop request unanswered this long
    shows as timed out (a later answer still shows). `stop_s`: an attempt still running this long after the executor
    took a stop is suspected of not stopping. The frame checks are ui/dashboard/frames.py's."""
    transition_s: float = 3.0
    stale_s: float = 8.0
    ack_timeout_s: float = 10.0
    stop_s: float = 20.0
    report_overdue_s: float = 15.0  # s without a report from the executor, while an attempt runs, before it is overdue
    frame_uniform_std: float = 2.0  # a camera frame's pixel spread below this is suspected invalid (ui/dashboard/frames.py)
    frame_repeat_s: float = 10.0  # simulated s the same camera image may repeat before it is suspected invalid


@dataclass(frozen=True)
class Robot:
    """The G1's hands: "dex3", Unitree's three-finger Dex3-1 (the robot in assets/robot), or "dex1", the Dex1-1
    parallel gripper (the robot profile assets/robot/dex1, built by tools/build_dex1_hand.py, with its own pick and
    place in skills/dex1)."""
    hands: str = "dex3"

    def __post_init__(self):
        if self.hands not in ("dex3", "dex1"):
            raise ValueError("robot.hands must be dex3 or dex1")

    def directory(self, root):
        """The robot's model, URDF and calibration folder under the release ROOT."""
        base = Path(root) / "assets/robot"
        return base if self.hands == "dex3" else base / self.hands


@dataclass(frozen=True)
class SharedMotion:
    arm_timeout: float = 15.0
    palm_tolerance: float = 0.015
    observation_age: float = 0.25


@dataclass(frozen=True)
class Navigate:
    """Drive speeds, stances and arrival. An arrival is verified within `arrival_distance`, or `arrival_distance + close_slack` once re-closed."""
    drive_speed: float = 0.35
    turn_speed: float = 0.45
    final_speed: float = 0.20
    loaded_drive_speed: float = 0.25
    loaded_turn_speed: float = 0.30
    min_close_speed: float = 0.15
    min_turn_speed: float = 0.15
    close_gain: float = 1.0
    heading_gain: float = 1.5
    turn_hold_gain: float = 1.0
    robot_radius: float = 0.28
    route_clearance_margin: float = 0.0
    stance_reach: float = 0.46
    stance_clearance: float = 0.15
    right_hand_offset: float = 0.14
    left_hand_offset: float = 0.14
    departure_distance: float = 0.30
    run_in_distance: float = 0.60
    aim_in_distance: float = 0.80
    goal_snap_max_distance: float = 2.0
    start_snap_distance: float = 0.15
    goal_snap_slack: float = 0.10
    facing_min_distance: float = 0.10
    waypoint_radius: float = 0.23
    route_lookahead: float = 0.50
    route_lateral_gain: float = 1.0
    turn_before_translate: float = 0.35
    stop_distance: float = 0.03
    arrival_distance: float = 0.10
    arrival_angle: float = 0.10
    settle_seconds: float = 1.0
    close_rounds: int = 3
    close_slack: float = 0.20
    close_min_gain: float = 0.02
    drive_timeout: float = 90.0
    stall_timeout: float = 6.0
    progress_distance: float = 0.03
    tick: float = 0.10


@dataclass(frozen=True)
class Grasp:
    """Grasp planning, approach and retention limits: the walk-in stands with the pre-grasp palm `grasp_range` from the shoulder, and a pick closes at least `squeeze` past where the fingers stopped."""
    shape_quantiles: tuple[float, float] = (0.02, 0.98)
    depth_max: float = 0.055
    depth_fraction: float = 0.5
    insertion_depth_scales: tuple[float, ...] = (1.0, 0.65, 0.3)
    approach_height: float = 0.10
    lift_height: float = 0.18
    lift_evidence: float = 0.05
    carry_palm_radius: float = 0.30
    carry_compact_min_delta: float = 0.03
    carry_compact_step: float = 0.01
    yaw_samples: int = 12
    finger_spread: float = 0.052
    pad_band: float = 0.015
    pad_band_points: int = 12
    approach_attempts: int = 2
    aperture_error: float = 0.008
    open_seconds: float = 1.0
    close_seconds: float = 1.5
    squeeze: float = 0.12
    target_shift: float = 0.08
    target_surface_tolerance: float = 0.015
    occlusion_min_rays: float = 0.4
    occlusion_missing_explained: float = 0.5
    occlusion_exposed_missing_max: float = 0.1
    occlusion_visible_match: float = 0.5
    occlusion_no_view_rays: float = 0.8
    wrist_camera: bool = True  # the hand's wrist camera when the head view loses the target
    hold_effort_fraction: float = 0.04
    hold_closure_gap: float = 0.04
    recovery_samples: int = 5
    recovery_confirmations: int = 3
    recovery_tick: float = 0.10
    recovery_separation: float = 0.08
    recovery_match: float = 0.05
    spline_out: tuple[float, ...] = (0.0, 0.10, 0.15)
    spline_over: tuple[float, ...] = (0.06, 0.10, 0.14)
    spline_descend: float = 0.08
    spline_samples: int = 4001
    spline_waypoints: int = 32
    spline_smooth_passes: int = 4
    spline_seed_bias: float = 0.75
    approach_start_lifts_m: tuple[float, ...] = (0.10, 0.20, 0.30)
    grasp_range: float = 0.36
    approach_short: float = 0.08
    approach_long: float = 0.03
    approach_across: float = 0.04
    approach_side: float = 0.24

    def __post_init__(self):
        if (min(self.grasp_range, self.approach_short, self.approach_long) <= 0
                or self.approach_across >= self.approach_side
                or min(self.approach_start_lifts_m, default=0) < 0):
            raise ValueError("Grasp range and approach band must be positive and ordered")


@dataclass(frozen=True)
class BlindFinish:
    """Limits on finishing a grasp whose target is partly visible or hidden by the arm: only the planned descent may continue, with the base and head still and the palm's travel budgeted."""
    partial_travel: float = 0.20
    occluded_reach: float = 0.30
    occluded_recency: float = 0.08
    travel_margin: float = 0.08
    palm_turn: float = 0.24
    base_turn: float = 0.10
    head_drift: float = 0.08
    head_turn: float = 0.24

    def __post_init__(self):
        if min(vars(self).values()) <= 0:
            raise ValueError("Blind finish limits must be positive")


@dataclass(frozen=True)
class Placement:
    release_height: float = 0.015
    withdraw_height: float = 0.03
    withdraw_back: float = 0.05
    withdraw_clearance: float = 0.03
    rest_tolerance: float = 0.35
    pull_step: float = 0.02
    max_pull: float = 0.20
    reach_margin_steps: int = 2
    release_drops: tuple[float, ...] = (0., 0.05, 0.10)
    release_twists: tuple[float, ...] = (0., 0.3, -0.3, 0.6, -0.6)
    # the walking policy's idle arm pose
    rest_arm_left: tuple[float, float, float, float, float, float, float] = (0.2, 0.2, 0., 0.6, 0., 0., 0.)
    rest_arm_right: tuple[float, float, float, float, float, float, float] = (0.2, -0.2, 0., 0.6, 0., 0., 0.)
    max_release_height: float = 0.30
    forward_reach: float = 0.42
    support_match: float = 0.04
    open_closure_limit: float = 0.15
    settle_count: int = 15
    tick: float = 0.10
    xy_tolerance: float = 0.12
    support_height_tolerance: float = 0.06
    still_distance: float = 0.01
    stable_frames: int = 3

    def rest_arm(self, side):
        return np.asarray(self.rest_arm_left if side == "left" else self.rest_arm_right, dtype=float)
