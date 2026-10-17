# Architecture

SIM and REAL2SIM share one skill contract, one Runner and the model providers in `vlm/`. Everything else is separate,
so neither release changes the other or inherits its retry, step-limit or recording policy.

| | SIM | REAL2SIM |
|---|---|---|
| Entry | `homebody` | `homebody real2sim explore` |
| Loop | `Session` in `session/session.py` | `ExplorationSession` in `session/exploration.py` |
| Robot | `MujocoBackend`, a `Robot` (`backends/mujoco.py`) | the G1: body host and driver in `backends/g1/`, offered to skills as live actions |
| Skills | `SKILLS` in `skills/registry.py` | `LIVE_SKILLS` in `real2sim/exploration/skills.py` |
| Settings | `session/config.py`, `configs/simulation.toml`, a scene's `settings.toml`, `--settings` | `real2sim/exploration/config.py`, `configs/exploration.toml`, `--settings` |
| Prompt | `vlm/agent_prompt.md` plus each skill's `PROMPT` | `real2sim/exploration/explore_prompt.md` |
| Recording | `runs/session-<time>-<id>/` | `runs/explore/` plus the sensor bag |
| Add a skill | one module in `skills/` and its name in `SKILLS`; with tunables, a settings group in `session/config.py` and `session/settings.py` (`tests/test_custom_skill.py` is a complete example) | one module in `skills/` and its name in `LIVE_SKILLS`; if the model may call it, also its branch in `vlm_pick.parse_decision` and in `decision_interpretation.py` (four places) |
| Add a setting | a field in `session/config.py`, never an environment variable | a field in `real2sim/exploration/config.py`, never an environment variable |
| Add a model, scene or backend | a `MODELS` row in `session/config.py` (a new service: a provider class in `vlm/` and a `PROVIDERS` entry); a folder under `assets/real2sim/` ([format](real2sim.md#homebody-scene-package)); a `Robot` implementation (`primitives/robot.py`) plus what CLAUDE.md lists for hardware | the same models; scenes come from [reconstruction](real2sim.md) |

A skill module in `src/homebody/skills/` declares `PROMPT`, `NEEDS`, `FAILURES`, `prepare`, `execute` and optionally
`OPTIONAL`, `SURFACE` and `abort`; `skills/contract.py` checks them (`declared`). `Runner.prepare` plans from one frame,
`Runner.execute` runs the plan against a `Context` (the skill's only robot handle), and each attempt ends in one
`Result` with a declared code. A skill never imports a backend or the app (shared click, raster, floor-network and
region geometry lives in `helpers/`): SIM skills reach `Robot`, exploration skills only their declared live actions
(`SURFACE = LiveActions`) and the explorer's `LiveWorld` in `skills/live.py`.

## Checked navigation and sensors

Checked navigation (`skills/navigate_real/`, separate from the SIM `navigate` skill) walks exploration routes:
`controller.py` prepares and follows them; `navigation.py` rechecks them, recovers a blocked departure, settles,
verifies arrival and checks facing, consuming primitives, never a backend (tunables: the `drive` and `gait` groups).
`pick_point` resolves the requested map coordinates and checks the current permission; a rejected destination stays
rejected even when recovery restores a usable departure. Primitives own acquisition and bounded motion:

| Owner | Responsibility |
|---|---|
| `primitives/localization/superodom.py` | SuperOdometry DDS readers, health continuity, registered pose and scan admission, dead reckoning, the exploration start frame |
| `primitives/localization/lidar.py` | Self filtering, floor exclusion and near-field distances |
| `primitives/cameras/context.py`, `color_stream.py` | Phone RGB client and its newest-frame colour stream subscriber, freshness, receipt-time image buffer and the live camera's lifetime |
| `primitives/navigation/` | The live actions (`live.py` is the `LiveActions` protocol) and observations skills see, the command envelope and live errors |
| `backends/g1/` | The DDS transport, the single motor writer and the body host that keeps the robot balancing |

Received and accepted observations carry separate counters and timestamps; a queued cloud keeps its observation time,
and a scan updates the live cloud or map only when it matches a buffered odometry sample of the same start-frame
generation. The phone image is uncalibrated RGB stamped at decode time, without metric depth or an exposure pose; SIM
`Frame` and `Measurement` come from `MujocoBackend` physics. No observation read starts or restarts a service.
