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

## Interventions on the dashboard

A request to stop a skill attempt (from the live monitor or the operator's Stop skill) is tracked as one
intervention (`ui/dashboard/interventions.py`), and every stage the dashboard shows comes from the component that owns
it.

**Attempt identity.** The session numbers a task's skill calls as the monitor does (`Phase.call`) and stamps the number
(`call`) on every skill and intervention event; the monitor feed sends the call each reading watched. The dashboard's
attempt is its tool call, with `execution` = `<task id>/call-<n>` and `number`, its goal's attempt count. Readings,
incidents and requests bind to the attempt by that number, never to whichever attempt runs when they arrive: a late
reading of attempt 3 stays attempt 3's, a new attempt's anomaly is its own incident, and an event that names no call
binds to none ("Unknown attempt"). The plan goal is the retry boundary: after a confirmed stop, the goal's next
attempt is recorded as the retry, with its outcome.

**Execution-state consistency** (`static/js/execution.js`) compares the tree's running attempt, the executor's status
(and when it last changed) and the controller's own report (the monitor app reads the controller plugin from the
simulation server; a local run's controller is in-process). Tolerances are `[dashboard]` settings (`transition_s`,
`stale_s`, `stop_s`, `ack_timeout_s`): a disagreement shorter than `transition_s` is a transition, a longer one a
suspected mismatch, a controller that reports stopped or failed under a running attempt a confirmed one (critical
while a stop on that attempt is open), and a controller status older than `stale_s` is Unknown, never a mismatch.

**Execution certainty** (`executionCertainty`) adds the executor's report freshness (`report_overdue_s`): an attempt
whose executor has not reported for that long is "execution status uncertain", never stopped or moving; its severity
grows with how overdue it is, with stale monitoring, and with an open stop on that attempt.

**Camera frames** (`ui/dashboard/frames.py`) are checked as they arrive: blank, dimensionless or undecodable frames are
invalid; a nearly uniform frame (`frame_uniform_std`) or the same image for `frame_repeat_s` of simulated time is
suspected invalid. Each monitor reading records the frame quality when it arrived, so a verdict on unusable frames is
marked. Transport, freshness and usability are shown apart.

**Task impact of a degraded service** (`ui/servers.py`) comes from established links only: the harness the dashboard
drives runs the task, and a simulation server is a critical dependency when that harness reports being attached to
its endpoint; without that report the impact is unknown.

| Stage | Reported by | Event |
| --- | --- | --- |
| Detected, decided, requested | Live monitor | a reading whose verdict is `abort_skill` or `reopen_goal` on the skill call it watched (`monitor_feed.py`: `call`, `request`) |
| Requested (operator) | Dashboard | the operator's Stop skill (`Controls.stop_skill`) |
| Accepted | Skill executor (session) | `{"type": "intervention", "stage": "accepted"}` when the skill's cancellation check takes the request (`Session.interruption`) |
| Confirmed, failed | Skill executor | the skill's result: `MONITOR_STOPPED` / `OPERATOR_STOPPED` confirms; another code after acceptance fails it |
| Cancelled | Skill executor, task service | the skill ended, or the task ended, before the request was taken |
| Rejected | Harness | a remote stop for a skill call that already ended (`MonitorLink.client_stop`) |
| Timed out | Dashboard (derived) | no acknowledgement within `ack_timeout_s`; a later confirmation still shows |
| Retry started, finished | Plan goal, skill executor | the stopped attempt's goal makes its next attempt, and how it ended |

The plan node's status stays the tree runtime's: a request is shown beside a running call, never as its failure.

What the dashboard cannot show yet, and shows as Unknown or Unverified instead:

* **Physical stop.** No base velocity, arm motion or safety-state telemetry reaches the dashboard, so a stopped skill
  is "confirmed by the executor" and the robot's motion stays "unverified".
* **Decision pending.** The monitor decides in the same step it detects; a policy engine that deliberates would need
  its own event.
* **Controller heartbeat.** Controller responsiveness is the age of the skill executor's last report (a stage, a
  result, an acknowledgement); a skill computing for a minute reports nothing. A heartbeat from the executor would
  separate "busy" from "unresponsive".
* **Recovery link.** After a stop the agent chooses the next step; no event links that decision to the intervention,
  so the inspector shows it as the next decision, not as a recovery the intervention caused.
* **Trace ids.** Interventions, incidents, tool calls and plan nodes are linked by the dashboard's own ids and the
  skill call number; a trace id shared with the harness's evidence (`events.jsonl`) would let the inspector open the
  recorded trace.
* **Controller execution state.** The controller plugin reports running, stopped or failed, not which skill it
  executes; so "controller executing while the tree finished" cannot be judged, and is not claimed.
* **Remote call numbers.** A client's monitor numbers calls as the harness does as long as no `skill_started` message
  is lost; the client keeps the harness's number beside its own (`HarnessClient.calls`) but readings carry the client's.
* **Controller on `homebody connect`.** The harness's controller status is not sent to a connect dashboard, so its
  Controller row is Unknown; the monitor app reads it from the simulation server.
* **Frames the monitor itself read.** The frame check sees the frames the dashboard receives; on a local run the
  monitor reads the session's own captures, which are the same camera but not provably the same frames.
* **Event sequence numbers.** Events are ordered by arrival and their recorded times; a sequence number from the
  harness would make out-of-order detection exact after a reconnect.
