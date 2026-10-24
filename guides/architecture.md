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
suspected invalid. The check keeps each state by simulation time, so a monitor reading carries the quality of the
frames at its own observation (the simulation time of the newest frame it read, within 1 s), not the camera's quality
when the reading arrived; a reading without a simulation time is checked at receipt and says so. Transport, freshness
and usability are shown apart.

**A monitor result and its evidence** are four separate properties, never one number (`static/js/assessment.js`): the
monitor's output (its score and what the score is: the decision model's uncalibrated answer probability, which the
rules average before acting), the model confidence (the top activity label's probability, labelled uncalibrated), the
input quality (above), and the assessment's reliability (Unknown: nothing checks the monitor independently). A
degraded input adds a warning beside the score; the score is never changed.

| Reading field | Meaning (V8 name) |
| --- | --- |
| `id` | `observation_id`, set by the store |
| `t`, `received`, `sim_time`, `lag_s` | `observation_timestamp` (dashboard clock), receipt, simulation time of the newest frame read, monitor lag |
| `input_source` | what the monitor read (the robot cameras' newest frames) |
| `frame_quality`, `quality_reason`, `quality_source` | `input_quality_status` and `quality_check_source` |
| `risk`, `level`, `signal`, `semantics` | `monitor_output` and `monitor_score_semantics` |
| `confidence`, `confidence_semantics`, `calibration` | `model_confidence`, `confidence_calibration_status` |
| `reliability` | `assessment_reliability_status` (unknown) |
| `scope`, `scope_goal` | what the reading is about (below) |

**Risk scope** follows the monitor's own rules (`monitor_feed.scope`): a reading taken during a skill call watches that
attempt (an abort stops only it), a standing-claim signal (`undone:<goal>`) and a reading during planning are about a
goal, anything else is unknown. Attempt-scoped risk never carries into the next attempt (`execution.continuousRisk`); a
goal's continues across its attempts, and task, robot or environment scope would continue too. The summary names the
context ("Stall risk high for 9 s · Pick attempt 4"); a previous attempt's risk is said to be the previous attempt's,
with the current one's assessment unknown. The risk tracks mark each attempt change with a boundary at the Skill
track's times.

**Investigation Mode** (`static/js/investigation.js`): opening an incident turns the workspace into recorded camera,
behavior tree and the incident inspector side by side (45 / 25 / 30 %); the inspector takes the robot panel's place
(`drawer.js` docks it), so it never covers the tree, and the monitoring timeline stays below. Narrower screens give the
tree and the inspector one column with tabs; a tablet stacks them. One toolbar under the header says what is inspected
and holds the only Return to Live, which restores the saved live workspace (the camera's width and its Auto or chosen
view), the live camera, the tree following execution and the rolling window. The camera's own "Live camera" shows the
stream in that panel without leaving the investigation. Nothing in the mode sends a robot command.

Investigation Mode adapts to its incident's evidence once, when the incident is selected (from the recorded frames'
index, so nothing waits on a load and moving the cursor never resizes panels): without a recording the camera column
becomes a compact placeholder and the tree and the inspector share the space; Expand camera gives it back for that
incident. A stale live camera does not count. Previous and Next walk the incidents by their own times
(`selection.incidentSequence`). While anything is inspected the header's controls are labelled "Live robot": they act on
the current execution (the store's running attempt), never on the selection.

**Timeline paint** (`static/js/timeline_style.js`) keeps three layers apart: the execution state is a left accent and an
icon (running, failed, succeeded, recovery), the selected incident's attempt and goal a quiet violet tint, the operator's
exact selection a thin border; labels are one high-contrast colour in every state, and nothing translucent is drawn over
text. The strongest emphasis is the cursor and the selected event's ring; risk and observation colours keep their
meaning under a selection. The skill track says "Pick · Attempt 4", the goal's attempt number the tree shows; the
skill's own call count and the execution id are named apart in the tooltip and the tree's technical view.

**Monitoring coverage** (`model.coverage`), shown while inspecting history: the task time a reading stands for (up to
three of its source's intervals) against the task time in the window, with the gaps and the readings on unusable
camera frames; without an established cadence it gives counts, not a percentage.

**Evidence attribution** (`static/js/evidence.js`): each piece of evidence says what it is to the detection. Detector
input is claimed only when an observation names the recorded frame the detector consumed (no monitor reports that yet);
a recorded robot-camera frame whose simulation time matches the observation's is a synchronized observation (the
history keeps each frame's simulation time); a frame near the moment on the dashboard's clock is contextual, with its
offset. Quality checks are attributed the same way (`quality_relation`): the dashboard's camera check is a "Robot camera
quality warning", and the detector's own input quality is Unknown.

**Request states** (`interventions.requestStatus`) are three axes, never one "timed out": the acknowledgement (awaiting,
acknowledged, acknowledged after deadline, deadline exceeded, rejected, ...), the command (pending, active, completed,
failed, cancelled, expired) and the physical outcome (unverified). The deadline is the request time plus
`ack_timeout_s`; an acknowledgement that arrives later joins its request, closed or not, and the exceeded deadline stays
marked on the timeline. "Expired" is a stage a controller can report (`{"type": "intervention", "stage": "expired"}`).

**Inspection** (`static/js/selection.js`) is one selection every panel follows, resolved by identity: intervention →
incident → attempt (its call) → plan node; the detecting observation (the incident's first, not the episode's newest);
the recorded robot-camera frame within two storage periods of the moment, else "Camera evidence unavailable". An
incident older than the live 60 s opens a recorded window around it. The tree highlights the attempt and unfolds its
parents while the rows keep their runtime state; its banner says the attempt's state then and now. Return to Live
restores the live camera, the tree following execution and the rolling window, and closes Investigation Mode. Inspecting
sends nothing to the robot.

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

**Commands while paused** (`Session.hold_for_operator`): while the operator's Pause lasts the robot waits and the
monitor goes on watching. Text sent meanwhile becomes the plan's next goal (`Plan.insert_operator`: an operator goal
before the first goal still to do, after earlier commands), and the pause ends so it runs; a skill frozen mid-motion
is stopped first (OPERATOR_STOPPED) so the command can run from a safe state. When the operator goal closes, done or
failed, the robot pauses again and waits. "continue" (or Resume) goes on with the rest of the plan, which never fails
because a command did. Without a plan the text goes to the agent's next prompt.

A stall (no progress for `stall_s` simulation seconds while a skill runs) is only flagged: the monitor labels it,
the agent's next prompt mentions it, and monitoring goes on. It never stops the skill, spends no abort, and raises no
alert; the monitoring summary says "Stalled for N s · <attempt> · monitoring continues".

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
* **Frames the monitor itself read.** The frame check sees the frames the dashboard receives, matched to a reading by
  simulation time; the monitor reads its own captures of the same camera, not provably the same frames. The inspector
  labels the recorded frame as context, not detection input. A monitor that sent its frames' ids (or their quality)
  would make the match exact.
* **Assessment reliability and calibration.** No component validates the monitor's assessments or calibrates its
  probabilities, so reliability is Unknown and scores and confidence are labelled uncalibrated.
* **Task-, robot- and environment-scoped risk.** This monitor defines attempt and goal scope only; the page handles the
  wider scopes when a monitor reports them.
* **Observation time on the monitor's clock.** A reading's time is when the dashboard received it, beside the
  simulation time of its newest frame; the monitor reports no wall-clock stamp of its own, so frame-level
  synchronization across systems is not claimed.
* **Detector input frames.** The monitor does not report which frames it consumed (or their quality), so no evidence is
  labelled detector input; the page supports it once an observation carries `detector_frame` and `quality_relation`
  `detector_input`.
* **Request expiry.** No component reports an expiry yet; a request ends when its skill's result arrives or the task
  ends. The store accepts an `expired` stage when a controller sends one.
* **Historical controller and tree state.** The controller's report and the plan node's status are current only; the
  inspector derives an attempt's state at a moment from its recorded start and end, and says so.
* **Event sequence numbers.** Events are ordered by arrival and their recorded times; a sequence number from the
  harness would make out-of-order detection exact after a reconnect.
