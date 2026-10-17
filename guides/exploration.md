# G1 exploration

Operator guide for the Linux/amd64 G1 + LiDAR + iPhone setup; the [SIM instructions](../README.md#-sim) are independent.
See also the [phone setup](phone-camera.md), [Real2Sim reconstruction](real2sim.md) and [Architecture](architecture.md).

## Quick run after installation
```bash
homebody real2sim explore --help
homebody real2sim explore check
homebody real2sim explore --plan --iface "$ROBOT_INTERFACE" --iphone-input "$PHONE_STREAM_URL" --iphone-relay-service "$PHONE_RELAY_SERVICE"
# Supervised setup: starts or reuses the video relay and the balance owner.
homebody real2sim explore setup --iface "$ROBOT_INTERFACE" --iphone-relay-service "$PHONE_RELAY_SERVICE"
# After the standing procedure and the phone broadcast are ready:
homebody real2sim explore --iface "$ROBOT_INTERFACE" --iphone-input "$PHONE_STREAM_URL" --iphone-relay-service "$PHONE_RELAY_SERVICE" \
  --budget-s 1200 --task "Explore the accessible parts of this room."
```

`--help`, `check` and `--plan` start no robot services, console, recording or provider call (`check` tests no robot
health, camera freshness, provider login or physical readiness).

| Option | Purpose |
|---|---|
| `--task "…"`, `--budget-s 1200` | The requested space, its progression and any intentional barriers; twenty minutes of active exploration, confirmed pauses not counted; walking starts only at **Start exploration** on **http://127.0.0.1:8099** |
| `--iface`, `--iphone-input`, `--iphone-relay-service`, `--iphone-endpoint` | Robot network interface (needed only when not exactly one interface has a `192.168.123.x` address), phone stream URL, the relay service to manage; or an already running receiver instead of `--iphone-input`, whose owner keeps its video storage |
| `--enable-ask-operator` | Let the model pause and ask a question; off unless given at launch (it can still end a run with `request_help`) |
| `--observe-only`, `--adopt` | Live sensors and preview without locomotion, not an offline check; keep an already healthy SuperOdometry instead of restarting it |

## Install
In the release checkout, with its Python environment active: fit the phone in [`hardware/g1_iphone_mount.stl`](../hardware/g1_iphone_mount.stl),
set up its [USB network and relay](phone-camera.md), run `homebody provider-check` (one real request; add `--model` if
you changed `agent.model`), and set `[robot]` and `[iphone]` in `configs/exploration.toml` or a `--settings` file:

```toml
[robot]
ssh_target = "unitree@192.168.123.164"  # the G1's SSH account and address
ssh_password = "123"                    # the Unitree factory password; use the robot's own
iface = "enp3s0"                        # this workstation's 192.168.123.x link; empty picks the only one
[iphone]
input_url = "rtsp://JETSON_LAN_IP:8554/iphone"
relay_service = "homebody-phone-rtsp.service"  # empty leaves externally managed video alone
hwaccel = "cuda"                               # decodes the phone video on the GPU
```

```bash
bash hardware/g1/install.sh --plan   # every command, password masked; runs nothing
bash hardware/g1/install.sh          # or: bash hardware/g1/install.sh --robot unitree@192.168.123.164 --password 123 --iface enp3s0
```

| Step | What it installs, and where |
|---|---|
| `prereqs`, `odometry`, `recorder` | Workstation: checks Docker, iproute2, OpenSSH, sshpass and FFmpeg; installs the UDP receive buffers (`99-homebody-udp.conf`, sudo); builds CycloneDDS 0.10.2 into `.cache/` and the `exploration` extra when the venv lacks `cyclonedds`; the `slam.image` SuperOdometry image, built from pinned public sources by `hardware/g1/superodom/build.sh` (pulled when you name another tag); the `recording.image` recorder that default full recording needs |
| `copy`, `phone` | Robot, only when `relay_service` names the shipped unit: `hardware/g1/phone/` into `~/homebody-phone` over sshpass and scp; the pinned MediaMTX v1.21.1, its configuration and `homebody-phone-rtsp.service`, enabled for boot but not started; the first `phone` run stops until you write the phone's USB address into `~/homebody-phone/phone.local.yml` on the robot, and an installed relay configuration is kept |
| `verify` | `explore check`, the offline image checks, robot SSH, the read-only LiDAR and IMU probes, and the SLAM and recorder status |

- Installation starts no robot motion, body host, relay or recording. A re-run skips finished steps; `--only <step>`
  re-runs one (`bash hardware/g1/install.sh --only recorder` after removing that image). Keep `CYCLONEDDS_HOME` set as
  the script reports and allow roughly 5 GB for images; the [build guide](../hardware/g1/superodom/README.md) has the
  source pins, serial-IMU variant and licensing.
- The Jetson's vendor LiDAR service, as commissioned, must publish the MID-360 streams on the factory Ethernet
  (`192.168.123.x`), DDS domain 0; a generic Livox publisher with other fields, units or frames does not work, and
  SuperOdometry runs under `/ws` on HomeBody's own unit conversion (feed it no converted values; run no second engine
  under that prefix). Read-only receipt probes (usage in the header of `run_superodom_ws.sh`):

```bash
# sensor_msgs/msg/PointCloud2, about 10 Hz; per-point FLOAT32 `time` in relative nanoseconds
hardware/g1/superodom/run_superodom_ws.sh probe /utlidar/cloud_livox_mid360 5
# sensor_msgs/msg/Imu, about 200 Hz; acceleration in g, angular velocity in rad/s
hardware/g1/superodom/run_superodom_ws.sh probe /utlidar/imu_livox_mid360 5
```

### Settings
Defaults in `src/homebody/real2sim/exploration/config.py` are overridden by the `[group]` tables of
`configs/exploration.toml`, then `--settings my.toml` (only what it names, e.g. `[agent]` `reasoning_effort = "high"`),
then command-line options; unknown groups and keys are refused. The only environment variable is
`HOMEBODY_G1_SSH_PASSWORD` (default: the Unitree factory password); runs record resolved values in
`meta/settings.json`, never read `configs/simulation.toml` and never fall back silently.

| Group | Holds |
|---|---|
| `exploration`, `agent` | task, budget (`budget_s`), console `host` and `port`, `output` root, `record`, `observe_only`, start zone and spin, completion fraction; `model`, `reasoning_effort`, `service_tier`, `timeout` (flags `--model`, `--reasoning-effort`, `--service-tier`, `--decision-timeout-s`) |
| `robot`, `body`, `iphone` | the interface and SSH target; the body host's address, CPU set, sudo, threads and writer lock; stream `input_url`, receiver `endpoint`, `relay_service`, `max_age_s`, `post_action_buffer_s`, `hwaccel` |
| `slam`, `recording`, `hands`, `lidar`, `map`, `gait`, `drive` | the SuperOdometry and recorder images, odometry hold, free-space minimum, frame rates; finger servo gains, self-filter radii, map extent and pictures, walking velocity commands, the navigation controller |

## Hardware setup and launch
- Setup and launch are supervised hardware operations (follow the robot's procedure and setup's output for the remote
  Y initialization and A standing sequence); a launch can open the hands and restart perception and SLAM.
- If setup refuses an occupied body-host address (default `127.0.0.1:29400`), another process may still be balancing
  the robot: secure it with physical support, stop its policy with remote **B**, run
  `PYTHONPATH=src:vendor python -m homebody.backends.g1.body_host stop` from the checkout that started the old host
  (refused while the policy is active or initializing), and retry once its process has exited and
  `ss -lunp 'sport = :29400'` shows no listener. Do not change the address or remove a lock to bypass the conflict.
- The console has no authentication: keep it on localhost or a trusted forward; expose it with `--host` only by intent.
- **Start exploration** surveys and lets the model set and revise its region; **Draw the region myself** fixes a
  boundary (**Let the model set the region** reverts) that is task scope, not proof of free floor. *Reachable floor*
  (light red) is where the robot can route; every destination is route-checked again from the actual pose. Barriers
  stated in the task or an answer ("stay on this side of the chair barrier") are instructions, not geometry.
- **Pause** holds and **Resume** rechecks the destination; an **Answer** (**Send answer and resume**) triggers a new
  decision but never moves the robot or edits the region. Paused or unanswered time is not budgeted; DONE or an
  expired budget (recorded as incomplete) holds in place.

## Stopping
- **STOP**, Ctrl-C or process termination ends the attempt and closes the console and recording while the body host
  keeps balancing; STOP is terminal, so use Pause for a cable adjustment.
- Stopping exploration does not sit the robot, release the legs or end the balance owner; use the G1's established
  procedure for those. Do not kill the body host or remove its writer lock (`body.writer_lock_path`) to force a restart.
- In the launch terminal **Ctrl-\\ is the separate immediate halt** (stops the policy, releases joint torque, ends
  the body host; not a graceful stop or sitting command), so follow the robot's supervised emergency procedure.
- Body loss, a failed hold or a frame reset blocks automatic resumption; power loss or SIGKILL may skip finalization.

## Where the run is
A run is a start-time directory under `runs/explore/` or `--output` (`output` in `[exploration]`). `--record` (the
default) keeps sensor MCAP, native phone video, clocks, gallery and model evidence, route, events, capture health,
settings and manifest; `--no-record` keeps only configuration, status and performance diagnostics, and planning still
sends evidence to the provider, so it is not offline. The recorder refuses to start below `recording.min_free_gb`
(40 GiB by default). STOP finalizes the archive; keep the **closed** original: only `homebody real2sim export` or
reconstruction make the portable package (schema in `configs/capture/`; see the
[audit](real2sim.md#audit-a-closed-run-first) and [skill](../.agents/skills/homebody-real2sim/SKILL.md),
`$homebody-real2sim`). `runs/` is not under source control. Code: `src/homebody/session/exploration.py`,
`src/homebody/real2sim/exploration/`, `src/homebody/motion/exploration/`, `src/homebody/backends/g1/`.
