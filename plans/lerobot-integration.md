# LeRobot integration plan

Goal: the HomeBody harness and its skills run every Hugging Face LeRobot robot in simulation, on both
MuJoCo and ManiSkill, and each robot is tested on a small, robot-specific subset of the val tasks just by
changing the robot.

Base: `superfast-agent` (v007: plan mode, traces, fix library; train 0.333, val 0.526), rebased onto
upstream Stanford-TML/homebody `main` (the real2sim release); see "Upstream" below.

## Decisions

| Topic | Decision |
|---|---|
| Simulators | MuJoCo and ManiSkill 3, both behind LeRobot's `Robot` interface; results reported per simulator |
| Configs | LeRobot style: draccus dataclasses, `RobotConfig.register_subclass`, chosen on the command line (`--robot.type=... --robot.backend=maniskill`) |
| Skills | A separate library in this repo, `packages/lerobot_skills/`, shaped like LeRobot policies: `SkillConfig` + `Skill.reset/select_action(observation) -> action` over LeRobot feature dicts |
| Robots | All tiers, except robots with no public URDF/MJCF (skipped: HopeJR, EarthRover) |
| G1 | Our current G1 (direct MuJoCo, AMO walking, Dex3 hands), refactored as a LeRobot-style `homebody_g1` robot with LeRobot's `unitree_g1` joint names plus hand and walking keys. Not LeRobot's own `unitree_g1` sim (no hand actions, DDS bridge) |
| G1 walking | AMO on both simulators. If AMO can't walk stably in ManiSkill after tuning, stop and decide about Holosoma's ONNX walker |
| G1 Dex1 | Model from `lerobot/unitree-g1-mujoco` (no licence tag): internal tests only, not redistributed |
| Fix library | G1 only; other robots start empty |
| Cameras | Arms without a head camera get a fixed front scene camera; the wrist camera only when needed (same rule as G1) |
| Val subset | pick-place, carry, small shelf; about 5 tasks per robot, scaled to reach, gripper opening and payload; 1 repeat; task lists reviewed before running |
| Pipeline | Full v007 pipeline on every robot (plan mode, auto coder, traces) |
| Real hardware | `SimRobot` uses LeRobot's interface, so hardware is a later `--robot.type` swap; no hardware runs in this plan |
| Environment | New Python 3.12 environment on the second A10's local disk (LeRobot needs Python >= 3.12, numpy < 2.3; the shared 3.11 environment stays untouched). ManiSkill in its own environment if dependencies conflict |

## Status

| Phase | State |
|---|---|
| 0 | Done 2026-10-18 on loki-sim1: `~/lerobot-work/.venv` (Python 3.12) with lerobot 0.6.1, mujoco 3.11.0, mani_skill 3.0.1, sapien 3.0.3, numpy 2.2.6, torch 2.11.0+cu128 (CUDA available); MuJoCo renders through EGL, ManiSkill through Vulkan (SO-100 PickCube frame); the harness's 598 tests pass in it. Built by `setup_lerobot_env.sh` (local operator script). Notes: the hosts' 570 driver needs the cu128 torch wheels; draccus 0.11.6 ships a stray top-level `tests/` package that shadows ours, removed in the environment |
| 1 | Done 2026-10-19: `homebody_g1` (packages/lerobot_robot_homebody_sim) is the G1 as a LeRobot robot, observations and actions as dicts named after `unitree_g1` (`homebody.robots.g1`); `LeRobotSurface` reads it as the harness's Robot; `--robot.type` (mujoco default, homebody_g1) in the CLI, the scripted check and the skill benchmark. Checks: 611 tests on loki-sim1; the scripted four-object run is identical line for line with both robot types; the held-out split with homebody_g1 on loki-sim1 scored 0.481 against v007r's 0.477 (+0.008 +- 0.046). Draccus configs cover the robot choice; the harness settings stay TOML (upstream's one-home rule) |
| 2 | Done 2026-10-19: `packages/lerobot_skills` (SkillConfig as a draccus choice, Skill with reset / select_action / status, run_skill, PolicySkill for any LeRobot policy). `skills.runtime = "stepwise"` runs the G1's skills, and any the coder writes, in that form: the skill's loop runs on a worker thread, its commands batch into one action dict per read or advance, and every render and command happens on the caller's thread (MuJoCo's GL context is thread-bound). Decision: the G1's Dex3/torso-specific pick and place are wrapped, not rewritten; robot-agnostic arm skills are written natively in phase 4. Checks: the scripted four-object run identical line for line; the free pick benchmark's 34 val episodes identical field for field (stage, held, lift height, simulated seconds), 15 held in both, and no slower (3211 s vs 3564 s of episode time, run side by side) |

## Layout

```
packages/
  lerobot_robot_homebody_sim/  # LeRobot plugin (found by its lerobot_robot_ prefix): SimRobotConfig + SimRobot, homebody_g1
  homebody_sim/
    backends/mujoco/           # generalised from src/homebody/backends
    backends/maniskill/
    robots/<name>/             # model files + embodiment (end-effector frames, gripper mapping, base type)
    tasks/<family>/<robot>/    # per-robot val tasks
  lerobot_skills/              # skills library
src/homebody/                  # harness: agents, plan mode, coder, traces; talks only to Robot + skills
```

## Phases

| # | Phase | Reused | Done when | Credits |
|---|---|---|---|---|
| 0 | Second A10: pinned lerobot 0.6.1, mujoco, mani_skill; both simulators render | — | one frame from each | 0 |
| 1 | LeRobot-style configs; `homebody_g1` plugin; harness only uses the `Robot` interface | LeRobot `Robot`/`RobotConfig`, plugin discovery, camera configs; our plan mode, coder, recovery, tree | unit tests pass; G1 val subset within +-1 SE of v007 | 1 small run |
| 2 | `lerobot_skills`: pick, place, look, navigate, carry as stepwise `select_action` skills; LeRobot policy wrapper; coder writes in this form | `RobotKinematics` + its end-effector processor steps, mink, RynnMotion's phased pick-place, our VLA chunk interface (`vla-skills` branch) | free G1 pick benchmark matches v007 | 0 |
| 3 | Tier A on MuJoCo: SO-101, Koch v1.1, OMX, each a LeRobot robot; the first native robot-agnostic skills (pick and place for arms on `RobotKinematics` and mink) in `lerobot_skills` | Menagerie SO-101, ROBOTIS OMX, Koch v1.1 converted from ManiSkill's URDF; `lerobot/so101-tabletop-mujoco` as env template | reachability check + one scripted pick per robot | 0 |
| 4 | Tier B on MuJoCo: LeKiwi, bimanual SO-101, OpenARM | Ekumen LeKiwi (arm swapped to SO-101), `enactic/openarm_mujoco` | same, plus base motion | 0 |
| 5 | Tier C on MuJoCo: Reachy2, reBot B601, G1 Dex1 | `pollen-robotics/reachy2_mujoco`, Menagerie `seeed_rebot_devarm`, Hub G1 Dex1 | same | 0 |
| 6 | Per-robot val tasks | `mujoco_scanned_objects`; new small-scale shelf scene | task lists reviewed | 0 |
| 7 | MuJoCo runs: 11 robots x their tasks; SO-101 first, its cost reported before the rest | — | results table (MuJoCo) | ~55 episodes |
| 8 | ManiSkill backend: our own registered 29-DoF G1 agent with Dex3 hands (the URDF the MuJoCo side uses, joints mapped by name, not sliced), AMO run each physics substep (`before_simulation_step` + `set_qf`, fallback PD drives with AMO's gains), MuJoCo's armature, damping, friction and foot contacts carried over; then every other robot as a ManiSkill agent (built-ins for SO-100, Koch, XLeRobot) | ManiSkill agent skeletons, holosoma's PD/IMU/odometry bridge | the G1 stands 60 s, walks 10 m and turns without falling; then the G1 val split on ManiSkill; each robot's reachability check and scripted pick | 1 small run |
| 9 | ManiSkill runs: the same robots and tasks | — | results table (ManiSkill), beside MuJoCo's | ~55 episodes |

Order (2026-10-19, user decision): every robot runs on MuJoCo first; the ManiSkill backend comes last, so its
risk (no learned humanoid walker has run in ManiSkill) holds up nothing else.

## Model sources

| Robot | MuJoCo | ManiSkill |
|---|---|---|
| SO-100 / SO-101 | Menagerie `trs_so_arm100`, `robotstudio_so101` | built-in `so100`; SO-101 adapted from it |
| Bimanual SO-101 | two Menagerie copies attached | from the `xlerobot` agent |
| Koch v1.1 | converted from ManiSkill's URDF | built-in `koch-v1.1` |
| OMX | ROBOTIS `open_manipulator` MJCF | from the URDF |
| LeKiwi | Ekumen `lekiwi_sim` | from `xlerobot` |
| OpenARM | `enactic/openarm_mujoco` v1/v2 | from the flattened URDF |
| Reachy2 | `reachy2_mujoco_assets` (prototype) | from the xacro URDF |
| reBot B601 | Menagerie `seeed_rebot_devarm` (RobStride build) | from the URDF |
| G1 Dex3 / Dex1 | ours / `lerobot/unitree-g1-mujoco` | new agent from holosoma's 29-DoF URDF |

## Coupling to remove (from the repo audit)

1. Fixed shapes: 29 joints, `ARM` slices, 7-joint/3-finger `HandState`, `ActionLimits.arm(..., 7)`.
2. `Robot.arm_target(side, 7 joints)`, `grip(side, ...)` with side in {left, right}, mandatory `base_velocity`.
3. Arm kinematics tied to the G1 URDF chain and the torso frame.
4. `calibration.json` read from fixed paths (controller, collision, `__main__`).
5. Controller and scene hard-wired to AMO, the upright check, `SPAWN_HEIGHT`, `WRIST_CAMERA_*`.
6. `Physics` mixes MuJoCo, Dex3 and AMO settings; `Placement.rest_arm_*` are 7-tuples.
7. `MujocoBackend` constructed directly in `__main__`, `check_manipulation`, `evolve/skill_bench`.
8. `pick.py` (655 lines) and `place.py` (473) are run-to-completion loops: the largest job is splitting them into stepwise phases.
9. Prompts say humanoid/G1, head + two wrist cameras, torso-frame top-down.

## Risks

- AMO in ManiSkill has no precedent (medium-high risk, about 1-2 weeks of tuning); gated by the walking check.
- ManiSkill Vulkan leak / driver error after many envs in one process (issues #1474, #1477): reuse envs or recycle the worker.
- Scores from the two simulators are not directly comparable (different contact physics).
- This branch will drift from `superfast-agent`; Astra rounds keep running from there.

## Upstream

`superfast-agent` is rebased onto Stanford-TML/homebody `main` at `0db5c8d` (the real2sim release: a G1
exploration, capture and reconstruction stack). Three files conflicted and were merged by keeping both
sides: `pyproject.toml` (their `exploration` extra, our `planning` extra), `CLAUDE.md` (their layout plus our
harness row) and `src/homebody/skills/contract.py` (their `Runner`, with our `skill=` override for
coder-written skills and our documented `SkillFailure`). The rebase is checked by `verify_rebase.sh`
(every file upstream did not touch is unchanged, unit tests, lint) and by a v007 baseline re-run on it.

What upstream changes for this plan:

- **Skill surfaces.** A skill may declare `SURFACE` (the interface its `NEEDS`/`OPTIONAL` commands come
  from) and the `Runner` takes a registry. Phase 1-2 build on this: each robot's `SimRobot` is a surface,
  and a skill declares the capabilities it needs against it instead of a new mechanism.
- **Prepare/execute split** in the `Runner` (`prepare`, `execute`, `cancel`): the stepwise
  `select_action` skills of Phase 2 keep this lifecycle.
- **"Every number has one home, and no environment variable sets it."** The draccus configs of Phase 1
  follow it; the `HOMEBODY_SKILLS` override is gone (nothing set it), so `SKILLS` in the registry is the one list until Phase 2 gives each robot its skill set.
- **Exploration skills and the live context** (`session/live_context.py`, `skills/live.py`) are a second,
  asynchronous skill runtime; the skills library leaves it untouched.
- The evolve loop now copies `hardware/` (upstream's tests read it), and the seed environment carries
  upstream's test-only packages (`seed_test_deps.sh`).

## Evaluation hosts

`loki-behavior-gh` and `loki-sim1` (A10, same group volume). loki-sim1 needed its NVIDIA driver upgraded to
570.195.03 for the EGL library (`upgrade_host_driver.sh`, then `setup_eval_host.sh`). A baseline runs train
on the first host and held-out on the second; a round evaluates one candidate per host. The LeRobot work
(Phase 0) gets its own environment on loki-sim1's local disk and runs there when no evaluation is.
