# LeRobot integration plan

Goal: the HomeBody harness and its skills run every Hugging Face LeRobot robot in simulation, on both
MuJoCo and ManiSkill, and each robot is tested on a small, robot-specific subset of the val tasks just by
changing the robot.

Base: `superfast-agent` (v007: plan mode, traces, fix library; train 0.333, val 0.526). Upstream
Stanford-TML/homebody `main` is not merged yet (see "Upstream" below).

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
| 3 | ManiSkill backend; new 29-DoF G1 with AMO driven each physics substep (`before_simulation_step` + `set_qf`) | holosoma `g1_29dof.urdf` and its PD/IMU/odometry bridge | stands 60 s, walks 10 m and turns without falling; then the G1 val subset on ManiSkill | 1 small run |
| 4 | Tier A: SO-101, Koch v1.1, OMX | Menagerie SO-101, ManiSkill Koch, ROBOTIS OMX; `lerobot/so101-tabletop-mujoco` as env template | reachability check + one scripted pick | 0 |
| 5 | Tier B: LeKiwi, bimanual SO-101, OpenARM | Ekumen LeKiwi (arm swapped to SO-101), ManiSkill XLeRobot agent, `enactic/openarm_mujoco` | same, plus base motion | 0 |
| 6 | Tier C: Reachy2, reBot B601, G1 Dex1 | `pollen-robotics/reachy2_mujoco`, Menagerie `seeed_rebot_devarm`, Hub G1 Dex1 | same | 0 |
| 7 | Per-robot val tasks | `mujoco_scanned_objects`; new small-scale shelf scene | task lists reviewed | 0 |
| 8 | Runs: 11 robots x 2 simulators x their tasks; SO-101 first, its cost reported before the rest | — | results table per simulator | ~110 episodes |

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

Stanford-TML/homebody `main` has two commits after our base (`c09fcf3` README dates, `0db5c8d` real2sim
release: 241 files, mostly a new real-robot exploration stack). A trial merge conflicts in `CLAUDE.md`,
`pyproject.toml` and `src/homebody/skills/contract.py`. Upstream refactored the skill `Runner`
(prepare/execute split, injected registry, `SURFACE`/`OPTIONAL` declarations); the simulator skills,
backend and session code are otherwise unchanged. Not merged for now.
