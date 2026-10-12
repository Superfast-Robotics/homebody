# Skill toolkit and robot interface reference

Skills live in `src/homebody/skills/<name>.py` and are listed in `skills/registry.py`. Each one
declares `PROMPT`, `ARGUMENTS`, `FAILURES`, `prepare(arguments, selected, current, planning)` and
`execute(plan, ctx)` / `abort` (see `skills/contract.py`). A skill reaches the robot only through
the `Context` (`ctx`), never the simulator.

## Context (`skills/contract.py`)

| Call | What it does |
| --- | --- |
| `ctx.observe(epoch)` | Current camera frame (`Frame`: rgb, depth, labels, intrinsics, `world_camera`, `world_torso`, hands, arm joints) |
| `ctx.measure(epoch)` | Current measurements without the camera (cheaper) |
| `ctx.advance(epoch, seconds=None)` | Let time pass (one servo tick by default) |
| `ctx.actions` | `base_velocity`, `arm_target`, `grip`, `stop` (see `primitives/robot.py`) |
| `ctx.arms[side]` | `Arm` kinematics: `forward(joints) -> (palm 4x4, J)`, `solve(pose, seed, position_only=False)`, `shoulder`, `reach`, `lower`, `upper` |
| `ctx.settings` | All tunables (`session/config.py`); skills may add their own settings group |
| `ctx.jaw_samples[side]` | The hand's calibrated jaw openings |
| `ctx.stage(name)` | Mark a skill stage (recorded) |
| `raise SkillFailure(CODE, message)` | End the skill with a declared failure code |

Shared motions: `motion/arm.py` (`servo_best_effort`, `move_joints`, `move_path(..., planned=True)`,
`close_or_open`, `reset_arm`), `motion/carry.py` (holding, tucking).

## Toolkit (`homebody.toolkit`)

| Function | Returns |
| --- | --- |
| `palm_in_torso(frame, arm, side)` | Current palm pose, torso frame |
| `diagnose_ik(arm, pose, seed)` | `{ok, reason: solved/beyond_reach/orientation/no_solution, joints, shoulder_distance_m, reach_m, ...}` |
| `reach_report(arm, poses, seed)` | Counts of those reasons over many poses |
| `clearance(frame, arms, side, settings, exempt_labels=(), hand="current"\|"open", carried=None)` | A collision gate: `.violation(joints)`, `.path_violation(a, b)` -> None or reason. `hand="current"` checks a preshaped hand as it is |
| `points_near(frame, center_torso, radius=0.25, exclude_labels=())` | Sensed points around a target (walls, neighbours) |
| `calibrated_grasps(frame, label, jaw_samples, settings.grasp, limit=48)` | `Grasp(tier, width, pre, grasp, jaw_in_palm)` list, best first |
| `trace(ctx, what, **data)` | Record a structured decision (`skill_trace` event) for later diagnosis |

cuRobo (`helpers/curobo.py`, needs the planning server): `CuroboPlanner(settings).plan(frame, side,
pre_world_poses, grasp_world_poses, exclude_labels)` -> `{success, index, approach, descent, status}`;
follow with `move_path(ctx, epoch, side, rows, planned=True)`.

Poses are 4x4 numpy arrays. Torso frame: x forward, y left, z up, metres. Palm frame: column 0
the finger direction, column 1 the closing axis.
