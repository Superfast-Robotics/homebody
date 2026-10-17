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

### Navigation carry repair

`from homebody.toolkit.navigation_carry import recover_navigation`

`recover_navigation(ctx, epoch, side, reclose=True, probe=False)` stops and settles the base,
then tries unchanged shared recovery. For an already lifted, uncertain carry, `reclose` permits
one stationary squeeze at the configured closure and duration, only with blocked loaded contact
and no visible separation. Repeated fresh opposing retention can restore verification once;
it does not refresh geometry or relax subsequent transport checks. Effort without closure gaps
is insufficient. Requires `grip` in the skill's `NEEDS`.

`probe=True` independently permits one upward motion after fresh opposing retention, visibility
of the original label, and overlap with the original capture. IK, camera bounds and collision
checks use the union of the old and observed payload. The bounded joint stream checks live
clearance and retention. Capture refresh requires repeated fresh views with measured hand rise,
measured target rise, and matching palm-relative surfaces. A commanded lift or stationary
contact alone cannot pass. The lift size comes from existing lift-evidence and surface tolerances.
No opening, target substitution or unchecked arm motion occurs.

The result has `status` (shared `Recovery` enum), `frame`, `gate`, and `history`. Failed gates
include `opposing_retention`, `visibility`, `capture_overlap`, `probe_clearance`, measured-motion
gates and `freshness`. Action history lives on the carry record (`navigation_recovery`) across
invocations: reclosing and probing each run at most once, even after interruption. Do not clear
that history to retry. `probe_required` after a previous squeeze is an opportunity to enable
the independent probe, subject to its live gates. A passed probe establishes current possession,
not future transport stability; keep `Transit.check_carry` running throughout navigation.

Built-in navigation enables reclosing only. To enable the separate probe in a customized
navigate, change `Transit.recover_carry` to call
`recover_navigation(self.ctx, self.plan.epoch, side, probe=True)`. Shared pick/place recovery
is unchanged. The helper is for already lifted navigation loads, not initial grasp verification.

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
`toolkit.release.release_alternatives(palm, record, points, angle, height, margin)` returns
at most twelve world palm poses with unchanged payload XY, tilted about the attached payload
centre at the original height and ±height. Supply world depth excluding the held label;
use the existing angular/height search allowances. Visible geometry under each footprint
ranks poses by intrusion above the payload bottom, then height change. Missing depth is
neutral evidence, not proof of clearance or a refusal. Callers must still check IK and
loaded collision clearance. The place skill invokes this bounded search before pullback.
