Higher-level primitives a skill module may use: `from homebody.harness import Env`, then in
`execute(plan, ctx)` build `env = Env(ctx, frame)` with the frame the skill was called on (keep
prepare's `selected` in the plan). Arm poses are 4x4 numpy arrays in the CURRENT torso frame (x
forward, y left, z up, metres), which moves whenever the robot walks; base goals are world (map)
coordinates; `side` is "right" (default) or "left". Things are chosen by clicks [x, y] (0-1000) on
that frame's head image.

Perception:
- `env.target(click)` -> a Target for the thing under the click, or None
- `env.visible()` -> [{target, click, pixels}] for what is in view now
- `env.locate(target, camera="head")` -> {center, top_z, bottom_z, radius, extent, n_points} (torso frame) or None;
  `camera="left"` / `"right"` reads that hand's wrist camera, rendered only on request: use it when the head
  view cannot see the target (the arm in the way, the inside of a container)
- `env.click_of(target)` -> its middle as a click on the current head image, or None
- `env.points_near(center, radius=0.25, exclude=(), camera="head")` -> (N,3) sensed points near a torso point
- `env.state()` -> {base: [x, y, yaw] world, hands: {side: {palm, closure, holding}}}
- `env.to_world(xyz)`, `env.to_torso(xyz)` convert points between frames

Built-in skills (each returns {ok, code, message}):
- `env.navigate(goal_xy, facing_xy, exact=True, hand="right")`: walk to a world goal facing a world point (exact=False lets it choose a clear stance)
- `env.pick(target, side)`, `env.place_on(target, side)`, `env.place_at(x_ahead, y_left, delta_z=0.0, side)`
- `env.skill(name, **arguments)`: any built-in skill with its raw arguments

Arm, hand and planning:
- `env.palm_pose(side)`, `env.arm_joints(side)`, `env.fk(joints, side)`
- `env.ik(pose, side, seed=None, position_only=False)` -> 7 joints or None
- `env.diagnose_ik(pose, side)` -> {ok, reason: solved/beyond_reach/orientation/no_solution, joints, shoulder_distance_m, reach_m}
- `env.make_pose(position, finger_direction, closing_axis)`: palm frame column 0 = fingers, column 1 = closing axis
- `env.grasp_poses(target, side, limit=24)` -> calibrated top grasps [{pre, grasp, width, tier, jaw_in_palm}], best first
- `env.collides(joints, side, exempt=(), hand="current")`, `env.path_collides(start, end, side, exempt=(), hand="current")` -> None or reason
- `env.plan_path(pre_poses, goal_poses, side, exempt=())` -> cuRobo {ok, index, approach, descent, status}
- `env.follow(joint_waypoints, side)` (planned path, no live check), `env.move_through(joint_waypoints, side)` (live-checked) -> None or why stopped
- `env.servo_to(pose, side, exempt=())` -> (reached_pose, message)
- `env.grip(closure, side)` (0 open .. 1 closed .. 1.12 squeeze), `env.holding(side)`
- `env.wait(seconds)`, `env.log(...)`
