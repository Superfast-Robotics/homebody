## Robot API (`env`)

The robot is a Unitree G1 humanoid with two Dex3 three-finger hands and a head RGB-D camera. Arm
poses are 4x4 numpy arrays in the CURRENT torso frame (x forward, y left, z up, metres); they change
whenever the robot walks. Base goals and scene memory are world (map) coordinates. `side` is
"right" (default) or "left". Only `np` and `math` are available: no imports, files or network.
`print(...)` and `env.log(...)` go to your report.

Perception (head camera; entity ids come from the scene memory):
- `env.memory()` -> scene memory: [{id, label, location, anchor_xyz (world), portable}]
- `env.visible()` -> {entity_id: visible pixel count} for what is in view now
- `env.locate(entity)` -> {center, top_z, bottom_z, radius, extent, n_points} in the torso frame, or None if not visible
- `env.points_near(center, radius=0.25, exclude=())` -> (N,3) sensed points near a torso point (walls, neighbours)
- `env.state()` -> {base: [x, y, yaw] world, hands: {side: {palm, closure, holding}}}
- `env.to_world(xyz)`, `env.to_torso(xyz)` convert points between frames

Base:
- `env.navigate(goal_xy, facing_xy, exact=True)` -> {ok, code, message, base}: walk to a world goal facing a world point. exact=False lets HomeBody pick a clear stance square to the faced surface. Arrival counts within ~10 cm and ~6 degrees; a blocked goal returns NO_ROUTE.

High-level skills (HomeBody's own; each returns {ok, code, message}):
- `env.pick(entity, side)`: grasp a visible entity and lift it into a carry pose (walks to a grasp stance when needed)
- `env.place_on(entity, side)`: put the held object down on a visible support entity
- `env.place_at(x_ahead, y_left, delta_z=0, side)`: release the held object's centre at a torso-frame point
- `env.skill(name, **arguments)`: any skill with its raw arguments

Arm, hand and planning (for writing your own manipulation):
- `env.palm_pose(side)`, `env.arm_joints(side)`, `env.fk(joints, side)`
- `env.ik(pose, side, seed=None, position_only=False)` -> 7 joints or None
- `env.diagnose_ik(pose, side)` -> {ok, reason: solved/beyond_reach/orientation/no_solution, joints, shoulder_distance_m, reach_m}
- `env.make_pose(position, finger_direction, closing_axis)`: palm frame column 0 = fingers, column 1 = closing axis
- `env.grasp_poses(entity, side, limit=24)` -> calibrated top grasps [{pre, grasp, width, tier, jaw_in_palm}], best first
- `env.collides(joints, side, exempt=(), hand="current"|"open")`, `env.path_collides(start, end, ...)` -> None or reason
- `env.plan_path(pre_poses, goal_poses, side, exempt=())` -> cuRobo {ok, index, approach, descent, status}
- `env.follow(joint_waypoints, side)` (planned, no live check), `env.move_through(joint_waypoints, side)` (live-checked) -> None or why stopped
- `env.servo_to(pose, side, exempt=())` -> (reached_pose, message)
- `env.grip(closure, side)` (0 open .. 1 closed .. 1.12 squeeze), `env.holding(side)`
- `env.wait(seconds)`, `env.log(...)`, `env.trace(what, **data)`
