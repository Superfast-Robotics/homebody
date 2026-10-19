# Brief: make the built-in `{skill}` skill more robust

HomeBody drives a simulated Unitree G1 humanoid in MuJoCo. The robot, its hands, sensors and calibration
are the ones the profile selects (see "This robot" below and the robot's calibration). Its built-in
skills (`src/homebody/skills/`: navigate, pick, place) are what the agent calls every turn, so a more
robust skill helps every task. You are improving **{skill}** (`{source}`) alone.

## The benchmark

{benchmark}

End to end, the costliest failure is a pick that reports OK and then loses the object on the walk: the
object usually lands where it cannot be reached again. When the skill refuses, nothing walks, so an object
it leaves in the hand is not held.

`results.md` lists every train episode: its stage, the skill's result code and message, seconds, and what
the robot's hand touched (`hand_contacts`, from the recorded physics). `episodes/<id>/` holds the skill's
own record (`trace.json`: each stage, its plan outline in `skill_prepared`, tracking and recovery events,
the final result), `result.json` (scores and `hand_contacts`), `arm_track.json` (every half second: the
arm joints the skill commanded, the rate-limited references the controller applied, the measured joints
and the base pose) and, for failed episodes, `keyframes/` (head and wrist camera images at each stage of
each skill call, `call-NNN-<stage>-head.jpg` / `-wrist-<side>.jpg`; open them to see what happened).
`hand_contacts`, `arm_track.json` and the keyframes are evidence for you only: the skill cannot read them.
`history.md` lists earlier versions and their train verdicts; `lessons.md` and `inspirations/` summarise
what earlier versions tried.

## How a version is judged

By {objective}. On held-out episodes you never see as well as the train episodes, compared episode by
episode with the current best: rejected when either split falls by more than two standard errors;
accepted when train and held-out together rise by more than one and a half standard errors (two and a
half when the skill becomes more than 30% slower per call); with the score shown no worse (within 0.02), also when the skill's seconds per call fall by a fifth. Episodes of one scene step (its stance variants) count as one sample when the standard error is computed. So fix failure mechanisms that recur across many episodes, in
general ways: never add anything about particular scenes, objects, colours or coordinates. More refusals
are not an improvement: an honest refusal only helps when it avoids a worse outcome.

## What you may change

Only these paths: {editable}. Anything else is rejected unseen, but you may read all of the repository
(the robot interface, controller, calibration and benchmark code included) to understand what the skill
can rely on. The skill reaches the robot only through the skill `Context` (the Robot interface): camera
frames, measured robot state, commands. Never use simulator ground truth in the skill (object poses from
the physics engine, scene files, the scorer, contacts). The contact model and the robot are fixed:
improve how the skill uses them (grasp choice, approach, closure and squeeze, lift, verification,
recovery), not the physics.
