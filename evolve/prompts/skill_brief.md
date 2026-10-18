# Brief: make the built-in `{skill}` skill more robust

HomeBody drives a simulated Unitree G1 humanoid (MuJoCo; Dex3 three-finger hands; a head RGB-D
camera). Its built-in skills (`src/homebody/skills/`: navigate, pick, place) are what the agent calls
every turn, so a more robust skill helps every task. You are improving **{skill}** alone.

## The benchmark

Each episode is one generated scene (three families: an object in an open bin; a cup, plate or tray on
a counter; an object near a backsplash beside a basket). A scripted driver walks the robot to the
scene's stance (some episodes shift it by up to 8 cm), calls `{skill}` once with a click on the
object's visible pixels (the click the agent would give), then carries the object along the task's
carry path (in some benchmarks walking back to the pick stance afterwards) and stands still. No model is involved, and the physics is deterministic, so every
version runs exactly the same episodes. Scored from physics: 0 not lifted, 1 lifted but not held at
the end (dropped during the pick or the carry), 2 still held: success. An episode's value is its stage
/ 2. Episodes whose stance or view failed before the skill ran are not scored. End to end, the costliest
failure is a pick that reports OK and then loses the object on the walk: the object usually lands where
it cannot be reached again. A grasp that will not survive carrying should be made firmer. Only a call
that returns OK and keeps the object through the walk scores as held: when the skill refuses, nothing
walks, so an object it leaves in the hand counts as lifted, the same as one dropped on the way.

`results.md` lists every train episode: its stage, the skill's refusal code and message, lifted,
held and seconds. `episodes/<id>/trace.json` is the skill's own record of that episode: each stage it
entered, its plan outline (`skill_prepared`), its tracking and recovery events and its final result;
`result.json` has the scores. `history.md` lists earlier versions and their verdicts.

## How a version is judged

On held-out episodes you never see as well as the train episodes, compared episode by episode with the
current version: accepted when neither split's value falls by more than two standard errors and the
value over train and val together rises by more than one and a half; with value unchanged, also when the skill's seconds per
call fall by a fifth. So fix failure mechanisms that recur across many episodes, in general ways:
never add anything about particular scenes, objects, colours or coordinates.

## What you may change

Only `src/homebody/skills/{skill}.py` and the shared code under it (`src/homebody/motion/`,
`src/homebody/helpers/`, `src/homebody/toolkit/`) and tests. Anything else is rejected unseen. Other
skills use the shared code too, so keep their behaviour working. The skill reaches the robot only
through the skill `Context` (the Robot interface): camera frames, measured robot state, commands.
Never use simulator ground truth (object poses from the physics engine, scene files, the scorer).
The contact model and the robot are fixed: improve how the skill uses them (grasp choice, approach,
closure and squeeze, lift, verification, recovery), not the physics.
