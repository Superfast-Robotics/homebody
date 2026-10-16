Each episode is one generated scene (three families: an object in an open bin; a cup, plate or tray on
a counter; an object near a backsplash beside a basket), with a destination (a carton, basket or similar
receptacle). A scripted driver stands the robot at the scene's stance (some episodes shifted by up to
8 cm), picks the object with the built-in pick and carries it to the place stance; an episode whose
setup fails (stance, pick or carry) is not scored. It then calls `{skill}` once with the release the
task's script would give (x_ahead_m, y_left_m and delta_z_m over the destination, in the torso frame); if
it refuses, the assistant gets one call ({coder_minutes} minutes, a few checks, two executes) limited to
`{skill}`. Scored from physics by the evaluator's own placement test against the destination's target
volume: 0 still in hand, 1 released but not placed in the target (outside it, unsupported or still
moving), 2 placed: inside the target, released, supported and still (success). The physics is
deterministic, so every version runs exactly the same episodes.

What the baseline shows: the largest group of failures is place returning OK while the object lands
outside the target (in about a third of the train episodes that were not placed). The assistant is
called only when the skill refuses, so these are never rescued. A place that verifies its own result
from what the robot senses (where the object came to rest relative to the destination it aimed for)
and refuses when it missed would both report honestly and hand those episodes to the assistant;
better targeting of the release would avoid the miss in the first place.
