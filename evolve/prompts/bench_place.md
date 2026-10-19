Each episode comes from the benchmark's episode files (its scene families appear in results.md), with a
destination (a shelf patch, a staging table, a carton, a basket or similar). A scripted driver stands the robot
at the episode's stance (some episodes shift it by up to 8 cm), picks the object with the built-in pick and
carries it to the place stance; an episode whose setup fails (stance, pick or carry) is not scored, for every
version alike. It then calls `{skill}` once with the release the task's script would give (x_ahead_m, y_left_m
and delta_z_m over the destination, in the torso frame). A coding-assistant rescue happens only when the profile
enables it (then one call of {coder_minutes} minutes, a few checks and two executes, limited to `{skill}`).
Scored from physics by the evaluator's own placement test against the destination's target volume: 0 still in
hand, 1 released but not placed in the target (outside it, unsupported or still moving), 2 placed: inside the
target, released, supported and still (success). The physics is deterministic, so every version runs exactly
the same episodes.
