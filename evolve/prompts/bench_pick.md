Each episode is one generated scene (three families: an object in an open bin; a cup, plate or tray on
a counter; an object near a backsplash beside a basket). A scripted driver stands the robot at the
scene's stance (some episodes shifted by up to 8 cm), calls `{skill}` once on a click on the object; if
it refuses, the assistant gets one call ({coder_minutes} minutes, a few checks, two executes) limited to `{skill}`.
A held object is then carried along the task's carry path. Scored from physics: 0 not lifted, 1 lifted
but not held at the end, 2 held: success. The physics is deterministic, so every version runs exactly
the same episodes.
