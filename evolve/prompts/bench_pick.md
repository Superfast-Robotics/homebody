Each episode comes from the benchmark's episode files (its scene families appear in results.md). A scripted
driver stands the robot at the episode's stance (some episodes shift it by up to 8 cm) and calls `{skill}`
once with a click on the object's visible pixels, the click the agent would give. A coding-assistant
rescue happens only when the profile enables it (then one call of {coder_minutes} minutes, a few checks
and two executes, limited to `{skill}`). A held object is then carried along the episode's carry path,
and back to the stance when the episode says so; the carry counts only when every leg arrives. Scored
from physics: 0 not lifted, 1 lifted but not held at the end (dropped, refused after lifting, or a carry
that did not arrive), 2 held through the whole carry. The physics is deterministic, so every version runs
exactly the same episodes. Episodes whose stance or view failed before the skill ran are not scored.
