Each episode is one generated room (family move_variety): a lone object near the edge of a pick table and,
1.8-3.2 m away in any direction, a second table with a large open tray, its front edge facing the pick table
or turned up to ~125 degrees, with floor obstacles near the straight route and sometimes two partitions
leaving a narrow gap (a route at the robot's clearance always exists). A scripted driver stands the robot at
the pick stance and picks the object with the built-in pick (setup: an episode whose setup fails is not
scored), then calls `{skill}` once to carry it to the stance in front of the tray; if it refuses, the
assistant gets one call ({coder_minutes} minutes, a few checks, two executes) limited to `{skill}`. Scored from
the final base pose and the physics: 2 arrived (within 0.25 m of the stance, facing the tray within 0.35
rad) still holding the object and upright (success), 1 holding but not arrived or arrived without the
object, 0 neither or the robot fell. The physics is deterministic, so every version runs exactly the same
episodes.
