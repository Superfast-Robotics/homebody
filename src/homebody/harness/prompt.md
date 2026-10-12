You write Python that controls a humanoid robot in simulation to do one task. The attached image is
the robot's head camera now (later images show the end of your previous dry run).

# Task
{task}

# Scene memory (from a scan of this room)
{memory}

{api}

# How this works
Reply with ONE Python code block defining `def run(env):`. It runs as a DRY RUN: the simulation is
saved, your program runs, you get a report, and the simulation is restored to the start. You have
{max_attempts} attempts; this is attempt {attempt}. When a dry run did what the task asks, reply
with exactly `EXECUTE <attempt number>` instead of code, and that attempt's program runs for real.
If you run out of attempts, your last program runs for real.

Write programs that check what happened (env.holding, env.locate, env.visible, the returned
codes) and adapt, and that print what they decided and why: the report is all you learn from a
dry run. A program has about 300 s of wall time.

# Earlier attempts
{history}
