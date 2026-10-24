You prepare a live monitor for a humanoid robot. A fast decision model will look at the robot's head camera a few
times a second and say which of your labels the robot is doing. Write the labels for the goal below.

Task: $task

Plan:
$tree

Goal to watch: $goal
$skill

Goals already done:
$done

Hands: $hands

Skills the robot may use:
$skills

Stages each skill has gone through so far:
$stages

What the monitor reported so far:
$findings

The first image is the robot's head camera now; the second, when present, is a top-down view of the space in
front of the robot.

Write:
- labels: at most $max_labels actions the robot may visibly be doing while it works on this goal. Each has
  - name: snake_case, unique;
  - kind: "progress" (a normal step toward the goal), "success" (the goal visibly achieved) or "failure"
    (something going wrong that the robot's own checks could miss: working on the wrong object, an object slipping
    or falling, a collision, knocking other things over, repeating a motion without effect);
  - description: one sentence a person could check in camera images;
  - stages: the stage names above that it matches, if any.
  Do not write unsafe, human_needed or none_of_these; they are always added.
- standing: for each done goal whose result a camera can see, the one visible fact that must stay true, as
  {"goal_id": ..., "text": ...}.
- expected_seconds: about how long this goal should take.

Reply with only one JSON object: {"labels": [...], "standing": [...], "expected_seconds": number}
