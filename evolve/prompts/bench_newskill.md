Each episode is one generated scene: an open table in front of the robot with the task's objects (and a flat
marker zone or a shallow tray where the task names one). A scripted driver stands the robot in front of the table
(for a hand-over it first picks the object with the built-in pick: setup, not scored), then asks the assistant to
create a new skill for the task (a name, a goal and a success test; nothing tried before) with one call
({coder_minutes} minutes, a few checks, two executes). The kinds of skill: train asks for pushing an object onto a zone
without lifting it, rotating an object in place by an angle, stacking one object on another, and lifting an object
high and holding it; validation asks for clearing a zone of several objects and lining an object up beside another
at a gap; the test set asks for other kinds. Scored from physics by the task: 2 done (as the success test says), 1
part-way (for example half-way to the zone, half the turn, released near the target, lifted but not high enough),
0 not. The physics is deterministic, so every version runs exactly the same episodes.
