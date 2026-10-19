You are combining two versions of the built-in `{skill}` skill ({source}): each succeeds on train
episodes the other fails.

Read {workspace}/BRIEF.md first: it says how versions are judged and what you may change. Then
{workspace}/lessons.md (what earlier versions taught: do not repeat a change that already failed unless
you say what differs) and the inspiration programs under {workspace}/inspirations/ (other scored versions,
with their code against the current best: borrow what worked). Edit files in the current directory, a git
checkout of version {parent}; its own results on the train episodes are in {parent_results}. Do not write anything outside the current directory; read anything in it and in {workspace}. You may change only: {editable}.

The current directory is version {parent}. The other version is {other}: its results are in
{other_results}, and its code against {parent} is the diff below (it may also include differences that
come from an older baseline; port only {other}'s own behaviour change). Treat complementary outcomes as a hypothesis: {other} was
measured in its own time (possibly older scenes or sensors). Compare the two result tables episode by
episode, identify the separable component of {other} that plausibly causes its wins, and port that
component alone into {parent}, keeping {parent}'s successes; say which component you ported and which you
left out, and why. If the two
changes conflict, make the choice between them depend on what the skill observes, not on the scene.

Implement one testable causal intervention, with the coordinated changes across stages it needs (not
a minimal tweak that cannot change an outcome): nothing tied to particular scenes, objects, colours or
coordinates; no simulator ground truth. State the episodes it should change, how their traces should
change, and which successful episodes must stay unchanged. Extra refusals are not an improvement. Keep the other skills that use the shared
code working.
A built-in skill must stay loadable as the coding assistant's editable copy: its module may import only
what `ALLOWED` in src/homebody/session/skill_module.py lists (numpy, scipy, standard math and data
modules, homebody.skills, motion, helpers, toolkit, harness and primitives), never homebody.session,
backends or remote, and it reads no files; calibrated hand data reaches it as `planning.jaw_samples`.
tests/test_coder.py checks this.
Add or update unit tests for logic worth pinning. Do basic testing of your change only
(the simulator is not available here, and the full suite runs later on the remote): run the unit
tests you added or changed and the tests of the modules you touched, with
`MUJOCO_GL=egl PYTHONPATH=src:vendor {python} -m pytest -q -x -p no:cacheprovider <test files>`
(tests/test_skills.py covers the skills, tests/test_coder.py that they load as editable copies), and `{ruff} check src tests`; fix every failure and
finding. Compare floats with a tolerance. Do not run the whole suite or anything long.

Make the new code report when it acts: where it changes a decision, emit
`ctx.emit({{"type": "mechanism_active", "mechanism": "<short name>", ...}})` with the values that decided
it. The probe runs the candidate on the train episodes your note names under Expected effect and on one
episode per outcome of its parent; one that reports no activation and changes no outcome there is
rejected as inert, so name the episodes where the change should act.

When done, reply with a short note in this form:
Mechanism: <what each version contributes>
Change: <what you changed, file by file>
Expected effect: <which episodes should now succeed, and why>
Risks: <what could get worse, including for other skills>

<diff from {parent} to {other}>
{other_diff}
</diff>

<diagnosis of the current best version>
{diagnosis}
</diagnosis>
