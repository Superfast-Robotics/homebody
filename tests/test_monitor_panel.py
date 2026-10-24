"""The console's monitor panel: the plan with its marks, the reading against the threshold, the labels, the
timeline and the alert; and the Resume button of a monitor pause."""
from dataclasses import replace

from fakes import mocked_console

from homebody.monitor.monitor import View
from homebody.monitor.targets import Label, Phase, Reading, Verdict, Vocabulary
from homebody.session.config import Monitor
from homebody.ui.monitor_panel import monitor_html

VOCAB = Vocabulary("g2", (Label("walking", "progress", "walks"), Label("dropped", "failure", "the cup falls")))
TREE = "sequence\n  [done] g1: pick the cup | check: name=holding hand=left\n  [ACTIVE ⚠ monitor: dropped p=0.93] g2: carry"


def view():
    reading = Reading("g2", 12.0, activity={"dropped": 0.93, "walking": 0.05, "none_of_these": 0.02}, unsafe=0.01,
                      human_needed=0.0, still_true={"g1": 0.1}, progress={"no_change": 0.7, "some": 0.2, "clear": 0.1},
                      phase=Phase("tool_call", "navigate", "Following route"), latency_s=0.5, lag_s=0.6)
    alert = Verdict("abort_skill", "g2", "dropped", 0.93, "the cup falls", 12.0)
    return View(status="watching", goal_id="g2", tree=TREE, phase=reading.phase, vocabulary=VOCAB, reading=reading,
                history=((11.0, "walking", 0.8, "continue", "progress"), (12.0, "dropped", 0.93, "abort_skill", "failure")),
                verdicts=(alert,), alert=alert, period_s=0.6, p95_s=0.5, calls=12, cost_usd=0.0031,
                spans=((2.0, 9.0, "pick", "g1", "OK"), (10.0, None, "navigate", "g2", None)),
                goals=((0.0, "g1"), (9.5, "g2")))


def test_the_panel_shows_the_alert_the_reading_and_the_labels():
    html = monitor_html(view(), tau=0.8, tau_unsafe=0.9)
    assert 'class="op-alert is-stop"' in html and "Stopped the skill in g2 at 12 s" in html and "the cup falls" in html
    assert "width:93%" in html and "left:80%" in html and "g1 no longer holds" in html and "no progress" in html
    assert 'class="is-failure is-now">dropped' in html and 'class="is-static">human needed' in html
    assert "12 readings, $0.003" in html
    assert "No alerts" in monitor_html(replace(view(), alert=None))


def test_the_panel_escapes_model_text():
    bad = Vocabulary("g2", (Label("<b>x</b>", "progress", "<script>"),))
    html = monitor_html(View(vocabulary=bad))
    assert "<b>x</b>" not in html and "&lt;b&gt;" in html


def test_the_console_draws_the_panel_and_shows_resume_only_while_paused(monkeypatch):
    console = mocked_console(monkeypatch, monitor=Monitor(enabled=True))
    console.monitor_update(view())
    assert "op-alert" in console.monitor_panel.content and console.dismiss_button.visible is True
    console.update("Paused", "The monitor asks for a person")
    assert console.resume_button.visible is True and console.submit_button.disabled
    assert "Paused" in console.strip.content
    console.update("Executing", "Walking.")
    assert console.resume_button.visible is False and "Running" in console.strip.content


def test_the_console_without_a_monitor_has_no_panel(monkeypatch):
    console = mocked_console(monkeypatch)
    console.monitor_update(view())
    assert console.monitor_panel is None and console.resume_button is None and console.pause_button is None


def test_the_timeline_draws_goals_skills_readings_and_verdicts_on_one_axis():
    from homebody.ui.monitor_panel import timeline_bar, timeline_svg
    svg = timeline_svg(view())
    assert "g1 from 00:00" in svg and "pick in g1: ok" in svg and "navigate in g2: running" in svg
    assert svg.count('width="2.5"') == 2 and "#C2271D" in svg  # two readings, the failure one red
    assert "monitor stopped: dropped" in svg and ">00:10<" in svg
    assert "going wrong" in timeline_bar(view()) and "fills in once a task" in timeline_svg(View())


def test_the_console_redraws_the_timeline_bar(monkeypatch):
    console = mocked_console(monkeypatch, monitor=Monitor(enabled=True))
    console.monitor_update(view())
    assert "op-recorder" in console.timeline.content and "navigate" in console.timeline.content


PLAN = {"tree": {"type": "sequence", "children": [
            {"type": "goal", "id": "g1", "goal": "pick the jam jar", "check": {"name": "holding", "hand": "right"}},
            {"type": "fallback", "children": [
                {"type": "goal", "id": "g2", "goal": "carry it to the cart", "check": {"name": "arrived"}},
                {"type": "goal", "id": "g3", "goal": "<b>place</b> it in the cart", "check": None}]}]},
        "status": {"g1": "done", "g2": "failed", "g3": "pending"}, "reasons": {"g2": "budget spent"},
        "flags": {"g3": "dropped p=0.93"}, "active": "g3"}


def test_the_tree_view_highlights_the_active_goal_and_its_running_skill():
    from homebody.ui.plan_view import plan_view
    html = plan_view(PLAN, Phase("tool_call", "place", "Approaching release point"))
    assert "1 done, 1 failed, 1 to go" in html
    assert "g3 · now" in html and "if it fails: g3" not in html  # a running fallback is a live card
    assert "g1 · done" in html and "g2 · failed" in html
    assert "&lt;b&gt;place" in html and "<b>place</b>" not in html
    assert "writes its plan as its first step" in plan_view(None)


def test_the_tree_view_draws_the_live_goal_with_its_skill_and_budget():
    from homebody.ui.plan_view import plan_view
    plan = {**PLAN, "active": "g1", "used": 2, "flags": {}}
    html = plan_view(plan, Phase("tool_call", "pick", "Lowering jaws"))
    assert "g1 · now" in html and 'stroke-width="2.5"' in html and "op-live-ring" not in html
    assert "pick · Lowering jaws" in html and "2 of 8 calls" in html


def test_a_tree_of_parts_is_drawn_as_numbered_rows_and_flags_show():
    from homebody.ui.plan_view import plan_view
    rows = {"type": "sequence", "children": [
        {"type": "sequence", "children": [{"type": "goal", "id": "a1", "goal": "pick the jam jar", "check": None}]},
        {"type": "sequence", "children": [{"type": "goal", "id": "b1", "goal": "pick the basil jar", "check": None}]}]}
    html = plan_view({"tree": rows, "status": {"a1": "done", "b1": "pending"}, "reasons": {},
                      "flags": {"b1": "slipping 0.91"}, "active": None})
    assert ">1</text>" in html and ">2</text>" in html and "pick the jam" in html
    assert "monitor: slipping 0.91" in html


def test_the_status_strip_says_what_runs_now():
    from homebody.ui.plan_view import status_strip
    strip = status_strip("Executing", replace(view(), plan={**PLAN, "active": "g1"}, sim_start=0.0, sim_now=75.0), 4)
    assert "Running" in strip and "g1 · pick the jam jar" in strip and "navigate: Following route" in strip
    assert "01:15" in strip and "4 decisions" in strip
    assert "No task running" in status_strip("Ready")


def test_the_console_draws_the_tree_as_its_main_view(monkeypatch):
    console = mocked_console(monkeypatch, monitor=Monitor(enabled=True))
    console.monitor_update(replace(view(), plan=PLAN))
    assert "op-map" in console.plan_view.content and "g3 · now" in console.plan_view.content


def test_a_long_sequence_of_steps_wraps_into_lines():
    from homebody.ui.plan_view import GOAL_H, LINE_GAP, PER_LINE, layout
    steps = {"type": "sequence", "children": [{"type": "goal", "id": f"g{i}", "goal": f"step {i}", "check": None}
                                              for i in range(1, 10)]}
    plan = {"tree": steps, "status": {}, "reasons": {}, "flags": {}, "active": "g1"}
    block = layout(steps, plan, None)
    lines = -(-9 // PER_LINE)
    assert block.height == lines * GOAL_H + (lines - 1) * LINE_GAP and block.width < 5 * 260


def test_a_reading_takes_its_alarm_colour_only_when_it_would_act():
    from homebody.ui.monitor_panel import QUIET, bar
    assert f"background:{QUIET}" in bar("unsafe", 0.63, "#C2271D", 0.9)
    assert "background:#C2271D" in bar("unsafe", 0.95, "#C2271D", 0.9)
    assert "background:#C2271D" in bar("g1 still holds", 0.1, "#C2271D", 0.2, acts=True)


def test_a_failed_goal_shows_its_reason_on_up_to_three_lines():
    from homebody.ui.plan_view import plan_view
    long = "Jam grasp succeeded, but navigation lost the jar on the way to the cart and the carry was cleared"
    html = plan_view({**PLAN, "reasons": {"g2": long}, "active": None})
    assert "Jam grasp succeeded, but" in html and "navigation lost the jar" in html


def test_rows_are_named_by_what_their_goal_ids_share():
    from homebody.ui.plan_view import row_label
    row = {"type": "sequence", "children": [{"type": "goal", "id": "jam_pick", "goal": "Return to the shelf"},
                                            {"type": "goal", "id": "jam_carry", "goal": "carry"}]}
    assert row_label(row) == "Jam"
    assert row_label({"type": "goal", "id": "g1", "goal": "Return to the shelf as needed"}) == "Return to the sh…"


def test_the_timeline_says_when_nothing_intervened_and_smooths_the_risk_line():
    from homebody.ui.monitor_panel import timeline_svg
    quiet = replace(view(), verdicts=(), marks=(), history=((1.0, "walking", 0.8, "continue", "progress", 0.0),
                                                            (2.0, "walking", 0.8, "continue", "progress", 1.0)))
    assert "None yet" in timeline_svg(quiet)
    raw, smooth = timeline_svg(quiet, ema=1.0), timeline_svg(quiet, ema=0.5)
    assert ",140.0" in raw and ",160.0" in smooth  # the second point: risk 1.0 raw, 0.5 once averaged


def test_a_goal_burning_its_budget_turns_the_strip_and_its_card_amber():
    from homebody.ui.plan_view import WATCH, plan_view, status_strip, trouble
    plan = {**PLAN, "active": "g1", "used": 3,
            "results": [{"skill": "navigate", "code": "NO_ROUTE"}, {"skill": "navigate", "code": "NO_ROUTE"},
                        {"skill": "pick", "code": "OK"}]}
    assert trouble(plan) == "2 navigate no route · 3 of 8 calls used"
    strip = status_strip("Thinking", replace(view(), plan=plan, phase=Phase("planning")))
    assert "Needs attention" in strip and "2 navigate no route" in strip and WATCH in strip
    assert f'fill="{WATCH}"' in plan_view(plan, Phase("planning"))
    assert trouble({**plan, "results": [], "used": 1}) == ""


def test_steps_after_a_failed_step_are_skipped_and_a_fallback_that_ran_says_so():
    from homebody.ui.plan_view import plan_view
    tree = {"type": "fallback", "children": [
        {"type": "sequence", "children": [{"type": "goal", "id": "a", "goal": "carry", "check": None},
                                          {"type": "goal", "id": "b", "goal": "place", "check": None}]},
        {"type": "goal", "id": "r", "goal": "record the failure", "check": None}]}
    html = plan_view({"tree": tree, "status": {"a": "failed", "b": "pending", "r": "done"}, "reasons": {},
                      "flags": {}, "active": None})
    assert "b · skipped" in html and "will not run" in html and "Fallback ran: r · done" in html


def test_cards_widen_into_the_main_view():
    from homebody.ui.plan_view import CARD_MAX, GOAL_W, card_width
    three = {"type": "sequence", "children": [{"type": "goal", "id": f"g{i}", "goal": "x"} for i in range(3)]}
    assert GOAL_W < card_width(three) <= CARD_MAX


def test_a_reading_from_an_earlier_step_is_marked_stale():
    from homebody.ui.monitor_panel import reading_html
    fresh = replace(view(), goal_id="g2", skill_call=4, reading=replace(view().reading,
                                                                        phase=Phase("tool_call", "navigate", call=4)))
    assert "is-stale" not in reading_html(fresh, 0.8, 0.9) and "during navigate in g2" in reading_html(fresh, 0.8, 0.9)
    stale = replace(fresh, skill_call=5)
    assert "is-stale" in reading_html(stale, 0.8, 0.9) and "from an earlier step" in reading_html(stale, 0.8, 0.9)


def test_refusals_are_ticks_and_stacked_decisions_are_counted():
    from homebody.ui.monitor_panel import timeline_svg
    busy = replace(view(), spans=((10.0, 10.0, "navigate", "g2", "NO_ROUTE"),), decisions=(10.0, 10.0, 10.0, 4.0))
    svg = timeline_svg(busy)
    assert 'width="3" height="22"' in svg and "navigate in g2: no route" in svg
    assert ">×3<" in svg and "3 decisions" in svg


def test_a_refusal_before_any_motion_reads_as_refused():
    from homebody.skills.contract import Result
    from homebody.ui.console import stage_progress
    html = stage_progress("navigate", ("Preparing",), Result("navigate", "NO_ROUTE", "No collision-free route"))
    assert "Navigate refused: No collision-free route" in html and "hb-refused" in html
    moved = stage_progress("navigate", ("Preparing", "Following route"), Result("navigate", "STALL", "stalled"))
    assert "hb-refused" not in moved and "hb-result" in moved


def test_stop_skill_is_enabled_only_while_a_skill_runs_and_the_task_box_folds_away(monkeypatch):
    from homebody.skills.contract import Result
    console = mocked_console(monkeypatch, monitor=Monitor(enabled=True))
    created = [c for c in console.server.gui.add_button.call_args_list if c.args and c.args[0] == "Stop skill"]
    assert created and created[0].kwargs.get("disabled") is True
    console.skill_event({"type": "skill_started", "skill": "pick"})
    assert console.stop_skill_button.disabled is False
    console.skill_event({"type": "skill_finished", "result": Result("pick", "OK", "held")})
    assert console.stop_skill_button.disabled is True
    console.update("Thinking")
    assert console.task.visible is False and console.submit_button.visible is False
    console.update("Ready")
    assert console.task.visible is True


def test_a_finished_run_shows_its_outcome_and_reason_not_the_last_phase(monkeypatch):
    from homebody.ui.plan_view import STOP, status_strip
    planning = replace(view(), plan={**PLAN, "active": "g1"}, phase=Phase("planning"))
    strip = status_strip("provider_error", planning, 4, "LiteLLM returned no decision: server_is_overloaded: busy " + "x" * 400)
    assert "Provider error" in strip and STOP in strip and "server_is_overloaded" in strip
    assert "choosing the next step" not in strip and "provider_error" not in strip and "x" * 300 not in strip
    console = mocked_console(monkeypatch, monitor=Monitor(enabled=True))
    console.update("provider_error", "LiteLLM returned no decision: server_is_overloaded")
    console.monitor_update(planning)
    assert "Provider error" in console.strip.content and "server_is_overloaded" in console.strip.content
