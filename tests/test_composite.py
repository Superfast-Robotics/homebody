import json
from dataclasses import dataclass, field

import pytest
from PIL import Image

from homebody.session.config import COMPOSITES, MODELS, Agent
from homebody.vlm.catalog import PROVIDERS, model_label
from homebody.vlm.composite import (Cascade, Composite, PlannerPointer, PlanRouter, Verifier, goal_of, qwen_size,
                                    resolve, to_1000, verdict)
from homebody.vlm.provider import Decision, ProviderError


@dataclass
class Fake:
    """A member model: answers in turn, records each prompt and writes a usage reply into its attempt."""
    answers: list
    prompts: list = field(default_factory=list)
    LOCAL = True

    def generate(self, prompt, *, images, names, attempt, tick, cancelled):
        self.prompts.append(prompt)
        attempt.mkdir(parents=True)
        (attempt / "reply.json").write_text(json.dumps({"usage": {"prompt_tokens": 100, "completion_tokens": 10}}))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@dataclass
class FakeEndpoint:
    format: str
    answers: list
    prompts: list = field(default_factory=list)

    def ask(self, prompt, images, *, attempt, tick, cancelled):
        self.prompts.append(prompt)
        attempt.mkdir(parents=True)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def turn(tmp_path, step=0):
    return {"images": (tmp_path / "head.png",), "names": ("pick", "done", "goal_done"),
            "attempt": tmp_path / f"step-{step:03d}", "tick": lambda: None, "cancelled": lambda: False}


@pytest.fixture
def image(tmp_path):
    Image.new("RGB", (640, 480)).save(tmp_path / "head.png")
    return tmp_path / "head.png"


def routing(tmp_path, step=0):
    return json.loads((tmp_path / f"step-{step:03d}" / "routing.json").read_text())["calls"]


def test_every_composite_is_a_models_row_of_a_known_kind():
    for name, spec in COMPOSITES.items():
        assert MODELS[name] == ("composite", name, None) and spec["kind"] in ("pointer", "cascade", "router", "verifier")
    assert PROVIDERS["composite"] is Composite
    assert model_label(Agent(model="e7-qwen30b-molmo")).startswith("Multi-model (e7-qwen30b-molmo")


def test_composite_builds_the_kind_its_row_names():
    kinds = {"e7-qwen30b-molmo": PlannerPointer, "e11-robobrain-sol": Cascade, "e17-qwen30b-small": PlanRouter,
             "e19-base-robobrain": Verifier}
    for name, kind in kinds.items():
        assert type(Composite(Agent(model=name), members={}, endpoints={})) is kind


def test_pointer_replaces_the_planners_point_and_logs_both(tmp_path, image):
    planner = Fake([Decision("pick it. TARGET: the red mug's handle", "pick", {"point_normalized_1000": [100, 100]})])
    pointer = FakeEndpoint("molmo", ['<point x="50.0" y="25.5" alt="handle">handle</point>'])
    agent = Composite(Agent(model="e7-qwen30b-molmo"), members={"planner": planner}, endpoints={"pointer": pointer})
    decision = agent.generate("prompt", **turn(tmp_path))
    assert decision.arguments["point_normalized_1000"] == [500, 255]
    assert "TARGET:" in planner.prompts[0] and pointer.prompts == ["Point to the red mug's handle."]
    calls = routing(tmp_path)
    assert [c["role"] for c in calls] == ["planner", "pointer"]
    assert calls[1]["planner_point"] == [100, 100] and calls[1]["pointer_point"] == [500, 255]
    assert calls[1]["used"] == "pointer" and calls[1]["local"] and calls[0]["dir"] == "01-planner"


def test_pointer_failure_keeps_the_planners_point(tmp_path, image):
    for step, answer in enumerate((ProviderError("down"), "I cannot see it")):
        planner = Fake([Decision("pick", "pick", {"point_normalized_1000": [1, 2]})])
        agent = Composite(Agent(model="e7-qwen30b-molmo"), members={"planner": planner},
                          endpoints={"pointer": FakeEndpoint("molmo", [answer])})
        assert agent.generate("p", **turn(tmp_path, step)).arguments["point_normalized_1000"] == [1, 2]
        assert routing(tmp_path, step)[-1]["used"] == "planner"


def test_pointer_leaves_decisions_without_a_point_alone(tmp_path, image):
    pointer = FakeEndpoint("molmo", [])
    agent = Composite(Agent(model="e7-qwen30b-molmo"), members={"planner": Fake([Decision("go", "done", {})])},
                      endpoints={"pointer": pointer})
    assert agent.generate("p", **turn(tmp_path)).skill == "done" and pointer.prompts == []


def test_point_formats(image):
    assert to_1000('<point x="10" y="90">', "molmo", image) == [100, 900]
    assert to_1000('<points coords="1 1 420 610"/>', "molmo", image) == [420, 610]
    assert to_1000("[312, 77]", "qwen1000", image) == [312, 77]
    width, height = qwen_size(640, 480)
    assert (width, height) == (644, 476)
    assert to_1000(f"<point>[[{width // 2}, {height // 4}]]</point>", "pixels", image) == [500, 250]
    assert to_1000('<points coords="1 1 523 412 2 600 300">red jars</points>', "molmo", image) == [523, 412]
    assert to_1000("[(612, 340)]", "robobrain", image) == [612, 340]
    assert to_1000('<points x1="677" y1="447" alt="jar">jar</points>', "qwen1000", image) == [677, 447]
    assert to_1000("<answer>\\point{498, 462}</answer>", "qwen1000", image) == [498, 462]
    reply = (f"<think>the jar at [10, 20] is behind</think><answer><point>[[{width // 2}, {height // 2}], "
             "[5, 5]]</point></answer>")
    assert to_1000(reply, "embodied_r1", image) == [500, 500]  # the answer's first point, not the reasoning's
    for answer, form in (("none", "qwen1000"), ("[1200, 5]", "qwen1000"), ("no", "molmo")):
        with pytest.raises(ProviderError):
            to_1000(answer, form, image)


def test_qwen_size_bounds():
    assert qwen_size(10, 10) == (56, 56)
    width, height = qwen_size(4000, 3000)
    assert width * height <= 28 * 28 * 1280 and width % 28 == 0 and height % 28 == 0


def cascade(*answers, fallback=()):
    routine, fall = Fake(list(answers)), Fake(list(fallback))
    return Composite(Agent(model="e11-robobrain-sol"), members={"routine": routine, "fallback": fall},
                     endpoints={}), routine, fall


def test_cascade_routine_answers_ordinary_turns(tmp_path):
    agent, routine, fallback = cascade(Decision("pick", "pick", {}))
    agent.turn(("pick", "done"))
    agent.observe({"code": "OK"})
    assert agent.generate("p", **turn(tmp_path)).skill == "pick" and not fallback.prompts
    assert [c["model"] for c in routing(tmp_path)] == ["robobrain25-8b-vllm"]


def test_cascade_escalates_after_a_refusal_and_on_plan_turns(tmp_path):
    agent, routine, fallback = cascade(fallback=[Decision("a", "pick", {}), Decision("b", "plan", {})])
    agent.observe({"code": "NO_GRASP"})
    agent.generate("p", **turn(tmp_path))
    assert routing(tmp_path)[0] | {"reason": "refusal"} == routing(tmp_path)[0]
    agent.observe({"code": "OK"})
    agent.turn(("plan",))
    agent.generate("p", **turn(tmp_path, 1))
    assert routing(tmp_path, 1)[0]["reason"] == "plan" and routine.prompts == []
    assert routing(tmp_path, 1)[0]["model"] == "sol61-litellm"


def test_cascade_rechecks_a_claim_with_the_fallback(tmp_path):
    agent, routine, fallback = cascade(Decision("done", "done", {}), fallback=[Decision("not yet", "pick", {})])
    agent.turn(("pick", "done"))
    assert agent.generate("p", **turn(tmp_path)).skill == "pick"
    calls = routing(tmp_path)
    assert [c["role"] for c in calls] == ["routine", "fallback"] and calls[1]["routine_skill"] == "done"


def test_cascade_triggers_are_optional(tmp_path, monkeypatch):
    monkeypatch.setitem(COMPOSITES, "e11-robobrain-sol", {**COMPOSITES["e11-robobrain-sol"], "escalate": ()})
    agent, routine, fallback = cascade(Decision("done", "done", {}))
    agent.observe({"code": "NO_GRASP"})
    agent.turn(("plan",))
    assert agent.generate("p", **turn(tmp_path)).skill == "done" and not fallback.prompts


def test_router_sends_plan_turns_to_the_planner(tmp_path):
    planner, executor = Fake([Decision("plan", "plan", {})]), Fake([Decision("pick", "pick", {})])
    agent = Composite(Agent(model="e17-qwen30b-small"), members={"planner": planner, "executor": executor},
                      endpoints={})
    agent.turn(("plan",))
    assert agent.generate("p", **turn(tmp_path)).skill == "plan"
    agent.turn(("pick", "goal_done"))
    assert agent.generate("p", **turn(tmp_path, 1)).skill == "pick"
    assert routing(tmp_path)[0]["role"] == "planner" and routing(tmp_path, 1)[0]["role"] == "executor"


PROMPT = "Current task:\nput the jar on the shelf\n\nPlan:\n...\nActive goal g2: jar on shelf (3 skill calls left for it)."


def verifier(base_answers, verdicts, vetoes=2):
    base, check = Fake(list(base_answers)), FakeEndpoint("yesno", list(verdicts))
    agent = Composite(Agent(model="e19-base-robobrain"), members={"base": base}, endpoints={"verifier": check})
    return agent, base, check


def test_verifier_accepts_a_confirmed_claim(tmp_path, image):
    agent, base, check = verifier([Decision("placed", "goal_done", {"reason": "placed"})], ["Yes, it is there."])
    assert agent.generate(PROMPT, **turn(tmp_path)).skill == "goal_done"
    assert "Goal: jar on shelf (part of the task: put the jar on the shelf)" in check.prompts[0]
    assert routing(tmp_path)[-1]["achieved"] is True


def test_verifier_veto_asks_again_with_its_reason(tmp_path, image):
    agent, base, check = verifier([Decision("placed", "goal_done", {}), Decision("again", "pick", {})],
                                  ["No. The jar is still on the table."])
    assert agent.generate(PROMPT, **turn(tmp_path)).skill == "pick"
    assert "The jar is still on the table." in base.prompts[1] and "goal_done claim" in base.prompts[1]
    assert [c["role"] for c in routing(tmp_path)] == ["base", "verifier", "base"]


def test_verifier_vetoes_are_bounded_per_goal(tmp_path, image):
    claims = [Decision("d", "goal_done", {})] * 4
    agent, base, check = verifier(claims, ["no: not yet", "no: still not"])
    assert agent.generate(PROMPT, **turn(tmp_path)).skill == "goal_done"
    assert len(base.prompts) == 3 and len(check.prompts) == 2  # two vetoes, then the claim stands
    assert agent.vetoed == {"jar on shelf (part of the task: put the jar on the shelf)": 2}
    agent2_prompt = PROMPT.replace("g2: jar on shelf (3 skill calls", "g3: can in tote (1 skill call")
    agent.members["base"].answers = [Decision("d", "goal_done", {})]
    check.answers = ["yes"]
    assert agent.generate(agent2_prompt, **turn(tmp_path, 1)).skill == "goal_done" and len(check.prompts) == 3


def test_verifier_unavailable_never_blocks(tmp_path, image):
    agent, base, check = verifier([Decision("d", "done", {"summary": "all placed"})], [ProviderError("down")])
    assert agent.generate(PROMPT, **turn(tmp_path)).skill == "done"


def test_verdicts_and_goals():
    assert verdict("No, the can fell.") == (False, "the can fell.")
    assert verdict("yes") [0] and verdict("Not sure")[0]
    assert goal_of("Current task:\ntidy\n") == "tidy"


def test_member_errors_are_routed_and_raised(tmp_path):
    agent, routine, fallback = cascade(ProviderError("boom"))
    with pytest.raises(ProviderError):
        agent.generate("p", **turn(tmp_path))
    assert routing(tmp_path)[0]["error"] == "boom"


def test_hooks_reach_nested_members():
    inner = Composite(Agent(model="e17-qwen30b-small"), members={}, endpoints={})
    outer = Composite(Agent(model="e19-base-robobrain"), members={"base": inner}, endpoints={})
    outer.turn(("plan",))
    outer.observe({"code": "OK"})
    assert inner.allowed == ("plan",) and inner.last == {"code": "OK"}


def test_best_picks_resolve_to_models_names():
    settings = Agent(model="e10-small-molmo", best_small="sol61-litellm")
    assert resolve(settings, "best-small") == "sol61-litellm" and resolve(settings, "astra") == "astra"
    with pytest.raises(ProviderError):
        resolve(settings, "nonesuch")


def test_an_endpoint_asks_with_the_image_first_and_thinking_off(tmp_path, image, monkeypatch):
    from homebody.vlm import composite
    sent = []
    monkeypatch.setenv("VLLM_API_KEY", "k")
    monkeypatch.setattr(composite, "run_request", lambda *a, **kw: sent.append(a[3]) or {
        "choices": [{"message": {"content": "<answer><point>[[1, 2]]</point></answer>"}}]})
    answer = composite.Endpoint("embodied-r1-3b", "embodied_r1", "http://x/v1", 5.0).ask(
        "p", (image,), attempt=tmp_path / "a", tick=lambda: None, cancelled=lambda: False)
    body = sent[0]
    assert answer.startswith("<answer>") and body["chat_template_kwargs"] == {"enable_thinking": False}
    assert [c["type"] for c in body["messages"][0]["content"]] == ["image_url", "text"]
    assert body["max_tokens"] == 2048 and body["repetition_penalty"] == 1.05 and body["temperature"] == 0.0
