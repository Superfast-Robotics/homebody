"""Multi-model agents: one provider made of others (session/config.py COMPOSITES). Each turn the composite asks its
members in sub-attempts of the step's attempt folder (attempt/01-planner, attempt/02-pointer, ...) and writes
attempt/routing.json: every call's role, model, whether it ran on our own GPUs (free) and its folder, whose reply.json
holds its usage (evolve/score_episode.py prices each call at its own model's rate).

pointer   the planner decides; when its skill takes point_normalized_1000 it names the target in words and a pointing
          model clicks it on image 1 (the planner's own point when the pointer fails; both are logged)
cascade   the routine model answers; the fallback answers instead after a refused result, on plan turns and when the
          routine model claims done or goal_done (each trigger optional)
router    plan turns go to the planner, step turns to the executor
verifier  a done or goal_done claim is checked by a verifier model on the latest images; a "no" sends the base model
          its reason and asks again, at most `vetoes` times per goal, then the claim stands

The session tells a composite each turn's allowed decisions (turn) and each result (observe)."""
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from homebody.session.config import COMPOSITES, MODELS
from homebody.session.settings import Agent

from .api import MEDIA, key, redact
from .provider import Decision, ProviderError, image_data, run_request

POINT = "point_normalized_1000"
CLAIMS = ("done", "goal_done")
PLAN_TURN = ("plan",)
TARGET = re.compile(r"TARGET:\s*(.+)", re.IGNORECASE)
TARGET_NOTE = ("When your skill takes point_normalized_1000, also end your text with `TARGET: <the exact spot to click "
               "on image 1, in a few words that name the object and its part>`.")
VETO_NOTE = ("A check of the latest images does not confirm your {skill} claim: {reason} Do not claim it again until "
             "the images show it; choose your next decision.")
# each pointer's own trained question (model cards and official inference code): Molmo2 "Point to ..."
# (allenai/Molmo2-4B card), RoboBrain2.5 inference.py's pointing task, Embodied-R1 inference_example.py's REG mode
POINTER_PROMPTS = {
    "molmo": "Point to {target}.",
    "qwen1000": "Point to {target}. Answer with the point as [x, y] in coordinates normalized to 0-1000.",
    "robobrain": ("{target}. Please provide its 2D coordinates. Your answer should be formatted as a tuple, i.e. "
                  "[(x, y)], where the tuple contains the x and y coordinates of a point satisfying the conditions "
                  "above."),
    "embodied_r1": ("Provide one or more points coordinate of objects region {target}. The results are presented in "
                    "a format <point>[[x1,y1], [x2,y2], ...]</point>. You FIRST think about the reasoning process as "
                    "an internal monologue and then provide the final answer. The reasoning process and answer are "
                    "enclosed within <think> </think> and <answer> </answer> tags. The answer consists only of "
                    "several coordinate points, with the overall format being: <think> reasoning process here "
                    "</think><answer><point>[[x1, y1], [x2, y2], ...]</point></answer>"),
    # 2026 models (pickxiguapi/Embodied-R1.5 inference/vllm_online_example.py; RynnBrain cookbooks
    # 6_affordance_location; MiMo-Embodied lmms_eval where2place_point with " /no_think"; NVIDIA's RefSpatial eval
    # suffix for Cosmos3; the Qwen3-VL cookbook's JSON points for Qwen3.6/3.8; Gemma 4's y-first point_2d as
    # NVIDIA's eval asks it)
    "embodied_r15": ("Provide one or more points coordinate of objects region this sentence describes: {target}. The "
                     "answer should be presented in JSON format as follows: [{{\"point_2d\": [x, y]}}].\nPlease answer "
                     "this question based on the visual content.You FIRST think about the reasoning process as an "
                     "internal monologue and then provide the final answer.At the end, you must output the final "
                     "answer in the format:\n<answer><your_answer_here></answer>\n"),
    "rynnbrain": ("{target}\nIdentify one affordance point and provide its normalized pixel coordinates. The x and y "
                  "values must be between 0 and 1000. Output format: <affordance> (x, y) </affordance>."),
    "mimo": ("Based on the description: \"{target}\", locate points matching the description. Output a JSON in the "
             "format [{{\"points\": [...], \"label\": \"{target}\"}}, ...]. /no_think"),
    "cosmos3": ("Point to {target}.\nOutput the point coordinates in JSON format.\nFor example:\n[\n  "
                "{{\"point_2d\": [x, y], \"label\": \"point_1\"}}\n]"),
    "qwen_json": "Locate {target} in this image. Output the point coordinates in JSON format.",
    "gemma": ("Point to {target}. Output the point coordinates in JSON format: [{{\"point\": [y, x], \"label\": "
              "\"point_1\"}}], with y and x normalized to 0-1000."),
    # HY-Embodied-0.5-X's training data (data_examples/data_demo.jsonl; its template adds /no_think)
    "hy05": ("Select a point on the {target}. Your answer should be formatted as a single point or a list of points, "
             "i.e. <point>(x,y)</point> or [<point>(x1,y1)</point>,<point>(x2,y2)</point>,...], where each point "
             "contain the x and y coordinates of a point satisfying the targeting object. The coordinates should be "
             "between 0 and 1000, indicating the normalized locations of the coordinates in the image."),
}
# the answers in 0-1000 [x, y] after any reasoning; gemma's are [y, x]; pixels are the Qwen2.5-VL resized image's
THOUSANDS = ("qwen1000", "robobrain", "embodied_r15", "rynnbrain", "cosmos3", "qwen_json", "hy05")
PIXELS = ("pixels", "embodied_r1", "mimo")
NO_THINK = {"mimo-embodied-7b": " /no_think"}  # a verifier question to a model that thinks unless told not to
# the yes/no check of a done claim: "plain" shows the goal and the claim; "evidence" adds the robot's last skill
# result, which a head-camera image alone often cannot confirm (a held object, a placement out of view)
VERIFY_PROMPTS = {
    "plain": ("A robot reports that it finished this goal.\nGoal: {goal}\nRobot's report: {claim}\nLook at the image "
              "(the robot's head camera, now). Is the goal really achieved? Answer yes or no first, then one sentence "
              "why."),
    "evidence": ("A robot reports that it finished this goal.\nGoal: {goal}\nRobot's report: {claim}\nIts last skill "
                 "result: {result}\nLook at the image (the robot's head camera, now) and weigh the result. Is the goal "
                 "really achieved? Answer yes or no first, then one sentence why."),
}
VERIFY_PROMPT = VERIFY_PROMPTS["plain"]
ENDPOINT_TOKENS = {"embodied_r1": 2048, "embodied_r15": 2048, "molmo": 1024}  # room for Embodied-R1's <think>; others 256
SAMPLING = {"embodied_r1": {"repetition_penalty": 1.05}}  # its reference decoding: greedy, 1.05
QWEN_FACTOR, QWEN_MIN_PIXELS, QWEN_MAX_PIXELS = 28, 56 * 56, 28 * 28 * 1280  # Qwen2.5-VL image resizing


@dataclass(frozen=True)
class Endpoint:
    """A model on one of our vLLM servers asked in its own words: a pointer or a verifier."""
    served: str
    format: str
    url: str
    timeout: float

    def ask(self, prompt: str, images: tuple[Path, ...], *, attempt: Path, tick: Callable[[], None],
            cancelled: Callable[[], bool]) -> str:
        content = [{"type": "image_url", "image_url": {"url": f"data:{MEDIA[image.suffix.lower()]};base64,{image_data(image)}"}}
                   for image in images]  # images first, as each pointer's own examples give them
        content.append({"type": "text", "text": prompt + (NO_THINK.get(self.served, "") if self.format == "yesno" else "")})
        body = {"model": self.served, "messages": [{"role": "user", "content": content}],
                "max_tokens": ENDPOINT_TOKENS.get(self.format, 256), "temperature": 0.0,
                # a model whose template thinks by default (Hy-Embodied) answers directly, as the pointers do
                "chat_template_kwargs": {"enable_thinking": False}, **SAMPLING.get(self.format, {})}
        headers = {"Authorization": f"Bearer {key('VLLM_API_KEY')}", "content-type": "application/json"}
        reply = run_request(f"vLLM {self.served}", self.url.rstrip("/") + "/chat/completions", headers, body,
                            prompt=prompt, attempt=attempt, timeout=self.timeout, tick=tick, cancelled=cancelled,
                            redacted=redact(body, images))
        try:
            return str(reply["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"vLLM {self.served} returned no answer: {json.dumps(reply)[:300]}") from exc


def qwen_size(width: int, height: int) -> tuple[int, int]:
    """The size Qwen2.5-VL resizes an image to (multiples of 28 within its pixel bounds): its pixel answers are in it."""
    w, h = (max(QWEN_FACTOR, round(side / QWEN_FACTOR) * QWEN_FACTOR) for side in (width, height))
    if w * h > QWEN_MAX_PIXELS:
        beta = math.sqrt(width * height / QWEN_MAX_PIXELS)
        w, h = (max(QWEN_FACTOR, math.floor(side / beta / QWEN_FACTOR) * QWEN_FACTOR) for side in (width, height))
    elif w * h < QWEN_MIN_PIXELS:
        beta = math.sqrt(QWEN_MIN_PIXELS / (width * height))
        w, h = (math.ceil(side * beta / QWEN_FACTOR) * QWEN_FACTOR for side in (width, height))
    return w, h


def numbers(text: str) -> list[float]:
    """The numbers in TEXT, not digits inside a name (x1="677" gives 677, not 1)."""
    return [float(n) for n in re.findall(r"(?<![A-Za-z_\d.])-?\d+(?:\.\d+)?", text)]


def answer_part(answer: str) -> str:
    """The answer without its reasoning: after </think>, inside <answer> and <point> when present."""
    answer = re.sub(r"^.*?</think>", "", answer, flags=re.DOTALL)
    for tag in ("answer", "point"):
        if inner := re.search(rf"<{tag}>(.*?)</{tag}>", answer, re.DOTALL):
            answer = inner[1]
    return answer


def to_1000(answer: str, form: str, image: Path) -> list[int]:
    """The pointer's ANSWER as point_normalized_1000 on IMAGE, its first point. molmo: Molmo2's
    <points coords="IMAGE ID X Y [ID X Y ...]"> in 0-1000 (or Molmo 1's x="%" y="%"); THOUSANDS: the first two
    numbers, 0-1000; gemma: the same as [y, x]; PIXELS: the first two numbers in the image's Qwen2.5-VL resized
    pixels. Any reasoning is ignored. ProviderError when no point is found or it is off the image."""
    answer = answer_part(answer)
    if form == "molmo":
        tag = re.search(r'x\d*="([\d.]+)"\s+y\d*="([\d.]+)"', answer)
        coords = re.search(r'coords="([^"]+)"', answer)
        if tag:
            x, y = float(tag[1]) * 10, float(tag[2]) * 10
        elif coords and len(values := numbers(coords[1].split(";")[0])) >= 4:
            x, y = values[2:4]  # image index, point id, x, y
        elif coords and len(values) >= 2:
            x, y = values[-2:]
        elif len(values := numbers(answer)) >= 2:  # a Molmo-trained model answering [x, y] (Mage-VL), 0-1000
            x, y = values[:2]
        else:
            raise ProviderError(f"no point in the pointer's answer: {answer[:200]}")
    else:
        values = numbers(answer)
        if len(values) < 2:
            raise ProviderError(f"no point in the pointer's answer: {answer[:200]}")
        group = re.search(r"\[([^\[\]]*)\]", answer)
        if group and len(box := numbers(group[1])) == 4:  # a box where a point was asked (Cosmos3-Nano): its centre
            values = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]  # gemma's y-first box gives y-first
        x, y = values[:2]
        if form == "gemma":
            y, x = values[:2]
        elif form in PIXELS:
            from PIL import Image  # noqa: PLC0415 -- only the pixel format reads the image size

            with Image.open(image) as picture:
                width, height = qwen_size(*picture.size)
            x, y = x / width * 1000, y / height * 1000
        elif form not in THOUSANDS:
            raise ProviderError(f"unknown point format {form}")
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        raise ProviderError(f"the pointer's point {[x, y]} is off the image")
    return [round(x), round(y)]


def verdict(answer: str) -> tuple[bool, str]:
    """(achieved, reason) from a yes/no answer; anything but a leading no counts as yes (a veto needs a clear no)."""
    answer = answer_part(answer)
    words = answer.strip().lower()
    no = bool(re.match(r"\W*no\b", words))
    return not no, re.sub(r"^\W*(yes|no)\W*", "", answer.strip(), flags=re.IGNORECASE) or answer.strip()


def section(prompt: str, title: str) -> str:
    """The first line after TITLE in the session prompt ("" when absent)."""
    match = re.search(rf"^{re.escape(title)}\s*\n(.+)$", prompt, re.MULTILINE)
    return match[1].strip() if match else ""


def goal_of(prompt: str) -> str:
    """The claim's goal: the plan's active goal when there is one, else the task."""
    active = re.search(r"^Active goal [^:\n]+:\s*(.+?)(?: \(\d+ skill calls? left for it\)\.)?$", prompt, re.MULTILINE)
    task = section(prompt, "Current task:")
    return f"{active[1].strip()} (part of the task: {task})" if active else task


def resolve(settings: Agent, name: str) -> str:
    """A member name: a MODELS name, or best-small / best-base for the settings' current pick."""
    model = {"best-small": settings.best_small, "best-base": settings.best_base}.get(name, name)
    if model not in MODELS:
        raise ProviderError(f"composite member {model} is not in MODELS")
    return model


def server_url(settings: Agent, server: str) -> str:
    """The URL of our model server SERVER: local (the vllm provider's agent.local_url) or second."""
    url = settings.second_url if server == "second" else getattr(settings, "local_url", None)
    if not url:
        raise ProviderError(f"no URL for model server {server}: the vllm provider sets agent.local_url")
    return url


class Composite:
    """What every multi-model agent shares: its members, the turn's allowed decisions, the last result, routing.
    Composite(settings) builds the kind settings.model names (a MODELS row with provider "composite")."""
    LABEL, KEY = "Multi-model", "VLLM_API_KEY"  # every composite has a model on our vLLM servers

    def __new__(cls, settings: Agent, *args, **kwargs):
        return super().__new__(KINDS[COMPOSITES[settings.model_id]["kind"]] if cls is Composite else cls)

    def __init__(self, settings: Agent, members: dict[str, Any] | None = None,
                 endpoints: dict[str, Endpoint] | None = None):
        self.settings = settings
        self.spec = COMPOSITES[settings.model_id]
        self.names: dict[str, str] = {}
        self.members = members if members is not None else self.build_members()
        self.endpoints = endpoints if endpoints is not None else self.build_endpoints()
        self.allowed: tuple[str, ...] = ()
        self.last: dict | None = None
        self.calls: list[dict] = []

    def build_members(self) -> dict[str, Any]:
        from .catalog import make  # noqa: PLC0415 -- the catalog imports this module

        members = {}
        for role, member in self.spec.items():
            if isinstance(member, dict) and "model" in member:
                name = resolve(self.settings, member["model"])
                self.names[role] = name
                local = ({"local_url": server_url(self.settings, member.get("server", "local"))}
                         if hasattr(self.settings, "local_url") else {})
                members[role] = make(replace(self.settings, model=name, **local))
        return members

    def build_endpoints(self) -> dict[str, Endpoint]:
        return {role: Endpoint(member["served"], member["format"],
                               server_url(self.settings, member.get("server", "second")), self.settings.timeout)
                for role, member in self.spec.items() if isinstance(member, dict) and "served" in member}

    # --- session hooks ---
    @property
    def cache_key(self):
        return next((member.cache_key for member in self.members.values() if hasattr(member, "cache_key")), None)

    @cache_key.setter
    def cache_key(self, value):
        for member in self.members.values():
            if hasattr(member, "cache_key"):
                member.cache_key = value

    def turn(self, allowed: tuple[str, ...]):
        self.allowed = tuple(allowed)
        for member in self.members.values():
            if hasattr(member, "turn"):
                member.turn(allowed)

    def observe(self, result: dict):
        self.last = result
        for member in self.members.values():
            if hasattr(member, "observe"):
                member.observe(result)

    # --- calls ---
    def model_name(self, role: str) -> str:
        return self.names.get(role) or self.spec[role].get("model") or self.spec[role]["served"]

    def sub_attempt(self, attempt: Path, role: str) -> Path:
        return attempt / f"{len(self.calls) + 1:02d}-{role}"

    def record(self, role: str, folder: Path, **extra):
        member = self.members.get(role)
        local = True if role in self.endpoints else bool(getattr(type(member), "LOCAL", False))
        model = self.model_name(role)
        model_id = MODELS[model][1] if model in MODELS else model
        self.calls.append({"role": role, "model": model, "model_id": model_id, "local": local,
                           "dir": folder.name, **extra})

    def ask(self, role: str, prompt: str, *, images, names, attempt, tick, cancelled, **extra) -> Decision:
        folder = self.sub_attempt(attempt, role)
        try:
            decision = self.members[role].generate(prompt, images=images, names=names, attempt=folder, tick=tick,
                                                   cancelled=cancelled)
        except ProviderError as error:
            self.record(role, folder, error=str(error)[:300], **extra)
            raise
        self.record(role, folder, skill=decision.skill, **extra)
        return decision

    def query(self, role: str, prompt: str, images, *, attempt, tick, cancelled) -> tuple[str, Path]:
        folder = self.sub_attempt(attempt, role)
        try:
            answer = self.endpoints[role].ask(prompt, images, attempt=folder, tick=tick, cancelled=cancelled)
        except ProviderError as error:
            self.record(role, folder, error=str(error)[:300])
            raise
        return answer, folder

    def generate(self, prompt: str, *, images: tuple[Path, ...], names: tuple[str, ...], attempt: Path,
                 tick: Callable[[], None], cancelled: Callable[[], bool]) -> Decision:
        self.calls = []
        attempt.mkdir(parents=True, exist_ok=True)
        try:
            return self.decide(prompt, images=images, names=names, attempt=attempt, tick=tick, cancelled=cancelled)
        finally:
            (attempt / "routing.json").write_text(json.dumps({"composite": self.settings.model,
                                                              "kind": self.spec["kind"], "calls": self.calls},
                                                             indent=2))

    def decide(self, prompt, **turn) -> Decision:
        raise NotImplementedError


class PlannerPointer(Composite):
    def decide(self, prompt, *, images, names, attempt, tick, cancelled):
        decision = self.ask("planner", f"{prompt}\n\n{TARGET_NOTE}", images=images, names=names, attempt=attempt,
                            tick=tick, cancelled=cancelled)
        if POINT not in decision.arguments or not images:
            return decision
        target = (match[1].strip() if (match := TARGET.search(decision.text)) else decision.text.strip())
        own = decision.arguments[POINT]
        endpoint = self.endpoints["pointer"]
        try:
            answer, folder = self.query("pointer", POINTER_PROMPTS[endpoint.format].format(target=target),
                                        images[:1], attempt=attempt, tick=tick, cancelled=cancelled)
        except ProviderError:
            self.calls[-1].update(target=target, planner_point=own, used="planner")
            return decision
        try:
            point = to_1000(answer, endpoint.format, images[0])
        except ProviderError as error:
            self.record("pointer", folder, target=target, answer=answer[:300], error=str(error)[:300],
                        planner_point=own, used="planner")
            return decision
        self.record("pointer", folder, target=target, answer=answer[:300], planner_point=own, pointer_point=point,
                    used="pointer")
        return replace(decision, arguments={**decision.arguments, POINT: point})


class Cascade(Composite):
    def decide(self, prompt, **turn):
        triggers = self.spec.get("escalate", ("refusal", "plan", "claim"))
        refused = self.last is not None and self.last.get("code") != "OK"
        reason = ("plan" if "plan" in triggers and self.allowed == PLAN_TURN
                  else "refusal" if "refusal" in triggers and refused else None)
        if reason:
            return self.ask("fallback", prompt, reason=reason, **turn)
        decision = self.ask("routine", prompt, **turn)
        if "claim" in triggers and decision.skill in CLAIMS:
            return self.ask("fallback", prompt, reason="claim", routine_skill=decision.skill, **turn)
        return decision


class PlanRouter(Composite):
    def decide(self, prompt, **turn):
        return self.ask("planner" if self.allowed == PLAN_TURN else "executor", prompt, **turn)


class Verifier(Composite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vetoed: dict[str, int] = {}

    def decide(self, prompt, *, images, names, attempt, tick, cancelled):
        limit, asked = int(self.spec.get("vetoes", 2)), prompt
        while True:
            decision = self.ask("base", asked, images=images, names=names, attempt=attempt, tick=tick,
                                cancelled=cancelled)
            goal = goal_of(prompt)
            if decision.skill not in CLAIMS or self.vetoed.get(goal, 0) >= limit or not images:
                return decision
            claim = str(decision.arguments.get("summary") or decision.arguments.get("reason") or decision.text)
            try:
                form = self.spec.get("prompt", "evidence")  # the bench: RoboBrain 43% -> 72% accurate with it
                result = json.dumps(self.last)[:400] if self.last else "none yet"
                asked_verifier = VERIFY_PROMPTS[form].format(goal=goal, claim=claim, result=result)
                answer, folder = self.query("verifier", asked_verifier, images[:1],
                                            attempt=attempt, tick=tick, cancelled=cancelled)
            except ProviderError:
                return decision  # an unavailable verifier never blocks a claim
            achieved, reason = verdict(answer)
            self.record("verifier", folder, claim_skill=decision.skill, achieved=achieved, answer=answer[:300])
            if achieved:
                return decision
            self.vetoed[goal] = self.vetoed.get(goal, 0) + 1
            asked = f"{prompt}\n\n{VETO_NOTE.format(skill=decision.skill, reason=reason)}"


KINDS = {"pointer": PlannerPointer, "cascade": Cascade, "router": PlanRouter, "verifier": Verifier}

