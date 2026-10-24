"""The decision model: the OpenAI Decisions API (POST /v1/decisions) reads the newest camera frames with the task,
the goal and the phase, and answers independent questions with probabilities in one call:

- activity (choice): which of the goal's labels the robot is doing, or none_of_these;
- unsafe and human_needed (predicates);
- still_true_<n> (predicate): whether a done goal's standing claim still holds;
- progress (score): how much the task moves on across the frames.

Requests go to `monitor.decision_base_url` with the key in OPENAI_DECISIONS_API_KEY, which nothing else reads (the
LiteLLM proxy cannot carry Decisions for its ChatGPT-backed models); every other model call goes through LiteLLM.
"""
import base64
import http.client
import io
import json
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from homebody.monitor.targets import NONE_OF_THESE, PROGRESS_LEVELS, DecisionRequest, Reading
from homebody.vlm.api import key
from homebody.vlm.provider import (
    BUSY_STATUS,
    ProviderBusy,
    ProviderError,
    error_body,
    failure,
    scrubbed,
)

KEY = "OPENAI_DECISIONS_API_KEY"
OUT_OF_CREDIT = ("insufficient_quota", "credit_balance_exhausted")
UNSAFE = ("The scene is beyond safe recovery: the robot has fallen or is falling, is colliding hard with furniture, "
          "a person or itself, or something is breaking, spilling or falling off a surface.")
HUMAN = ("Only a person can fix the situation now: an object the robot needs is on the floor or out of reach, "
         "something is jammed, or the robot or its hand is trapped.")
PROGRESS = (("no_change", "nothing relevant changes, or the same motion repeats without effect"),
            ("some", "the robot moves toward the goal, slowly or partly"),
            ("clear", "the robot visibly gets closer to the goal across the frames"))


def jpeg_url(rgb: np.ndarray, short_side: int) -> str:
    """RGB as a base64 JPEG data URL, scaled down so its short side is at most SHORT_SIDE."""
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    scale = short_side / min(image.size)
    if scale < 1:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def context(request: DecisionRequest) -> str:
    """The text part: frame times, task, goal, phase and hands."""
    node, phase = request.node, request.phase
    times = ", ".join(f"{shot.view} t={shot.sim_time:.2f}s" for shot in request.shots)
    check = "the agent closes it" if not node.check else " ".join(f"{k}={v}" for k, v in node.check.items())
    doing = ("the robot runs skill " + phase.skill + (f", stage {phase.stage}" if phase.stage else "")
             if phase.name == "tool_call" and phase.skill else
             "the agent is choosing the robot's next step" if phase.name == "planning" else "the robot is idle")
    budget = f", {node.budget_left} skill calls left" if node.budget_left is not None else ""
    inside = f", inside {node.context}" if node.context else ""
    return (f"Camera frames from a humanoid robot, oldest first: {times}.\n"
            f"Task: {request.task}\n"
            f"Goal being worked on: {node.goal_id}: {node.goal} (done when: {check}; {node.status}{budget}{inside})\n"
            f"Now: {doing}.\nHands: {request.hands}")


def questions(request: DecisionRequest) -> list[dict]:
    choices = [{"value": label.name, "description": label.description} for label in request.vocabulary.labels]
    choices.append({"value": NONE_OF_THESE, "description": "none of the other choices describes what the robot does"})
    asked = [{"type": "choice", "name": "activity", "choices": choices,
              "instructions": "Which of these is the robot doing in the newest frames?"},
             {"type": "predicate", "name": "unsafe", "instructions": UNSAFE},
             {"type": "predicate", "name": "human_needed", "instructions": HUMAN}]
    asked += [{"type": "predicate", "name": f"still_true_{index}",
               "instructions": f"Is this still true: {claim.text}? Answer no only if the frames show it is no "
                               "longer true; if they cannot show it (out of view, hidden by the robot), answer yes."}
              for index, claim in enumerate(request.vocabulary.standing)]
    asked.append({"type": "score", "name": "progress", "levels": [
        {"label": label, "description": text} for label, text in PROGRESS],
        "instructions": "How much does the robot's work on the goal move on across the frames?"})
    return asked


def body(request: DecisionRequest, model: str, short_side: int) -> dict:
    content = [{"type": "input_text", "text": context(request)}]
    content += [{"type": "input_image", "image_url": jpeg_url(shot.rgb, short_side)} for shot in request.shots]
    return {"model": model, "input": [{"role": "user", "content": content}], "questions": questions(request)}


def redacted(record: dict) -> dict:
    """BODY as recorded: each image replaced by its index."""
    record = json.loads(json.dumps(record))
    for message in record.get("input") or []:
        index = 0
        for part in message["content"]:
            if part.get("type") == "input_image":
                part["image_url"] = f"frame {index}"
                index += 1
    return record


def probability(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and 0 <= value <= 1 else None


def parse(reply: dict, request: DecisionRequest, *, latency_s=0.0, lag_s=0.0, usd_per_mtok=0.0) -> Reading:
    """The reply's answers as a Reading; a refused or malformed answer is listed in `refusals`."""
    answers = {answer.get("name"): answer for answer in reply.get("answers") or [] if isinstance(answer, dict)}
    refusals = []

    def answer(name):
        found = answers.get(name)
        if found is None or found.get("type") == "refusal":
            refusals.append(name)
            return None
        return found

    activity = {}
    if (found := answer("activity")) is not None:
        for item in found.get("probabilities") or []:
            if isinstance(item, dict) and (p := probability(item.get("probability"))) is not None:
                activity[str(item.get("value"))] = p
        if not activity:
            refusals.append("activity")
    flags = {}
    for name in ("unsafe", "human_needed"):
        found = answer(name)
        flags[name] = probability(found.get("probability")) if found is not None else None
    still_true = {}
    for index, claim in enumerate(request.vocabulary.standing):
        found = answer(f"still_true_{index}")
        if found is not None and (p := probability(found.get("probability"))) is not None:
            still_true[claim.goal_id] = p
    progress = {}
    if (found := answer("progress")) is not None:
        for item in found.get("probabilities") or []:
            label = item.get("label") if isinstance(item, dict) else None
            if label in PROGRESS_LEVELS and (p := probability(item.get("probability"))) is not None:
                progress[label] = p
        if not progress:
            refusals.append("progress")
    tokens = (reply.get("usage") or {}).get("input_tokens") if isinstance(reply.get("usage"), dict) else 0
    tokens = int(tokens) if isinstance(tokens, (int, float)) and 0 <= tokens < 1e9 else 0
    newest = max((shot.sim_time for shot in request.shots), default=0.0)
    return Reading(goal_id=request.node.goal_id, sim_time=newest, activity=activity, unsafe=flags["unsafe"],
                   human_needed=flags["human_needed"], still_true=still_true, progress=progress,
                   refusals=tuple(dict.fromkeys(refusals)), phase=request.phase, latency_s=latency_s, lag_s=lag_s,
                   input_tokens=tokens, cost_usd=tokens * usd_per_mtok / 1e6)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect is an error: urllib would send the key on to wherever it points."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, f"redirect to {newurl} refused", headers, fp)


OPENER = urllib.request.build_opener(NoRedirect)


class OutOfCredit(ProviderError):
    """The account has no credit left: asking again cannot succeed."""


class OpenAiDecisions:
    """One /v1/decisions call per reading, on the caller's thread (the monitor's worker)."""

    def __init__(self, settings):
        self.settings = settings

    def decide(self, request: DecisionRequest, lag_s: float = 0.0) -> tuple[Reading, dict]:
        """The reading for REQUEST and the request as recorded (images redacted). ProviderBusy / ProviderError
        on failure, with credentials scrubbed."""
        s = self.settings
        sent = body(request, s.decision_model, s.image_short_side)
        headers = {"Authorization": f"Bearer {key(KEY)}", "content-type": "application/json"}
        http_request = urllib.request.Request(s.decision_base_url.rstrip("/") + "/decisions", method="POST",
                                              data=json.dumps(sent).encode(), headers=headers)
        started = time.monotonic()
        try:
            with OPENER.open(http_request, timeout=s.decision_timeout) as response:
                text = response.read().decode(errors="replace")
        except urllib.error.HTTPError as error:
            message = scrubbed(f"Decisions HTTP {error.code}: {error_body(error)}", headers)
            if any(code in message for code in OUT_OF_CREDIT):
                raise OutOfCredit(message) from error
            raise (ProviderBusy(message) if error.code in BUSY_STATUS else failure(message)) from error
        except (OSError, http.client.HTTPException) as error:
            raise failure(scrubbed(f"Decisions transport failed: {error}", headers)) from error
        latency = time.monotonic() - started
        try:
            reply = json.loads(text)
        except (json.JSONDecodeError, ValueError) as error:
            raise failure(scrubbed(f"Decisions returned no JSON: {text[:300]}", headers)) from error
        if not isinstance(reply, dict):
            raise failure(scrubbed(f"Decisions returned no JSON object: {text[:300]}", headers))
        return parse(reply, request, latency_s=latency, lag_s=lag_s, usd_per_mtok=s.decision_usd_per_mtok), redacted(sent)
