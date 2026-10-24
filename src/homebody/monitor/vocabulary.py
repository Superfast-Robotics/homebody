"""The vocabulary model: when a goal opens (and when a skill starts in it) it writes the goal's action labels and
the standing claims of the done goals. Chat completions at `monitor.vocab_base_url`: a vLLM server, its key read
from `monitor.vocab_key_file`, or the LiteLLM proxy with LITELLM_API_KEY when that file is absent. The reply is
parsed leniently, and anything unusable falls back to one generic progress label, so the monitor always has a
vocabulary."""
import json
import os
import re
import time
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from string import Template

from homebody.monitor.targets import KINDS, STATIC, Claim, Label, Vocabulary, VocabularyRequest
from homebody.vlm.api import MEDIA, redact
from homebody.vlm.provider import ProviderError, image_data, run_request

KEY = "LITELLM_API_KEY"
NAME = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
SCHEMA = {
    "type": "object", "required": ["labels", "standing", "expected_seconds"], "additionalProperties": False,
    "properties": {
        "labels": {"type": "array", "items": {
            "type": "object", "required": ["name", "kind", "description", "stages"], "additionalProperties": False,
            "properties": {"name": {"type": "string"}, "kind": {"enum": list(KINDS)},
                           "description": {"type": "string"},
                           "stages": {"type": "array", "items": {"type": "string"}}}}},
        "standing": {"type": "array", "items": {
            "type": "object", "required": ["goal_id", "text"], "additionalProperties": False,
            "properties": {"goal_id": {"type": "string"}, "text": {"type": "string"}}}},
        "expected_seconds": {"type": "number"}}}


def api_key(path: str) -> str:
    """The vLLM key from PATH, else LITELLM_API_KEY, else "" (a server without a key)."""
    file = Path(path).expanduser()
    if file.is_file():
        return file.read_text().strip()
    return os.environ.get(KEY, "").strip()


def check_text(check: dict | None) -> str:
    return "the agent closes it" if not check else " ".join(f"{k}={v}" for k, v in check.items())


def prompt(request: VocabularyRequest, max_labels: int) -> str:
    template = Template(files("homebody.monitor").joinpath("vocabulary_prompt.md").read_text())
    node = request.node
    skill = (f"The robot is starting skill {request.skill} with {json.dumps(request.arguments or {})}."
             if request.skill else "")
    return template.safe_substitute(
        task=request.task, tree=request.tree or "(no plan: the task is the goal)",
        goal=f"{node.goal_id}: {node.goal} (check: {check_text(node.check)})", skill=skill,
        done="\n".join(f"- {n.goal_id}: {n.goal} (check: {check_text(n.check)})" for n in request.done) or "none",
        hands=request.hands, skills="\n".join(f"- {line}" for line in request.skills) or "none",
        stages="\n".join(f"- {name}: {', '.join(names)}" for name, names in request.stages.items()) or "none yet",
        findings="\n".join(f"- {line}" for line in request.findings) or "nothing yet", max_labels=max_labels)


def json_object(text: str) -> dict | None:
    """The reply's JSON object: the whole text, a fenced block, or the outermost braces."""
    for candidate in (text, *re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL),
                      text[text.find("{"):text.rfind("}") + 1] if "{" in text else ""):
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def fallback(request: VocabularyRequest, why: str = "") -> Vocabulary:
    """One generic progress label for the goal: the monitor still watches unsafe, human_needed and stalls."""
    node = request.node
    return Vocabulary(node.goal_id, (Label("working", "progress", f"the robot works toward: {node.goal}"),),
                      source=f"fallback{f': {why}' if why else ''}")


def parse(text: str, request: VocabularyRequest, max_labels: int) -> Vocabulary:
    """The vocabulary in the model's reply TEXT; malformed entries are dropped, nothing usable falls back."""
    value = json_object(text)
    if value is None:
        return fallback(request, "no JSON object in the reply")
    labels, seen = [], set[str](STATIC)
    for item in value.get("labels") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip().lower().replace(" ", "_").replace("-", "_")
        kind, description = item.get("kind"), str(item.get("description", "")).strip()
        if not NAME.match(name) or name in seen or kind not in KINDS or not description:
            continue
        stages = tuple(str(s) for s in item.get("stages") or () if isinstance(s, str))
        labels.append(Label(name, kind, description[:300], stages))
        seen.add(name)
        if len(labels) == max_labels:
            break
    if not labels:
        return fallback(request, "no usable labels")
    done = {node.goal_id for node in request.done}
    standing = tuple(Claim(item["goal_id"], str(item["text"])[:300]) for item in value.get("standing") or []
                     if isinstance(item, dict) and item.get("goal_id") in done and str(item.get("text", "")).strip())
    seconds = value.get("expected_seconds")
    seconds = float(seconds) if isinstance(seconds, (int, float)) and seconds > 0 else None
    return Vocabulary(request.node.goal_id, tuple(labels), standing, seconds)


class VocabularyModel:
    """Qwen3.8-27B (or any chat-completions model) writing vocabularies, recorded under each call's attempt."""

    def __init__(self, settings):
        self.settings = settings

    def body(self, text: str, images: tuple[Path, ...]) -> dict:
        s = self.settings
        content = [{"type": "text", "text": text}]
        content += [{"type": "image_url", "image_url": {
            "url": f"data:{MEDIA[image.suffix.lower()]};base64,{image_data(image)}"}} for image in images]
        return {"model": s.vocab_model, "messages": [{"role": "user", "content": content}],
                "temperature": s.vocab_temperature, "top_p": s.vocab_top_p, "top_k": s.vocab_top_k,
                "presence_penalty": s.vocab_presence_penalty,
                "chat_template_kwargs": {"enable_thinking": s.vocab_thinking},
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "vocabulary", "strict": True, "schema": SCHEMA}}}

    def generate(self, request: VocabularyRequest, attempt: Path,
                 cancelled: Callable[[], bool] = lambda: False) -> Vocabulary:
        """The vocabulary for REQUEST; a transport or reply failure falls back (and is recorded under ATTEMPT)."""
        text = prompt(request, self.settings.max_labels)
        images = tuple(image for image in request.images if image.suffix.lower() in MEDIA)
        body = self.body(text, images)
        headers = {"content-type": "application/json"}
        if secret := api_key(self.settings.vocab_key_file):
            headers["Authorization"] = f"Bearer {secret}"
        try:
            reply = run_request("Vocabulary model", self.settings.vocab_base_url.rstrip("/") + "/chat/completions",
                                headers, body, prompt=text, attempt=attempt, timeout=self.settings.vocab_timeout,
                                tick=lambda: time.sleep(.02), cancelled=cancelled,
                                redacted=redact(body, images))
            content = reply["choices"][0]["message"]["content"] or ""
        except ProviderError as error:
            return fallback(request, str(error)[:200])
        except (KeyError, IndexError, TypeError):
            return fallback(request, "the reply had no message")
        return parse(content, request, self.settings.max_labels)
