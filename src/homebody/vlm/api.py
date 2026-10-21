"""One structured decision per turn from the Anthropic or OpenAI API. The key comes from the environment."""
import json
import os
import re
from collections.abc import Callable
from pathlib import Path

from homebody.session.settings import Agent

from .provider import Decision, ProviderError, image_data, run_request, schema

MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def key(variable: str) -> str:
    """The stripped key from the environment, or ProviderError when unset or not visible ASCII."""
    value = os.environ.get(variable, "").strip()
    if not value:
        raise ProviderError(f"{variable} is not set")
    if not all("!" <= character <= "~" for character in value):
        raise ProviderError(f"{variable} holds characters no API key has; set it again")
    return value


class ClaudeApi:
    """Anthropic Messages API; the decision is the forced input of one tool."""
    URL = "https://api.anthropic.com/v1/messages"
    LABEL, KEY = "Claude", "ANTHROPIC_API_KEY"

    def __init__(self, settings: Agent):
        self.settings = settings

    def request(self, prompt: str, images: tuple[Path, ...], names: tuple[str, ...]) -> dict:
        content = [{"type": "image", "source": {"type": "base64", "media_type": MEDIA[image.suffix.lower()],
                                                "data": image_data(image)}} for image in images]
        content.append({"type": "text", "text": prompt})
        return {"model": self.settings.model_id, "max_tokens": 1024,
                "messages": [{"role": "user", "content": content}],
                "tools": [{"name": "decide", "description": "The one structured decision for this turn.",
                           "input_schema": schema(names)}],
                "tool_choice": {"type": "tool", "name": "decide"}}

    def generate(self, prompt: str, *, images: tuple[Path, ...], names: tuple[str, ...],
                 attempt: Path, tick: Callable[[], None], cancelled: Callable[[], bool]) -> Decision:
        headers = {"x-api-key": key(self.KEY), "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        body = self.request(prompt, images, names)
        reply = run_request("Claude API", self.URL, headers, body, prompt=prompt, attempt=attempt,
                            timeout=self.settings.timeout, tick=tick, cancelled=cancelled,
                            redacted=redact(body, images))
        try:
            used = next((block for block in reply["content"] if block["type"] == "tool_use"), None)
            if used is None:
                raise ProviderError(f"Claude API returned no decision: {json.dumps(reply)[:500]}")
            return Decision.parse(json.dumps(used["input"]), names)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ProviderError(f"Claude API reply had an unexpected shape: {json.dumps(reply)[:500]}") from exc


class OpenAiApi:
    """OpenAI Chat Completions API with a strict JSON schema response."""
    URL = "https://api.openai.com/v1/chat/completions"
    LABEL, KEY = "OpenAI", "OPENAI_API_KEY"

    def __init__(self, settings: Agent):
        self.settings = settings

    def request(self, prompt: str, images: tuple[Path, ...], names: tuple[str, ...]) -> dict:
        content = [{"type": "image_url", "image_url": {"url": f"data:{MEDIA[image.suffix.lower()]};base64,{image_data(image)}"}}
                   for image in images]
        content.append({"type": "text", "text": prompt})
        return {"model": self.settings.model_id, "messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "decision", "strict": True, "schema": schema(names)}}}

    def generate(self, prompt: str, *, images: tuple[Path, ...], names: tuple[str, ...],
                 attempt: Path, tick: Callable[[], None], cancelled: Callable[[], bool]) -> Decision:
        headers = {"Authorization": f"Bearer {key(self.KEY)}", "content-type": "application/json"}
        body = self.request(prompt, images, names)
        reply = run_request("OpenAI API", self.URL, headers, body, prompt=prompt, attempt=attempt,
                            timeout=self.settings.timeout, tick=tick, cancelled=cancelled,
                            redacted=redact(body, images))
        try:
            content = reply["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"OpenAI API returned no decision: {json.dumps(reply)[:500]}") from exc
        return Decision.parse(content, names)


class LocalVlm(OpenAiApi):
    """An open-weight model served by vLLM's OpenAI-compatible Chat Completions API at `agent.local_url` (the key
    in VLLM_API_KEY): free (LOCAL). Points are the model's own normalised-1000 coordinates (the Qwen3-VL
    convention), unchanged.

    The reply's arguments are a JSON object in the response schema, not a JSON-encoded string, so guided decoding
    guarantees they parse (a small model drops braces in a long escaped plan tree); they are encoded back into
    arguments_json for Decision.parse. Only the turn's allowed decisions are offered (the schema is not part of the
    cached text). agent.local_layout puts the text before the images (the server's prefix cache serves the opening
    every turn repeats) or after them; agent.local_image_labels names each image ("Image 1:") for models that see
    them only by position. Sampling is agent.local_temperature, not the checkpoint's defaults. A MODELS row with
    effort "think" lets the model reason before its answer (a reasoning parser on the server keeps it out of the
    content; any <think> block left in the content is dropped)."""
    LABEL, KEY = "vLLM", "VLLM_API_KEY"
    LOCAL = True
    MAX_TOKENS, THINK_TOKENS = 2048, 4096
    THINK_NOTE = ("Reason briefly inside <think></think> about the images and the last results first; then give "
                  "the decision as the JSON object alone.")

    @property
    def URL(self) -> str:  # noqa: N802 -- the parent's class constant, from settings here
        return self.settings.local_url.rstrip("/") + "/chat/completions"

    allowed: tuple[str, ...] = ()

    def turn(self, allowed: tuple[str, ...]):
        """The turn's allowed decisions (the session says them each turn): the response schema offers only these,
        so a small model cannot answer with another plan phase's skill."""
        self.allowed = tuple(allowed)

    @property
    def thinking(self) -> bool:
        return self.settings.effort == "think"

    def request(self, prompt: str, images: tuple[Path, ...], names: tuple[str, ...]) -> dict:
        offered = tuple(n for n in names if n in self.allowed) or names
        pictures = []
        for number, image in enumerate(images, 1):
            if self.settings.local_image_labels:
                pictures.append({"type": "text", "text": f"Image {number}:"})
            pictures.append({"type": "image_url",
                             "image_url": {"url": f"data:{MEDIA[image.suffix.lower()]};base64,{image_data(image)}"}})
        text = {"type": "text", "text": prompt}
        content = [text, *pictures] if self.settings.local_layout == "text-first" else [*pictures, text]
        messages = [{"role": "user", "content": content}]
        if self.thinking:
            messages.insert(0, {"role": "system", "content": self.THINK_NOTE})
        decision = schema(offered)
        decision["properties"] = {**{k: v for k, v in decision["properties"].items() if k != "arguments_json"},
                                  "arguments_json": {"type": "object"}}
        return {"model": self.settings.model_id, "messages": messages,
                "response_format": {"type": "json_schema", "json_schema": {"name": "decision", "schema": decision}},
                "max_tokens": self.THINK_TOKENS if self.thinking else self.MAX_TOKENS,
                "temperature": self.settings.local_temperature,
                "chat_template_kwargs": {"enable_thinking": self.thinking}}

    @staticmethod
    def content(raw: str) -> str:
        """The reply as Decision.parse reads it: any <think> block dropped, the arguments object encoded."""
        raw = re.sub(r"^.*?</think>", "", raw, flags=re.DOTALL).strip()
        try:
            value = json.loads(raw)
        except ValueError:
            return raw  # Decision.parse names the fault
        if isinstance(value, dict) and isinstance(value.get("arguments_json"), dict):
            value = {**value, "arguments_json": json.dumps(value["arguments_json"])}
        return json.dumps(value)

    def generate(self, prompt: str, *, images: tuple[Path, ...], names: tuple[str, ...],
                 attempt: Path, tick: Callable[[], None], cancelled: Callable[[], bool]) -> Decision:
        headers = {"Authorization": f"Bearer {key(self.KEY)}", "content-type": "application/json"}
        body = self.request(prompt, images, names)
        reply = run_request("vLLM", self.URL, headers, body, prompt=prompt, attempt=attempt,
                            timeout=self.settings.timeout, tick=tick, cancelled=cancelled, redacted=redact(body, images))
        try:
            raw = reply["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"vLLM returned no decision: {json.dumps(reply)[:500]}") from exc
        return Decision.parse(self.content(raw), names)


class LiteLlm:
    """OpenAI Responses API through a LiteLLM proxy (LITELLM_BASE_URL, default 127.0.0.1:4000) holding a
    ChatGPT sign-in: only this request reaches the model. The text goes before the images, and the cache key is
    the session's own (cache_key, set by the session), so the instructions every turn repeats are served from the
    prompt cache without every concurrent session crowding one key."""
    LABEL, KEY = "LiteLLM", "LITELLM_API_KEY"
    BASE = "http://127.0.0.1:4000"
    CACHE_KEY = "homebody-agent"

    def __init__(self, settings: Agent):
        self.settings = settings
        self.cache_key = self.CACHE_KEY

    @staticmethod
    def effort(settings: Agent) -> str:
        return {"max": "xhigh"}.get(settings.effort, settings.effort)

    def request(self, prompt: str, images: tuple[Path, ...], names: tuple[str, ...]) -> dict:
        content = [{"type": "input_text", "text": prompt}]
        content += [{"type": "input_image", "image_url": f"data:{MEDIA[image.suffix.lower()]};base64,{image_data(image)}"}
                    for image in images]
        return {"model": self.settings.model_id, "input": [{"role": "user", "content": content}],
                "reasoning": {"effort": self.effort(self.settings)}, "prompt_cache_key": self.cache_key,
                "text": {"format": {"type": "json_schema", "name": "decision", "strict": True, "schema": schema(names)}}}

    def generate(self, prompt: str, *, images: tuple[Path, ...], names: tuple[str, ...],
                 attempt: Path, tick: Callable[[], None], cancelled: Callable[[], bool]) -> Decision:
        headers = {"Authorization": f"Bearer {key(self.KEY)}", "content-type": "application/json"}
        url = os.environ.get("LITELLM_BASE_URL", self.BASE).rstrip("/") + "/v1/responses"
        body = self.request(prompt, images, names)
        reply = run_request("LiteLLM", url, headers, body, prompt=prompt, attempt=attempt,
                            timeout=self.settings.timeout, tick=tick, cancelled=cancelled,
                            redacted=redact(body, images))
        texts = [part.get("text", "") for item in reply.get("output") or [] if item.get("type") == "message"
                 for part in item.get("content") or [] if part.get("type") == "output_text"]
        if not texts:
            raise ProviderError(f"LiteLLM returned no decision: {json.dumps(reply)[:500]}")
        return Decision.parse(texts[-1], names)


def redact(body: dict, images: tuple[Path, ...]) -> dict:
    """The request as recorded: image bytes replaced by the image file names, in order."""
    names = iter(image.name for image in images)
    record = json.loads(json.dumps(body))
    for message in record.get("messages") or record.get("input") or []:
        for part in message["content"]:
            if part.get("type") == "image":
                part["source"] = {"file": next(names)}
            elif part.get("type") == "image_url":
                part["image_url"] = {"file": next(names)}
            elif part.get("type") == "input_image":
                part["image_url"] = next(names)
    return record
