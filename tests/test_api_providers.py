"""The Anthropic and OpenAI API providers: one request, the schema, the images, no fallback."""
import http.client
import io
import json
import threading
import time
import urllib.error

import pytest

from homebody.session.settings import Agent
from homebody.vlm import api, provider
from homebody.vlm.api import ClaudeApi, LocalVlm, OpenAiApi
from homebody.vlm.provider import Cancelled, ProviderError

PNG = (b"\x89PNG\r\n\x1a\n" + bytes(16))
DECISION = {"text": "Nothing left to do.", "skill": "done", "arguments_json": "{}"}
REPLIES = {ClaudeApi: {"content": [{"type": "tool_use", "name": "decide", "input": DECISION}]},
           OpenAiApi: {"choices": [{"message": {"content": json.dumps(DECISION)}}]}}
PROVIDERS = [(ClaudeApi, "ANTHROPIC_API_KEY"), (OpenAiApi, "OPENAI_API_KEY")]


def fake_transport(monkeypatch, reply, status=200, delay=.05):
    """Replace urlopen with a recorder that waits DELAY seconds, long enough for the caller's
    loop to tick the physics at least once, then answers REPLY (a JSON value, raw bytes, or an
    exception to raise) or an HTTP error whose body echoes the request's headers, as a careless
    proxy would."""
    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return self.payload if isinstance(self.payload, bytes) else json.dumps(self.payload).encode()

    def urlopen(request, timeout):
        calls.append((request.full_url, dict(request.header_items()), json.loads(request.data), timeout))
        time.sleep(delay)
        if isinstance(reply, Exception):
            raise reply
        if status != 200:
            body = f"denied with headers {dict(request.header_items())}".encode()
            raise urllib.error.HTTPError(request.full_url, status, "boom", {}, io.BytesIO(body))
        return Response(reply)

    monkeypatch.setattr(provider.urllib.request, "urlopen", urlopen)
    return calls


def images(tmp_path, count=2):
    paths = []
    for i in range(count):
        path = tmp_path / f"observation-{i}.png"
        path.write_bytes(PNG)
        paths.append(path)
    return tuple(paths)


def request(tmp_path, name="a", **overrides):
    """One run_request call with every argument explicit; OVERRIDES replace any of them."""
    arguments = {"prompt": "p", "attempt": tmp_path / name, "timeout": 5., "tick": lambda: None,
                 "cancelled": lambda: False, **overrides}
    return provider.run_request("Test API", "https://api.invalid/v1", {"x-api-key": "k"},
                                {"model": "m"}, **arguments)


def attempt_files(attempt):
    return {path.name: path.read_text() for path in attempt.iterdir()}


def test_claude_api_forces_the_decision_tool_and_attaches_the_images(tmp_path, monkeypatch):
    """The recorded request names each image file instead of holding its bytes a second time,
    and the physics owner keeps ticking while the request runs."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k-test")
    calls = fake_transport(monkeypatch, {"content": [{"type": "tool_use", "name": "decide", "input": {
        "text": "The carton is in reach.", "skill": "pick", "arguments_json": '{"hand": "right"}'}}]})
    ticks = []
    decision = ClaudeApi(Agent(model="opus-api")).generate(
        "prompt text", images=images(tmp_path), names=("pick", "done"), attempt=tmp_path / "attempt",
        tick=lambda: ticks.append(1), cancelled=lambda: False)
    assert (decision.skill, decision.arguments, decision.text) == ("pick", {"hand": "right"}, "The carton is in reach.")
    url, headers, body, _ = calls[0]
    assert url == ClaudeApi.URL and headers["X-api-key"] == "k-test" and body["tool_choice"] == {"type": "tool", "name": "decide"}
    assert body["tools"][0]["input_schema"]["properties"]["skill"]["enum"] == ["pick", "done"]
    parts = body["messages"][0]["content"]
    assert [part["type"] for part in parts] == ["image", "image", "text"] and parts[-1]["text"] == "prompt text"
    assert parts[0]["source"]["data"] == provider.image_data(images(tmp_path)[0])
    recorded = json.loads((tmp_path / "attempt/request.json").read_text())
    assert recorded["messages"][0]["content"][0]["source"] == {"file": "observation-0.png"}
    assert json.loads((tmp_path / "attempt/status.json").read_text())["status"] == "ok"
    assert ticks


def test_openai_api_requests_a_strict_schema_and_parses_the_content(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k-test")
    calls = fake_transport(monkeypatch, REPLIES[OpenAiApi])
    decision = OpenAiApi(Agent(model="astra-api")).generate(
        "prompt text", images=images(tmp_path, 1), names=("pick", "done"), attempt=tmp_path / "attempt",
        tick=lambda: None, cancelled=lambda: False)
    assert (decision.skill, decision.arguments) == ("done", {})
    url, headers, body, _ = calls[0]
    assert url == OpenAiApi.URL and headers["Authorization"] == "Bearer k-test" and body["model"] == "gpt-6-astra"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize("cls, variable", PROVIDERS)
def test_api_providers_refuse_without_a_key_and_never_fall_back(tmp_path, monkeypatch, cls, variable):
    monkeypatch.delenv(variable, raising=False)
    with pytest.raises(ProviderError, match=variable):
        cls(Agent()).generate("p", images=(), names=("done",), attempt=tmp_path / "a", tick=lambda: None,
                              cancelled=lambda: False)
    monkeypatch.setenv(variable, "k")
    fake_transport(monkeypatch, {}, status=403)
    with pytest.raises(ProviderError, match="HTTP 403"):
        cls(Agent()).generate("p", images=(), names=("done",), attempt=tmp_path / "b", tick=lambda: None,
                              cancelled=lambda: False)
    assert json.loads((tmp_path / "b/status.json").read_text())["status"] == "error"


@pytest.mark.parametrize("cls, variable", PROVIDERS)
def test_attempt_evidence_never_holds_the_key(tmp_path, monkeypatch, cls, variable):
    """A key sourced from a file often ends in a newline: it is sent without it, and no
    attempt file ever holds the key."""
    monkeypatch.setenv(variable, "k-secret-zz\n")
    fake_transport(monkeypatch, {}, status=401)
    with pytest.raises(ProviderError, match="HTTP 401"):
        cls(Agent()).generate("p", images=images(tmp_path, 1), names=("done",), attempt=tmp_path / "a",
                              tick=lambda: None, cancelled=lambda: False)
    files = attempt_files(tmp_path / "a")
    assert set(files) == {"prompt.txt", "request.json", "status.json"}
    assert not any("k-secret-zz" in text for text in files.values())
    assert "[redacted]" in json.loads(files["status.json"])["error"]
    calls = fake_transport(monkeypatch, REPLIES[cls])
    decision = cls(Agent()).generate("p", images=images(tmp_path, 1), names=("done",), attempt=tmp_path / "b",
                                     tick=lambda: None, cancelled=lambda: False)
    assert decision.skill == "done"
    assert any(value.endswith("k-secret-zz") for value in calls[0][1].values())
    files = attempt_files(tmp_path / "b")
    assert set(files) == {"prompt.txt", "request.json", "reply.json", "status.json"}
    assert not any("k-secret-zz" in text for text in files.values())


@pytest.mark.parametrize("cls, variable", PROVIDERS)
@pytest.mark.parametrize("value", ["k-secret\nzz", "k-secret…zz"])
def test_a_key_no_header_can_carry_is_refused_before_any_request(tmp_path, monkeypatch, cls, variable, value):
    """http.client refuses such a header inside the request thread, quoting the whole value
    in its error on stderr; the key must stop at the environment."""
    monkeypatch.setenv(variable, value)
    calls = fake_transport(monkeypatch, REPLIES[cls])
    with pytest.raises(ProviderError, match=f"{variable} holds") as refused:
        cls(Agent()).generate("p", images=(), names=("done",), attempt=tmp_path / "a", tick=lambda: None,
                              cancelled=lambda: False)
    assert not calls and "k-secret" not in str(refused.value) and not (tmp_path / "a").exists()


def test_api_decision_outside_the_schema_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    fake_transport(monkeypatch, {"content": [{"type": "text", "text": "I would rather chat."}]})
    with pytest.raises(ProviderError, match="no decision"):
        ClaudeApi(Agent()).generate("p", images=(), names=("done",), attempt=tmp_path / "a", tick=lambda: None,
                                    cancelled=lambda: False)
    assert api.key("ANTHROPIC_API_KEY") == "k"


def request_thread():
    return next(thread for thread in threading.enumerate() if thread.name == "Test API request")


def test_a_cancelled_request_returns_at_once_and_leaves_only_a_daemon_thread(tmp_path, monkeypatch):
    """The worker is joined at the end so its fake request ends before the next test."""
    fake_transport(monkeypatch, {}, delay=.4)
    started = time.monotonic()
    with pytest.raises(Cancelled):
        request(tmp_path, cancelled=lambda: True)
    assert time.monotonic() - started < .3
    assert json.loads((tmp_path / "a/status.json").read_text())["status"] == "cancelled"
    worker = request_thread()
    assert worker.daemon and worker.is_alive()
    worker.join()


def test_a_timed_out_request_is_a_provider_error_without_a_retry(tmp_path, monkeypatch):
    calls = fake_transport(monkeypatch, {}, delay=.3)
    with pytest.raises(ProviderError, match="timed out after 0.05s"):
        request(tmp_path, timeout=.05)
    assert json.loads((tmp_path / "a/status.json").read_text())["status"] == "error"
    request_thread().join()
    assert len(calls) == 1


def test_an_error_reply_cut_off_in_transit_is_a_provider_error(tmp_path, monkeypatch):
    """An overloaded server's error body can end early; reading it must not kill the thread."""
    class CutOff(io.BytesIO):
        def read(self, *_):
            raise http.client.IncompleteRead(b"over", 40)

    def urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 529, "overloaded", {}, CutOff())
    monkeypatch.setattr(provider.urllib.request, "urlopen", urlopen)
    with pytest.raises(ProviderError, match="HTTP 529: body unreadable"):
        request(tmp_path)
    assert json.loads((tmp_path / "a/status.json").read_text())["status"] == "error"


@pytest.mark.parametrize("reply, message", [
    ([], "returned no JSON object"), (b"<html>", "returned no JSON"),
    (http.client.IncompleteRead(b"partial"), "transport failed: IncompleteRead")])
def test_a_reply_that_is_not_a_json_object_or_a_protocol_failure_is_a_provider_error(
        tmp_path, monkeypatch, reply, message):
    """A raw reply, when one arrived, stays in the attempt as evidence."""
    fake_transport(monkeypatch, reply)
    with pytest.raises(ProviderError, match=message):
        request(tmp_path)
    assert json.loads((tmp_path / "a/status.json").read_text())["status"] == "error"
    if not isinstance(reply, Exception):
        assert (tmp_path / "a/reply.json").read_text() in ("[]", "<html>")


def test_the_litellm_request_carries_the_sessions_own_cache_key():
    from homebody.session.config import Agent
    from homebody.vlm.api import LiteLlm
    provider = LiteLlm(Agent(model="sol61-litellm"))
    assert provider.request("p", (), ("pick", "done"))["prompt_cache_key"] == "homebody-agent"
    provider.cache_key = "homebody-session-20261021T000000Z-abcd1234"
    assert provider.request("p", (), ("pick", "done"))["prompt_cache_key"] == provider.cache_key


def test_a_local_vllm_model_sends_text_before_images_to_its_configured_server(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "k-local")
    calls = fake_transport(monkeypatch, REPLIES[OpenAiApi])
    settings = Agent(model="robobrain25-8b-vllm", local_url="http://10.0.0.5:8000/v1/")
    decision = LocalVlm(settings).generate(
        "prompt text", images=images(tmp_path, 2), names=("pick", "done"), attempt=tmp_path / "attempt",
        tick=lambda: None, cancelled=lambda: False)
    assert (decision.skill, decision.arguments) == ("done", {})
    url, headers, body, _ = calls[0]
    assert url == "http://10.0.0.5:8000/v1/chat/completions" and headers["Authorization"] == "Bearer k-local"
    assert body["model"] == "robobrain2.5-8b" and body["max_tokens"] == LocalVlm.MAX_TOKENS
    content = body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "prompt text"}
    assert [c.get("text") or c["type"] for c in content[1:]] == ["Image 1:", "image_url", "Image 2:", "image_url"]
    assert body["temperature"] == 0.0 and body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"]["json_schema"]["schema"]["properties"]["arguments_json"] == {"type": "object"}
    assert LocalVlm.LOCAL and len(body["messages"]) == 1


def test_a_local_model_may_answer_only_the_turns_allowed_decisions():
    provider = LocalVlm(Agent(model="robobrain25-8b-vllm"))
    offered = ("navigate", "pick", "plan", "done")
    enum = lambda: provider.request("p", (), offered)["response_format"]["json_schema"]["schema"]["properties"]["skill"]["enum"]  # noqa: E731
    assert enum() == list(offered)
    provider.turn(("plan",))
    assert enum() == ["plan"]
    provider.turn(("navigate", "pick", "done"))
    assert enum() == ["navigate", "pick", "done"]


def test_a_local_models_arguments_are_an_object_the_grammar_keeps_valid(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "k-local")
    tree = {"tree": {"type": "sequence", "children": [{"type": "goal", "id": "g1", "goal": "pick", "budget": 2}]}}
    reply = {"choices": [{"message": {"content": json.dumps({"text": "t", "skill": "plan", "arguments_json": tree})}}]}
    fake_transport(monkeypatch, reply)
    decision = LocalVlm(Agent(model="qwen3vl-8b-vllm")).generate(
        "p", images=(), names=("plan", "done"), attempt=tmp_path / "a", tick=lambda: None, cancelled=lambda: False)
    assert (decision.skill, decision.arguments) == ("plan", tree)
    assert LocalVlm.content('{"text": "t", "skill": "done", "arguments_json": "{}"}') == (
        '{"text": "t", "skill": "done", "arguments_json": "{}"}')  # an encoded string still passes through


def test_a_thinking_local_model_reasons_first_and_its_think_block_is_dropped():
    provider = LocalVlm(Agent(model="cosmos-r2-8b-think-vllm", local_layout="images-first", local_temperature=0.6))
    body = provider.request("p", (), ("done",))
    assert body["messages"][0] == {"role": "system", "content": LocalVlm.THINK_NOTE}
    assert body["max_tokens"] == LocalVlm.THINK_TOKENS and body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["temperature"] == 0.6 and body["messages"][1]["content"][-1] == {"type": "text", "text": "p"}
    raw = '<think>the can is left</think>\n{"text": "t", "skill": "done", "arguments_json": {}}'
    assert json.loads(LocalVlm.content(raw)) == {"text": "t", "skill": "done", "arguments_json": "{}"}
    with pytest.raises(ValueError, match="local_layout"):
        Agent(model="qwen3vl-8b-vllm", local_layout="sideways")
