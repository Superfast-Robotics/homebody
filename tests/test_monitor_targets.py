"""The monitor's model contracts: the vocabulary model's prompt and lenient parse, the Decisions API request and
its answers as probabilities."""
import json

import numpy as np
import pytest

from homebody.monitor import decision, vocabulary
from homebody.monitor.targets import (
    NONE_OF_THESE,
    Claim,
    DecisionRequest,
    Label,
    NodeInfo,
    Phase,
    Shot,
    Vocabulary,
    VocabularyRequest,
)
from homebody.session.config import Monitor
from homebody.vlm.provider import ProviderBusy, ProviderError

NODE = NodeInfo("g2", "carry the cup to the table", {"name": "arrived"}, "pending", 6, "sequence")
REQUEST = VocabularyRequest(task="put the cup on the table", tree="sequence\n  [done] g1: pick the cup",
                            node=NODE, done=(NodeInfo("g1", "pick the cup", {"name": "holding", "hand": "left"},
                                                      "done"),),
                            skills=("navigate: walk to a map point",), hands="the left hand HOLDS the cup",
                            stages={"navigate": ("Following route",)})
VOCAB = Vocabulary("g2", (Label("walking", "progress", "the robot walks"), Label("dropped", "failure", "it falls")),
                   standing=(Claim("g1", "the cup is in the left hand"),))


def decision_request():
    shots = tuple(Shot(np.zeros((48, 64, 3), np.uint8), t / 10) for t in range(5))
    return DecisionRequest("put the cup on the table", NODE, Phase("tool_call", "navigate", "Following route"),
                           "the left hand HOLDS the cup", VOCAB, shots)


def test_the_vocabulary_prompt_carries_the_goal_and_never_scene_state():
    text = vocabulary.prompt(REQUEST, 6)
    assert "g2: carry the cup to the table (check: name=arrived)" in text
    assert "navigate: Following route" in text and "at most 6 actions" in text and "$" not in text


def test_the_vocabulary_parse_keeps_good_labels_and_valid_claims():
    reply = "Sure.\n```json\n" + json.dumps({
        "labels": [{"name": "Walking", "kind": "progress", "description": "the robot walks", "stages": ["x"]},
                   {"name": "walking", "kind": "progress", "description": "dup", "stages": []},
                   {"name": "unsafe", "kind": "failure", "description": "static", "stages": []},
                   {"name": "dropped", "kind": "oops", "description": "bad kind", "stages": []},
                   {"name": "dropped", "kind": "failure", "description": "the cup falls", "stages": []}],
        "standing": [{"goal_id": "g1", "text": "the cup is in the left hand"}, {"goal_id": "g9", "text": "no"}],
        "expected_seconds": 20}) + "\n```"
    vocab = vocabulary.parse(reply, REQUEST, 8)
    assert [label.name for label in vocab.labels] == ["walking", "dropped"]
    assert vocab.standing == (Claim("g1", "the cup is in the left hand"),)
    assert vocab.expected_seconds == 20 and vocab.source == "model" and vocab.failures() == ("dropped",)


@pytest.mark.parametrize("reply", ["no json here", json.dumps({"labels": []}), json.dumps([1, 2])])
def test_an_unusable_vocabulary_falls_back(reply):
    vocab = vocabulary.parse(reply, REQUEST, 8)
    assert vocab.source.startswith("fallback") and [label.kind for label in vocab.labels] == ["progress"]


def test_the_vocabulary_model_falls_back_when_the_service_fails(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise ProviderError("down")
    monkeypatch.setattr(vocabulary, "run_request", broken)
    vocab = vocabulary.VocabularyModel(Monitor()).generate(REQUEST, tmp_path / "vocab-000")
    assert vocab.source == "fallback: down"


def test_the_vocabulary_request_turns_thinking_off_and_asks_for_the_schema():
    sent = vocabulary.VocabularyModel(Monitor()).body("hello", ())
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["response_format"]["json_schema"]["schema"] is vocabulary.SCHEMA
    assert (sent["temperature"], sent["top_p"], sent["top_k"], sent["presence_penalty"]) == (0.7, 0.8, 20, 1.5)


def test_the_decisions_request_has_the_documented_shape():
    sent = decision.body(decision_request(), "gpt-6-luna-decisions", 32)
    assert sent["model"] == "gpt-6-luna-decisions"
    parts = sent["input"][0]["content"]
    assert parts[0]["type"] == "input_text" and "skill navigate, stage Following route" in parts[0]["text"]
    assert [p["type"] for p in parts[1:]] == ["input_image"] * 5
    assert all(p["image_url"].startswith("data:image/jpeg;base64,") for p in parts[1:])
    asked = {q["name"]: q for q in sent["questions"]}
    assert list(asked) == ["activity", "unsafe", "human_needed", "still_true_0", "progress"]
    assert [c["value"] for c in asked["activity"]["choices"]] == ["walking", "dropped", NONE_OF_THESE]
    assert asked["unsafe"]["type"] == asked["still_true_0"]["type"] == "predicate"
    assert [level["label"] for level in asked["progress"]["levels"]] == ["no_change", "some", "clear"]
    assert decision.redacted(sent)["input"][0]["content"][1]["image_url"] == "frame 0"


def test_frames_are_scaled_to_the_short_side():
    import base64
    import io

    from PIL import Image
    url = decision.jpeg_url(np.zeros((480, 640, 3), np.uint8), 240)
    image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert image.size == (320, 240)


def test_the_answers_become_a_reading():
    reply = {"answers": [
        {"name": "activity", "type": "choice", "choice": "walking", "confidence": 0.9,
         "probabilities": [{"value": "walking", "probability": 0.9}, {"value": "dropped", "probability": 0.08},
                           {"value": NONE_OF_THESE, "probability": 0.02}]},
        {"name": "unsafe", "type": "predicate", "probability": 0.01},
        {"name": "human_needed", "type": "refusal", "refusal": "no"},
        {"name": "still_true_0", "type": "predicate", "probability": 0.97},
        {"name": "progress", "type": "score", "score": 1.8, "probabilities": [
            {"value": 0, "label": "no_change", "probability": 0.05}, {"value": 1, "label": "some", "probability": 0.1},
            {"value": 2, "label": "clear", "probability": 0.85}]}],
        "usage": {"input_tokens": 5000}}
    reading = decision.parse(reply, decision_request(), latency_s=0.4, usd_per_mtok=0.10)
    assert reading.top() == ("walking", 0.9) and reading.unsafe == 0.01 and reading.human_needed is None
    assert reading.refusals == ("human_needed",) and reading.still_true == {"g1": 0.97}
    assert reading.progress["clear"] == 0.85 and reading.sim_time == pytest.approx(0.4)
    assert reading.cost_usd == pytest.approx(0.0005) and reading.phase.skill == "navigate"


def test_missing_answers_are_refusals():
    reading = decision.parse({"answers": []}, decision_request())
    assert set(reading.refusals) == {"activity", "unsafe", "human_needed", "still_true_0", "progress"}


class Response:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.text.encode()


def test_the_decision_call_posts_to_the_proxy_with_the_key(monkeypatch):
    seen = {}

    def urlopen(request, timeout):
        seen.update(url=request.full_url, auth=request.headers["Authorization"], timeout=timeout)
        return Response(json.dumps({"answers": []}))
    monkeypatch.setenv("OPENAI_DECISIONS_API_KEY", "sk-test")
    monkeypatch.setattr(decision.OPENER, "open", urlopen)
    reading, recorded = decision.OpenAiDecisions(Monitor()).decide(decision_request())
    assert seen == {"url": "https://api.openai.com/v1/decisions", "auth": "Bearer sk-test", "timeout": 15.0}
    assert "base64" not in json.dumps(recorded) and reading.goal_id == "g2"


def test_a_busy_service_raises_busy_with_the_key_scrubbed(monkeypatch):
    import io
    import urllib.error
    from email.message import Message

    def urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 429, "busy", Message(), io.BytesIO(b"slow down sk-test"))
    monkeypatch.setenv("OPENAI_DECISIONS_API_KEY", "sk-test")
    monkeypatch.setattr(decision.OPENER, "open", urlopen)
    with pytest.raises(ProviderBusy) as caught:
        decision.OpenAiDecisions(Monitor()).decide(decision_request())
    assert "sk-test" not in str(caught.value)


def test_no_credit_is_a_hard_error_not_busy(monkeypatch):
    import io
    import urllib.error
    from email.message import Message

    def urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 429, "quota", Message(),
                                     io.BytesIO(b'{"error": {"code": "credit_balance_exhausted"}}'))
    monkeypatch.setenv("OPENAI_DECISIONS_API_KEY", "sk-test")
    monkeypatch.setattr(decision.OPENER, "open", urlopen)
    with pytest.raises(decision.OutOfCredit):
        decision.OpenAiDecisions(Monitor()).decide(decision_request())


def test_decisions_read_only_their_own_key(monkeypatch):
    monkeypatch.delenv("OPENAI_DECISIONS_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-shared")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-proxy")
    with pytest.raises(ProviderError, match="OPENAI_DECISIONS_API_KEY is not set"):
        decision.OpenAiDecisions(Monitor()).decide(decision_request())


def test_a_redirect_is_refused_so_the_key_stays_with_its_host():
    import urllib.request
    from email.message import Message
    handler = decision.NoRedirect()
    request = urllib.request.Request("https://api.openai.com/v1/decisions")
    with pytest.raises(decision.urllib.error.HTTPError, match="redirect to https://elsewhere"):
        handler.redirect_request(request, None, 302, "Found", Message(), "https://elsewhere/x")


@pytest.mark.parametrize("field", ["decision_base_url", "vocab_base_url"])
def test_the_monitor_sends_frames_and_keys_only_over_https_or_to_this_machine(field):
    Monitor(**{field: "http://127.0.0.1:4000/v1"})
    with pytest.raises(ValueError, match="https"):
        Monitor(**{field: "http://example.com/v1"})


def test_malformed_usage_does_not_break_a_reading():
    reading = decision.parse({"answers": [], "usage": {"input_tokens": "lots"}}, decision_request())
    assert reading.input_tokens == 0


def test_the_vocabulary_key_comes_from_its_file_else_the_proxy_key(tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "sk-proxy")
    assert vocabulary.api_key(str(tmp_path / "missing")) == "sk-proxy"
    (tmp_path / "api_key").write_text("sk-vllm\n")
    assert vocabulary.api_key(str(tmp_path / "api_key")) == "sk-vllm"


def test_a_private_network_server_is_allowed_but_not_a_public_http_one():
    Monitor(vocab_base_url="http://10.19.89.218:9002/v1")
    with pytest.raises(ValueError, match="private network"):
        Monitor(vocab_base_url="http://8.8.8.8/v1")
