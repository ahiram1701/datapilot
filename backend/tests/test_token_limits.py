from types import SimpleNamespace

from app.agents.react import ReActAgent
from app.agents.tools import ToolRegistry, schema
from app.llm.base import TRUNCATED_NOTE, LLMResponse, ToolCall
from app.llm.mock_p import MockProvider
from app.llm.openai_p import OpenAIProvider
from app.main import friendly_error


class FakeCompletions:
    """Sustituye client.chat.completions: guarda los kwargs y responde fijo."""

    def __init__(self, finish_reason="stop", content="hola"):
        self.kwargs = None
        self.finish_reason = finish_reason
        self.content = content

    def create(self, **kwargs):
        self.kwargs = kwargs
        msg = SimpleNamespace(content=self.content, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=self.finish_reason)])


def make_provider(monkeypatch, fake, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    p = OpenAIProvider(api_key="test")
    p.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return p


def test_max_tokens_and_reasoning_effort_are_sent(monkeypatch):
    fake = FakeCompletions()
    make_provider(monkeypatch, fake, LLM_MAX_TOKENS="800", LLM_REASONING_EFFORT="low").chat("s", [])
    assert fake.kwargs["max_tokens"] == 800
    assert fake.kwargs["reasoning_effort"] == "low"


def test_defaults_when_env_missing(monkeypatch):
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    fake = FakeCompletions()
    make_provider(monkeypatch, fake).chat("s", [])
    assert fake.kwargs["max_tokens"] == 1024
    assert "reasoning_effort" not in fake.kwargs  # no se envía si no está configurado


def test_truncated_response_is_marked(monkeypatch):
    fake = FakeCompletions(finish_reason="length", content="respuesta a medi")
    resp = make_provider(monkeypatch, fake).chat("s", [])
    assert resp.text == "respuesta a medi" + TRUNCATED_NOTE


def test_observation_limit_from_env(monkeypatch):
    monkeypatch.setenv("LLM_MAX_OBSERVATION_CHARS", "10")
    tools = ToolRegistry()
    tools.add("big", "devuelve mucho texto", schema(), lambda: "x" * 100)
    agent = ReActAgent("A", "s", MockProvider([
        LLMResponse(text=None, tool_calls=[ToolCall("1", "big", {})]),
        LLMResponse(text="fin"),
    ]), tools)
    steps = []
    agent.run("t", steps.append)
    obs = next(s for s in steps if s.type == "observation")
    assert obs.content == "x" * 10 + "... [truncado]"


def test_friendly_rate_limit_message():
    RateLimitError = type("RateLimitError", (Exception,), {})
    msg = friendly_error(RateLimitError("429 tokens per minute"))
    assert "LLM_MAX_TOKENS" in msg and "429 tokens per minute" in msg
    assert friendly_error(ValueError("x")) == "ValueError: x"
