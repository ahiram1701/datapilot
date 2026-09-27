import pytest

from app.llm.factory import get_provider
from app.llm.mock_p import MockProvider


@pytest.mark.parametrize("name, key_var, url_part, model", [
    ("groq", "GROQ_API_KEY", "api.groq.com", "llama-3.3-70b-versatile"),
    ("gemini", "GEMINI_API_KEY", "generativelanguage.googleapis.com", "gemini-3.8-flash"),
])
def test_free_providers_use_compatible_endpoint(monkeypatch, name, key_var, url_part, model):
    monkeypatch.setenv(key_var, "test-key")
    monkeypatch.delenv(f"{name.upper()}_MODEL", raising=False)
    provider = get_provider(name)
    assert provider.name == name
    assert url_part in str(provider.client.base_url)
    assert provider.model == model


def test_model_can_be_overridden(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
    assert get_provider("groq").model == "llama-3.1-8b-instant"


def test_missing_key_gives_helpful_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY.*aistudio"):
        get_provider("gemini")


def test_ollama_needs_no_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert "11434" in str(get_provider("ollama").client.base_url)


def test_default_is_mock(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert isinstance(get_provider(), MockProvider)


def test_unknown_provider():
    with pytest.raises(ValueError):
        get_provider("nope")


def test_empty_env_values_fall_back_to_defaults(monkeypatch):
    # En .env es común dejar "GROQ_MODEL=" vacío: debe usarse el modelo por defecto
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "")
    monkeypatch.setenv("GROQ_BASE_URL", "")
    provider = get_provider("groq")
    assert provider.model == "llama-3.3-70b-versatile"
    assert "api.groq.com" in str(provider.client.base_url)
