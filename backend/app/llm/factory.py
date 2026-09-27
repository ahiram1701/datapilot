import os

from .base import LLMProvider


def get_provider(name: str | None = None) -> LLMProvider:
    """Crea el proveedor según LLM_PROVIDER. Imports perezosos: no hace falta
    instalar el SDK de un proveedor que no usas."""
    name = (name or os.getenv("LLM_PROVIDER", "mock")).lower()
    if name == "anthropic":
        from .anthropic_p import AnthropicProvider
        return AnthropicProvider()
    if name == "openai":
        from .openai_p import OpenAIProvider
        return OpenAIProvider()
    if name == "ollama":
        from .openai_p import OllamaProvider
        return OllamaProvider()
    if name == "mock":
        from .mock_p import MockProvider, demo_script
        return MockProvider(demo_script())
    raise ValueError(f"Proveedor LLM desconocido: {name}")
