"""Proveedor falso y determinista: permite correr tests y CI sin API keys.

Recibe un "guion" de respuestas; si se acaba, responde con texto final.
"""
from typing import Any

from .base import LLMProvider, LLMResponse, ToolCall


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self, script: list[LLMResponse] | None = None):
        self.script = list(script or [])
        self.calls: list[dict[str, Any]] = []  # registro para aserciones en tests

    def chat(self, system, messages, tools=None) -> LLMResponse:
        self.calls.append({"system": system, "messages": list(messages),
                           "tools": [t.name for t in tools or []]})
        if self.script:
            return self.script.pop(0)
        return LLMResponse(text="(mock) Análisis completado.")


def demo_script() -> list[LLMResponse]:
    """Guion usado cuando LLM_PROVIDER=mock en la app, para una demo sin keys."""
    # Las respuestas se consumen en orden a través de todos los agentes:
    # Orchestrator -> DataAnalyst (2 tools + final) -> Orchestrator final.
    return [
        LLMResponse(text="Delego la exploración al analista de datos.",
                    tool_calls=[ToolCall("m1", "ask_data_analyst",
                                         {"task": "Describe el dataset y sus correlaciones"})]),
        LLMResponse(text="Primero exploro el dataset.",
                    tool_calls=[ToolCall("m2", "describe_dataset", {})]),
        LLMResponse(text="Ahora veo correlaciones.",
                    tool_calls=[ToolCall("m3", "correlations", {})]),
        LLMResponse(text="(mock) Dataset descrito y correlaciones calculadas."),
        LLMResponse(text="(mock) El analista revisó el dataset. Configura "
                         "LLM_PROVIDER=groq|gemini|anthropic|openai|ollama para un análisis real."),
    ]
