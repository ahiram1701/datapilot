"""Interfaz común para cualquier proveedor de LLM.

Los agentes solo conocen estos tipos neutrales; cada proveedor traduce
hacia/desde el formato de su API. Así cambiar de Claude a OpenAI u Ollama
es una variable de entorno, no un refactor.

Formato neutral de mensajes (lista de dicts):
    {"role": "user", "content": str}
    {"role": "assistant", "content": str | None, "tool_calls": [ToolCall], "raw": Any}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    """Descripción de una herramienta en JSON Schema (lo que ve el LLM)."""
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Respuesta original del proveedor: algunos (Anthropic) necesitan que
    # se reenvíe tal cual en el siguiente turno (p.ej. bloques de thinking).
    raw: Any = None

    def to_message(self) -> dict[str, Any]:
        return {"role": "assistant", "content": self.text,
                "tool_calls": self.tool_calls, "raw": self.raw}


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def chat(self, system: str, messages: list[dict[str, Any]],
             tools: list[ToolSpec] | None = None) -> LLMResponse:
        """Envía la conversación y devuelve una respuesta normalizada."""
