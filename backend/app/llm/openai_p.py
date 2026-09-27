"""Proveedor para la API de OpenAI y cualquier servidor compatible
(Ollama, Groq, Gemini): mismo protocolo, distinta base_url."""
import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .base import LLMProvider, LLMResponse, ToolCall, ToolSpec


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, max_retries: int = 2):
        # max_retries: el SDK reintenta 429/5xx con backoff exponencial
        self.client = OpenAI(base_url=base_url, api_key=api_key or os.getenv("OPENAI_API_KEY"),
                             max_retries=max_retries)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    @staticmethod
    def _to_api_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("content")}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{
                        "id": tc.id, "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    } for tc in m["tool_calls"]]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"],
                            "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def chat(self, system, messages, tools=None) -> LLMResponse:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters,
            }} for t in tools]
        resp = self.client.chat.completions.create(
            model=self.model, messages=self._to_api_messages(system, messages), **kwargs,
        )
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(tc.id, tc.function.name, args))
        return LLMResponse(text=msg.content, tool_calls=calls)


@dataclass(frozen=True)
class CompatPreset:
    """Configuración de un servicio compatible con la API de OpenAI."""
    env_prefix: str            # GROQ -> GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL
    base_url: str
    default_model: str
    needs_key: bool = True
    key_url: str = ""          # dónde conseguir la key (para el mensaje de error)


COMPAT_PRESETS: dict[str, CompatPreset] = {
    "ollama": CompatPreset("OLLAMA", "http://localhost:11434/v1", "qwen2.5:7b", needs_key=False),
    "groq": CompatPreset("GROQ", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile",
                         key_url="https://console.groq.com/keys"),
    "gemini": CompatPreset("GEMINI", "https://generativelanguage.googleapis.com/v1beta/openai/",
                           "gemini-3.8-flash", key_url="https://aistudio.google.com/apikey"),
}


def make_compat_provider(name: str) -> OpenAIProvider:
    p = COMPAT_PRESETS[name]
    key = os.getenv(f"{p.env_prefix}_API_KEY")
    if p.needs_key and not key:
        raise ValueError(f"Falta {p.env_prefix}_API_KEY en .env (consíguela gratis en {p.key_url})")
    provider = OpenAIProvider(
        model=os.getenv(f"{p.env_prefix}_MODEL", p.default_model),
        base_url=os.getenv(f"{p.env_prefix}_BASE_URL", p.base_url),
        api_key=key or name,  # Ollama ignora la key, pero el cliente exige una
        max_retries=5,        # los planes gratuitos devuelven 429 con frecuencia
    )
    provider.name = name
    return provider
