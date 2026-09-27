import os
from typing import Any

import anthropic

from .base import LLMProvider, LLMResponse, ToolCall, ToolSpec


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str | None = None):
        self.client = anthropic.Anthropic()
        self.model = model or (os.getenv("ANTHROPIC_MODEL") or "claude-opus-5")

    def _to_api_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                # Reenviamos el contenido original (incluye thinking/tool_use)
                if m.get("raw") is not None:
                    out.append({"role": "assistant", "content": m["raw"]})
                else:
                    blocks: list[dict[str, Any]] = []
                    if m.get("content"):
                        blocks.append({"type": "text", "text": m["content"]})
                    for tc in m.get("tool_calls", []):
                        blocks.append({"type": "tool_use", "id": tc.id,
                                       "name": tc.name, "input": tc.arguments})
                    out.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"],
                         "content": m["content"]}
                # Todos los tool_result consecutivos deben ir en UN solo mensaje user
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def chat(self, system, messages, tools=None) -> LLMResponse:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = [{"name": t.name, "description": t.description,
                                "input_schema": t.parameters} for t in tools]
        resp = self.client.messages.create(
            model=self.model, max_tokens=16000, system=system,
            messages=self._to_api_messages(messages), **kwargs,
        )
        text = "".join(b.text for b in resp.content if b.type == "text") or None
        calls = [ToolCall(b.id, b.name, dict(b.input))
                 for b in resp.content if b.type == "tool_use"]
        return LLMResponse(text=text, tool_calls=calls, raw=resp.content)
