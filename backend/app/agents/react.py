"""Agente ReAct (Reason + Act) implementado a mano.

Ciclo: el LLM razona (Thought), pide una herramienta (Action), recibe el
resultado (Observation) y repite hasta dar una respuesta final sin tools.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from ..llm.base import LLMProvider, env_int
from .tools import ToolRegistry


@dataclass
class Step:
    agent: str
    type: str  # thought | action | observation | final | error
    content: str
    tool: str | None = None
    args: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


StepCallback = Callable[[Step], None]


class ReActAgent:
    def __init__(self, name: str, system: str, llm: LLMProvider, tools: ToolRegistry,
                 max_steps: int = 8):
        self.name = name
        self.system = system
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def run(self, task: str, on_step: StepCallback | None = None) -> str:
        emit = on_step or (lambda s: None)
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]

        for _ in range(self.max_steps):
            resp = self.llm.chat(self.system, messages, self.tools.specs)

            if not resp.tool_calls:  # sin acciones pendientes => respuesta final
                answer = resp.text or ""
                emit(Step(self.name, "final", answer))
                return answer

            if resp.text:
                emit(Step(self.name, "thought", resp.text))
            messages.append(resp.to_message())

            for call in resp.tool_calls:
                emit(Step(self.name, "action", f"{call.name}({call.arguments})",
                          tool=call.name, args=call.arguments))
                result = self.tools.execute(call.name, call.arguments)
                # Evita inundar el contexto: cada observación se reenvía en todos
                # los pasos siguientes, así que su tamaño multiplica los tokens.
                limit = env_int("LLM_MAX_OBSERVATION_CHARS", 4000)
                if len(result) > limit:
                    result = result[:limit] + "... [truncado]"
                emit(Step(self.name, "observation", result, tool=call.name))
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "name": call.name, "content": result})

        msg = f"{self.name} alcanzó el límite de {self.max_steps} pasos sin respuesta final."
        emit(Step(self.name, "error", msg))
        return msg
