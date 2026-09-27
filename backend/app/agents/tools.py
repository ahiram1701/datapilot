import json
from dataclasses import dataclass
from typing import Any, Callable

from ..llm.base import ToolSpec


@dataclass
class Tool:
    spec: ToolSpec
    fn: Callable[..., Any]


class ToolRegistry:
    """Colección de herramientas que un agente puede invocar por nombre."""

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {t.spec.name: t for t in tools or []}

    def add(self, name: str, description: str, parameters: dict[str, Any],
            fn: Callable[..., Any]) -> None:
        self._tools[name] = Tool(ToolSpec(name, description, parameters), fn)

    @property
    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Ejecuta la tool y devuelve SIEMPRE un string (lo que verá el LLM).
        Los errores se devuelven como texto para que el agente pueda corregirse."""
        tool = self._tools.get(name)
        if tool is None:
            return f"ERROR: la herramienta '{name}' no existe. Disponibles: {list(self._tools)}"
        try:
            result = tool.fn(**arguments)
        except Exception as exc:  # noqa: BLE001 - el agente debe ver el fallo
            return f"ERROR ejecutando {name}: {type(exc).__name__}: {exc}"
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)


def schema(properties: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    """Atajo para escribir JSON Schemas de parámetros."""
    return {"type": "object", "properties": properties or {}, "required": required or []}
