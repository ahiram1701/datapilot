from app.agents.react import ReActAgent
from app.agents.team import run_team
from app.agents.tools import ToolRegistry, schema
from app.llm.base import LLMResponse, ToolCall
from app.llm.mock_p import MockProvider


def make_calc_agent(script):
    tools = ToolRegistry()
    tools.add("add", "Suma dos números", schema({"a": {"type": "number"}, "b": {"type": "number"}}),
              lambda a, b: a + b)
    llm = MockProvider(script)
    return ReActAgent("Calc", "sistema", llm, tools, max_steps=3), llm


def test_react_loop_executes_tool_and_returns_final_answer():
    agent, llm = make_calc_agent([
        LLMResponse(text="Necesito sumar", tool_calls=[ToolCall("1", "add", {"a": 2, "b": 3})]),
        LLMResponse(text="La suma es 5"),
    ])
    steps = []
    assert agent.run("¿2+3?", steps.append) == "La suma es 5"
    assert [s.type for s in steps] == ["thought", "action", "observation", "final"]
    assert steps[2].content == "5"
    # la observación vuelve al LLM como mensaje "tool"
    last_msgs = llm.calls[-1]["messages"]
    assert last_msgs[-1] == {"role": "tool", "tool_call_id": "1", "name": "add", "content": "5"}


def test_tool_errors_are_returned_to_the_agent_not_raised():
    agent, _ = make_calc_agent([
        LLMResponse(text=None, tool_calls=[ToolCall("1", "no_existe", {})]),
        LLMResponse(text=None, tool_calls=[ToolCall("2", "add", {"a": 1})]),
        LLMResponse(text="ok"),
    ])
    steps = []
    agent.run("x", steps.append)
    observations = [s.content for s in steps if s.type == "observation"]
    assert observations[0].startswith("ERROR: la herramienta 'no_existe'")
    assert observations[1].startswith("ERROR ejecutando add")


def test_max_steps_stops_infinite_loops():
    loop = [LLMResponse(text=None, tool_calls=[ToolCall(str(i), "add", {"a": 1, "b": 1})])
            for i in range(10)]
    agent, _ = make_calc_agent(loop)
    assert "límite" in agent.run("x")


def test_max_steps_asks_for_partial_answer_instead_of_losing_work():
    calls = [LLMResponse(text=None, tool_calls=[ToolCall(str(i), "add", {"a": 1, "b": 1})])
             for i in range(3)]
    agent, llm = make_calc_agent(calls + [LLMResponse(text="Parcial: 1+1=2")])
    steps = []
    assert agent.run("x", steps.append) == "Parcial: 1+1=2"
    assert steps[-1].type == "final"
    assert "NO uses más herramientas" in llm.calls[-1]["messages"][-1]["content"]


def test_orchestrator_delegates_to_specialist(housing):
    llm = MockProvider([
        # Orquestador delega
        LLMResponse(text=None, tool_calls=[ToolCall("o1", "ask_data_analyst", {"task": "describe"})]),
        # DataAnalyst usa su tool y responde
        LLMResponse(text=None, tool_calls=[ToolCall("d1", "describe_dataset", {})]),
        LLMResponse(text="800 filas"),
        # Orquestador sintetiza
        LLMResponse(text="El dataset tiene 800 filas."),
    ])
    steps = []
    answer = run_team(llm, housing, "¿Cuántas filas?", steps.append)
    assert answer == "El dataset tiene 800 filas."
    assert {s.agent for s in steps} == {"Orchestrator", "DataAnalyst"}
