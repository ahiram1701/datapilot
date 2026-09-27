"""Sistema multi-agente: un Orquestador que delega en especialistas.

Patrón "agents as tools": para el orquestador, cada especialista es solo
otra herramienta. Cada especialista es a su vez un agente ReAct con su
propio set de herramientas, así que la traza es jerárquica.
"""
from __future__ import annotations

import pandas as pd

from ..llm.base import LLMProvider
from ..ml import tools_ml
from ..rag.store import search_knowledge
from .react import ReActAgent, StepCallback
from .tools import ToolRegistry, schema

TARGET = {"target": {"type": "string", "description": "Columna objetivo a predecir"}}
MODEL = {"model_type": {"type": "string", "enum": tools_ml.MODEL_TYPES}}


def _columns_hint(df: pd.DataFrame) -> str:
    return ", ".join(f"{c} ({t})" for c, t in df.dtypes.astype(str).items())


def build_data_analyst(llm: LLMProvider, df: pd.DataFrame) -> ReActAgent:
    tools = ToolRegistry()
    tools.add("describe_dataset", "Resumen: filas, tipos, nulos, estadísticas.",
              schema(), lambda: tools_ml.describe_dataset(df))
    tools.add("correlations", "Correlaciones de Pearson; con target, ranking contra esa columna.",
              schema({"target": {"type": "string"}}),
              lambda target=None: tools_ml.correlations(df, target))
    tools.add("segment_analysis", "Promedio del objetivo por segmento (categorías y cuartiles "
              "numéricos). Con objetivo 0/1 da la tasa por grupo, p. ej. abandono por tipo de contrato.",
              schema(TARGET, ["target"]), lambda target: tools_ml.segment_analysis(df, target))
    return ReActAgent(
        "DataAnalyst",
        "Eres un analista de datos. Explora el dataset con tus herramientas y reporta "
        "hallazgos concretos con números (tamaño, nulos, distribuciones, correlaciones). "
        "Si la pregunta gira en torno a una variable objetivo, usa correlations Y "
        "segment_analysis: correlations ignora las columnas categóricas. "
        f"Columnas: {_columns_hint(df)}. Responde en español y de forma concisa.",
        llm, tools)


def build_ml_engineer(llm: LLMProvider, df: pd.DataFrame) -> ReActAgent:
    tools = ToolRegistry()
    tools.add("train_model", "Entrena y evalúa (train/test split) un modelo. Detecta "
              "regresión vs clasificación automáticamente.",
              schema({**TARGET, **MODEL, "test_size": {"type": "number"}}, ["target"]),
              lambda target, model_type="random_forest", test_size=0.2:
                  tools_ml.train_model(df, target, model_type, test_size))
    tools.add("cross_validate", "Validación cruzada k-fold para estimar la generalización.",
              schema({**TARGET, **MODEL, "folds": {"type": "integer"}}, ["target"]),
              lambda target, model_type="random_forest", folds=5:
                  tools_ml.cross_validate(df, target, model_type, folds))
    tools.add("gradient_descent_regression", "Regresión lineal hecha a mano con descenso de "
              "gradiente (NumPy), comparada con scikit-learn. Solo objetivos numéricos.",
              schema({**TARGET, "learning_rate": {"type": "number"},
                      "epochs": {"type": "integer"}}, ["target"]),
              lambda target, learning_rate=0.05, epochs=500:
                  tools_ml.gradient_descent_regression(df, target, learning_rate, epochs))
    return ReActAgent(
        "MLEngineer",
        "Eres ingeniero de ML. Compara al menos dos tipos de modelo, valida con "
        "cross-validation y detecta overfitting (métrica train vs test). "
        f"Columnas: {_columns_hint(df)}. Reporta solo las métricas exactas que devuelvan "
        "tus herramientas, de forma concisa. Responde en español.",
        llm, tools)


def build_explainer(llm: LLMProvider) -> ReActAgent:
    tools = ToolRegistry()
    tools.add("search_knowledge", "Busca en la base de conocimiento de ML (base vectorial).",
              schema({"query": {"type": "string"}}, ["query"]),
              lambda query: search_knowledge(query))
    return ReActAgent(
        "Explainer",
        "Explicas resultados de ML a personas no técnicas. Consulta search_knowledge para "
        "basar tus explicaciones de métricas y conceptos. Responde en español, claro y breve.",
        llm, tools, max_steps=4)


def build_orchestrator(llm: LLMProvider, df: pd.DataFrame, on_step: StepCallback | None = None) -> ReActAgent:
    specialists = {
        "ask_data_analyst": (build_data_analyst(llm, df), "Exploración y estadística descriptiva."),
        "ask_ml_engineer": (build_ml_engineer(llm, df), "Entrenar, validar y comparar modelos."),
        "ask_explainer": (build_explainer(llm), "Traducir resultados técnicos a lenguaje simple."),
    }
    tools = ToolRegistry()
    for name, (agent, desc) in specialists.items():
        tools.add(name, f"Delegar al agente {agent.name}: {desc}",
                  schema({"task": {"type": "string", "description": "Instrucción concreta"}}, ["task"]),
                  # default arg `a=agent` fija el agente en cada iteración del loop
                  lambda task, a=agent: a.run(task, on_step))
    return ReActAgent(
        "Orchestrator",
        "Coordinas un equipo de agentes para responder preguntas sobre un dataset. "
        "Divide el problema, delega a los especialistas con instrucciones concretas y "
        "sintetiza una respuesta final en español con los números clave. "
        "Usa SOLO cifras y métricas que tus especialistas hayan obtenido con sus "
        "herramientas, con el nombre exacto que devuelven (p. ej. accuracy_test, "
        "f1_macro_test); nunca inventes métricas, segmentos ni modelos que no se calcularon. "
        "Respuesta final breve (máximo ~200 palabras) en markdown: hallazgos con números "
        "y, si aplica, recomendaciones accionables. "
        f"El dataset tiene {len(df)} filas; columnas: {_columns_hint(df)}.",
        llm, tools, max_steps=10)


def run_team(llm: LLMProvider, df: pd.DataFrame, question: str,
             on_step: StepCallback | None = None) -> str:
    return build_orchestrator(llm, df, on_step).run(question, on_step)
