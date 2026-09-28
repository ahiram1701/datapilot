"""Sistema multi-agente: un Orquestador que delega en especialistas.

Patrón "agents as tools": para el orquestador, cada especialista es solo
otra herramienta. Cada especialista es a su vez un agente ReAct con su
propio set de herramientas, así que la traza es jerárquica.
"""
from __future__ import annotations

import os

import pandas as pd

from ..db.postgres import SQLSource
from ..llm.base import LLMProvider
from ..ml import tools_ml
from ..rag.store import search_knowledge
from .react import ReActAgent, StepCallback
from .tools import ToolRegistry, schema

TARGET = {"target": {"type": "string", "description": "Columna objetivo a predecir"}}
MODEL = {"model_type": {"type": "string", "enum": tools_ml.MODEL_TYPES}}
SQL_PREVIEW_ROWS = 50


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


# --- Modo SQL en vivo: las tools reciben SQL en vez de un DataFrame fijo ---

QUERY = {"query": {"type": "string", "description": "Consulta SELECT de PostgreSQL"}}
SQL_RULES = ("El esquema de abajo ya lista tablas y columnas: úsalo directo, con nombres "
             "calificados (esquema.tabla); llama describe_table solo si necesitas llaves para un JOIN. "
             "Escribe SQL de PostgreSQL de solo lectura (SELECT o WITH, una sola sentencia). "
             "Agrega en SQL (GROUP BY, COUNT, AVG) en vez de traer filas crudas, y resuelve "
             "varias preguntas en una sola consulta cuando se pueda: tienes pocos pasos. ")
SQL_MAX_STEPS = 12  # explorar un esquema y corregir SQL consume más pasos que un CSV


def _sql_df(source: SQLSource, query: str) -> pd.DataFrame:
    """Materializa una consulta solo para el cálculo que la pide (con tope de filas)."""
    return source.query(query, limit=int(os.getenv("SQL_ML_MAX_ROWS", "50000")))


def _run_sql(source: SQLSource, query: str) -> dict:
    df = source.query(query, limit=SQL_PREVIEW_ROWS)
    return {"rows_returned": len(df), "truncated_at": SQL_PREVIEW_ROWS if len(df) == SQL_PREVIEW_ROWS else None,
            "columns": list(df.columns), "rows": df.to_dict(orient="records")}


def build_sql_analyst(llm: LLMProvider, source: SQLSource) -> ReActAgent:
    tools = ToolRegistry()
    tools.add("list_tables", "Tablas y vistas con columnas y filas estimadas.",
              schema(), source.list_tables)
    tools.add("describe_table", "Columnas, tipos, llave primaria y llaves foráneas de una tabla.",
              schema({"table": {"type": "string"}}, ["table"]), source.describe_table)
    tools.add("run_sql", f"Ejecuta un SELECT y devuelve hasta {SQL_PREVIEW_ROWS} filas.",
              schema(QUERY, ["query"]), lambda query: _run_sql(source, query))
    tools.add("correlations", "Correlaciones de Pearson sobre el resultado de una consulta; "
              "con target, ranking contra esa columna.",
              schema({**QUERY, "target": {"type": "string"}}, ["query"]),
              lambda query, target=None: tools_ml.correlations(_sql_df(source, query), target))
    tools.add("segment_analysis", "Promedio del objetivo por segmento sobre el resultado de una "
              "consulta. Con objetivo 0/1 da la tasa por grupo.",
              schema({**QUERY, **TARGET}, ["query", "target"]),
              lambda query, target: tools_ml.segment_analysis(_sql_df(source, query), target))
    return ReActAgent(
        "DataAnalyst",
        "Eres un analista de datos con acceso a una base PostgreSQL. " + SQL_RULES +
        "Reporta hallazgos concretos con números. "
        f"Esquema: {source.schema_hint()}. Responde en español y de forma concisa.",
        llm, tools, max_steps=SQL_MAX_STEPS)


def build_sql_ml_engineer(llm: LLMProvider, source: SQLSource) -> ReActAgent:
    tools = ToolRegistry()
    tools.add("train_model", "Entrena y evalúa un modelo sobre el resultado de una consulta (una "
              "fila por ejemplo, con la columna objetivo). Detecta regresión vs clasificación.",
              schema({**QUERY, **TARGET, **MODEL, "test_size": {"type": "number"}}, ["query", "target"]),
              lambda query, target, model_type="random_forest", test_size=0.2:
                  tools_ml.train_model(_sql_df(source, query), target, model_type, test_size))
    tools.add("cross_validate", "Validación cruzada k-fold sobre el resultado de una consulta.",
              schema({**QUERY, **TARGET, **MODEL, "folds": {"type": "integer"}}, ["query", "target"]),
              lambda query, target, model_type="random_forest", folds=5:
                  tools_ml.cross_validate(_sql_df(source, query), target, model_type, folds))
    tools.add("gradient_descent_regression", "Regresión lineal con descenso de gradiente (NumPy) "
              "comparada con scikit-learn. Solo objetivos numéricos.",
              schema({**QUERY, **TARGET, "learning_rate": {"type": "number"},
                      "epochs": {"type": "integer"}}, ["query", "target"]),
              lambda query, target, learning_rate=0.05, epochs=500:
                  tools_ml.gradient_descent_regression(_sql_df(source, query), target,
                                                       learning_rate, epochs))
    return ReActAgent(
        "MLEngineer",
        "Eres ingeniero de ML con acceso a una base PostgreSQL. Cada tool recibe un SELECT que "
        "arma la tabla de entrenamiento (JOINs y columnas derivadas en SQL; excluye IDs). "
        "Compara al menos dos tipos de modelo, valida con cross-validation y detecta "
        "overfitting (métrica train vs test). " + SQL_RULES +
        f"Esquema: {source.schema_hint()}. Reporta solo las métricas exactas que devuelvan "
        "tus herramientas, de forma concisa. Responde en español.",
        llm, tools, max_steps=SQL_MAX_STEPS)


def build_orchestrator(llm: LLMProvider, data: pd.DataFrame | SQLSource,
                       on_step: StepCallback | None = None) -> ReActAgent:
    if isinstance(data, SQLSource):
        analyst, engineer = build_sql_analyst(llm, data), build_sql_ml_engineer(llm, data)
        data_desc = f"una base de datos PostgreSQL con este esquema: {data.schema_hint()}"
    else:
        analyst, engineer = build_data_analyst(llm, data), build_ml_engineer(llm, data)
        data_desc = f"un dataset de {len(data)} filas; columnas: {_columns_hint(data)}"
    specialists = {
        "ask_data_analyst": (analyst, "Exploración y estadística descriptiva."),
        "ask_ml_engineer": (engineer, "Entrenar, validar y comparar modelos."),
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
        "Coordinas un equipo de agentes para responder preguntas sobre datos. "
        "Divide el problema, delega a los especialistas con instrucciones concretas y "
        "sintetiza una respuesta final en español con los números clave. "
        "Usa SOLO cifras y métricas que tus especialistas hayan obtenido con sus "
        "herramientas, con el nombre exacto que devuelven (p. ej. accuracy_test, "
        "f1_macro_test); nunca inventes métricas, segmentos ni modelos que no se calcularon. "
        "Respuesta final breve (máximo ~200 palabras) en markdown: hallazgos con números "
        "y, si aplica, recomendaciones accionables. "
        f"Los datos son {data_desc}.",
        llm, tools, max_steps=10)


def run_team(llm: LLMProvider, data: pd.DataFrame | SQLSource, question: str,
             on_step: StepCallback | None = None) -> str:
    return build_orchestrator(llm, data, on_step).run(question, on_step)
