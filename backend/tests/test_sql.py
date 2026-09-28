import pytest
from sqlalchemy import create_engine, text

from app import main
from app.agents.team import run_team
from app.db.postgres import SQLSource, ensure_read_only, normalize_url
from app.llm.base import LLMResponse, ToolCall
from app.llm.mock_p import MockProvider
from tests.test_api import client, parse_sse


@pytest.fixture
def source(tmp_path, churn) -> SQLSource:
    """SQLite en archivo como stand-in de Postgres: SQLSource usa SQLAlchemy genérico."""
    engine = create_engine(f"sqlite:///{tmp_path / 'demo.db'}")
    churn.to_sql("clientes", engine, index=False)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE pedidos (id INTEGER PRIMARY KEY, "
                          "cliente_id INTEGER REFERENCES clientes(rowid), total REAL)"))
    return SQLSource(engine)


# --- ensure_read_only ---

@pytest.mark.parametrize("sql", [
    "SELECT * FROM clientes",
    "  select contrato, avg(abandona) from clientes group by contrato;  ",
    "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
    "SELECT 'no borres; DROP TABLE x' AS texto",  # palabras peligrosas dentro de un string
])
def test_read_only_accepts_queries(sql):
    cleaned = ensure_read_only(sql)
    assert not cleaned.rstrip().endswith(";")


@pytest.mark.parametrize("sql", [
    "DELETE FROM clientes",
    "UPDATE clientes SET abandona = 0",
    "DROP TABLE clientes",
    "INSERT INTO clientes VALUES (1)",
    "SELECT 1; DROP TABLE clientes",
    "WITH d AS (DELETE FROM clientes RETURNING *) SELECT * FROM d",
    "",
])
def test_read_only_rejects_writes(sql):
    with pytest.raises(ValueError):
        ensure_read_only(sql)


# --- SQLSource ---

def test_normalize_url():
    assert normalize_url("postgresql://u:p@h:5432/db").startswith("postgresql+psycopg://")
    assert normalize_url("postgres://u:p@h/db").startswith("postgresql+psycopg://")
    with pytest.raises(ValueError):
        normalize_url("mysql://u:p@h/db")


def test_list_and_describe(source):
    names = {t["name"] for t in source.list_tables()}
    assert names == {"clientes", "pedidos"}
    desc = source.describe_table("pedidos")
    assert desc["primary_key"] == ["id"]
    assert desc["foreign_keys"][0]["references"].startswith("clientes")
    with pytest.raises(ValueError):
        source.describe_table("no_existe")


def test_query_enforces_outer_limit(source):
    df = source.query("SELECT * FROM clientes", limit=7)
    assert len(df) == 7


def test_query_error_is_short_and_actionable(source):
    with pytest.raises(ValueError) as exc:
        source.query("SELECT columna_inexistente FROM clientes")
    msg = str(exc.value)
    assert "columna_inexistente" in msg and "[SQL:" not in msg


def test_sql_team_trains_from_query(source):
    """El MLEngineer materializa la consulta y reutiliza tools_ml sin cambios."""
    q = "SELECT meses_cliente, cargo_mensual, tickets_soporte, abandona FROM clientes"
    llm = MockProvider([
        LLMResponse(text="", tool_calls=[ToolCall("1", "ask_ml_engineer", {"task": "entrena"})]),
        LLMResponse(text="", tool_calls=[ToolCall("2", "train_model",
                                         {"query": q, "target": "abandona", "model_type": "tree"})]),
        LLMResponse(text="listo"),
        LLMResponse(text="final"),
    ])
    steps = []
    assert run_team(llm, source, "¿Qué predice el abandono?", steps.append) == "final"
    obs = next(s for s in steps if s.type == "observation" and s.tool == "train_model")
    assert "accuracy_test" in obs.content


# --- API ---

def test_connect_rejects_non_postgres():
    assert client.post("/connections", json={"url": "mysql://u:secreta@h/db"}).status_code == 400


def test_connect_error_hides_password():
    r = client.post("/connections", json={"url": "postgresql://u:secreta123@127.0.0.1:1/db"})
    assert r.status_code == 400
    assert "secreta123" not in r.text


def test_chat_over_connection_with_mock(source, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setitem(main.CONNECTIONS, "db1", source)
    r = client.post("/chat", json={"dataset_id": "db1", "question": "¿Qué tablas hay?"})
    events = parse_sse(r.text)
    assert events[-1][0] == "answer"
    obs = [d for k, d in events if k == "step" and d["type"] == "observation"]
    assert obs[0]["tool"] == "list_tables" and "clientes" in obs[0]["content"]
