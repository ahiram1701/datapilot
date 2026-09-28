"""Fuente de datos SQL en vivo: los agentes consultan la base del usuario
sin cargar tablas completas a memoria.

Dos capas de seguridad, porque el SQL lo escribe un LLM:
1. `ensure_read_only` rechaza en la app lo que no sea una consulta (y le da
   al agente un error legible para corregirse).
2. En PostgreSQL la sesión se abre con `default_transaction_read_only=on` y
   `statement_timeout`: aunque algo evada el filtro, el servidor no escribe.
"""
from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import DBAPIError

MAX_SCHEMA_HINT_CHARS = 3000


def normalize_url(url: str) -> str:
    """Acepta `postgres://` o `postgresql://` y fuerza el driver psycopg v3."""
    try:
        parsed = make_url(url.strip())
    except Exception as exc:  # noqa: BLE001 - el mensaje original podría incluir la clave
        raise ValueError("URL de conexión inválida") from exc
    if parsed.get_backend_name() not in ("postgresql", "postgres"):
        raise ValueError("Solo se admiten URLs de PostgreSQL "
                         "(postgresql://usuario:clave@host:5432/base)")
    return parsed.set(drivername="postgresql+psycopg").render_as_string(hide_password=False)


# Strings '...', identificadores "..." y comentarios, en ese orden de prioridad:
# un `--` dentro de un string no es comentario y un `;` dentro de un string no separa sentencias.
_LEXEMES = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|--[^\n]*|/\*.*?\*/", re.S)
_WRITE_WORDS = re.compile(r"\b(INSERT|UPDATE|DELETE|MERGE|UPSERT|DROP|ALTER|TRUNCATE|CREATE|"
                          r"GRANT|REVOKE|COPY|CALL|VACUUM|REINDEX|LOCK|INTO)\b", re.I)


def ensure_read_only(sql: str) -> str:
    """Valida que `sql` sea UNA sola consulta de lectura y la devuelve limpia
    (sin `;` final), lista para envolverse en un subquery. Lanza ValueError
    con un mensaje que el agente pueda entender si no lo es."""
    # 1. Quita comentarios (un `-- ...` final comentaría el `)` del subquery) pero conserva strings.
    cleaned = _LEXEMES.sub(lambda m: m.group() if m.group()[0] in "'\"" else " ", sql or "")
    cleaned = cleaned.strip().rstrip(";").strip()
    # 2. Versión sin strings ni identificadores entre comillas: solo queda "código" SQL.
    code = _LEXEMES.sub(" ", cleaned)
    if not code.strip():
        raise ValueError("La consulta está vacía.")
    if ";" in code:
        raise ValueError("Solo se permite UNA sentencia por llamada (sin ';' intermedios).")
    first = code.split(None, 1)[0].upper()
    if first not in ("SELECT", "WITH"):
        raise ValueError(f"Solo se permiten consultas de lectura (SELECT o WITH), no {first}.")
    if match := _WRITE_WORDS.search(code):
        raise ValueError(f"'{match.group().upper()}' no está permitido: la conexión es de solo lectura.")
    return cleaned


class SQLSource:
    """Conexión viva a una base de datos, compartida por todas las tools de una sesión."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.display_name = engine.url.render_as_string(hide_password=True)

    @classmethod
    def connect(cls, url: str) -> "SQLSource":
        connect_args: dict[str, Any] = {}
        if url.startswith("postgresql"):
            timeout = int(os.getenv("SQL_STATEMENT_TIMEOUT_MS", "15000"))
            connect_args = {"options": f"-c default_transaction_read_only=on "
                                       f"-c statement_timeout={timeout}",
                            "connect_timeout": 10}
        source = cls(create_engine(url, pool_pre_ping=True, connect_args=connect_args))
        with source.engine.connect() as conn:  # falla aquí si la URL o credenciales son malas
            conn.execute(text("SELECT 1"))
        return source

    @property
    def _is_postgres(self) -> bool:
        return self.engine.dialect.name == "postgresql"

    def _qualified_tables(self) -> list[tuple[str | None, str]]:
        if not self._is_postgres:
            insp = inspect(self.engine)
            return [(None, t) for t in insp.get_table_names() + insp.get_view_names()]
        # information_schema.tables ya filtra por privilegios; además exigimos USAGE en el
        # esquema. Así un usuario de solo lectura no ve (ni falla en) esquemas ajenos.
        sql = text("SELECT table_schema, table_name FROM information_schema.tables "
                   "WHERE table_schema <> 'information_schema' AND table_schema NOT LIKE 'pg\\_%' "
                   "AND has_schema_privilege(table_schema, 'USAGE') "
                   "AND has_table_privilege(quote_ident(table_schema) || '.' || quote_ident(table_name), 'SELECT') "
                   "ORDER BY 1, 2")
        with self.engine.connect() as conn:
            return [(s, t) for s, t in conn.execute(sql)]

    @staticmethod
    def _full_name(schema: str | None, table: str) -> str:
        return table if schema in (None, "public") else f"{schema}.{table}"

    def _resolve(self, table: str) -> tuple[str | None, str]:
        for schema, name in self._qualified_tables():
            if self._full_name(schema, name) == table:
                return schema, name
        raise ValueError(f"La tabla '{table}' no existe. Usa list_tables para ver las disponibles.")

    def _row_estimates(self) -> dict[str, int]:
        """Estimado barato de filas (pg_class.reltuples) para no hacer COUNT(*) en tablas grandes."""
        if not self._is_postgres:
            return {}
        sql = text("SELECT n.nspname, c.relname, c.reltuples::bigint FROM pg_class c "
                   "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relkind IN ('r','p','m')")
        with self.engine.connect() as conn:
            return {self._full_name(s, t): max(int(r), 0) for s, t, r in conn.execute(sql)}

    def list_tables(self) -> list[dict[str, Any]]:
        insp = inspect(self.engine)
        estimates = self._row_estimates()
        tables = []
        for schema, table in self._qualified_tables():
            name = self._full_name(schema, table)
            cols = insp.get_columns(table, schema=schema)
            tables.append({"name": name, "rows_estimate": estimates.get(name),
                           "columns": [f"{c['name']} {c['type']}" for c in cols]})
        return tables

    def describe_table(self, table: str) -> dict[str, Any]:
        schema, name = self._resolve(table)
        insp = inspect(self.engine)
        return {
            "table": table,
            "columns": [{"name": c["name"], "type": str(c["type"]), "nullable": c["nullable"]}
                        for c in insp.get_columns(name, schema=schema)],
            "primary_key": insp.get_pk_constraint(name, schema=schema).get("constrained_columns", []),
            # las FK le dicen al LLM cómo hacer JOINs sin adivinar
            "foreign_keys": [{"columns": fk["constrained_columns"],
                              "references": f"{self._full_name(fk.get('referred_schema'), fk['referred_table'])}"
                                            f"({', '.join(fk['referred_columns'])})"}
                             for fk in insp.get_foreign_keys(name, schema=schema)],
        }

    def query(self, sql: str, limit: int = 50) -> pd.DataFrame:
        """Ejecuta una consulta de lectura con un LIMIT externo que el LLM no puede saltarse."""
        inner = ensure_read_only(sql)
        wrapped = text(f"SELECT * FROM ({inner}) AS datapilot_q LIMIT :n")
        try:
            with self.engine.connect() as conn:
                result = conn.execute(wrapped, {"n": int(limit)})
                # coerce_float: NUMERIC llega como Decimal; sin esto tools_ml no lo vería numérico
                return pd.DataFrame.from_records(result.fetchall(), columns=list(result.keys()),
                                                 coerce_float=True)
        except DBAPIError as exc:
            # Solo el mensaje de la base (p. ej. "column x does not exist" + pista), sin el
            # SQL generado ni el enlace de SQLAlchemy: menos tokens y el agente corrige antes.
            detail = "\n".join(str(exc.orig).splitlines()[:3])
            raise ValueError(f"PostgreSQL rechazó la consulta: {detail}") from exc

    def schema_hint(self) -> str:
        """Esquema compacto para los system prompts: `tabla(col tipo, ...)`."""
        hint = "; ".join(f"{t['name']}({', '.join(t['columns'])})" for t in self.list_tables())
        if len(hint) > MAX_SCHEMA_HINT_CHARS:
            hint = hint[:MAX_SCHEMA_HINT_CHARS] + "… (recortado; usa list_tables/describe_table)"
        return hint

    def close(self) -> None:
        self.engine.dispose()
