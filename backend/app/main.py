"""API REST de DataPilot."""
from __future__ import annotations

import io
import json
import os
import queue
import threading
import uuid
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

from .agents.team import run_team  # noqa: E402  (después de cargar .env)
from .db.postgres import SQLSource, normalize_url  # noqa: E402
from .llm.factory import get_provider  # noqa: E402

SAMPLE_DIR = Path(os.getenv("SAMPLE_DIR", Path(__file__).resolve().parents[2] / "sample_data"))
MAX_UPLOAD_MB = 20

app = FastAPI(title="DataPilot API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])

# Almacenamiento en memoria: suficiente para una demo de un solo proceso.
DATASETS: dict[str, pd.DataFrame] = {}
# Conexiones vivas a bases de datos. Las credenciales solo viven en el engine:
# nunca se devuelven al cliente ni se escriben en logs.
CONNECTIONS: dict[str, SQLSource] = {}


def _dataset_info(dataset_id: str, name: str, df: pd.DataFrame) -> dict:
    return {"id": dataset_id, "name": name, "rows": len(df),
            "columns": [{"name": c, "dtype": str(t)} for c, t in df.dtypes.items()],
            "preview": json.loads(df.head(5).to_json(orient="records"))}


def friendly_error(exc: Exception) -> str:
    """Traduce errores comunes de los proveedores a un mensaje accionable."""
    detail = f"{type(exc).__name__}: {exc}"
    if type(exc).__name__ == "RateLimitError":  # openai y anthropic usan este nombre
        return ("Se alcanzó el límite de uso del proveedor (tokens o peticiones por minuto). "
                "Prueba: esperar un minuto, bajar LLM_MAX_TOKENS (p. ej. 800) o "
                "LLM_MAX_OBSERVATION_CHARS (p. ej. 2000), usar LLM_REASONING_EFFORT=low/none, "
                f"o cambiar de modelo.\n\nDetalle: {detail}")
    return detail


def _register(name: str, df: pd.DataFrame) -> dict:
    dataset_id = uuid.uuid4().hex[:8]
    DATASETS[dataset_id] = df
    return _dataset_info(dataset_id, name, df)


@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": os.getenv("LLM_PROVIDER", "mock")}


@app.get("/samples")
def list_samples():
    return sorted(p.name for p in SAMPLE_DIR.glob("*.csv"))


@app.post("/samples/{name}")
def load_sample(name: str):
    path = SAMPLE_DIR / Path(name).name  # Path(name).name evita path traversal
    if not path.exists():
        raise HTTPException(404, "Dataset de ejemplo no encontrado")
    return _register(path.name, pd.read_csv(path))


@app.post("/datasets")
async def upload_dataset(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Solo se aceptan archivos .csv")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"El archivo supera {MAX_UPLOAD_MB} MB")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"No se pudo leer el CSV: {exc}") from exc
    return _register(file.filename, df)


class ConnectionRequest(BaseModel):
    url: str


def _connection_info(connection_id: str, source: SQLSource) -> dict:
    return {"id": connection_id, "kind": "sql", "name": source.display_name,
            "tables": source.list_tables()}


@app.post("/connections")
def connect_database(req: ConnectionRequest):
    try:
        url = normalize_url(req.url)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        source = SQLSource.connect(url)
        connection_id = uuid.uuid4().hex[:8]
        info = _connection_info(connection_id, source)
    except Exception as exc:  # noqa: BLE001
        # El mensaje del driver puede incluir la URL: solo exponemos el tipo de error.
        raise HTTPException(400, f"No se pudo conectar a la base de datos ({type(exc).__name__}). "
                                 "Revisa host, puerto, usuario y contraseña.") from exc
    CONNECTIONS[connection_id] = source
    return info


@app.delete("/connections/{connection_id}")
def disconnect_database(connection_id: str):
    source = CONNECTIONS.pop(connection_id, None)
    if source is None:
        raise HTTPException(404, "Conexión no encontrada")
    source.close()
    return {"ok": True}


class ChatRequest(BaseModel):
    dataset_id: str  # id de un dataset CSV o de una conexión a base de datos
    question: str


@app.post("/chat")
def chat(req: ChatRequest):
    """Ejecuta el equipo de agentes y transmite cada paso como Server-Sent Events."""
    data = DATASETS.get(req.dataset_id)
    if data is None:
        data = CONNECTIONS.get(req.dataset_id)
    if data is None:
        raise HTTPException(404, "Dataset no encontrado; súbelo de nuevo")
    is_sql = isinstance(data, SQLSource)

    events: queue.Queue = queue.Queue()

    def worker():
        try:
            answer = run_team(get_provider(sql=is_sql), data, req.question,
                              on_step=lambda s: events.put(("step", s.to_dict())))
            events.put(("answer", {"answer": answer}))
        except Exception as exc:  # noqa: BLE001
            events.put(("error", {"error": friendly_error(exc)}))
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while (item := events.get()) is not None:
            kind, data = item
            yield f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
