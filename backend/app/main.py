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
from .llm.factory import get_provider  # noqa: E402

SAMPLE_DIR = Path(os.getenv("SAMPLE_DIR", Path(__file__).resolve().parents[2] / "sample_data"))
MAX_UPLOAD_MB = 20

app = FastAPI(title="DataPilot API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])

# Almacenamiento en memoria: suficiente para una demo de un solo proceso.
DATASETS: dict[str, pd.DataFrame] = {}


def _dataset_info(dataset_id: str, name: str, df: pd.DataFrame) -> dict:
    return {"id": dataset_id, "name": name, "rows": len(df),
            "columns": [{"name": c, "dtype": str(t)} for c, t in df.dtypes.items()],
            "preview": json.loads(df.head(5).to_json(orient="records"))}


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


class ChatRequest(BaseModel):
    dataset_id: str
    question: str


@app.post("/chat")
def chat(req: ChatRequest):
    """Ejecuta el equipo de agentes y transmite cada paso como Server-Sent Events."""
    df = DATASETS.get(req.dataset_id)
    if df is None:
        raise HTTPException(404, "Dataset no encontrado; súbelo de nuevo")

    events: queue.Queue = queue.Queue()

    def worker():
        try:
            answer = run_team(get_provider(), df, req.question,
                              on_step=lambda s: events.put(("step", s.to_dict())))
            events.put(("answer", {"answer": answer}))
        except Exception as exc:  # noqa: BLE001
            events.put(("error", {"error": f"{type(exc).__name__}: {exc}"}))
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while (item := events.get()) is not None:
            kind, data = item
            yield f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
