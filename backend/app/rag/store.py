"""Base de conocimiento vectorial (ChromaDB) con conceptos de ML.

Los documentos en docs/*.md se dividen por secciones "## " y se indexan con
embeddings. El agente Explainer la consulta para justificar sus explicaciones
(RAG: Retrieval-Augmented Generation).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

DOCS_DIR = Path(__file__).parent / "docs"


def chunk_markdown(text: str, source: str) -> list[dict[str, str]]:
    chunks, title, buf = [], source, []
    for line in text.splitlines():
        if line.startswith("## "):
            if buf:
                chunks.append({"title": title, "text": "\n".join(buf).strip(), "source": source})
            title, buf = line[3:].strip(), []
        elif not line.startswith("# "):
            buf.append(line)
    if buf:
        chunks.append({"title": title, "text": "\n".join(buf).strip(), "source": source})
    return [c for c in chunks if c["text"]]


@lru_cache(maxsize=1)
def get_collection():
    import chromadb  # import perezoso: carga pesada

    path = os.getenv("CHROMA_PATH")
    client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
    col = client.get_or_create_collection("ml_knowledge", metadata={"hnsw:space": "cosine"})
    if col.count() == 0:
        chunks = [c for f in sorted(DOCS_DIR.glob("*.md"))
                  for c in chunk_markdown(f.read_text(encoding="utf-8"), f.stem)]
        col.add(ids=[f"{c['source']}-{i}" for i, c in enumerate(chunks)],
                documents=[f"{c['title']}\n{c['text']}" for c in chunks],
                metadatas=[{"title": c["title"], "source": c["source"]} for c in chunks])
    return col


def search_knowledge(query: str, k: int = 3) -> list[dict[str, Any]]:
    res = get_collection().query(query_texts=[query], n_results=k)
    return [{"title": m["title"], "source": m["source"], "text": d,
             "distance": round(dist, 3)}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]
