from app.rag.store import chunk_markdown, search_knowledge


def test_chunk_markdown_splits_by_h2():
    chunks = chunk_markdown("# T\n## Uno\nA\n## Dos\nB", "src")
    assert [(c["title"], c["text"]) for c in chunks] == [("Uno", "A"), ("Dos", "B")]


def test_semantic_search_finds_overfitting():
    results = search_knowledge("el modelo memoriza los datos de entrenamiento", k=3)
    assert any("Overfitting" in r["title"] for r in results)
