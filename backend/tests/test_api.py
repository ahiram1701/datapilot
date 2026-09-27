import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_upload_rejects_non_csv():
    r = client.post("/datasets", files={"file": ("x.txt", b"hola", "text/plain")})
    assert r.status_code == 400


def test_upload_and_chat_with_mock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    csv = b"a,b,y\n1,2,3\n2,3,5\n3,4,7\n4,5,9\n"
    ds = client.post("/datasets", files={"file": ("d.csv", csv, "text/csv")}).json()
    assert ds["rows"] == 4

    r = client.post("/chat", json={"dataset_id": ds["id"], "question": "¿Qué hay?"})
    events = parse_sse(r.text)
    kinds = [k for k, _ in events]
    assert "step" in kinds and kinds[-1] == "answer"


def test_chat_unknown_dataset():
    assert client.post("/chat", json={"dataset_id": "nope", "question": "x"}).status_code == 404


def test_sample_path_traversal_blocked():
    assert client.post("/samples/..%2F..%2Fsecret.csv").status_code == 404
