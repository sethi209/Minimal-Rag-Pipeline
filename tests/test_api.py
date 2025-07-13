import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_query_endpoint_success():
    response = client.post("/query", json={"query": "What is a transformer model?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["context_documents"], list)

def test_query_endpoint_empty_query():
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Query cannot be empty."

def test_query_endpoint_no_results(monkeypatch):
    # simulate empty retrieval
    from app import retrieval_chain

    def fake_invoke(*args, **kwargs):
        return {"answer": "Some answer"}

    # patch retrieval_chain to simulate no retrieval results
    monkeypatch.setattr("app.retriever.vectorstore.similarity_search_with_score", lambda *args, **kwargs: [])

    response = client.post("/query", json={"query": "random gibberish"})
    assert response.status_code == 404
    assert response.json()["detail"] == "No relevant documents found for the query."
