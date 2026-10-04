import numpy as np
import pytest
from fastapi.testclient import TestClient

from enterprise_ai_search import dataset, dense, reranker, service as service_module
from enterprise_ai_search.api import create_app
from enterprise_ai_search.authorization import DocumentAccessPolicy, PolicyStore
from enterprise_ai_search.hybrid import DocumentCandidate, HybridResult
from enterprise_ai_search.models import Document, SearchResult
from enterprise_ai_search.rag import SYSTEM_PROMPT
from enterprise_ai_search.service import SearchService, load_service


HEADERS = {"X-Tenant-ID": "tenant-a", "X-Principal-ID": "fixture-user"}


def fixture_policy() -> PolicyStore:
    policies = {identifier: DocumentAccessPolicy(identifier, "tenant-a", "tenant") for identifier in ("1", "2", "3", "doc")}
    return PolicyStore(frozenset({"tenant-a"}), policies)


class FakeIndex:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[HybridResult]:
        self.calls.append((query, top_k))
        return [
            HybridResult(number, 0.01, str(number), None, DocumentCandidate(
                number, str(number), SearchResult(number, 0.1, f"{number}::0", str(number), str(number)),
            ))
            for number in range(1, 4)
        ]


class FakeReranker:
    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> np.ndarray:
        return np.array([float(text) for query, text in pairs])


class FakeGenerator:
    def __init__(self, answer: str = "Fixture answer [1].") -> None:
        self.answer = answer
        self.messages: list[dict[str, str]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.messages = messages
        return self.answer


@pytest.fixture
def service() -> SearchService:
    return SearchService(FakeIndex(), FakeReranker(), FakeGenerator(), 1.25, policy_store=fixture_policy())


@pytest.mark.parametrize("configured", [True, False])
def test_health_is_cheap_and_reports_generation_configuration(service, configured) -> None:
    if not configured:
        service.generator = None
    with TestClient(create_app(lambda: service), headers=HEADERS) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ready", "retrieval_initialized": True,
            "generation_configured": configured, "authorization_initialized": True, "preparation_seconds": 1.25,
        }
        assert service.index.calls == []


def test_lifespan_initializes_once_and_releases_resources(service) -> None:
    calls = []

    def factory() -> SearchService:
        calls.append(1)
        return service

    app = create_app(factory)
    assert calls == []
    with TestClient(app, headers=HEADERS) as client:
        index = service.index
        for _ in range(2):
            assert client.post("/search", json={"query": "question"}).status_code == 200
        assert calls == [1] and service.index is index and len(index.calls) == 2
    assert app.state.service is None
    assert service.index is service.reranker is service.generator is None


def test_search_returns_reranked_passage_provenance_and_bounded_results(service) -> None:
    with TestClient(create_app(lambda: service), headers=HEADERS) as client:
        response = client.post("/search", json={"query": " question ", "top_k": 2})
        assert response.status_code == 200
        body = response.json()
        assert body["results"] == [
            {"rank": 1, "document_id": "3", "chunk_id": "3::0", "text": "3", "reranker_score": 3.0},
            {"rank": 2, "document_id": "2", "chunk_id": "2::0", "text": "2", "reranker_score": 2.0},
        ]
        assert service.index.calls == [("question", 50)]
        assert body["timings"]["handler_seconds"] >= body["timings"]["retrieval_reranking_seconds"] >= 0


@pytest.mark.parametrize("body", [
    {}, {"query": ""}, {"query": " \n"}, {"query": 7}, {"query": "x" * 2001},
    {"query": "q", "top_k": 0}, {"query": "q", "top_k": 11},
    {"query": "q", "top_k": True}, {"query": "q", "top_k": "2"},
    {"query": "q", "top_k": 1.5}, {"query": "q", "api_key": "fixture-secret"},
])
def test_search_validation_rejects_invalid_requests_without_echoing_input(service, body) -> None:
    with TestClient(create_app(lambda: service), headers=HEADERS) as client:
        response = client.post("/search", json=body)
        assert response.status_code == 422
        assert "fixture-secret" not in response.text
        assert all("input" not in error for error in response.json()["detail"])
        assert service.index.calls == []


@pytest.mark.parametrize("answer,passed", [("Fixture answer [1].", True), ("Fixture [99].", False)])
def test_ask_reuses_free_form_rag_and_preserves_citation_status(service, answer, passed) -> None:
    generator = FakeGenerator(answer)
    service.generator = generator
    with TestClient(create_app(lambda: service), headers=HEADERS) as client:
        response = client.post("/ask", json={"question": " question "})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == answer
        assert body["citation_validation"]["passed"] is passed
        assert body["evidence"][0] == {
            "source_number": 1, "document_id": "3", "chunk_id": "3::0", "rank": 1, "text": "3",
        }
        if passed:
            assert body["citation_validation"]["citations"][0]["document_id"] == "3"
        else:
            assert body["citation_validation"]["invalid_source_numbers"] == [99]
        assert service.index.calls == [("question", 50)]
        assert generator.messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert body["timings"]["handler_seconds"] >= body["timings"]["online_seconds"] >= 0
        assert body["timings"]["generation_request_seconds"] >= 0


def test_missing_generation_config_does_not_disable_search_or_health(service) -> None:
    service.generator = None
    with TestClient(create_app(lambda: service), headers=HEADERS) as client:
        assert client.post("/ask", json={"question": "question"}).status_code == 503
        assert service.index.calls == []
        assert client.post("/search", json={"query": "question"}).status_code == 200
        assert client.get("/health").status_code == 200


@pytest.mark.parametrize("body", [{}, {"question": " \n"}, {"question": None}, {"question": "x" * 2001}])
def test_ask_validates_question(service, body) -> None:
    with TestClient(create_app(lambda: service), headers=HEADERS) as client:
        assert client.post("/ask", json=body).status_code == 422
        assert service.index.calls == []


def test_initialization_failure_is_visible_but_sanitized() -> None:
    def factory() -> SearchService:
        raise OSError("fixture-secret /private/path")

    with TestClient(create_app(factory), headers=HEADERS) as client:
        health = client.get("/health")
        assert health.status_code == 503 and not health.json()["retrieval_initialized"]
        for route, body in [("/search", {"query": "q"}), ("/ask", {"question": "q"})]:
            response = client.post(route, json=body)
            assert response.status_code == 503
            assert "fixture-secret" not in response.text and "/private/path" not in response.text


@pytest.mark.parametrize("route,method,error,status", [
    ("/search", "search", ValueError, 500), ("/ask", "answer", OSError, 500),
    ("/search", "search", TypeError, 500),
])
def test_request_errors_do_not_expose_details(service, monkeypatch, caplog, route, method, error, status) -> None:
    def fail(*args: object) -> None:
        raise error("fixture-secret /private/path")

    monkeypatch.setattr(service, method, fail)
    with TestClient(create_app(lambda: service), headers=HEADERS) as client:
        response = client.post(route, json={"query" if route == "/search" else "question": "q"})
        assert response.status_code == status
        assert "fixture-secret" not in response.text and "/private/path" not in response.text
        assert "fixture-secret" not in caplog.text and "/private/path" not in caplog.text


def test_default_service_loader_uses_existing_preparation_once_without_models(monkeypatch) -> None:
    for name in ("GENERATION_ENDPOINT", "GENERATION_MODEL", "GENERATION_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    calls = []
    monkeypatch.setenv("AUTHORIZATION_POLICY_PATH", "fixture-policy.json")
    monkeypatch.setattr(service_module, "load_policy_store", lambda path: fixture_policy())

    def corpus(path):
        calls.append("corpus")
        return [Document("doc", "fixture text")]

    def encoder(path):
        calls.append("encoder")
        return object(), 0

    def vectors(chunks, model, path):
        calls.append("vectors")
        return np.ones((len(chunks), 384), dtype=np.float32) / np.sqrt(384), {}

    def reranking_model(path):
        calls.append("reranker")
        return FakeReranker(), 0

    monkeypatch.setattr(dataset, "load_corpus", corpus)
    monkeypatch.setattr(dense, "load_encoder", encoder)
    monkeypatch.setattr(dense, "prepare_embeddings", vectors)
    monkeypatch.setattr(reranker, "load_reranker", reranking_model)
    with TestClient(create_app(load_service), headers=HEADERS) as client:
        for _ in range(2):
            assert client.get("/health").json()["retrieval_initialized"]
        assert calls == ["corpus", "encoder", "vectors", "reranker"]
        assert not client.get("/health").json()["generation_configured"]
