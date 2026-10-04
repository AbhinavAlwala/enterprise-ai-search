import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID, uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient

from enterprise_ai_search import observability
from enterprise_ai_search.api import create_app
from enterprise_ai_search.authorization import DocumentAccessPolicy, PolicyStore
from enterprise_ai_search.hybrid import DocumentCandidate, HybridResult
from enterprise_ai_search.models import SearchResult
from enterprise_ai_search.observability import (
    Metrics, ObservedGenerator, RequestTrace, RequestTracingMiddleware, current_trace, record_timings,
)
from enterprise_ai_search.service import SearchService

IDENTITY = {"X-Tenant-ID": "tenant-private", "X-Principal-ID": "principal-private", "X-Groups": "group-private"}
CALLER_ID = "5996a386-cbd8-40bd-a8b8-2e86559fe286"


class FakeIndex:
    def search(self, query: str, top_k: int) -> list[HybridResult]:
        return [HybridResult(i, 0.1, identifier, None, DocumentCandidate(
            i, identifier, SearchResult(i, 0.1, identifier + "::0", identifier, text),
        )) for i, (identifier, text) in enumerate([
            ("allowed-document", "protected-evidence"), ("hidden-document", "hidden-evidence"),
        ], 1)]


class FakeReranker:
    def __init__(self) -> None:
        self.inputs = []

    def predict(self, pairs, **kwargs):
        self.inputs.append(pairs)
        return np.ones(len(pairs))


class FakeGenerator:
    def __init__(self) -> None:
        self.inputs = []
        self.answer = "protected-answer [1]."

    def generate(self, messages):
        self.inputs.append(messages)
        return self.answer


@pytest.fixture
def service() -> SearchService:
    policies = {name: DocumentAccessPolicy(name, tenant, "tenant") for name, tenant in [
        ("allowed-document", "tenant-private"), ("hidden-document", "tenant-hidden"),
    ]}
    store = PolicyStore(frozenset({"tenant-private", "tenant-hidden"}), policies)
    return SearchService(FakeIndex(), FakeReranker(), FakeGenerator(), 1.25, policy_store=store)


@pytest.mark.parametrize("supplied,preserved", [
    (None, False), (CALLER_ID, True), ("tenant-private", False), ("x" * 200, False),
    (CALLER_ID.upper(), False), ("5996a386-cbd8-30bd-a8b8-2e86559fe286", False),
    ("00000000-0000-0000-0000-000000000000", False), (" " + CALLER_ID, False),
])
def test_request_id_policy_and_response_header(service, supplied, preserved, caplog) -> None:
    caplog.set_level(logging.INFO)
    headers = {"X-Request-ID": supplied} if supplied is not None else {}
    with TestClient(create_app(lambda: service)) as client:
        response = client.get("/health", headers=headers)
        effective = response.headers["X-Request-ID"]
        assert UUID(effective).version == 4 and str(UUID(effective)) == effective
        assert (effective == supplied) is preserved
        record = next(r for r in caplog.records if getattr(r, "event", None) == "request_completed")
        assert record.request_id == effective and record.status == 200
        assert json.loads(record.getMessage())["request_id"] == effective


@pytest.mark.parametrize("values", [[CALLER_ID, str(uuid4())], ["bad\nvalue"]])
def test_ambiguous_or_unsafe_request_ids_are_replaced(service, values) -> None:
    with TestClient(create_app(lambda: service)) as client:
        response = client.get("/health", headers=[("X-Request-ID", value) for value in values])
        effective = response.headers["X-Request-ID"]
        assert UUID(effective).version == 4 and effective not in values


def test_non_ascii_request_id_bytes_are_replaced() -> None:
    effective = observability.request_id([(b"x-request-id", b"\xff" * 36)])
    assert UUID(effective).version == 4


@pytest.mark.parametrize("route,status", [("/not-found", 404), ("/search", 405)])
def test_non_business_responses_have_request_ids(service, route, status) -> None:
    with TestClient(create_app(lambda: service)) as client:
        response = client.get(route, headers={"X-Request-ID": CALLER_ID})
        assert response.status_code == status and response.headers["X-Request-ID"] == CALLER_ID


def test_logs_are_structured_and_omit_protected_inputs(service, caplog) -> None:
    caplog.set_level(logging.INFO)
    headers = {**IDENTITY, "X-Request-ID": CALLER_ID, "Authorization": "Bearer secret-key"}
    with TestClient(create_app(lambda: service)) as client:
        response = client.post("/ask", json={"question": "private-question"}, headers=headers)
        assert response.status_code == 200
        assert {r["document_id"] for r in response.json()["evidence"]} == {"allowed-document"}
        assert service.reranker.inputs == [[("private-question", "protected-evidence")]]
        assert "hidden-evidence" not in json.dumps(service.generator.inputs)
        completed = next(r for r in caplog.records if getattr(r, "event", None) == "request_completed")
        assert completed.request_id == CALLER_ID and completed.method == "POST" and completed.route == "ask"
        assert completed.timings.keys() == set(observability.STAGES)
        assert completed.duration_seconds >= response.json()["timings"]["handler_seconds"]
        snapshot = client.get("/metrics").json()
        for secret in [*IDENTITY.values(), "secret-key", "private-question", "protected-evidence",
                       "hidden-evidence", "protected-answer", "allowed-document", "hidden-document"]:
            assert secret not in caplog.text and secret not in json.dumps(snapshot)


def test_counters_and_metrics_poll_boundary(service) -> None:
    app = create_app(lambda: service)
    with TestClient(app, headers=IDENTITY) as client:
        assert client.get("/metrics").json()["total_requests"] == 0
        client.get("/health")
        client.post("/search", json={"query": "q"})
        client.post("/ask", json={"question": "q"})
        snapshot = client.get("/metrics").json()
        assert snapshot["total_requests"] == 4
        assert snapshot["search_requests"] == snapshot["ask_requests"] == 1
        assert snapshot["request_duration_seconds"]["count"] == 4
        assert snapshot["request_duration_seconds"]["mean"] > 0
        assert snapshot["requests_by_route"]["metrics"]["status_families"]["2xx"] == 1
        assert snapshot["stages_seconds"]["retrieval_reranking_seconds"]["count"] == 2
        assert snapshot["stages_seconds"]["generation_request_seconds"]["count"] == 1
        assert snapshot["stages_seconds"]["online_seconds"]["count"] == 1
        assert app.state.metrics.snapshot()["total_requests"] == 5


def test_identity_denials_count_without_document_or_configuration_denials(service, caplog) -> None:
    caplog.set_level(logging.INFO)
    app = create_app(lambda: service)
    with TestClient(app) as client:
        assert client.post("/search", json={"query": "q"}).status_code == 400
        assert client.post("/ask", headers={**IDENTITY, "X-Tenant-ID": "unknown-tenant"},
                           json={"question": "q"}).status_code == 403
        assert client.post("/search", headers=IDENTITY, json={"query": ""}).status_code == 422
        service.policy_store = None
        assert client.post("/search", headers=IDENTITY, json={"query": "q"}).status_code == 503
        snapshot = app.state.metrics.snapshot()
        assert snapshot["authorization_denied_requests"] == 2
        assert snapshot["unexpected_errors"] == snapshot["generation_failures"] == 0
        assert all(r.exc_info is None for r in caplog.records)
        assert "unknown-tenant" not in caplog.text


@pytest.mark.parametrize("error", [ValueError, TypeError])
def test_internal_errors_are_sanitized_and_correlated(service, monkeypatch, caplog, error) -> None:
    caplog.set_level(logging.INFO)

    def fail(*args):
        raise error("secret-key private-query private-evidence")

    monkeypatch.setattr(service, "search", fail)
    app = create_app(lambda: service)
    with TestClient(app, headers={**IDENTITY, "X-Request-ID": CALLER_ID}) as client:
        response = client.post("/search", json={"query": "q"})
        assert response.status_code == 500 and response.headers["X-Request-ID"] == CALLER_ID
        assert "Traceback" not in response.text and "secret-key" not in response.text
        snapshot = app.state.metrics.snapshot()
        assert snapshot["unexpected_errors"] == 1 and snapshot["generation_failures"] == 0
        errors = [r for r in caplog.records if getattr(r, "event", None) == "request_error"]
        assert len(errors) == 1 and errors[0].request_id == CALLER_ID
        assert errors[0].error_type == error.__name__ and errors[0].exc_info is None
        assert "secret-key" not in caplog.text and "private-query" not in caplog.text


@pytest.mark.parametrize("failure,status,unexpected", [(ValueError, 502, 0), (TypeError, 500, 1), (None, 502, 0)])
def test_generation_failure_count_detects_actual_generator_calls(service, monkeypatch, failure, status, unexpected) -> None:
    def fail(messages):
        if failure:
            raise failure("private-answer private-key")
        return ""

    monkeypatch.setattr(service.generator, "generate", fail)
    app = create_app(lambda: service)
    with TestClient(app, headers=IDENTITY) as client:
        response = client.post("/ask", json={"question": "q"})
        assert response.status_code == status and "private-key" not in response.text
        snapshot = app.state.metrics.snapshot()
        assert snapshot["generation_failures"] == 1 and snapshot["unexpected_errors"] == unexpected
        assert snapshot["stages_seconds"]["generation_request_seconds"]["count"] == 1


@pytest.mark.parametrize("stage", ["retrieval", "reranking", "configuration"])
def test_failures_before_generation_are_classified_and_correlated(service, monkeypatch, caplog, stage) -> None:
    caplog.set_level(logging.INFO)
    def fail(*args):
        raise ValueError("private-evidence")

    if stage == "retrieval":
        monkeypatch.setattr(service.index, "search", fail)
    elif stage == "reranking":
        monkeypatch.setattr(service.reranker, "predict", fail)
    else:
        service.generator = None
    app = create_app(lambda: service)
    with TestClient(app, headers={**IDENTITY, "X-Request-ID": CALLER_ID}) as client:
        response = client.post("/ask", json={"question": "q"})
        assert response.status_code == (503 if stage == "configuration" else 500)
        assert response.headers["X-Request-ID"] == CALLER_ID
        snapshot = app.state.metrics.snapshot()
        assert snapshot["generation_failures"] == 0
        assert snapshot["unexpected_errors"] == (0 if stage == "configuration" else 1)
        assert snapshot["stages_seconds"]["generation_request_seconds"]["count"] == 0
        assert service.generator is None or service.generator.inputs == []
        errors = [r for r in caplog.records if getattr(r, "event", None) == "request_error"]
        assert len(errors) == (0 if stage == "configuration" else 1)
        assert all(r.request_id == CALLER_ID and r.unexpected for r in errors)
        assert "private-evidence" not in response.text and "private-evidence" not in caplog.text


def test_snapshot_shape_is_bounded_and_copied(service, caplog) -> None:
    caplog.set_level(logging.INFO)
    app = create_app(lambda: service)
    with TestClient(app) as client:
        for number in range(20):
            assert client.get(f"/private-path-{number}?secret=private-query").status_code == 404
        snapshot = client.get("/metrics").json()
        assert set(snapshot) == {"total_requests", "search_requests", "ask_requests", "authorization_denied_requests",
                                 "unexpected_errors", "generation_failures", "request_duration_seconds",
                                 "requests_by_route", "stages_seconds"}
        assert set(snapshot["requests_by_route"]) == set(observability.ROUTE_NAMES)
        assert set(snapshot["stages_seconds"]) == set(observability.STAGES)
        for route in snapshot["requests_by_route"].values():
            assert set(route["status_families"]) == set(observability.STATUS_FAMILIES)
            assert set(route["duration_seconds"]) == {"count", "total", "mean"}
        assert snapshot["requests_by_route"]["other"]["status_families"]["4xx"] == 20
        snapshot["requests_by_route"]["other"]["status_families"]["4xx"] = 99
        assert app.state.metrics.snapshot()["requests_by_route"]["other"]["status_families"]["4xx"] == 20
        server_logs = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("enterprise_ai_search"))
        assert "private-path" not in server_logs and "private-query" not in server_logs


def test_health_metrics_are_cheap_and_new_apps_reset(service) -> None:
    app = create_app(lambda: service)
    with TestClient(app) as client:
        assert client.get("/metrics").json()["total_requests"] == 0
        assert client.get("/health").status_code == 200
        assert service.reranker.inputs == [] and service.generator.inputs == []
        assert app.state.metrics.snapshot()["total_requests"] == 2
    fresh_app = create_app(lambda: SearchService(None, None, None))
    with TestClient(fresh_app) as client:
        assert client.get("/metrics").json()["total_requests"] == 0
        assert client.get("/health").status_code == 503
        assert fresh_app.state.metrics.snapshot()["total_requests"] == 2


def test_metrics_updates_are_thread_safe() -> None:
    metrics = Metrics()

    def update(number):
        trace = RequestTrace(str(uuid4()), "POST", "ask", generation_failed=True,
                             timings={"generation_request_seconds": 0.5})
        metrics.record(trace, 502, 1.0)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(update, range(1000)))
    snapshot = metrics.snapshot()
    assert snapshot["total_requests"] == snapshot["ask_requests"] == snapshot["generation_failures"] == 1000
    assert snapshot["request_duration_seconds"] == {"count": 1000, "total": 1000.0, "mean": 1.0}
    assert snapshot["requests_by_route"]["ask"]["status_families"]["5xx"] == 1000
    assert snapshot["stages_seconds"]["generation_request_seconds"]["total"] == 500.0


def test_concurrent_http_requests_keep_distinct_request_contexts(service, caplog) -> None:
    caplog.set_level(logging.INFO)
    app = create_app(lambda: service)
    ids = [str(uuid4()) for _ in range(20)]
    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=4) as executor:
            responses = list(executor.map(lambda value: client.get("/health", headers={"X-Request-ID": value}), ids))
        assert [r.headers["X-Request-ID"] for r in responses] == ids
        assert app.state.metrics.snapshot()["total_requests"] == 20
    records = [r for r in caplog.records if getattr(r, "event", None) == "request_completed"]
    assert {r.request_id for r in records} == set(ids) and current_trace.get() is None


def test_request_duration_includes_response_body_and_context_resets(monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(observability, "perf_counter", lambda: clock[0])
    messages = []
    metrics = Metrics()

    async def backend(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        clock[0] += 0.25
        await send({"type": "http.response.body", "body": b"a", "more_body": True})
        clock[0] += 0.75
        await send({"type": "http.response.body", "body": b"b"})

    async def send(message):
        messages.append(message)

    asyncio.run(RequestTracingMiddleware(backend, metrics)(
        {"type": "http", "method": "GET", "path": "/health", "headers": []}, None, send,
    ))
    assert metrics.snapshot()["request_duration_seconds"] == {"count": 1, "total": 1.0, "mean": 1.0}
    assert dict(messages[0]["headers"])[b"x-request-id"] and current_trace.get() is None


def test_observer_preserves_original_generator_exception_and_messages() -> None:
    error = ValueError("original error")
    messages = [{"role": "user", "content": "protected evidence"}]

    class FailedGenerator:
        def generate(self, received):
            assert received is messages
            raise error

    trace = RequestTrace(CALLER_ID, "POST", "ask")
    token = current_trace.set(trace)
    try:
        with pytest.raises(ValueError) as raised:
            ObservedGenerator(FailedGenerator()).generate(messages)
        assert raised.value is error and trace.generation_failed
        record_timings({"unknown-secret": 1.0, "online_seconds": float("nan")})
        assert set(trace.timings) == {"generation_request_seconds"}
    finally:
        current_trace.reset(token)


def test_lifecycle_flags_and_access_log_privacy(service, caplog) -> None:
    caplog.set_level(logging.INFO)
    with TestClient(create_app(lambda: service)):
        assert logging.getLogger("uvicorn.access").disabled
    events = {r.event: r for r in caplog.records if hasattr(r, "event")}
    assert {"initialization_started", "initialization_succeeded", "shutdown_completed"} <= events.keys()
    ready = events["initialization_succeeded"]
    assert ready.preparation_seconds == 1.25 and ready.authorization_initialized and ready.generation_configured


def test_error_after_response_start_propagates_sanitized_failure(caplog) -> None:
    caplog.set_level(logging.INFO)
    messages = []
    metrics = Metrics()

    async def backend(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise ValueError("private-query private-key")

    async def send(message):
        messages.append(message)

    with pytest.raises(RuntimeError, match="Response failed after headers sent"):
        asyncio.run(RequestTracingMiddleware(backend, metrics)(
            {"type": "http", "method": "GET", "path": "/health", "headers": []}, None, send,
        ))
    assert len(messages) == 1 and metrics.snapshot()["unexpected_errors"] == 1
    assert "private-key" not in caplog.text and current_trace.get() is None
