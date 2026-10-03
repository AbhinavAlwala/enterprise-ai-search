import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from enterprise_ai_search import dataset, dense, reranker, service as service_module
from enterprise_ai_search.api import create_app
from enterprise_ai_search.authorization import (
    AuthorizedIndex, DocumentAccessPolicy, PolicyStore, PrincipalContext,
    load_policy_store, parse_principal,
)
from enterprise_ai_search.hybrid import DocumentCandidate, HybridResult
from enterprise_ai_search.models import Document, SearchResult
from enterprise_ai_search.rag import SYSTEM_PROMPT
from enterprise_ai_search.service import SearchService


def headers(tenant="tenant-a", principal="alice", groups="researchers") -> dict[str, str]:
    return {"X-Tenant-ID": tenant, "X-Principal-ID": principal, "X-Groups": groups}


def policy_store() -> PolicyStore:
    policies = [
        DocumentAccessPolicy("a-public", "tenant-a", "tenant"),
        DocumentAccessPolicy("a-private", "tenant-a", "restricted", frozenset({"alice"})),
        DocumentAccessPolicy("a-group", "tenant-a", "restricted", allowed_groups=frozenset({"researchers"})),
        DocumentAccessPolicy("a-empty", "tenant-a", "restricted"),
        DocumentAccessPolicy("b-public", "tenant-b", "tenant"),
        DocumentAccessPolicy("b-private", "tenant-b", "restricted", frozenset({"alice"}), frozenset({"researchers"})),
    ]
    return PolicyStore(frozenset({"tenant-a", "tenant-b"}), {p.document_id: p for p in policies})


def candidate(identifier: str, rank: int, text: str) -> HybridResult:
    passage = SearchResult(rank, 0.2, identifier + "::0", identifier, text)
    representative = DocumentCandidate(rank, identifier, passage)
    return HybridResult(rank, 0.01, identifier, representative, representative)


class FakeIndex:
    def __init__(self, candidates: list[HybridResult] | None = None) -> None:
        self.calls = []
        identifiers = ["b-private", "a-private", "b-public", "a-group", "unlisted-doc", "a-public", "a-empty"]
        self.candidates = candidates if candidates is not None else [
            candidate(identifier, rank, "TEXT-" + identifier) for rank, identifier in enumerate(identifiers, 1)
        ]

    def search(self, query: str, top_k: int) -> list[HybridResult]:
        self.calls.append((query, top_k))
        return self.candidates


class RecordingReranker:
    def __init__(self) -> None:
        self.inputs = []

    def predict(self, pairs, **kwargs):
        self.inputs.append(pairs)
        return np.arange(len(pairs), dtype=float)


class RecordingGenerator:
    def __init__(self, answer="Fixture [1].") -> None:
        self.inputs = []
        self.answer = answer

    def generate(self, messages):
        self.inputs.append(messages)
        return self.answer


@pytest.fixture
def service() -> SearchService:
    return SearchService(FakeIndex(), RecordingReranker(), RecordingGenerator(), policy_store=policy_store())


@pytest.mark.parametrize("document,tenant,principal,groups,allowed", [
    ("a-public", "tenant-a", "bob", (), True),
    ("a-private", "tenant-a", "alice", (), True),
    ("a-private", "tenant-a", "bob", (), False),
    ("a-group", "tenant-a", "bob", ("researchers",), True),
    ("a-group", "tenant-a", "bob", ("other",), False),
    ("a-empty", "tenant-a", "alice", ("researchers",), False),
    ("unlisted-doc", "tenant-a", "alice", ("researchers",), False),
    ("b-private", "tenant-a", "alice", ("researchers",), False),
    ("a-public", "tenant-b", "alice", ("researchers",), False),
    ("b-private", "tenant-b", "alice", (), True),
    ("b-private", "tenant-b", "bob", ("researchers",), True),
    ("b-private", "tenant-b", "bob", (), False),
    ("a-public", "unknown", "alice", ("researchers",), False),
])
def test_exact_access_rules(document, tenant, principal, groups, allowed) -> None:
    context = PrincipalContext(tenant, principal, frozenset(groups))
    assert policy_store().allows(document, context) is allowed


@pytest.mark.parametrize("tenant,principal,groups", [
    (None, "alice", None), ("tenant-a", None, None), ("", "alice", None),
    ("tenant-a", " ", None), ("tenant*", "alice", None), ("tenant-a", "a/b", None),
    ("a" * 65, "alice", None), ("tenant-a", "alice", "researchers,,other"),
    ("tenant-a", "alice", "*"), ("tenant-a", "alice", " "+"x" * 2048),
    ("tenant-a", "alice", " " * 2049),
    ("tenant-a", "alice", ",".join(f"group-{i}" for i in range(33))),
])
def test_identity_parser_rejects_invalid_inputs(tenant, principal, groups) -> None:
    with pytest.raises(ValueError):
        parse_principal(tenant, principal, groups)


def test_identity_parser_trims_and_deduplicates_groups_without_case_folding() -> None:
    assert parse_principal(" tenant-a ", " Alice ", " researchers, reviewers, researchers ") == (
        PrincipalContext("tenant-a", "Alice", frozenset({"researchers", "reviewers"}))
    )
    assert parse_principal("tenant-a", "alice", None).groups == frozenset()


def valid_file() -> dict:
    return {
        "schema_version": 1, "description": "Fixture metadata", "tenants": ["tenant-a"],
        "documents": [{"document_id": "a", "tenant_id": "tenant-a", "visibility": "tenant",
                       "allowed_principals": [], "allowed_groups": []}],
    }


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(schema_version=True), lambda p: p.update(tenants=[]),
    lambda p: p.update(tenants=["tenant-a", "tenant-a"]), lambda p: p.update(allow_all=True),
    lambda p: p["documents"][0].update(visibility="public"),
    lambda p: p["documents"][0].update(tenant_id="unknown"),
    lambda p: p["documents"][0].update(allowed_groups="researchers"),
    lambda p: p["documents"][0].update(allowed_principals=["alice"]),
    lambda p: p["documents"][0].update(allowed_groups=[None]),
    lambda p: p["documents"][0].update(allowed_groups=["*"]),
    lambda p: p["documents"][0].pop("visibility"),
    lambda p: p["documents"].append(dict(p["documents"][0])),
    lambda p: p["documents"].append({"document_id": "invalid"}),
])
def test_invalid_policy_rejects_entire_file(tmp_path, mutation) -> None:
    data = valid_file()
    mutation(data)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_policy_store(path)


def test_duplicate_json_fields_and_missing_file_fail_closed(tmp_path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(valid_file()).replace('"visibility": "tenant"', '"visibility": "restricted", "visibility": "tenant"'))
    with pytest.raises(ValueError, match="Duplicate"):
        load_policy_store(path)
    with pytest.raises(OSError):
        load_policy_store(tmp_path / "missing.json")


def test_policy_store_is_a_validated_immutable_snapshot() -> None:
    policies = {"a": DocumentAccessPolicy("a", "tenant-a", "tenant")}
    store = PolicyStore(frozenset({"tenant-a"}), policies)
    policies.clear()
    assert store.allows("a", PrincipalContext("tenant-a", "alice"))
    with pytest.raises(TypeError):
        store.policies["a"] = None
    with pytest.raises(ValueError):
        PolicyStore(frozenset({"tenant-a"}), {"a": None})


def test_demo_file_is_explicit_and_reproducible() -> None:
    store = load_policy_store(Path("config/demo_access.json"))
    assert len(store.policies) == 5
    assert store.allows("9745001", PrincipalContext("tenant-a", "alice"))
    assert not store.allows("9745001", PrincipalContext("tenant-b", "alice"))
    assert store == load_policy_store(Path("config/demo_access.json"))


def test_same_query_is_scoped_for_both_tenants_before_reranking(service) -> None:
    with TestClient(create_app(lambda: service)) as client:
        a = client.post("/search", json={"query": "same query", "top_k": 10}, headers=headers()).json()
        b = client.post("/search", json={"query": "same query", "top_k": 10}, headers=headers("tenant-b")).json()
        assert {r["document_id"] for r in a["results"]} == {"a-public", "a-private", "a-group"}
        assert {r["document_id"] for r in b["results"]} == {"b-public", "b-private"}
        assert [r["rank"] for r in a["results"]] == [1, 2, 3]
        assert {text for query, text in service.reranker.inputs[0]} == {"TEXT-a-public", "TEXT-a-private", "TEXT-a-group"}
        assert {text for query, text in service.reranker.inputs[1]} == {"TEXT-b-public", "TEXT-b-private"}
        assert service.index.calls == [("same query", 50), ("same query", 50)]
        assert "TEXT-b" not in json.dumps(a) and "b-private" not in json.dumps(a)
        assert "tenant-b" not in json.dumps(a) and "allowed_groups" not in json.dumps(a)


def test_rag_prompt_and_citations_use_only_authorized_sources(service) -> None:
    with TestClient(create_app(lambda: service)) as client:
        response = client.post("/ask", json={"question": "question"}, headers=headers())
        assert response.status_code == 200
        body = response.json()
        allowed = {"a-public", "a-private", "a-group"}
        assert {item["document_id"] for item in body["evidence"]} == allowed
        assert {item["document_id"] for item in body["citation_validation"]["citations"]} <= allowed
        assert body["citation_validation"]["passed"]
        messages = service.generator.inputs[0]
        assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert all("TEXT-" + identifier in messages[1]["content"] for identifier in allowed)
        for forbidden in ("b-public", "b-private", "unlisted-doc", "a-empty", "tenant-b", "alice", "researchers"):
            assert forbidden not in json.dumps(messages) and forbidden not in response.text
        assert {text for query, text in service.reranker.inputs[0]} == {"TEXT-" + identifier for identifier in allowed}


def test_forbidden_original_source_number_does_not_map_to_hidden_document(service) -> None:
    service.generator.answer = "Fixture [99]."
    with TestClient(create_app(lambda: service)) as client:
        body = client.post("/ask", json={"question": "q"}, headers=headers(principal="bob", groups="")).json()
        assert [item["document_id"] for item in body["evidence"]] == ["a-public"]
        assert body["citation_validation"]["citations"] == []
        assert not body["citation_validation"]["passed"]
        assert body["answer"] == "Fixture [99]."


@pytest.mark.parametrize("tenant,expected", [("tenant-a", "a-public"), ("tenant-b", "b-public")])
def test_identical_cross_tenant_passages_do_not_collide(service, tenant, expected) -> None:
    service.index = FakeIndex([candidate("a-public", 1, "identical text"), candidate("b-public", 2, "identical text")])
    with TestClient(create_app(lambda: service)) as client:
        body = client.post("/search", json={"query": "identical"}, headers=headers(tenant)).json()
        assert [r["document_id"] for r in body["results"]] == [expected]
        assert service.reranker.inputs == [[("identical", "identical text")]]


def test_top_k_applies_after_authorization_and_reranking(service) -> None:
    with TestClient(create_app(lambda: service)) as client:
        results = client.post("/search", json={"query": "q", "top_k": 1}, headers=headers()).json()["results"]
        assert len(results) == 1 and results[0]["document_id"].startswith("a-")
        assert len(service.reranker.inputs[0]) == 3


def test_candidate_depth_is_bounded_and_no_backfill_occurs(service) -> None:
    service.index = FakeIndex([candidate("hidden-" + str(i), i + 1, "hidden") for i in range(50)] + [candidate("a-public", 51, "allowed")])
    with TestClient(create_app(lambda: service)) as client:
        assert client.post("/search", json={"query": "q"}, headers=headers()).json()["results"] == []
        assert service.index.calls == [("q", 50)]
        assert service.reranker.inputs == []


def test_mismatched_parent_is_removed_before_any_model_sees_text(service) -> None:
    valid = candidate("a-public", 1, "allowed")
    wrong = candidate("b-private", 1, "TEXT-b-private").dense
    service.index = FakeIndex([replace(valid, dense=wrong)])
    with TestClient(create_app(lambda: service)) as client:
        body = client.post("/ask", json={"question": "q"}, headers=headers()).json()
        assert body["evidence"] == [] and body["citation_validation"]["citations"] == []
        assert service.reranker.inputs == []
        assert "TEXT-b-private" not in json.dumps(service.generator.inputs)


@pytest.mark.parametrize("route,body", [("/search", {"query": "q"}), ("/ask", {"question": "q"})])
@pytest.mark.parametrize("identity,status", [
    ({}, 400), ({"X-Tenant-ID": "tenant-a"}, 400), (headers(tenant="unknown"), 403),
    (headers(principal="*"), 400), (headers(groups="researchers,,"), 400),
    ([("X-Tenant-ID", "tenant-a"), ("X-Tenant-ID", "tenant-b"), ("X-Principal-ID", "alice")], 400),
    ([("X-Tenant-ID", "tenant-a"), ("X-Principal-ID", "alice"), ("X-Groups", "one"), ("X-Groups", "two")], 400),
])
def test_bad_identity_fails_before_retrieval_or_generation(service, route, body, identity, status, caplog) -> None:
    with TestClient(create_app(lambda: service)) as client:
        response = client.post(route, json=body, headers=identity)
        assert response.status_code == status
        assert service.index.calls == [] and service.reranker.inputs == [] and service.generator.inputs == []
        assert all(value not in response.text for value in ("tenant-a", "tenant-b", "unknown", "alice"))
        assert "alice" not in caplog.text and "tenant-a" not in caplog.text


def test_no_policy_store_disables_protected_requests_and_health_is_private(service) -> None:
    service.policy_store = None
    with TestClient(create_app(lambda: service)) as client:
        health = client.get("/health")
        assert health.status_code == 503 and not health.json()["authorization_initialized"]
        assert health.json()["retrieval_initialized"]
        assert "tenant" not in health.text and "alice" not in health.text
        for route, body in [("/search", {"query": "q"}), ("/ask", {"question": "q"})]:
            assert client.post(route, json=body, headers=headers()).status_code == 503
        assert service.index.calls == [] and service.generator.inputs == []


@pytest.mark.parametrize("identifier", ["b-private", "unlisted-doc", "a-empty", "nonexistent"])
def test_empty_search_does_not_reveal_denied_document_existence(service, identifier) -> None:
    service.index = FakeIndex([candidate(identifier, 1, "forbidden passage")])
    with TestClient(create_app(lambda: service)) as client:
        response = client.post("/search", json={"query": "q"}, headers=headers())
        assert response.status_code == 200 and response.json()["results"] == []
        assert identifier not in response.text and "forbidden passage" not in response.text
        assert service.reranker.inputs == []


def test_service_cannot_bypass_policy_for_unknown_tenant(service) -> None:
    with pytest.raises(PermissionError):
        service.search("q", 1, PrincipalContext("unknown", "alice"))
    assert service.index.calls == []


@pytest.mark.parametrize("configuration", ["unset", "missing-file", "invalid-file"])
def test_default_loader_never_falls_back_to_broad_access(tmp_path, monkeypatch, configuration) -> None:
    for name in ("GENERATION_ENDPOINT", "GENERATION_MODEL", "GENERATION_API_KEY", "AUTHORIZATION_POLICY_PATH"):
        monkeypatch.delenv(name, raising=False)
    if configuration != "unset":
        path = tmp_path / "policy.json"
        monkeypatch.setenv("AUTHORIZATION_POLICY_PATH", str(path))
        if configuration == "invalid-file":
            path.write_text('{"allow_all": true}')
    monkeypatch.setattr(dataset, "load_corpus", lambda path: [Document("a-public", "fixture")])
    monkeypatch.setattr(dense, "load_encoder", lambda path: (object(), 0))
    monkeypatch.setattr(dense, "prepare_embeddings", lambda chunks, model, path: (np.ones((1, 384), dtype=np.float32) / np.sqrt(384), {}))
    monkeypatch.setattr(reranker, "load_reranker", lambda path: (RecordingReranker(), 0))
    service = service_module.load_service()
    assert service.retrieval_initialized and not service.authorization_initialized
    with TestClient(create_app(lambda: service)) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/search", json={"query": "q"}, headers=headers()).status_code == 503
