import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from enterprise_ai_search.bm25 import BM25Index
from enterprise_ai_search.claim_verification import VERIFICATION_PROMPT, build_verification_messages, parse_verdict, verify_claim
from enterprise_ai_search.dataset import StanceClaim, load_queries, load_stance_claims
from enterprise_ai_search.dense import DenseIndex
from enterprise_ai_search.hybrid import HybridIndex
from enterprise_ai_search.models import Chunk
from enterprise_ai_search.rag import EvidenceItem


def query_file(path: Path, metadata: object) -> Path:
    records = [
        {"_id": "test", "text": "Claim", "metadata": metadata},
        {"_id": "empty", "text": "Unlabeled", "metadata": {}},
        {"_id": "train", "text": "Not selected", "metadata": {"broken": []}},
    ]
    content = "\n".join(json.dumps(r) for r in records)
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(content.encode()))
    else:
        path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz"])
@pytest.mark.parametrize("stance", ["SUPPORT", "CONTRADICT"])
def test_load_explicit_stance_without_changing_query_behavior(tmp_path: Path, suffix: str, stance: str) -> None:
    metadata = {"b": [{"label": stance, "sentences": [0, 3]}], "a": [{"label": stance, "sentences": [1]}]}
    path = query_file(tmp_path / ("queries" + suffix), metadata)
    assert load_queries(path) == {"test": "Claim", "empty": "Unlabeled", "train": "Not selected"}
    assert load_stance_claims(path, {"test", "empty"}) == [StanceClaim("test", "Claim", stance, ("a", "b"))]
    with pytest.raises(ValueError, match="unknown query"):
        load_stance_claims(path, {"missing"})


@pytest.mark.parametrize("metadata", [
    [], {"a": []}, {"a": [{}]}, {"a": [{"label": "ABSTAIN", "sentences": [0]}]},
    {"a": [{"label": "SUPPORT", "sentences": [-1]}]},
    {"a": [{"label": "SUPPORT", "sentences": [True]}]},
    {"a": [{"label": "SUPPORT", "sentences": []}]},
    {"a": [{"label": "SUPPORT", "sentences": [0]}, {"label": "CONTRADICT", "sentences": [1]}]},
    {"a": [{"label": "SUPPORT", "sentences": [0]}], "b": [{"label": "CONTRADICT", "sentences": [0]}]},
])
def test_reject_invalid_or_inconsistent_gold(tmp_path: Path, metadata: object) -> None:
    with pytest.raises(ValueError, match=r":1:"):
        load_stance_claims(query_file(tmp_path / "queries.jsonl", metadata), {"test"})


@pytest.fixture
def evidence() -> tuple[EvidenceItem, ...]:
    return (EvidenceItem(1, "doc", "doc::0", 1, "Evidence text"),)


@pytest.mark.parametrize("verdict", ["SUPPORT", "CONTRADICT", "ABSTAIN"])
def test_parse_all_verdicts_with_source_mappings(verdict: str, evidence: tuple) -> None:
    parsed = parse_verdict(json.dumps({"verdict": verdict, "explanation": "Reason [1]."}), evidence)
    assert parsed.verdict == verdict and parsed.citation_validation.passed
    assert parsed.citation_validation.citations[0].document_id == "doc"


@pytest.mark.parametrize("raw", [
    "SUPPORT", '```json\n{"verdict":"SUPPORT","explanation":"x [1]"}\n```', "{", "[]", "null", None,
    '{"verdict":"support","explanation":"x"}', '{"verdict":"UNKNOWN","explanation":"x"}',
    '{"verdict":[],"explanation":"x"}', '{"verdict":"SUPPORT","explanation":" "}',
    '{"verdict":"SUPPORT","explanation":1}', '{"verdict":"SUPPORT"}',
    '{"verdict":"SUPPORT","explanation":"x","extra":1}',
    '{"verdict":"SUPPORT","verdict":"CONTRADICT","explanation":"x"}',
])
def test_no_repair_or_freeform_coercion(raw: str, evidence: tuple) -> None:
    with pytest.raises(ValueError):
        parse_verdict(raw, evidence)


@pytest.mark.parametrize("explanation", ["Reason [99].", "Reason [1][-1].", "Insufficient evidence."])
def test_reference_failure_keeps_structured_prediction(explanation: str, evidence: tuple) -> None:
    parsed = parse_verdict(json.dumps({"verdict": "ABSTAIN", "explanation": explanation}), evidence)
    assert parsed.verdict == "ABSTAIN" and not parsed.citation_validation.passed


def test_prompt_contains_only_claim_and_supplied_evidence(evidence: tuple) -> None:
    messages = build_verification_messages(" Claim\n text ", evidence)
    assert messages[0]["content"] == VERIFICATION_PROMPT
    assert messages[1]["content"] == "Claim:\nClaim text\n\nEvidence:\n[1] Document: doc; Chunk: doc::0; Rank: 1\nEvidence text"
    assert "No evidence passages" in build_verification_messages("claim", ())[1]["content"]
    with pytest.raises(ValueError):
        build_verification_messages(" ", evidence)


class FakeEncoder:
    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        return np.array([[1, 0] if text in ("cat", "kitty") else [0, 1] for text in texts], dtype=np.float32)


class FakeReranker:
    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> np.ndarray:
        return np.array([10 if text == "cat" else -10 for _, text in pairs])


@pytest.mark.parametrize("raw,status,predicted", [
    ('{"verdict":"SUPPORT","explanation":"Cat [1]."}', "ok", "SUPPORT"),
    ('{"verdict":"ABSTAIN","explanation":"Unclear."}', "ok", "ABSTAIN"),
    ("free-form SUPPORT [1]", "failed", None),
])
def test_fake_generation_with_real_retrieval(raw: str, status: str, predicted: str | None) -> None:
    chunks = [Chunk("a0", "a", "cat"), Chunk("b0", "b", "dog")]
    encoder = FakeEncoder()
    index = HybridIndex(BM25Index(chunks), DenseIndex(chunks, encoder.encode([c.text for c in chunks]), encoder))
    class FakeGenerator:
        def generate(self, messages: list[dict[str, str]]) -> str:
            assert "gold" not in messages[1]["content"]
            assert "[1] Document: a; Chunk: a0; Rank: 1\ncat" in messages[1]["content"]
            return raw
    record = verify_claim(StanceClaim("q", "kitty", "CONTRADICT", ("a",)), index, FakeReranker(), FakeGenerator())
    assert record["parse_status"] == status and record["predicted_stance"] == predicted
    assert record["gold_document_present"] and record["supplied_document_ids"] == ["a", "b"]
    assert record["abstained"] == (predicted == "ABSTAIN")
    assert record["timings"]["online_seconds"] >= record["timings"]["retrieval_reranking_seconds"] + record["timings"]["generation_request_seconds"]


def test_generation_error_is_not_a_parse_error_or_abstention() -> None:
    class EmptyIndex:
        def search(self, query: str, top_k: int) -> list:
            assert top_k == 50
            return []
    class FailingGenerator:
        def generate(self, messages: list[dict[str, str]]) -> str:
            raise ValueError("failed request")
    record = verify_claim(StanceClaim("q", "claim", "SUPPORT", ("a",)), EmptyIndex(), FakeReranker(), FailingGenerator())
    assert record["generation_status"] == "failed" and record["parse_status"] == "not_attempted"
    assert record["predicted_stance"] is None and not record["abstained"]
