import gzip
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from enterprise_ai_search import dense, generation, rag, reranker
from enterprise_ai_search.bm25 import BM25Index
from enterprise_ai_search.cli import main
from enterprise_ai_search.dense import DenseIndex
from enterprise_ai_search.hybrid import DocumentCandidate, HybridIndex, HybridResult
from enterprise_ai_search.models import Chunk, SearchResult
from enterprise_ai_search.rag import EVIDENCE_COUNT, SYSTEM_PROMPT, ask, build_context, build_messages, select_evidence, validate_citations
from enterprise_ai_search.reranker import PassageScore, RerankedResult


def ranked_result(document_id: str, rank: int) -> RerankedResult:
    passage = SearchResult(20, 0.5, document_id + "::chunk::1", document_id, "Winning passage for " + document_id)
    candidate = HybridResult(30 - rank, 0.02, document_id, None, DocumentCandidate(2, document_id, passage))
    return RerankedResult(rank, -float(rank), document_id, passage, candidate, (PassageScore(passage, -float(rank), ("dense",)),))


@pytest.fixture
def results() -> list[RerankedResult]:
    return [ranked_result(str(rank), rank) for rank in range(1, 8)]


def test_top_five_source_numbering_and_selected_passage_provenance(results: list[RerankedResult]) -> None:
    evidence = select_evidence(results)
    assert len(evidence) == EVIDENCE_COUNT == 5
    assert [item.source_number for item in evidence] == [1, 2, 3, 4, 5]
    for item, result in zip(evidence, results):
        assert item.document_id == result.document_id
        assert item.chunk_id == result.passage.chunk_id
        assert item.text == result.passage.text
        assert item.rank == result.rank
    assert select_evidence([]) == ()


def test_context_is_deterministic_and_preserves_text(results: list[RerankedResult]) -> None:
    evidence = select_evidence(results[:2])
    expected = "[1] Document: 1; Chunk: 1::chunk::1; Rank: 1\nWinning passage for 1\n\n[2] Document: 2; Chunk: 2::chunk::1; Rank: 2\nWinning passage for 2"
    assert build_context(evidence) == expected == build_context(evidence)
    assert build_context(()) == ""


def test_evidence_rejects_duplicate_parents_or_mismatched_passages(results: list[RerankedResult]) -> None:
    with pytest.raises(ValueError, match="unique"):
        select_evidence([results[0], results[0]])
    with pytest.raises(ValueError, match="belong"):
        select_evidence([replace(results[0], document_id="other")])


def test_prompt_has_one_grounding_instruction_location_and_exact_context(results: list[RerankedResult]) -> None:
    evidence = select_evidence(results[:1])
    messages = build_messages("  What\n happens? ", evidence)
    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Question:\nWhat happens?\n\nEvidence:\n" + build_context(evidence)},
    ]
    assert "without outside knowledge" in SYSTEM_PROMPT
    assert "insufficient" in SYSTEM_PROMPT and "do not invent sources" in SYSTEM_PROMPT
    assert "No evidence passages were retrieved." in build_messages("question", ())[1]["content"]
    with pytest.raises(ValueError):
        build_messages(" \n", evidence)


def test_valid_citations_map_ids_and_deduplicate_in_first_appearance_order(results: list[RerankedResult]) -> None:
    evidence = select_evidence(results)
    validation = validate_citations("Fixture claim [2][1], repeated [2].", evidence)
    assert validation.passed and not validation.missing_citations
    assert [(citation.source_number, citation.document_id, citation.chunk_id) for citation in validation.citations] == [
        (2, "2", "2::chunk::1"), (1, "1", "1::chunk::1"),
    ]
    assert validation.invalid_source_numbers == ()


@pytest.mark.parametrize("answer,invalid", [("Fixture [6]", (6,)), ("Fixture [0]", (0,)), ("Fixture [1][-1][99]", (-1, 99))])
def test_out_of_range_citations_fail_without_discarding_valid_mappings(results: list[RerankedResult], answer: str, invalid: tuple) -> None:
    validation = validate_citations(answer, select_evidence(results))
    assert not validation.passed and not validation.missing_citations
    assert validation.invalid_source_numbers == invalid
    assert all(citation.source_number == 1 for citation in validation.citations)


@pytest.mark.parametrize("answer", ["Fixture answer without citations.", "Grouped syntax [1, 2] is unsupported.", "The supplied evidence is insufficient to answer this question."])
def test_no_supported_citation_markers_fail_even_for_insufficiency(results: list[RerankedResult], answer: str) -> None:
    validation = validate_citations(answer, select_evidence(results))
    assert not validation.passed and validation.missing_citations
    assert validation.citations == ()


class FakeGenerator:
    def __init__(self, answer: str = "Fixture answer [1].") -> None:
        self.answer = answer
        self.messages: list[dict[str, str]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.messages = messages
        return self.answer


class FakeEncoder:
    tokenizer = staticmethod(lambda texts, **kwargs: {"input_ids": [[1] for _ in texts]})

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for number, text in enumerate(texts):
            vectors[number, 0 if text in ("cat", "kitty") else 1] = 1
        return vectors


class FakeReranker:
    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> np.ndarray:
        return np.array([10 if text == "cat" else -10 for query, text in pairs])


def test_fake_generator_integration_uses_actual_frozen_retrieval_functions() -> None:
    chunks = [Chunk("a0", "a", "cat"), Chunk("b0", "b", "dog")]
    encoder = FakeEncoder()
    index = HybridIndex(BM25Index(chunks), DenseIndex(chunks, encoder.encode([chunk.text for chunk in chunks]), encoder))
    generator = FakeGenerator()
    result = ask("kitty", index, FakeReranker(), generator)
    assert result.answer == generator.answer and result.citation_validation.passed
    assert result.evidence[0].document_id == "a" and result.evidence[0].chunk_id == "a0"
    assert result.evidence[0].text == "cat"
    assert "Question:\nkitty" in generator.messages[1]["content"]
    assert "[1] Document: a; Chunk: a0; Rank: 1\ncat" in generator.messages[1]["content"]
    assert result.timings["online_seconds"] >= result.timings["retrieval_reranking_seconds"] + result.timings["generation_request_seconds"]


def test_ask_requests_fifty_candidates_and_uses_top_five_after_reranking() -> None:
    class FakeIndex:
        def search(self, query: str, top_k: int) -> list[HybridResult]:
            assert top_k == 50
            return [
                HybridResult(number, 0.01, str(number), None, DocumentCandidate(number, str(number), SearchResult(number, 0.1, str(number), str(number), str(number))))
                for number in range(1, 51)
            ]
    class NumericReranker:
        def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> np.ndarray:
            return np.array([float(text) for query, text in pairs])
    result = ask("question", FakeIndex(), NumericReranker(), FakeGenerator("Fixture [5]"))
    assert [item.document_id for item in result.evidence] == ["50", "49", "48", "47", "46"]
    assert result.citation_validation.citations[0].document_id == "46"


@pytest.mark.parametrize("answer", ["Fixture answer without citations.", "Fixture [99]", "The supplied evidence is insufficient to answer this question."])
def test_answer_preserved_when_citation_validation_fails(answer: str) -> None:
    class EmptyIndex:
        def search(self, *args: object) -> list:
            return []
    generator = FakeGenerator(answer)
    result = ask("question", EmptyIndex(), FakeReranker(), generator)
    assert result.answer == answer and not result.citation_validation.passed
    assert result.evidence == ()
    assert "No evidence passages were retrieved." in generator.messages[1]["content"]


def test_empty_question_fails_before_retrieval_or_generation() -> None:
    generator = FakeGenerator()
    with pytest.raises(ValueError, match="nonempty"):
        ask(" \n", None, None, generator)
    assert generator.messages == []


@pytest.mark.parametrize("answer", ["", "   ", None])
def test_empty_or_nontext_generator_output_rejected(answer: str) -> None:
    class EmptyIndex:
        def search(self, *args: object) -> list:
            return []
    with pytest.raises(ValueError, match="nonempty answer"):
        ask("question", EmptyIndex(), FakeReranker(), FakeGenerator(answer))


def test_cli_ask_with_fake_models_and_generator_has_provenance_and_timings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "corpus.jsonl.gz").write_bytes(gzip.compress(b'{"_id":"a","text":"cat"}\n{"_id":"b","text":"dog"}\n'))
    monkeypatch.setenv("GENERATION_ENDPOINT", "http://localhost/v1/chat/completions")
    monkeypatch.setenv("GENERATION_MODEL", "fixture-model")
    monkeypatch.setenv("GENERATION_API_KEY", "fixture-secret")
    monkeypatch.setattr(generation, "HttpGenerator", lambda config: FakeGenerator())
    monkeypatch.setattr(dense, "load_encoder", lambda path: (FakeEncoder(), 0))
    monkeypatch.setattr(reranker, "load_reranker", lambda path: (FakeReranker(), 0))
    monkeypatch.setattr(sys, "argv", ["enterprise-search", "ask", "kitty", "--data-dir", str(tmp_path), "--cache-dir", str(tmp_path / "cache")])
    main()
    output = capsys.readouterr().out
    result = json.loads(output)
    assert "fixture-secret" not in output
    assert result["citation_validation"]["passed"]
    assert result["citation_validation"]["citations"][0]["document_id"] == "a"
    assert result["evidence"][0]["text"] == "cat"
    assert result["timings"]["total_end_to_end_seconds"] >= result["timings"]["preparation_seconds"] + result["timings"]["online_seconds"]


def test_cli_missing_config_exits_before_loading_models(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    for name in ("GENERATION_ENDPOINT", "GENERATION_MODEL", "GENERATION_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dense, "load_encoder", lambda path: pytest.fail("Must not load a model without configuration"))
    monkeypatch.setattr(sys, "argv", ["enterprise-search", "ask", "question"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "GENERATION_ENDPOINT, GENERATION_MODEL" in capsys.readouterr().err
