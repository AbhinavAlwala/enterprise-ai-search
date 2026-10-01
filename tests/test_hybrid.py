import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from enterprise_ai_search import hybrid_evaluation
from enterprise_ai_search.bm25 import BM25Index
from enterprise_ai_search.cli import main
from enterprise_ai_search.comparison import _hybrid_examples, compare_reports
from enterprise_ai_search.dense import DenseIndex
from enterprise_ai_search.evaluation import recall_at_k
from enterprise_ai_search.hybrid import (
    CANDIDATE_DEPTH, RRF_K, DocumentCandidate, HybridIndex,
    document_candidates, reciprocal_rank_fusion,
)
from enterprise_ai_search.models import Chunk, SearchResult


def candidates(document_ids: list[str], raw_score: float = 1.0) -> list[DocumentCandidate]:
    return [
        DocumentCandidate(rank, doc_id, SearchResult(rank, raw_score, doc_id + "-chunk", doc_id, doc_id))
        for rank, doc_id in enumerate(document_ids, 1)
    ]


def test_document_extraction_preserves_best_passage_and_consecutive_document_ranks() -> None:
    chunks = [
        SearchResult(1, 9, "a0", "a", "best a"),
        SearchResult(2, 8, "a1", "a", "other a"),
        SearchResult(3, 7, "b0", "b", "best b"),
        SearchResult(4, 6, "c0", "c", "best c"),
    ]
    result = document_candidates(chunks, 2)
    assert [candidate.document_id for candidate in result] == ["a", "b"]
    assert [candidate.rank for candidate in result] == [1, 2]
    assert result[0].passage is chunks[0]
    assert result[1].passage is chunks[2]
    assert document_candidates([], 100) == []
    with pytest.raises(ValueError):
        document_candidates(chunks, 0)


def test_rrf_scores_shared_and_single_list_documents() -> None:
    bm25, dense = candidates(["a", "b"]), candidates(["b", "c"])
    results = reciprocal_rank_fusion(bm25, dense, 10)
    assert [result.document_id for result in results] == ["b", "a", "c"]
    assert [result.score for result in results] == pytest.approx([
        1 / 62 + 1 / 61, 1 / 61, 1 / 62,
    ])
    assert results[0].bm25 is bm25[1] and results[0].dense is dense[0]
    assert results[1].dense is None and results[2].bm25 is None
    assert RRF_K == 60 and CANDIDATE_DEPTH == 100


def test_rrf_ignores_raw_scores_and_uses_stable_document_id_ties() -> None:
    first = reciprocal_rank_fusion(candidates(["z", "a"], 10000), candidates(["a", "z"], -100), 2)
    second = reciprocal_rank_fusion(candidates(["z", "a"], 0.001), candidates(["a", "z"], 0.9), 2)
    assert [result.document_id for result in first] == ["a", "z"]
    assert [result.score for result in first] == [result.score for result in second]


def test_fusion_retains_each_retrievers_own_best_passage() -> None:
    lexical = document_candidates([SearchResult(1, 20, "a0", "a", "lexical passage")], 100)
    semantic = document_candidates([SearchResult(1, 0.8, "a1", "a", "semantic passage")], 100)
    result = reciprocal_rank_fusion(lexical, semantic, 1)[0]
    assert result.bm25.passage.chunk_id == "a0"
    assert result.dense.passage.chunk_id == "a1"


def test_rrf_empty_rankings_and_top_k() -> None:
    assert reciprocal_rank_fusion([], [], 10) == []
    assert len(reciprocal_rank_fusion(candidates(["a", "b"]), [], 1)) == 1
    assert len(reciprocal_rank_fusion([], candidates(["a"]), 99)) == 1
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], [], 0)


def test_duplicate_or_inconsistent_document_candidates_rejected() -> None:
    ranking = candidates(["a"])
    with pytest.raises(ValueError, match="unique"):
        reciprocal_rank_fusion(ranking * 2, [], 10)
    wrong_rank = DocumentCandidate(2, "a", ranking[0].passage)
    wrong_parent = DocumentCandidate(1, "b", ranking[0].passage)
    for candidate in (wrong_rank, wrong_parent):
        with pytest.raises(ValueError, match="consecutive"):
            reciprocal_rank_fusion([candidate], [], 10)


class FixtureEncoder:
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.tokenizer = lambda texts, **kwargs: {"input_ids": [[1, 2] for _ in texts]}

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i, 0 if text in ("cat", "kitty") else 1] = 1
        return vectors


def test_duplicate_chunks_do_not_consume_document_candidate_depth() -> None:
    chunks = [Chunk(f"a{i:02}", "a", "cat") for i in range(12)] + [Chunk("b", "b", "cat")]
    vectors = np.array([[1, 0]] * len(chunks), dtype=np.float32)
    bm25, dense = BM25Index(chunks), DenseIndex(chunks, vectors, FixtureEncoder(2))
    index = HybridIndex(bm25, dense, candidate_depth=2)
    results = index.search("cat", 10)
    assert [result.document_id for result in results] == ["a", "b"]
    assert results[0].score == pytest.approx(2 / 61)
    assert results[1].score == pytest.approx(2 / 62)
    assert results[0].bm25.passage.chunk_id == "a00"
    assert results[0].dense.passage.chunk_id == "a00"
    shallow = HybridIndex(bm25, dense, candidate_depth=1).search("cat", 10)
    assert [result.document_id for result in shallow] == ["a"]
    assert index.search(" ") == []
    with pytest.raises(ValueError):
        index.search("cat", 0)
    with pytest.raises(ValueError):
        HybridIndex(bm25, dense, candidate_depth=0)
    with pytest.raises(ValueError, match="same ordered"):
        HybridIndex(bm25, DenseIndex(list(reversed(chunks)), vectors, FixtureEncoder(2)))


@pytest.fixture
def hybrid_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    monkeypatch.setattr(hybrid_evaluation, "load_encoder", lambda path: (FixtureEncoder(), 0.0))
    documents = [{"_id": "a", "text": "cat"}, {"_id": "b", "text": "dog"}]
    queries = [{"_id": "1", "text": "cat"}, {"_id": "2", "text": "kitty"}]
    for name, records in [("corpus.jsonl.gz", documents), ("queries.jsonl.gz", queries)]:
        content = "\n".join(json.dumps(record) for record in records)
        (tmp_path / name).write_bytes(gzip.compress(content.encode()))
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels/test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n1\ta\t1\n2\ta\t1\n", encoding="utf-8"
    )
    report = hybrid_evaluation.evaluate_hybrid_scifact(tmp_path, tmp_path / "cache")
    return tmp_path, report


def test_hybrid_evaluation_reuses_metrics_and_cache(hybrid_report: tuple[Path, dict]) -> None:
    data_dir, report = hybrid_report
    assert report["counts"]["evaluated_queries"] == 2
    assert all(value == 1 for value in report["metrics"].values())
    assert report["settings"]["rrf_k"] == 60
    assert report["settings"]["candidate_depth_documents_per_retriever"] == 100
    assert report["settings"]["chunking"] == {"size": 180, "overlap": 30}
    assert report["settings"]["bm25"] == {"k1": 1.2, "b": 0.75}
    record = report["per_query"]["2"]
    assert record["fusion_details"][0]["bm25_rank"] is None
    assert record["fusion_details"][0]["dense_rank"] == 1
    assert recall_at_k(record["document_ids"], {"a": 1}, 10) == record["metrics"]["Recall@10"]
    repeated = hybrid_evaluation.evaluate_hybrid_scifact(data_dir, data_dir / "cache")
    assert repeated["preparation"]["cache_hit"]
    assert repeated["metrics"] == report["metrics"]


def test_cli_hybrid_evaluation_writes_json(
    hybrid_report: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir, _ = hybrid_report
    output = data_dir / "hybrid.json"
    monkeypatch.setattr("sys.argv", [
        "enterprise-search", "evaluate-hybrid", "--data-dir", str(data_dir),
        "--cache-dir", str(data_dir / "cache"), "--output", str(output),
    ])
    main()
    assert json.loads(output.read_text())["method"] == "hybrid"
    assert "SciFact hybrid test: 2 queries" in capsys.readouterr().out


def _component_reports(report: dict) -> tuple[dict, dict]:
    bm25, dense = copy.deepcopy(report), copy.deepcopy(report)
    dense["method"] = "dense"
    return bm25, dense


def test_three_way_comparison_uses_compatible_artifacts(hybrid_report: tuple[Path, dict]) -> None:
    data_dir, report = hybrid_report
    bm25, dense = _component_reports(report)
    paths = [data_dir / f"{method}.json" for method in ("bm25", "dense", "hybrid")]
    for path, contents in zip(paths, (bm25, dense, report)):
        path.write_text(json.dumps(contents))
    summary = compare_reports(paths[0], paths[1], data_dir, paths[2])
    assert set(summary["artifact_sha256"]) == {"bm25", "dense", "hybrid"}
    assert summary["metrics"]["Recall@10"]["hybrid_minus_dense"] == 0
    assert summary["examples"] == []


def test_qualitative_selection_uses_first_numeric_id_and_measured_ranks() -> None:
    queries = {query_id: f"query {query_id}" for query_id in ("10", "3", "2", "1")}
    qrels = {query_id: {"gold": 1} for query_id in queries}
    rankings = {
        "bm25": {"1": ["x"], "2": ["x", "y", "gold"], "3": ["gold"], "10": ["x"]},
        "dense": {"1": ["gold"], "2": ["x", "gold"], "3": ["x"], "10": ["gold"]},
        "hybrid": {"1": ["gold"], "2": ["gold"], "3": ["x"], "10": ["gold"]},
    }
    reports = {
        method: {"per_query": {query_id: {"document_ids": ranking} for query_id, ranking in rows.items()}}
        for method, rows in rankings.items()
    }
    examples = _hybrid_examples(reports, queries, qrels)
    assert [(row["category"], row["query_id"]) for row in examples] == [
        ("hybrid_hit_component_miss", "1"),
        ("hybrid_improves_both_first_hit_ranks", "2"),
        ("hybrid_miss_component_hit", "3"),
    ]
    assert examples[1]["first_relevant_ranks"] == {"bm25": 3, "dense": 2, "hybrid": 1}


@pytest.mark.parametrize("change", ["inputs", "embedding", "bm25", "queries", "metric"])
def test_three_way_comparison_rejects_inconsistency(hybrid_report: tuple[Path, dict], change: str) -> None:
    data_dir, report = hybrid_report
    bm25, dense = _component_reports(report)
    bad = copy.deepcopy(report)
    if change == "inputs":
        bad["input_sha256"]["corpus.jsonl.gz"] = "different"
    elif change == "embedding":
        bad["settings"]["embedding"]["revision"] = "different"
    elif change == "bm25":
        bad["settings"]["bm25"]["k1"] = 2
    elif change == "queries":
        bad["per_query"].pop("1")
    else:
        bad["metrics"]["Recall@5"] = 0.5
    paths = [data_dir / f"{method}.json" for method in ("bm25", "dense", "hybrid")]
    for path, contents in zip(paths, (bm25, dense, bad)):
        path.write_text(json.dumps(contents))
    with pytest.raises(ValueError):
        compare_reports(paths[0], paths[1], data_dir, paths[2])


@pytest.mark.parametrize("output", ["results/scifact_bm25_test.json", "results/scifact_dense_test.json"])
def test_hybrid_cannot_overwrite_frozen_reports(monkeypatch: pytest.MonkeyPatch, output: str) -> None:
    monkeypatch.setattr("sys.argv", ["enterprise-search", "evaluate-hybrid", "--output", output])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


@pytest.mark.parametrize("command", ["evaluate", "evaluate-dense"])
def test_default_baseline_outputs_are_frozen(monkeypatch: pytest.MonkeyPatch, command: str) -> None:
    monkeypatch.setattr("sys.argv", ["enterprise-search", command])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
