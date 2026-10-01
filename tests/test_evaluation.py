import gzip
import json
import math
from pathlib import Path

import pytest

from enterprise_ai_search import dataset
from enterprise_ai_search.bm25 import BM25Index
from enterprise_ai_search.cli import main
from enterprise_ai_search.dataset import load_qrels, load_queries
from enterprise_ai_search.evaluation import (
    evaluate_scifact,
    ndcg_at_k,
    ranked_document_ids,
    recall_at_k,
    reciprocal_rank_at_k,
)
from enterprise_ai_search.models import Chunk, SearchResult


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Evaluation tests must not access the network")
    monkeypatch.setattr(dataset, "urlopen", forbidden)


def test_document_deduplication_keeps_best_occurrence_and_compacts_ranks() -> None:
    results = [
        SearchResult(1, 5, "a0", "a", ""),
        SearchResult(2, 4, "a1", "a", ""),
        SearchResult(3, 3, "b0", "b", ""),
        SearchResult(4, 2, "a2", "a", ""),
        SearchResult(5, 1, "c0", "c", ""),
    ]
    assert ranked_document_ids(results, 2) == ["a", "b"]
    assert ranked_document_ids(results, 10) == ["a", "b", "c"]
    assert ranked_document_ids([], 10) == []
    with pytest.raises(ValueError):
        ranked_document_ids(results, 0)


def test_all_candidates_find_documents_beyond_repeated_top_chunks() -> None:
    chunks = [Chunk(f"a{i:02}", "a", "cat cat") for i in range(12)]
    chunks.append(Chunk("b0", "b", "cat filler filler filler"))
    index = BM25Index(chunks)
    assert ranked_document_ids(index.search("cat", 10), 2) == ["a"]
    assert ranked_document_ids(index.search("cat", len(chunks)), 2) == ["a", "b"]


def test_tied_document_ranking_is_deterministic() -> None:
    chunks = [Chunk("z", "b", "cat"), Chunk("a", "a", "cat"), Chunk("b", "a", "cat")]
    expected = ["a", "b"]
    for ordering in (chunks, list(reversed(chunks))):
        assert ranked_document_ids(BM25Index(ordering).search("cat", 3), 10) == expected


def test_recall_multiple_relevant_documents_uses_all_judged_positives() -> None:
    judgments = {"a": 1, "b": 1, "c": 1, "zero": 0}
    ranking = ["unjudged", "b", "zero", "a"]
    assert recall_at_k(ranking, judgments, 1) == 0
    assert recall_at_k(ranking, judgments, 2) == pytest.approx(1 / 3)
    assert recall_at_k(ranking, judgments, 10) == pytest.approx(2 / 3)


def test_reciprocal_rank_uses_first_positive_and_obeys_cutoff() -> None:
    ranking = ["zero", "unjudged", "b", "a"]
    judgments = {"a": 1, "b": 2, "zero": 0}
    assert reciprocal_rank_at_k(ranking, judgments, 2) == 0
    assert reciprocal_rank_at_k(ranking, judgments, 3) == pytest.approx(1 / 3)
    assert reciprocal_rank_at_k(ranking, judgments, 10) == pytest.approx(1 / 3)


def test_ndcg_binary_multiple_positives_and_missing_relevant_documents() -> None:
    judgments = {"a": 1, "b": 1, "c": 1}
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (
        1 + 1 / math.log2(3) + 1 / math.log2(4)
    )
    assert ndcg_at_k(["unjudged", "a", "b"], judgments, 10) == pytest.approx(expected)
    assert ndcg_at_k(["a", "b", "c"], judgments, 10) == pytest.approx(1)
    assert ndcg_at_k(["a"], judgments, 1) == pytest.approx(1)


def test_ndcg_uses_linear_graded_gain_and_full_ideal_ranking() -> None:
    judgments = {"a": 3, "b": 1, "c": 2}
    dcg = 1 + 3 / math.log2(3)
    ideal = 3 + 2 / math.log2(3)
    assert ndcg_at_k(["b", "a"], judgments, 2) == pytest.approx(dcg / ideal)


@pytest.mark.parametrize("metric", [recall_at_k, reciprocal_rank_at_k, ndcg_at_k])
def test_empty_no_hit_and_cutoff_misses_score_zero(metric) -> None:
    judgments = {"a": 1}
    assert metric([], judgments, 10) == 0
    assert metric(["unjudged"], judgments, 10) == 0
    assert metric([f"x{i}" for i in range(10)] + ["a"], judgments, 10) == 0


@pytest.mark.parametrize("metric", [recall_at_k, reciprocal_rank_at_k, ndcg_at_k])
@pytest.mark.parametrize("ranking,judgments,k", [
    (["a", "a"], {"a": 1}, 10),
    ([], {"a": 1}, 0),
    ([], {}, 10),
    ([], {"a": 0}, 10),
    ([], {"a": -1}, 10),
    ([], {"a": 0.5}, 10),
])
def test_invalid_metric_inputs_are_not_silently_scored(metric, ranking, judgments, k) -> None:
    with pytest.raises(ValueError):
        metric(ranking, judgments, k)


def test_qrels_loader_preserves_ids_and_grades(tmp_path: Path) -> None:
    path = tmp_path / "qrels.tsv"
    path.write_text("query-id\tcorpus-id\tscore\n01\t0002\t1\n01\t3\t0\n2\t4\t2\n", encoding="utf-8")
    assert load_qrels(path) == {"01": {"0002": 1, "3": 0}, "2": {"4": 2}}


@pytest.mark.parametrize("content", [
    "",
    "query-id\tcorpus-id\twrong\n",
    "query-id\tcorpus-id\tscore\n",
    "query-id\tcorpus-id\tscore\nq\td\t1\nq\td\t1\n",
    "query-id\tcorpus-id\tscore\nq\td\t1\nq\td\t2\n",
    "query-id\tcorpus-id\tscore\nq\td\t-1\n",
    "query-id\tcorpus-id\tscore\nq\td\t1.5\n",
    "query-id\tcorpus-id\tscore\nq\td\t0\n",
    "query-id\tcorpus-id\tscore\n\td\t1\n",
    "query-id\tcorpus-id\tscore\nq\t\t1\n",
    "query-id\tcorpus-id\tscore\nq\td\t1\textra\n",
])
def test_malformed_qrels_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "qrels.tsv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_qrels(path)


def test_query_loader_duplicate_ids_rejected(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    record = json.dumps({"_id": "q", "text": "cat"}) + "\n"
    path.write_text(record * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_queries(path)


@pytest.fixture
def local_dataset(tmp_path: Path) -> Path:
    records = [{"_id": "a", "text": "cat"}, {"_id": "b", "text": "dog"}]
    queries = [
        {"_id": "q1", "text": "cat"},
        {"_id": "q2", "text": "unknown"},
        {"_id": "unjudged_query", "text": "dog"},
    ]
    for filename, rows in [("corpus.jsonl.gz", records), ("queries.jsonl.gz", queries)]:
        content = "\n".join(json.dumps(record) for record in rows)
        (tmp_path / filename).write_bytes(gzip.compress(content.encode()))
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels/test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\ta\t1\nq2\tb\t1\n", encoding="utf-8"
    )
    return tmp_path


def test_evaluation_aggregates_all_judged_queries_and_preserves_defaults(local_dataset: Path) -> None:
    report = evaluate_scifact(local_dataset)
    assert report["counts"] == {
        "documents": 2, "chunks": 2, "available_queries": 3,
        "evaluated_queries": 2, "judgments": 2,
    }
    assert all(value == pytest.approx(0.5) for value in report["metrics"].values())
    assert report["per_query"]["q1"]["document_ids"] == ["a"]
    assert report["per_query"]["q2"]["document_ids"] == []
    assert report["settings"]["chunking"] == {"size": 180, "overlap": 30}
    assert report["settings"]["bm25"] == {"k1": 1.2, "b": 0.75}
    assert report["performance"]["average_query_seconds"] > 0
    assert report["performance"]["total_evaluation_seconds"] > 0
    assert len(report["input_sha256"]["qrels/test.tsv"]) == 64
    repeated = evaluate_scifact(local_dataset)
    assert repeated["metrics"] == report["metrics"]
    assert [r["document_ids"] for r in repeated["per_query"].values()] == [
        r["document_ids"] for r in report["per_query"].values()
    ]


@pytest.mark.parametrize("query_id,document_id", [("missing", "a"), ("q1", "missing")])
def test_inconsistent_qrels_references_rejected(local_dataset: Path, query_id: str, document_id: str) -> None:
    (local_dataset / "qrels/test.tsv").write_text(
        f"query-id\tcorpus-id\tscore\n{query_id}\t{document_id}\t1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown IDs"):
        evaluate_scifact(local_dataset)


def test_cli_evaluate_writes_actual_json_and_summary(
    local_dataset: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = local_dataset / "results/report.json"
    monkeypatch.setattr("sys.argv", [
        "enterprise-search", "evaluate", "--data-dir", str(local_dataset), "--output", str(output)
    ])
    main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["metrics"]["Recall@5"] == pytest.approx(0.5)
    assert "MRR@10: 0.500000" in capsys.readouterr().out
