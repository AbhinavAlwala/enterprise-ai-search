import copy
import gzip
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from enterprise_ai_search import dense, reranked_evaluation, reranker
from enterprise_ai_search.cli import main
from enterprise_ai_search.comparison import _reranked_examples, compare_reports
from enterprise_ai_search.hybrid import DocumentCandidate, HybridResult
from enterprise_ai_search.models import SearchResult
from enterprise_ai_search.reranker import (
    RerankerConfig, load_reranker, representative_passages, rerank_candidates,
)


def candidate(doc: str, rank: int, texts: tuple[str, ...]) -> HybridResult:
    components = [
        DocumentCandidate(rank, doc, SearchResult(rank, 1000, f"{doc}::{i}", doc, text))
        for i, text in enumerate(texts)
    ]
    return HybridResult(rank, 0.03, doc, components[0], components[-1])


class FakeCrossEncoder:
    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = scores
        self.batches: list[list[tuple[str, str]]] = []
        self.calls = 0

    def predict(self, pairs: list[tuple[str, str]], batch_size: int, **kwargs: object) -> np.ndarray:
        self.calls += 1
        values = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            self.batches.append(batch)
            for query, text in batch:
                values.append(self.scores[text] if self.scores is not None else float(query == text or (query == "kitty" and text == "cat")))
        return np.array(values)


def test_reranking_uses_max_passage_score_and_preserves_provenance() -> None:
    candidates = [candidate("a", 1, ("bad", "best")), candidate("b", 2, ("good",))]
    model = FakeCrossEncoder({"bad": -4, "best": 3, "good": 2})
    results, info = rerank_candidates("query", candidates, model, 10)
    assert [result.document_id for result in results] == ["a", "b"]
    assert results[0].score == 3
    assert results[0].passage.chunk_id == "a::1"
    assert results[0].hybrid is candidates[0]
    assert [entry.score for entry in results[0].passage_scores] == [-4, 3]
    assert results[0].passage_scores[1].retrievers == ("dense",)
    assert results[1].passage_scores[0].retrievers == ("bm25", "dense")
    assert info["pair_count"] == 3 and model.calls == 1


def test_duplicate_representative_chunk_is_scored_once_and_dense_only_is_allowed() -> None:
    both = candidate("a", 1, ("same",))
    dense_only = HybridResult(2, 0.02, "b", None, candidate("b", 2, ("other",)).dense)
    model = FakeCrossEncoder({"same": 0.2, "other": 0.8})
    results, info = rerank_candidates("query", [both, dense_only], model, 2)
    assert info["pair_count"] == 2
    assert results[0].document_id == "b" and results[0].passage_scores[0].retrievers == ("dense",)
    assert len(results[1].passage_scores) == 1


def test_document_and_passage_score_ties_use_ids_without_blending_hybrid_scores() -> None:
    candidates = [candidate("z", 1, ("one", "two")), candidate("a", 2, ("three",))]
    model = FakeCrossEncoder({"one": -2, "two": -2, "three": -2})
    results, _ = rerank_candidates("query", candidates, model, 10)
    assert [result.document_id for result in results] == ["a", "z"]
    assert results[1].passage.chunk_id == "z::0"
    assert all(result.score == -2 for result in results)
    assert [result.rank for result in results] == [1, 2]


def test_candidate_cutoff_top_k_and_batching() -> None:
    candidates = [candidate(str(i), i + 1, (str(i),)) for i in range(51)]
    model = FakeCrossEncoder({str(i): float(i) for i in range(51)})
    results, info = rerank_candidates("query", candidates, model, 3, RerankerConfig(batch_size=16))
    assert [result.document_id for result in results] == ["49", "48", "47"]
    assert info["pair_count"] == 50
    assert [len(batch) for batch in model.batches] == [16, 16, 16, 2]
    assert all(pair[1] != "50" for batch in model.batches for pair in batch)


def test_empty_queries_candidates_and_invalid_top_k() -> None:
    model = FakeCrossEncoder()
    assert rerank_candidates("query", [], model)[0] == []
    assert rerank_candidates(" \n", [candidate("a", 1, ("a",))], model)[0] == []
    assert model.calls == 0
    with pytest.raises(ValueError):
        rerank_candidates("query", [], model, 0)


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], [], [[1]], [1, 2]])
def test_invalid_model_output_rejected(scores: list) -> None:
    model = SimpleNamespace(predict=lambda *args, **kwargs: scores)
    with pytest.raises(ValueError, match="finite scalar"):
        rerank_candidates("query", [candidate("a", 1, ("a",))], model)


def test_invalid_candidates_and_passage_identity_rejected() -> None:
    a = candidate("a", 1, ("a",))
    with pytest.raises(ValueError, match="unique"):
        rerank_candidates("query", [a, a], FakeCrossEncoder())
    with pytest.raises(ValueError, match="consecutive"):
        rerank_candidates("query", [candidate("a", 2, ("a",))], FakeCrossEncoder())
    with pytest.raises(ValueError, match="representative"):
        representative_passages(HybridResult(1, 0.03, "a", None, None))
    with pytest.raises(ValueError, match="belong"):
        representative_passages(HybridResult(1, 0.03, "b", a.bm25, None))
    conflicting = DocumentCandidate(1, "a", SearchResult(1, 1, "a::0", "a", "different"))
    with pytest.raises(ValueError, match="different passage"):
        representative_passages(HybridResult(1, 0.03, "a", a.bm25, conflicting))


@pytest.mark.parametrize("setting", ["batch_size", "cpu_threads", "max_sequence_length"])
def test_invalid_config_rejected(setting: str) -> None:
    with pytest.raises(ValueError):
        RerankerConfig(**{setting: 0})


def test_model_loading_is_revision_pinned_cpu_identity_without_real_transformer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = {}
    identity = object()
    def construct(name: str, **kwargs: object) -> SimpleNamespace:
        recorded.update(model=name, **kwargs)
        return SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(num_labels=1)))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda threads: recorded.update(threads=threads), nn=SimpleNamespace(Identity=lambda: identity),
    ))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=construct))
    _, seconds = load_reranker(tmp_path)
    assert recorded["revision"] == RerankerConfig().revision
    assert recorded["max_length"] == 512 and recorded["threads"] == 4
    assert recorded["device"] == "cpu" and recorded["activation_fn"] is identity
    assert recorded["trust_remote_code"] is False and seconds >= 0


class FakeEncoder:
    tokenizer = staticmethod(lambda texts, **kwargs: {"input_ids": [[1] for _ in texts]})

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for row, text in enumerate(texts):
            vectors[row, 0 if text in ("cat", "kitty") else 1] = 1
        return vectors


@pytest.fixture
def reranked_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    monkeypatch.setattr(reranked_evaluation, "load_encoder", lambda path: (FakeEncoder(), 0.0))
    monkeypatch.setattr(reranked_evaluation, "load_reranker", lambda path: (FakeCrossEncoder(), 0.0))
    for name, records in [
        ("corpus.jsonl.gz", [{"_id": "a", "text": "cat"}, {"_id": "b", "text": "dog"}]),
        ("queries.jsonl.gz", [{"_id": "1", "text": "cat"}, {"_id": "2", "text": "kitty"}]),
    ]:
        (tmp_path / name).write_bytes(gzip.compress("\n".join(json.dumps(record) for record in records).encode()))
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\n1\ta\t1\n2\ta\t1\n", encoding="utf-8")
    return tmp_path, reranked_evaluation.evaluate_reranked_scifact(tmp_path, tmp_path / "cache", tmp_path / "models")


def test_evaluation_metrics_candidate_recall_and_provenance(reranked_report: tuple[Path, dict]) -> None:
    data_dir, report = reranked_report
    assert all(value == 1 for value in report["metrics"].values())
    assert report["candidate_metrics"]["Recall@50"] == 1
    assert report["settings"]["reranker"] == asdict(RerankerConfig())
    assert report["settings"]["reranker_candidate_depth"] == 50
    assert report["settings"]["candidate_depth_documents_per_retriever"] == 100
    for record in report["per_query"].values():
        assert set(record["document_ids"]) <= set(record["candidate_document_ids"])
        assert record["query_seconds"] >= record["candidate_generation_seconds"] + record["reranker_inference_seconds"]
        detail = record["reranking_details"][0]
        assert detail["hybrid_rank"] == record["candidate_document_ids"].index(detail["document_id"]) + 1
        assert detail["reranker_score"] == max(entry["score"] for entry in detail["passage_scores"])
    repeated = reranked_evaluation.evaluate_reranked_scifact(data_dir, data_dir / "cache", data_dir / "models")
    assert repeated["preparation"]["cache_hit"] and repeated["preparation"]["encoding_seconds_this_run"] == 0


def test_cli_evaluation_writes_report(reranked_report: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir, _ = reranked_report
    output = data_dir / "reranked.json"
    monkeypatch.setattr(sys, "argv", [
        "enterprise-search", "evaluate-reranked", "--data-dir", str(data_dir), "--cache-dir", str(data_dir / "cache"),
        "--model-cache-dir", str(data_dir / "models"), "--output", str(output),
    ])
    main()
    assert json.loads(output.read_text())["method"] == "reranked"


def test_metric_integration_when_reranker_demotes_relevant_document(
    reranked_report: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, _ = reranked_report
    monkeypatch.setattr(reranked_evaluation, "load_reranker", lambda path: (FakeCrossEncoder({"cat": -5, "dog": 5}), 0.0))
    report = reranked_evaluation.evaluate_reranked_scifact(data_dir, data_dir / "cache", data_dir / "models")
    assert report["candidate_metrics"]["Recall@50"] == 1
    assert report["metrics"]["Recall@10"] == 1
    assert report["metrics"]["MRR@10"] == 0.5
    assert report["metrics"]["nDCG@10"] == pytest.approx(1 / np.log2(3))
    assert all(row["document_ids"][0] == "b" for row in report["per_query"].values())


def test_cli_search_returns_winning_passage_and_hybrid_provenance(
    reranked_report: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir, _ = reranked_report
    monkeypatch.setattr(dense, "load_encoder", lambda path: (FakeEncoder(), 0.0))
    monkeypatch.setattr(reranker, "load_reranker", lambda path: (FakeCrossEncoder(), 0.0))
    monkeypatch.setattr(sys, "argv", [
        "enterprise-search", "search-reranked", "kitty", "--top-k", "1", "--data-dir", str(data_dir),
        "--cache-dir", str(data_dir / "cache"),
    ])
    main()
    result = json.loads(capsys.readouterr().out)[0]
    assert result["document_id"] == "a" and result["passage"]["text"] == "cat"
    assert result["hybrid"]["dense"]["document_id"] == "a"
    assert result["passage_scores"][0]["retrievers"] == ["dense"]


def component_reports(report: dict) -> tuple[dict, dict, dict]:
    hybrid = copy.deepcopy(report)
    for row in hybrid["per_query"].values():
        row["document_ids"] = row["candidate_document_ids"][:10]
    return copy.deepcopy(hybrid), copy.deepcopy(hybrid), hybrid


def write_reports(data_dir: Path, reports: tuple[dict, ...]) -> list[Path]:
    paths = [data_dir / f"{name}.json" for name in ("bm25", "dense", "hybrid", "reranked")]
    for path, report in zip(paths, reports):
        path.write_text(json.dumps(report), encoding="utf-8")
    return paths


def test_four_way_comparison_uses_compatible_artifacts(reranked_report: tuple[Path, dict]) -> None:
    data_dir, report = reranked_report
    paths = write_reports(data_dir, (*component_reports(report), report))
    summary = compare_reports(paths[0], paths[1], data_dir, paths[2], paths[3])
    assert set(summary["artifact_sha256"]) == {"bm25", "dense", "hybrid", "reranked"}
    assert summary["metrics"]["MRR@10"]["reranked_minus_hybrid"] == 0
    assert summary["candidate_recall_at_50"] == 1 and summary["examples"] == []


@pytest.mark.parametrize("change", ["model", "depth", "hybrid_order", "outside_pool", "candidate_recall", "aggregate", "queries"])
def test_four_way_comparison_rejects_inconsistency(reranked_report: tuple[Path, dict], change: str) -> None:
    data_dir, report = reranked_report
    bad = copy.deepcopy(report)
    if change == "model":
        bad["settings"]["reranker"]["revision"] = "changed"
    elif change == "depth":
        bad["settings"]["reranker_candidate_depth"] = 100
    elif change == "hybrid_order":
        bad["per_query"]["1"]["candidate_document_ids"].reverse()
    elif change == "outside_pool":
        bad["per_query"]["1"]["reranked_document_ids"][0] = "unknown"
    elif change == "candidate_recall":
        bad["candidate_metrics"]["Recall@50"] = 0
    elif change == "aggregate":
        bad["metrics"]["MRR@10"] = 0
    else:
        bad["per_query"].pop("1")
    paths = write_reports(data_dir, (*component_reports(report), bad))
    with pytest.raises(ValueError):
        compare_reports(paths[0], paths[1], data_dir, paths[2], paths[3])


def test_example_selection_uses_same_relevant_document_and_fixed_five_rank_rule() -> None:
    candidate_ids = ["a", "b", "c", "d", "e", "gold"]
    rows = {
        "10": {"candidate_document_ids": candidate_ids, "reranked_document_ids": ["gold", "a", "b", "c", "d", "e"]},
        "2": {"candidate_document_ids": candidate_ids, "reranked_document_ids": ["gold", "a", "b", "c", "d", "e"]},
        "3": {"candidate_document_ids": ["gold", "a", "b", "c", "d", "e"], "reranked_document_ids": candidate_ids},
        "4": {"candidate_document_ids": ["a"], "reranked_document_ids": ["a"]},
    }
    for record in rows.values():
        record.update(document_ids=record["reranked_document_ids"][:10], candidate_recall_at_50=float("gold" in record["candidate_document_ids"]))
    qrels = {query_id: {"gold": 1} for query_id in rows}
    examples = _reranked_examples({"per_query": rows}, {query_id: query_id for query_id in rows}, qrels)
    assert [(row["category"], row["query_id"]) for row in examples] == [
        ("relevant_upward", "2"), ("relevant_downward", "3"), ("relevant_absent_from_candidates", "4"),
    ]
    assert examples[0]["hybrid_candidate_rank"] == 6 and examples[0]["reranked_rank"] == 1
    assert examples[2]["reranked_rank"] is None


@pytest.mark.parametrize("output", [
    "results/scifact_bm25_test.json", "results/scifact_dense_test.json", "results/scifact_hybrid_test.json",
    "results/scifact_comparison.json", "results/scifact_hybrid_comparison.json",
])
def test_existing_reports_are_protected(monkeypatch: pytest.MonkeyPatch, output: str) -> None:
    monkeypatch.setattr(sys, "argv", ["enterprise-search", "evaluate-reranked", "--output", output])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
