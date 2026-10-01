import hashlib
import logging
import math
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from enterprise_ai_search.bm25 import BM25Index
from enterprise_ai_search.dataset import load_corpus, load_qrels, load_queries
from enterprise_ai_search.models import SearchResult
from enterprise_ai_search.text import ChunkingConfig, chunk_documents

logger = logging.getLogger(__name__)
DOCUMENT_CUTOFF = 10


def ranked_document_ids(results: list[SearchResult], top_k: int) -> list[str]:
    """Keep the first occurrence of each parent in an already-ranked chunk list."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    seen: set[str] = set()
    document_ids = []
    for result in results:
        if result.document_id not in seen:
            seen.add(result.document_id)
            document_ids.append(result.document_id)
            if len(document_ids) == top_k:
                break
    return document_ids


def _validate_metric_inputs(ranking: list[str], judgments: dict[str, int], k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(set(ranking)) != len(ranking):
        raise ValueError("Document ranking must contain unique IDs")
    if any(type(grade) is not int or grade < 0 for grade in judgments.values()):
        raise ValueError("Relevance grades must be nonnegative integers")
    if not any(grade > 0 for grade in judgments.values()):
        raise ValueError("At least one relevant document is required")


def recall_at_k(ranking: list[str], judgments: dict[str, int], k: int) -> float:
    _validate_metric_inputs(ranking, judgments, k)
    relevant = {document_id for document_id, grade in judgments.items() if grade > 0}
    return len(set(ranking[:k]) & relevant) / len(relevant)


def reciprocal_rank_at_k(ranking: list[str], judgments: dict[str, int], k: int) -> float:
    _validate_metric_inputs(ranking, judgments, k)
    for rank, document_id in enumerate(ranking[:k], start=1):
        if judgments.get(document_id, 0) > 0:
            return 1 / rank
    return 0.0


def ndcg_at_k(ranking: list[str], judgments: dict[str, int], k: int) -> float:
    _validate_metric_inputs(ranking, judgments, k)
    dcg = sum(
        judgments.get(document_id, 0) / math.log2(rank + 1)
        for rank, document_id in enumerate(ranking[:k], start=1)
    )
    ideal_dcg = sum(
        grade / math.log2(rank + 1)
        for rank, grade in enumerate(sorted(judgments.values(), reverse=True)[:k], start=1)
    )
    return dcg / ideal_dcg


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate_scifact(data_dir: Path) -> dict[str, Any]:
    """Evaluate the unchanged Milestone 1 defaults on every test-qrels query."""
    started = perf_counter()
    corpus_path = data_dir / "corpus.jsonl.gz"
    queries_path = data_dir / "queries.jsonl.gz"
    qrels_path = data_dir / "qrels/test.tsv"
    documents = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)
    document_ids = {document.document_id for document in documents}
    missing_queries = qrels.keys() - queries.keys()
    missing_documents = {doc_id for grades in qrels.values() for doc_id in grades} - document_ids
    if missing_queries or missing_documents:
        raise ValueError(
            f"Qrels reference unknown IDs: queries={sorted(missing_queries)}, "
            f"documents={sorted(missing_documents)}"
        )
    config = ChunkingConfig()
    index = BM25Index(chunk_documents(documents, config))
    logger.info("Evaluating %d test queries on %d chunks", len(qrels), len(index.chunks))
    per_query: dict[str, Any] = {}
    for number, query_id in enumerate(sorted(qrels), start=1):
        query_started = perf_counter()
        # All matching chunks guarantee a complete top-document ranking, even
        # when a single parent contributes many high-ranked chunks.
        results = index.search(queries[query_id], top_k=max(1, len(index.chunks)))
        ranking = ranked_document_ids(results, DOCUMENT_CUTOFF)
        query_seconds = perf_counter() - query_started
        judgments = qrels[query_id]
        per_query[query_id] = {
            "document_ids": ranking,
            "relevant_document_count": sum(grade > 0 for grade in judgments.values()),
            "metrics": {
                "Recall@5": recall_at_k(ranking, judgments, 5),
                "Recall@10": recall_at_k(ranking, judgments, 10),
                "MRR@10": reciprocal_rank_at_k(ranking, judgments, 10),
                "nDCG@10": ndcg_at_k(ranking, judgments, 10),
            },
            "query_seconds": query_seconds,
        }
        if number % 50 == 0 or number == len(qrels):
            logger.info("Evaluated %d/%d queries", number, len(qrels))
    metrics = {
        name: math.fsum(record["metrics"][name] for record in per_query.values()) / len(per_query)
        for name in ("Recall@5", "Recall@10", "MRR@10", "nDCG@10")
    }
    source_dir = Path(__file__).parent
    report = {
        "schema_version": 1,
        "dataset": "BEIR SciFact",
        "split": "test",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "counts": {
            "documents": len(documents),
            "chunks": len(index.chunks),
            "available_queries": len(queries),
            "evaluated_queries": len(per_query),
            "judgments": sum(len(grades) for grades in qrels.values()),
        },
        "settings": {
            "chunking": asdict(config),
            "bm25": {"k1": index.k1, "b": index.b},
            "document_cutoff": DOCUMENT_CUTOFF,
            "candidate_chunks": "all positive-score matches",
            "document_ranking": "first parent occurrence in BM25 chunk order",
            "aggregation": "macro mean over every test-qrels query",
            "relevant_grade_threshold": "> 0",
            "ndcg_gain": "raw relevance grade",
            "ndcg_discount": "1 / log2(rank + 1)",
            "unjudged_documents": "zero gain",
        },
        "input_sha256": {
            "corpus.jsonl.gz": _file_hash(corpus_path),
            "queries.jsonl.gz": _file_hash(queries_path),
            "qrels/test.tsv": _file_hash(qrels_path),
        },
        "source_sha256": {
            name: _file_hash(source_dir / name)
            for name in ("bm25.py", "text.py", "models.py", "dataset.py", "evaluation.py")
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "metrics": metrics,
        "performance": {
            "average_query_seconds": math.fsum(
                record["query_seconds"] for record in per_query.values()
            ) / len(per_query),
            "query_timing_scope": "BM25 search plus document deduplication; excludes index build and metrics",
            "total_timing_scope": "load, validate, index, retrieve, metrics, report assembly; excludes JSON write",
        },
        "per_query": per_query,
    }
    report["performance"]["total_evaluation_seconds"] = perf_counter() - started
    return report
