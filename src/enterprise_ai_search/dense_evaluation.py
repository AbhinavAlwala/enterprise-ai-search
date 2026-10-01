import logging
import math
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from enterprise_ai_search.dataset import load_corpus, load_qrels, load_queries
from enterprise_ai_search.dense import DenseIndex, EmbeddingConfig, load_encoder, prepare_embeddings
from enterprise_ai_search.evaluation import (
    DOCUMENT_CUTOFF, _file_hash, ndcg_at_k, ranked_document_ids,
    recall_at_k, reciprocal_rank_at_k,
)
from enterprise_ai_search.text import ChunkingConfig, chunk_documents

logger = logging.getLogger(__name__)


def evaluate_dense_scifact(
    data_dir: Path, cache_dir: Path, rebuild: bool = False,
) -> dict[str, Any]:
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
    chunks = chunk_documents(documents, config)
    encoder, model_load_seconds = load_encoder(cache_dir / "models")
    preparation_started = perf_counter()
    vectors, preparation = prepare_embeddings(chunks, encoder, cache_dir / "chunks.npz", config, rebuild)
    index = DenseIndex(chunks, vectors, encoder)
    preparation["index_preparation_seconds"] = perf_counter() - preparation_started
    preparation["model_load_seconds"] = model_load_seconds
    logger.info("Dense preparation: model %.3f s, index %.3f s, cache hit=%s",
                model_load_seconds, preparation["index_preparation_seconds"], preparation["cache_hit"])
    per_query: dict[str, Any] = {}
    for number, query_id in enumerate(sorted(qrels), start=1):
        query_started = perf_counter()
        results = index.search(queries[query_id], top_k=max(1, len(chunks)))
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
            logger.info("Dense evaluated %d/%d queries", number, len(qrels))
    # Aggregate definitions and query weighting match the frozen M2 report.
    metrics = {
        name: math.fsum(record["metrics"][name] for record in per_query.values()) / len(per_query)
        for name in ("Recall@5", "Recall@10", "MRR@10", "nDCG@10")
    }
    source_dir = Path(__file__).parent
    report = {
        "schema_version": 1, "dataset": "BEIR SciFact", "split": "test", "method": "dense",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "counts": {
            "documents": len(documents), "chunks": len(chunks), "available_queries": len(queries),
            "evaluated_queries": len(per_query), "judgments": sum(len(g) for g in qrels.values()),
        },
        "settings": {
            "chunking": asdict(config), "embedding": asdict(EmbeddingConfig()),
            "document_cutoff": DOCUMENT_CUTOFF, "candidate_chunks": "all chunks",
            "document_ranking": "first parent occurrence in cosine chunk order",
            "aggregation": "macro mean over every test-qrels query", "relevant_grade_threshold": "> 0",
            "ndcg_gain": "raw relevance grade", "ndcg_discount": "1 / log2(rank + 1)",
            "unjudged_documents": "zero gain", "similarity": "cosine", "query_prompt": "",
            "passage_prompt": "", "query_normalization": "existing NFC/whitespace normalization",
        },
        "input_sha256": {
            "corpus.jsonl.gz": _file_hash(corpus_path), "queries.jsonl.gz": _file_hash(queries_path),
            "qrels/test.tsv": _file_hash(qrels_path),
        },
        "source_sha256": {
            name: _file_hash(source_dir / name)
            for name in ("dense.py", "dense_evaluation.py", "evaluation.py", "text.py", "models.py", "dataset.py")
        },
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "runtime_versions": preparation["artifact"]["identity"]["runtime_versions"],
            "device": "cpu", "torch_threads": EmbeddingConfig().cpu_threads,
        },
        "metrics": metrics,
        "preparation": preparation,
        "performance": {
            "average_query_seconds": math.fsum(r["query_seconds"] for r in per_query.values()) / len(per_query),
            "query_timing_scope": "query embedding, exact cosine search, full chunk results, document deduplication",
            "total_timing_scope": "load, validate, model load, cache/index preparation, retrieve, metrics, report; excludes JSON write",
        },
        "per_query": per_query,
    }
    report["performance"]["total_evaluation_seconds"] = perf_counter() - started
    return report
