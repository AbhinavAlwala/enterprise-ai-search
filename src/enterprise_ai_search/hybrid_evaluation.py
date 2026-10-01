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
from enterprise_ai_search.dense import DenseIndex, EmbeddingConfig, load_encoder, prepare_embeddings
from enterprise_ai_search.evaluation import DOCUMENT_CUTOFF, _file_hash, ndcg_at_k, recall_at_k, reciprocal_rank_at_k
from enterprise_ai_search.hybrid import CANDIDATE_DEPTH, RRF_K, HybridIndex, HybridResult
from enterprise_ai_search.text import ChunkingConfig, chunk_documents

logger = logging.getLogger(__name__)


def _fusion_details(result: HybridResult) -> dict[str, Any]:
    return {
        "document_id": result.document_id, "rrf_score": result.score,
        "bm25_rank": result.bm25.rank if result.bm25 else None,
        "dense_rank": result.dense.rank if result.dense else None,
        "bm25_chunk_id": result.bm25.passage.chunk_id if result.bm25 else None,
        "dense_chunk_id": result.dense.passage.chunk_id if result.dense else None,
    }


def evaluate_hybrid_scifact(data_dir: Path, cache_dir: Path) -> dict[str, Any]:
    started = perf_counter()
    paths = {
        "corpus.jsonl.gz": data_dir / "corpus.jsonl.gz",
        "queries.jsonl.gz": data_dir / "queries.jsonl.gz",
        "qrels/test.tsv": data_dir / "qrels/test.tsv",
    }
    documents = load_corpus(paths["corpus.jsonl.gz"])
    queries = load_queries(paths["queries.jsonl.gz"])
    qrels = load_qrels(paths["qrels/test.tsv"])
    document_ids = {document.document_id for document in documents}
    missing_queries = qrels.keys() - queries.keys()
    missing_documents = {doc_id for grades in qrels.values() for doc_id in grades} - document_ids
    if missing_queries or missing_documents:
        raise ValueError(f"Qrels reference unknown IDs: queries={sorted(missing_queries)}, documents={sorted(missing_documents)}")
    config = ChunkingConfig()
    chunks = chunk_documents(documents, config)
    encoder, model_seconds = load_encoder(cache_dir / "models")
    preparation_started = perf_counter()
    vectors, preparation = prepare_embeddings(chunks, encoder, cache_dir / "chunks.npz", config)
    index = HybridIndex(BM25Index(chunks), DenseIndex(chunks, vectors, encoder))
    preparation["model_load_seconds"] = model_seconds
    preparation["index_preparation_seconds"] = perf_counter() - preparation_started
    logger.info("Hybrid preparation: model %.3f s; index %.3f s; cache hit=%s",
                model_seconds, preparation["index_preparation_seconds"], preparation["cache_hit"])
    per_query: dict[str, Any] = {}
    for number, query_id in enumerate(sorted(qrels), start=1):
        query_started = perf_counter()
        results = index.search(queries[query_id], DOCUMENT_CUTOFF)
        ranking = [result.document_id for result in results]
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
            "fusion_details": [_fusion_details(result) for result in results],
        }
        if number % 50 == 0 or number == len(qrels):
            logger.info("Hybrid evaluated %d/%d queries", number, len(qrels))
    source_dir = Path(__file__).parent
    report = {
        "schema_version": 1, "dataset": "BEIR SciFact", "split": "test", "method": "hybrid",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "counts": {
            "documents": len(documents), "chunks": len(chunks), "available_queries": len(queries),
            "evaluated_queries": len(per_query), "judgments": sum(len(g) for g in qrels.values()),
        },
        "settings": {
            "chunking": asdict(config), "bm25": {"k1": index.bm25.k1, "b": index.bm25.b},
            "embedding": asdict(EmbeddingConfig()), "document_cutoff": DOCUMENT_CUTOFF,
            "candidate_depth_documents_per_retriever": CANDIDATE_DEPTH, "rrf_k": RRF_K,
            "retriever_weights": {"bm25": 1, "dense": 1}, "fusion_level": "document",
            "candidate_chunks": "full component chunk rankings before document truncation",
            "document_ranking": "RRF of first-parent-occurrence document ranks",
            "tie_breaking": "ascending document ID", "aggregation": "macro mean over every test-qrels query",
            "relevant_grade_threshold": "> 0", "ndcg_gain": "raw relevance grade",
            "ndcg_discount": "1 / log2(rank + 1)", "unjudged_documents": "zero gain",
        },
        "input_sha256": {name: _file_hash(path) for name, path in paths.items()},
        "source_sha256": {
            name: _file_hash(source_dir / name)
            for name in ("hybrid.py", "hybrid_evaluation.py", "bm25.py", "dense.py", "evaluation.py", "text.py", "models.py", "dataset.py")
        },
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(), "device": "cpu",
            "runtime_versions": preparation["artifact"]["identity"]["runtime_versions"],
            "torch_threads": EmbeddingConfig().cpu_threads,
        },
        "metrics": {
            name: math.fsum(record["metrics"][name] for record in per_query.values()) / len(per_query)
            for name in ("Recall@5", "Recall@10", "MRR@10", "nDCG@10")
        },
        "preparation": preparation,
        "performance": {
            "average_query_seconds": math.fsum(record["query_seconds"] for record in per_query.values()) / len(per_query),
            "query_timing_scope": "sequential BM25 retrieval, query embedding, exact dense retrieval, component document extraction, RRF, final document ranking",
            "total_timing_scope": "load, validate, model load, cache/index preparation, retrieve/fuse, metrics, report; excludes JSON write",
        },
        "per_query": per_query,
    }
    report["performance"]["total_evaluation_seconds"] = perf_counter() - started
    return report
