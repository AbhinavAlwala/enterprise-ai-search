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
from enterprise_ai_search.hybrid import CANDIDATE_DEPTH as HYBRID_DEPTH, RRF_K, HybridIndex
from enterprise_ai_search.hybrid_evaluation import _fusion_details
from enterprise_ai_search.reranker import CANDIDATE_DEPTH, RerankedResult, RerankerConfig, load_reranker, rerank_candidates
from enterprise_ai_search.text import ChunkingConfig
from enterprise_ai_search.text import chunk_documents

logger = logging.getLogger(__name__)


def _reranking_details(result: RerankedResult) -> dict[str, Any]:
    return {
        **_fusion_details(result.hybrid), "hybrid_rank": result.hybrid.rank,
        "reranker_score": result.score, "selected_chunk_id": result.passage.chunk_id,
        "passage_scores": [
            {"chunk_id": entry.passage.chunk_id, "score": entry.score, "retrievers": list(entry.retrievers)}
            for entry in result.passage_scores
        ],
    }


def evaluate_reranked_scifact(data_dir: Path, cache_dir: Path, model_cache: Path) -> dict[str, Any]:
    started = perf_counter()
    paths = {
        "corpus.jsonl.gz": data_dir / "corpus.jsonl.gz", "queries.jsonl.gz": data_dir / "queries.jsonl.gz",
        "qrels/test.tsv": data_dir / "qrels/test.tsv",
    }
    documents, queries = load_corpus(paths["corpus.jsonl.gz"]), load_queries(paths["queries.jsonl.gz"])
    qrels = load_qrels(paths["qrels/test.tsv"])
    document_ids = {document.document_id for document in documents}
    missing_queries = qrels.keys() - queries.keys()
    missing_documents = {doc for grades in qrels.values() for doc in grades} - document_ids
    if missing_queries or missing_documents:
        raise ValueError(f"Qrels reference unknown IDs: queries={sorted(missing_queries)}, documents={sorted(missing_documents)}")
    chunking = ChunkingConfig()
    chunks = chunk_documents(documents, chunking)
    encoder, encoder_seconds = load_encoder(cache_dir / "models")
    preparation_started = perf_counter()
    vectors, preparation = prepare_embeddings(chunks, encoder, cache_dir / "chunks.npz", chunking)
    index = HybridIndex(BM25Index(chunks), DenseIndex(chunks, vectors, encoder))
    preparation["index_preparation_seconds"] = perf_counter() - preparation_started
    model, model_seconds = load_reranker(model_cache)
    preparation.update(encoder_model_load_seconds=encoder_seconds, reranker_model_load_seconds=model_seconds)
    logger.info("Reranker loaded %.3f s; encoder %.3f s; index %.3f s; cache hit=%s",
                model_seconds, encoder_seconds, preparation["index_preparation_seconds"], preparation["cache_hit"])
    per_query = {}
    for number, query_id in enumerate(sorted(qrels), 1):
        query_started = perf_counter()
        candidates = index.search(queries[query_id], CANDIDATE_DEPTH)
        candidate_seconds = perf_counter() - query_started
        reranking_started = perf_counter()
        results, inference = rerank_candidates(queries[query_id], candidates, model, CANDIDATE_DEPTH)
        reranking_seconds = perf_counter() - reranking_started
        ranking = [result.document_id for result in results[:DOCUMENT_CUTOFF]]
        query_seconds = perf_counter() - query_started
        candidate_ids = [candidate.document_id for candidate in candidates]
        grades = qrels[query_id]
        per_query[query_id] = {
            "document_ids": ranking, "candidate_document_ids": candidate_ids,
            "reranked_document_ids": [result.document_id for result in results],
            "relevant_document_count": sum(grade > 0 for grade in grades.values()),
            "candidate_recall_at_50": recall_at_k(candidate_ids, grades, CANDIDATE_DEPTH),
            "metrics": {
                "Recall@5": recall_at_k(ranking, grades, 5), "Recall@10": recall_at_k(ranking, grades, 10),
                "MRR@10": reciprocal_rank_at_k(ranking, grades, 10), "nDCG@10": ndcg_at_k(ranking, grades, 10),
            },
            "query_seconds": query_seconds, "candidate_generation_seconds": candidate_seconds,
            "reranker_inference_seconds": inference["inference_seconds"], "reranking_seconds": reranking_seconds,
            "pair_count": inference["pair_count"],
            "reranking_details": [_reranking_details(result) for result in results[:DOCUMENT_CUTOFF]],
        }
        if number % 10 == 0 or number == len(qrels):
            logger.info("Reranked evaluated %d/%d queries; elapsed %.1f s", number, len(qrels), perf_counter() - started)
    source_dir = Path(__file__).parent
    report = {
        "schema_version": 1, "dataset": "BEIR SciFact", "split": "test", "method": "reranked",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "counts": {
            "documents": len(documents), "chunks": len(chunks), "available_queries": len(queries),
            "evaluated_queries": len(per_query), "judgments": sum(len(grades) for grades in qrels.values()),
        },
        "settings": {
            "chunking": asdict(chunking), "bm25": {"k1": index.bm25.k1, "b": index.bm25.b},
            "embedding": asdict(EmbeddingConfig()), "document_cutoff": DOCUMENT_CUTOFF,
            "candidate_depth_documents_per_retriever": HYBRID_DEPTH, "rrf_k": RRF_K,
            "retriever_weights": {"bm25": 1, "dense": 1}, "reranker": asdict(RerankerConfig()),
            "reranker_candidate_depth": CANDIDATE_DEPTH, "passage_deduplication": "chunk ID within each document",
            "passage_selection": "maximum raw cross-encoder score over existing representative passages",
            "passage_tie_breaking": "ascending chunk ID", "tie_breaking": "ascending document ID",
            "score_activation": "identity (raw logits); no combination with retrieval scores",
            "pair_truncation": "longest_first to 512 tokens including query, passage, and special tokens",
            "aggregation": "macro mean over every test-qrels query", "relevant_grade_threshold": "> 0",
            "ndcg_gain": "raw relevance grade", "ndcg_discount": "1 / log2(rank + 1)", "unjudged_documents": "zero gain",
        },
        "input_sha256": {name: _file_hash(path) for name, path in paths.items()},
        "source_sha256": {
            name: _file_hash(source_dir / name)
            for name in ("reranker.py", "reranked_evaluation.py", "hybrid.py", "hybrid_evaluation.py", "bm25.py", "dense.py", "evaluation.py", "text.py", "models.py", "dataset.py")
        },
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(), "device": "cpu",
            "runtime_versions": preparation["artifact"]["identity"]["runtime_versions"],
            "torch_threads": RerankerConfig().cpu_threads,
        },
        "metrics": {
            name: math.fsum(record["metrics"][name] for record in per_query.values()) / len(per_query)
            for name in ("Recall@5", "Recall@10", "MRR@10", "nDCG@10")
        },
        "candidate_metrics": {
            "Recall@50": math.fsum(record["candidate_recall_at_50"] for record in per_query.values()) / len(per_query),
        },
        "preparation": preparation,
        "performance": {
            "average_query_seconds": math.fsum(record["query_seconds"] for record in per_query.values()) / len(per_query),
            "average_candidate_generation_seconds": math.fsum(record["candidate_generation_seconds"] for record in per_query.values()) / len(per_query),
            "average_reranker_inference_seconds": math.fsum(record["reranker_inference_seconds"] for record in per_query.values()) / len(per_query),
            "average_reranking_seconds": math.fsum(record["reranking_seconds"] for record in per_query.values()) / len(per_query),
            "query_timing_scope": "unchanged hybrid candidates, pair preparation, batched cross-encoder prediction, max passage aggregation, final document ranking",
            "inference_timing_scope": "CrossEncoder.predict plus score array conversion; includes pair tokenization/inference",
            "total_timing_scope": "load/validate, model loads, cache/index preparation, online retrieval/reranking, metrics/report; excludes JSON write",
        },
        "per_query": per_query,
    }
    report["performance"]["total_evaluation_seconds"] = perf_counter() - started
    return report
