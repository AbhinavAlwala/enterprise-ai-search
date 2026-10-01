import json
import math
from pathlib import Path
from typing import Any

from enterprise_ai_search.dataset import load_qrels, load_queries
from enterprise_ai_search.evaluation import _file_hash

METRICS = ("Recall@5", "Recall@10", "MRR@10", "nDCG@10")


def compare_reports(bm25_path: Path, dense_path: Path, data_dir: Path) -> dict[str, Any]:
    bm25 = json.loads(bm25_path.read_text(encoding="utf-8"))
    dense = json.loads(dense_path.read_text(encoding="utf-8"))
    if bm25["dataset"] != dense["dataset"] or bm25["split"] != dense["split"]:
        raise ValueError("Reports must describe the same dataset/split")
    if bm25["input_sha256"] != dense["input_sha256"] or bm25["counts"] != dense["counts"]:
        raise ValueError("Reports must use identical inputs and corpus/query counts")
    if bm25["settings"]["chunking"] != dense["settings"]["chunking"]:
        raise ValueError("Reports must use identical chunking")
    for name in ("document_cutoff", "aggregation", "relevant_grade_threshold", "ndcg_gain", "ndcg_discount", "unjudged_documents"):
        if bm25["settings"][name] != dense["settings"][name]:
            raise ValueError(f"Evaluation policy mismatch: {name}")
    if bm25["per_query"].keys() != dense["per_query"].keys():
        raise ValueError("Reports must evaluate identical query IDs")
    for report in (bm25, dense):
        if len(report["per_query"]) != report["counts"]["evaluated_queries"]:
            raise ValueError("Query count does not match per-query results")
        for name in METRICS:
            value = report["metrics"][name]
            mean = math.fsum(record["metrics"][name] for record in report["per_query"].values()) / len(report["per_query"])
            if not 0 <= value <= 1 or not math.isclose(value, mean, abs_tol=1e-12):
                raise ValueError(f"Invalid aggregate metric: {name}")
    queries_path, qrels_path = data_dir / "queries.jsonl.gz", data_dir / "qrels/test.tsv"
    if _file_hash(queries_path) != bm25["input_sha256"]["queries.jsonl.gz"] or _file_hash(qrels_path) != bm25["input_sha256"]["qrels/test.tsv"]:
        raise ValueError("Local query/qrels files differ from evaluated inputs")
    queries, qrels = load_queries(queries_path), load_qrels(qrels_path)
    if set(qrels) != set(bm25["per_query"]):
        raise ValueError("Local qrels query IDs differ from reports")
    examples = []
    # Choose the first numeric query ID for each hit/miss direction at rank 10.
    for winner, loser, winning_report, losing_report in [
        ("dense", "bm25", dense, bm25), ("bm25", "dense", bm25, dense),
    ]:
        for query_id in sorted(qrels, key=lambda value: (int(value) if value.isdecimal() else math.inf, value)):
            winning = winning_report["per_query"][query_id]
            losing = losing_report["per_query"][query_id]
            if winning["metrics"]["MRR@10"] > 0 and losing["metrics"]["MRR@10"] == 0:
                relevant = sorted(doc_id for doc_id, grade in qrels[query_id].items() if grade > 0)
                examples.append({
                    "query_id": query_id, "query": queries[query_id], "winner": winner, "loser": loser,
                    "relevant_document_ids": relevant,
                    "bm25_document_ids": bm25["per_query"][query_id]["document_ids"],
                    "dense_document_ids": dense["per_query"][query_id]["document_ids"],
                    "first_relevant_rank": next(i for i, doc_id in enumerate(winning["document_ids"], 1) if doc_id in relevant),
                })
                break
    return {
        "dataset": bm25["dataset"], "split": bm25["split"], "evaluated_queries": len(qrels),
        "artifact_sha256": {"bm25": _file_hash(bm25_path), "dense": _file_hash(dense_path)},
        "metrics": {
            name: {"bm25": bm25["metrics"][name], "dense": dense["metrics"][name],
                   "dense_minus_bm25": dense["metrics"][name] - bm25["metrics"][name]}
            for name in METRICS
        },
        "query_latency_ms": {
            "bm25": bm25["performance"]["average_query_seconds"] * 1000,
            "dense": dense["performance"]["average_query_seconds"] * 1000,
        },
        "timing_caveat": "Separate sequential local runs; BM25 runtime predates ML dependencies; dense includes query encoding; neither includes index setup.",
        "example_selection": "First numeric query ID in each direction with a top-10 relevant hit versus no hit",
        "examples": examples,
    }
