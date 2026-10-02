import hashlib
import json
import logging
import platform
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from enterprise_ai_search.claim_verification import VERIFICATION_PROMPT, verify_claim
from enterprise_ai_search.dataset import StanceClaim, load_qrels, load_stance_claims
from enterprise_ai_search.generation import GenerationConfig, Generator
from enterprise_ai_search.rag import EVIDENCE_COUNT

logger = logging.getLogger(__name__)
GOLD_LABELS = ("SUPPORT", "CONTRADICT")
FULL_RUN_LIMIT_SECONDS = 90 * 60


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_predictions(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot score an empty evaluation")
    for record in records:
        if record["gold_stance"] not in GOLD_LABELS or record["predicted_stance"] not in (*GOLD_LABELS, "ABSTAIN", None):
            raise ValueError("Unknown evaluation stance")
    count = len(records)
    correct = sum(r["gold_stance"] == r["predicted_stance"] for r in records)
    covered = sum(r["predicted_stance"] in GOLD_LABELS for r in records)
    class_metrics = {}
    confusion = {gold: {predicted: 0 for predicted in (*GOLD_LABELS, "ABSTAIN")} for gold in GOLD_LABELS}
    failures_by_gold = dict.fromkeys(GOLD_LABELS, 0)
    for record in records:
        predicted = record["predicted_stance"]
        if predicted is None:
            failures_by_gold[record["gold_stance"]] += 1
        else:
            confusion[record["gold_stance"]][predicted] += 1
    for label in GOLD_LABELS:
        tp = sum(r["gold_stance"] == label and r["predicted_stance"] == label for r in records)
        fp = sum(r["gold_stance"] != label and r["predicted_stance"] == label for r in records)
        fn = sum(r["gold_stance"] == label and r["predicted_stance"] != label for r in records)
        class_metrics[label] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "gold_count": tp + fn,
        }
    citation_passes = sum(bool(r["citation_validation"] and r["citation_validation"]["passed"]) for r in records)
    parsed = sum(r["parse_status"] == "ok" for r in records)
    diagnostics = {}
    for name, present in (("gold_document_present", True), ("gold_document_absent", False)):
        subset = [r for r in records if r["gold_document_present"] == present]
        diagnostics[name] = {
            "count": len(subset), "overall_accuracy": _ratio(sum(r["gold_stance"] == r["predicted_stance"] for r in subset), len(subset)),
        }
    return {
        "evaluated_claims": count, "correct_predictions": correct,
        "overall_accuracy": correct / count, "non_abstained_accuracy": _ratio(correct, covered),
        "macro_f1": sum(m["f1"] for m in class_metrics.values()) / 2,
        "per_class": class_metrics,
        "abstention_rate": sum(r["predicted_stance"] == "ABSTAIN" for r in records) / count,
        "coverage": covered / count,
        "parsing_failure_rate": sum(r["parse_status"] == "failed" for r in records) / count,
        "generation_failure_rate": sum(r["generation_status"] == "failed" for r in records) / count,
        "citation_validation_pass_rate": citation_passes / count,
        "citation_pass_rate_among_parsed": _ratio(citation_passes, parsed),
        "confusion_matrix": confusion, "failures_by_gold": failures_by_gold,
        "gold_document_present_rate": diagnostics["gold_document_present"]["count"] / count,
        "diagnostics": diagnostics,
    }


def load_test_claims(data_dir: Path) -> list[StanceClaim]:
    qrels = load_qrels(data_dir / "qrels/test.tsv")
    claims = load_stance_claims(data_dir / "queries.jsonl.gz", set(qrels))
    if len(claims) != 188 or Counter(c.gold_stance for c in claims) != {"SUPPORT": 124, "CONTRADICT": 64}:
        raise ValueError("Expected the frozen SciFact test subset: 188 claims (124 SUPPORT, 64 CONTRADICT)")
    return claims


def prepare_retrieval(data_dir: Path, cache_dir: Path, model_cache: Path) -> tuple[Any, Any, dict[str, Any]]:
    from enterprise_ai_search.bm25 import BM25Index
    from enterprise_ai_search.dataset import load_corpus
    from enterprise_ai_search.dense import DenseIndex, load_encoder, prepare_embeddings
    from enterprise_ai_search.hybrid import HybridIndex
    from enterprise_ai_search.reranker import load_reranker
    from enterprise_ai_search.text import chunk_documents

    started = perf_counter()
    documents = load_corpus(data_dir / "corpus.jsonl.gz")
    chunks = chunk_documents(documents)
    encoder, encoder_seconds = load_encoder(cache_dir / "models")
    vectors, cache = prepare_embeddings(chunks, encoder, cache_dir / "chunks.npz")
    index = HybridIndex(BM25Index(chunks), DenseIndex(chunks, vectors, encoder))
    reranker, reranker_seconds = load_reranker(model_cache)
    return index, reranker, {
        "preparation_seconds": perf_counter() - started, "encoder_load_seconds": encoder_seconds,
        "reranker_load_seconds": reranker_seconds, "embedding_cache_hit": cache["cache_hit"],
        "documents": len(documents), "chunks": len(chunks),
    }


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluation_identity(data_dir: Path, cache_dir: Path, claims: list[StanceClaim], config: GenerationConfig) -> dict[str, Any]:
    source_dir = Path(__file__).parent
    return {
        "schema_version": 1, "dataset": "BEIR SciFact", "split": "test",
        "claim_ids": [c.query_id for c in claims], "prompt": VERIFICATION_PROMPT,
        "generation": {
            "endpoint": config.endpoint, "model": config.model, "temperature": config.temperature,
            "max_output_tokens": config.max_output_tokens, "timeout_seconds": config.timeout_seconds,
        },
        "evidence_count": EVIDENCE_COUNT,
        "input_sha256": {name: _file_hash(data_dir / name) for name in ("corpus.jsonl.gz", "queries.jsonl.gz", "qrels/test.tsv")},
        "embedding_cache_sha256": _file_hash(cache_dir / "chunks.npz"),
        "source_sha256": {p.name: _file_hash(p) for p in sorted(source_dir.glob("*.py"))},
        "python": platform.python_version(),
    }


def _records_hash(records: dict[str, Any]) -> str:
    payload = json.dumps(records, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_checkpoint(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report["records_sha256"] = _records_hash(report["per_query"])
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path, identity: dict[str, Any], claims: list[StanceClaim]) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["identity"] != identity:
            raise ValueError("Checkpoint settings, inputs, or code differ")
        records = report["per_query"]
        if report["records_sha256"] != _records_hash(records):
            raise ValueError("Checkpoint record checksum mismatch")
        by_id = {c.query_id: c for c in claims}
        for query_id, record in records.items():
            claim = by_id[query_id]
            if (record["query_id"], record["claim"], record["gold_stance"], record["gold_document_ids"]) != (claim.query_id, claim.text, claim.gold_stance, list(claim.gold_document_ids)):
                raise ValueError("Checkpoint claim mismatch")
        if records:
            summarize_predictions(list(records.values()))
        return report
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Malformed checkpoint") from error


def estimate_full_runtime(report: dict[str, Any], total_claims: int) -> float:
    records = list(report["per_query"].values())
    if not records:
        raise ValueError("Need completed smoke predictions to estimate runtime")
    mean_online = sum(r["timings"]["online_seconds"] for r in records) / len(records)
    return report["preparation_sessions"][0]["preparation_seconds"] + total_claims * mean_online


def run_evaluation(
    claims: list[StanceClaim], index: Any, reranker: Any, generator: Generator,
    output: Path, identity: dict[str, Any], preparation: dict[str, Any], *,
    smoke: bool = False, resume_from: Path | None = None,
) -> dict[str, Any]:
    if output.exists() and (resume_from is None or output.resolve() != resume_from.resolve()):
        raise ValueError("Output exists; use a new path or explicitly resume it")
    if resume_from is None:
        report: dict[str, Any] = {
            "identity": identity, "created_at_utc": datetime.now(UTC).isoformat(),
            "per_query": {}, "preparation_sessions": [], "evaluation_session_seconds": [],
            "benchmark_scope": "explicit SciFact stance classification, not free-form answer correctness",
        }
    else:
        report = load_checkpoint(resume_from, identity, claims)
    if not smoke:
        if len(report["per_query"]) < 5 or not report.get("smoke_completed"):
            raise ValueError("Run and resume a five-claim smoke evaluation before a full run")
        estimate = report["estimated_full_runtime_seconds"]
        if estimate > FULL_RUN_LIMIT_SECONDS:
            raise ValueError("Smoke estimate exceeds 90 minutes; full evaluation not launched")
        if report["metrics"]["generation_failure_rate"]:
            raise ValueError("Smoke had generation failures; resolve the endpoint before a full run")
    selected = claims[:5] if smoke else claims
    started = perf_counter()
    report["preparation_sessions"].append(preparation)
    report["evaluation_session_seconds"].append(0.0)
    for claim in selected:
        if claim.query_id in report["per_query"]:
            continue
        record = verify_claim(claim, index, reranker, generator)
        report["per_query"][claim.query_id] = record
        report["metrics"] = summarize_predictions(list(report["per_query"].values()))
        report["evaluation_session_seconds"][-1] = perf_counter() - started
        report["status"] = "in_progress"
        save_checkpoint(output, report)
        logger.info("Claims evaluated %d/%d; query %s: %s (%s), %.1f s",
                    len(report["per_query"]), len(claims), claim.query_id,
                    record["predicted_stance"], record["parse_status"], record["timings"]["online_seconds"])
        if record["generation_status"] == "failed":
            raise ValueError("Generation failed; saved completed records and stopped evaluation")
    report["metrics"] = summarize_predictions(list(report["per_query"].values()))
    report["evaluation_session_seconds"][-1] = perf_counter() - started
    if smoke:
        report["smoke_completed"] = True
        report["estimated_full_runtime_seconds"] = estimate_full_runtime(report, len(claims))
    report["status"] = "complete" if len(report["per_query"]) == len(claims) else "smoke_complete"
    report["performance"] = {
        "preparation_seconds": sum(s["preparation_seconds"] for s in report["preparation_sessions"]),
        "online_query_seconds": sum(r["timings"]["online_seconds"] for r in report["per_query"].values()),
        "total_evaluation_seconds": sum(s["preparation_seconds"] for s in report["preparation_sessions"]) + sum(report["evaluation_session_seconds"]),
    }
    save_checkpoint(output, report)
    return report
