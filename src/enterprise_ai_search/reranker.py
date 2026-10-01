from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from enterprise_ai_search.hybrid import HybridResult
from enterprise_ai_search.models import SearchResult
from enterprise_ai_search.text import normalize_text

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

CANDIDATE_DEPTH = 50


@dataclass(frozen=True)
class RerankerConfig:
    model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    max_sequence_length: int = 512
    batch_size: int = 16
    cpu_threads: int = 4

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.cpu_threads <= 0 or self.max_sequence_length <= 0:
            raise ValueError("Reranker batch size, thread count, and token limit must be positive")


@dataclass(frozen=True)
class PassageScore:
    passage: SearchResult
    score: float
    retrievers: tuple[str, ...]


@dataclass(frozen=True)
class RerankedResult:
    rank: int
    score: float
    document_id: str
    passage: SearchResult
    hybrid: HybridResult
    passage_scores: tuple[PassageScore, ...]


def load_reranker(
    model_cache: Path, config: RerankerConfig = RerankerConfig(),
) -> tuple["CrossEncoder", float]:
    started = perf_counter()
    import torch
    from sentence_transformers import CrossEncoder

    torch.set_num_threads(config.cpu_threads)
    model = CrossEncoder(
        config.model, revision=config.revision, device="cpu", cache_folder=str(model_cache),
        max_length=config.max_sequence_length, activation_fn=torch.nn.Identity(),
        trust_remote_code=False, model_kwargs={"use_safetensors": True},
    )
    if model.model.config.num_labels != 1:
        raise ValueError("Reranker must return one score per query/passage pair")
    return model, perf_counter() - started


def representative_passages(candidate: HybridResult) -> list[tuple[SearchResult, tuple[str, ...]]]:
    passages: dict[str, SearchResult] = {}
    sources: dict[str, list[str]] = {}
    for name, component in (("bm25", candidate.bm25), ("dense", candidate.dense)):
        if component is None:
            continue
        passage = component.passage
        if passage.document_id != candidate.document_id:
            raise ValueError("Representative passage must belong to its candidate document")
        previous = passages.get(passage.chunk_id)
        if previous is not None and previous.text != passage.text:
            raise ValueError("One chunk ID cannot represent different passage text")
        passages.setdefault(passage.chunk_id, passage)
        sources.setdefault(passage.chunk_id, []).append(name)
    if not passages:
        raise ValueError("Every reranker candidate needs a representative passage")
    return [(passages[chunk_id], tuple(sources[chunk_id])) for chunk_id in sorted(passages)]


def rerank_candidates(
    query: str, candidates: list[HybridResult], model: "CrossEncoder", top_k: int = 5,
    config: RerankerConfig = RerankerConfig(),
) -> tuple[list[RerankedResult], dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query = normalize_text(query)
    selected = candidates[:CANDIDATE_DEPTH]
    if not query or not selected:
        return [], {"inference_seconds": 0.0, "pair_count": 0}
    if len({candidate.document_id for candidate in selected}) != len(selected):
        raise ValueError("Reranking requires unique document candidates")
    if any(candidate.rank != rank for rank, candidate in enumerate(selected, 1)):
        raise ValueError("Candidates must preserve consecutive hybrid ranks")
    passages = [representative_passages(candidate) for candidate in selected]
    pairs = [(query, passage.text) for group in passages for passage, _ in group]
    inference_started = perf_counter()
    scores = np.asarray(model.predict(
        pairs, batch_size=config.batch_size, show_progress_bar=False, convert_to_numpy=True,
    ), dtype=np.float64)
    inference_seconds = perf_counter() - inference_started
    if scores.shape != (len(pairs),) or not np.isfinite(scores).all():
        raise ValueError("CrossEncoder must return one finite scalar score per pair")
    scored_documents = []
    offset = 0
    for candidate, group in zip(selected, passages):
        passage_scores = tuple(
            PassageScore(passage, float(scores[offset + number]), sources)
            for number, (passage, sources) in enumerate(group)
        )
        offset += len(group)
        best = min(passage_scores, key=lambda entry: (-entry.score, entry.passage.chunk_id))
        scored_documents.append((best.score, candidate, best.passage, passage_scores))
    scored_documents.sort(key=lambda entry: (-entry[0], entry[1].document_id))
    results = [
        RerankedResult(rank, score, candidate.document_id, passage, candidate, passage_scores)
        for rank, (score, candidate, passage, passage_scores) in enumerate(scored_documents[:top_k], 1)
    ]
    return results, {"inference_seconds": inference_seconds, "pair_count": len(pairs)}
