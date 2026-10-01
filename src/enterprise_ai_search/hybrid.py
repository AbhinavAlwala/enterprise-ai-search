from dataclasses import dataclass

from enterprise_ai_search.bm25 import BM25Index
from enterprise_ai_search.dense import DenseIndex
from enterprise_ai_search.evaluation import ranked_document_ids
from enterprise_ai_search.models import SearchResult

CANDIDATE_DEPTH = 100
RRF_K = 60


@dataclass(frozen=True)
class DocumentCandidate:
    rank: int
    document_id: str
    passage: SearchResult


@dataclass(frozen=True)
class HybridResult:
    rank: int
    score: float
    document_id: str
    bm25: DocumentCandidate | None
    dense: DocumentCandidate | None


def document_candidates(results: list[SearchResult], depth: int) -> list[DocumentCandidate]:
    document_ids = ranked_document_ids(results, depth)
    selected = set(document_ids)
    passages: dict[str, SearchResult] = {}
    for result in results:
        if result.document_id in selected and result.document_id not in passages:
            passages[result.document_id] = result
            if len(passages) == len(document_ids):
                break
    return [
        DocumentCandidate(rank, document_id, passages[document_id])
        for rank, document_id in enumerate(document_ids, start=1)
    ]


def reciprocal_rank_fusion(
    bm25: list[DocumentCandidate], dense: list[DocumentCandidate], top_k: int,
) -> list[HybridResult]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    scores: dict[str, float] = {}
    for ranking in (bm25, dense):
        if len({candidate.document_id for candidate in ranking}) != len(ranking):
            raise ValueError("Fusion requires unique documents per retriever")
        for rank, candidate in enumerate(ranking, start=1):
            if candidate.rank != rank or candidate.document_id != candidate.passage.document_id:
                raise ValueError("Document candidates must have consecutive ranks and matching parent IDs")
            scores[candidate.document_id] = scores.get(candidate.document_id, 0.0) + 1 / (RRF_K + rank)
    bm25_by_id = {candidate.document_id: candidate for candidate in bm25}
    dense_by_id = {candidate.document_id: candidate for candidate in dense}
    document_ids = sorted(scores, key=lambda document_id: (-scores[document_id], document_id))[:top_k]
    return [
        HybridResult(rank, scores[document_id], document_id,
                     bm25_by_id.get(document_id), dense_by_id.get(document_id))
        for rank, document_id in enumerate(document_ids, start=1)
    ]


class HybridIndex:
    """Equal-weight document RRF over the existing lexical and dense indexes."""

    def __init__(self, bm25: BM25Index, dense: DenseIndex, candidate_depth: int = CANDIDATE_DEPTH) -> None:
        if candidate_depth <= 0:
            raise ValueError("candidate_depth must be positive")
        if bm25.chunks != dense.chunks:
            raise ValueError("Hybrid indexes must contain the same ordered chunks")
        self.bm25 = bm25
        self.dense = dense
        self.candidate_depth = candidate_depth

    def search(self, query: str, top_k: int = 5) -> list[HybridResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        # A document depth cannot be guaranteed by requesting that many chunks.
        chunk_limit = max(1, len(self.bm25.chunks))
        bm25 = document_candidates(self.bm25.search(query, chunk_limit), self.candidate_depth)
        dense = document_candidates(self.dense.search(query, chunk_limit), self.candidate_depth)
        return reciprocal_rank_fusion(bm25, dense, top_k)
