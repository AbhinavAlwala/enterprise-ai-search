import logging
from _thread import LockType
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING

from enterprise_ai_search.generation import GenerationConfig, Generator, HttpGenerator
from enterprise_ai_search.hybrid import HybridIndex
from enterprise_ai_search.rag import GeneratedAnswer, ask
from enterprise_ai_search.reranker import CANDIDATE_DEPTH, RerankedResult, rerank_candidates

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@dataclass
class SearchService:
    index: HybridIndex | None = field(repr=False)
    reranker: "CrossEncoder | None" = field(repr=False)
    generator: Generator | None = field(repr=False)
    preparation_seconds: float = 0.0
    lock: LockType = field(default_factory=Lock, repr=False)

    @property
    def retrieval_initialized(self) -> bool:
        return self.index is not None and self.reranker is not None

    @property
    def generation_configured(self) -> bool:
        return self.generator is not None

    def search(self, query: str, top_k: int) -> tuple[list[RerankedResult], float]:
        if not self.retrieval_initialized:
            raise RuntimeError("Retrieval unavailable")
        # Shared CPU models should not run overlapping inference requests.
        with self.lock:
            started = perf_counter()
            candidates = self.index.search(query, CANDIDATE_DEPTH)
            results, _ = rerank_candidates(query, candidates, self.reranker, top_k)
            return results, perf_counter() - started

    def answer(self, question: str) -> GeneratedAnswer:
        if not self.retrieval_initialized or self.generator is None:
            raise RuntimeError("Answer generation unavailable")
        with self.lock:
            return ask(question, self.index, self.reranker, self.generator)

    def close(self) -> None:
        self.index = None
        self.reranker = None
        self.generator = None


def load_service() -> SearchService:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    started = perf_counter()
    generator = None
    try:
        generator = HttpGenerator(GenerationConfig.from_env())
    except ValueError:
        logger.warning("Generation configuration unavailable; search remains enabled")

    index, reranker = None, None
    try:
        from enterprise_ai_search.bm25 import BM25Index
        from enterprise_ai_search.dataset import load_corpus
        from enterprise_ai_search.dense import DenseIndex, load_encoder, prepare_embeddings
        from enterprise_ai_search.reranker import load_reranker
        from enterprise_ai_search.text import chunk_documents

        chunks = chunk_documents(load_corpus(Path("data/scifact/corpus.jsonl.gz")))
        encoder, _ = load_encoder(Path("data/dense/models"))
        vectors, _ = prepare_embeddings(chunks, encoder, Path("data/dense/chunks.npz"))
        index = HybridIndex(BM25Index(chunks), DenseIndex(chunks, vectors, encoder))
        reranker, _ = load_reranker(Path("data/reranker/models"))
    except (OSError, ValueError, RuntimeError) as error:
        index, reranker = None, None
        logger.error("Retrieval initialization failed (%s)", type(error).__name__)
    service = SearchService(index, reranker, generator, perf_counter() - started)
    logger.info("Service initialized in %.3f s; retrieval=%s; generation configured=%s",
                service.preparation_seconds, service.retrieval_initialized, service.generation_configured)
    return service
