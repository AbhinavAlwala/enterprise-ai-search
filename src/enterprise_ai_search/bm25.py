import math
from collections import Counter

from enterprise_ai_search.models import Chunk, SearchResult
from enterprise_ai_search.text import tokenize


class BM25Index:
    """In-memory chunk index using positive IDF and saturated term frequency."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75) -> None:
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("Require finite k1 > 0 and 0 <= b <= 1")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Chunk IDs must be unique")
        if any(not chunk.chunk_id or not chunk.document_id for chunk in chunks):
            raise ValueError("Chunk and document IDs must be nonempty")
        self.chunks = tuple(chunks)
        self.k1 = k1
        self.b = b
        self.frequencies = [Counter(tokenize(chunk.text)) for chunk in chunks]
        if any(not counts for counts in self.frequencies):
            raise ValueError("Each chunk must contain at least one searchable token")
        self.lengths = [sum(counts.values()) for counts in self.frequencies]
        self.average_length = sum(self.lengths) / len(chunks) if chunks else 0.0
        document_frequency: Counter[str] = Counter()
        for counts in self.frequencies:
            document_frequency.update(counts.keys())
        self.idf = {
            term: math.log1p((len(chunks) - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        # Repeated query words do not boost their weight in this baseline.
        terms = sorted(set(tokenize(query)) & self.idf.keys())
        if not terms:
            return []
        scored: list[tuple[float, Chunk]] = []
        for chunk, counts, length in zip(self.chunks, self.frequencies, self.lengths):
            length_factor = self.k1 * (1 - self.b + self.b * length / self.average_length)
            score = sum(
                self.idf[term] * counts[term] * (self.k1 + 1) / (counts[term] + length_factor)
                for term in terms
                if counts[term]
            )
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            SearchResult(rank, score, chunk.chunk_id, chunk.document_id, chunk.text)
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]
