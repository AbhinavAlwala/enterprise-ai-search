import copy
import json
from pathlib import Path

import numpy as np
import pytest

from enterprise_ai_search.dense import (
    DenseIndex, EmbeddingConfig, cache_identity, normalize_vectors,
    prepare_embeddings, read_embedding_cache,
)
from enterprise_ai_search.evaluation import ranked_document_ids
from enterprise_ai_search.models import Chunk
from enterprise_ai_search.text import ChunkingConfig


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.tokenizer = lambda texts, **kwargs: {"input_ids": [[1, 2] for _ in texts]}

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append(texts)
        vectors = np.zeros((len(texts), EmbeddingConfig().dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i, 0 if "cat" in text else 1] = 1
        return vectors


class TinyEncoder:
    def __init__(self, vector: np.ndarray | None = None) -> None:
        self.vector = np.array([[3, 0]], dtype=np.float32) if vector is None else vector
        self.calls = 0

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls += 1
        return self.vector


def test_vector_normalization_preserves_direction() -> None:
    result = normalize_vectors(np.array([[3, 4], [-2, 0]], dtype=np.float32))
    np.testing.assert_allclose(result, [[0.6, 0.8], [-1, 0]])
    assert result.dtype == np.float32
    assert normalize_vectors(np.empty((0, 2), dtype=np.float32)).shape == (0, 2)


@pytest.mark.parametrize("vectors", [
    np.array([1, 2]), np.array([[0, 0]]), np.empty((1, 0)),
    np.array([[np.nan, 1]]), np.array([[np.inf, 1]]),
])
def test_invalid_vectors_rejected(vectors: np.ndarray) -> None:
    with pytest.raises(ValueError):
        normalize_vectors(vectors)


def test_exact_cosine_ranking_top_k_scores_and_document_deduplication() -> None:
    chunks = [Chunk("a", "parent", "a"), Chunk("b", "parent", "b"), Chunk("c", "other", "c")]
    index = DenseIndex(chunks, np.array([[4, 0], [1, 1], [-1, 0]], dtype=np.float32), TinyEncoder())
    results = index.search("query", 99)
    assert [result.chunk_id for result in results] == ["a", "b", "c"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.score for result in results] == pytest.approx([1, 1 / np.sqrt(2), -1])
    assert results[0].text == "a"
    assert ranked_document_ids(results, 10) == ["parent", "other"]
    assert len(index.search("query", 1)) == 1


def test_ties_use_chunk_ids_independently_of_input_order() -> None:
    chunks = [Chunk("z", "z", "z"), Chunk("a", "a", "a")]
    for ordering in (chunks, list(reversed(chunks))):
        index = DenseIndex(ordering, np.array([[1, 0], [1, 0]], dtype=np.float32), TinyEncoder())
        assert [result.chunk_id for result in index.search("query")] == ["a", "z"]


def test_empty_search_does_not_encode() -> None:
    encoder = TinyEncoder()
    index = DenseIndex([Chunk("a", "a", "cat")], np.array([[1, 0]], dtype=np.float32), encoder)
    assert index.search(" \n ") == []
    empty = DenseIndex([], np.empty((0, 2), dtype=np.float32), encoder)
    assert empty.search("cat") == []
    assert encoder.calls == 0
    with pytest.raises(ValueError):
        index.search("cat", 0)


def test_invalid_embedding_shapes_and_duplicate_chunks() -> None:
    chunks = [Chunk("a", "a", "cat")]
    with pytest.raises(ValueError, match="row"):
        DenseIndex(chunks, np.array([[1, 0], [1, 0]], dtype=np.float32), TinyEncoder())
    with pytest.raises(ValueError, match="unique"):
        DenseIndex(chunks * 2, np.array([[1, 0], [1, 0]], dtype=np.float32), TinyEncoder())
    for vector in (np.array([1, 0]), np.array([[1, 0, 0]]), np.array([[1, 0], [1, 0]])):
        index = DenseIndex(chunks, np.array([[1, 0]], dtype=np.float32), TinyEncoder(vector))
        with pytest.raises(ValueError):
            index.search("query")


def test_cache_round_trip_hit_and_forced_rebuild(tmp_path: Path) -> None:
    chunks = [Chunk("a", "a", "cat"), Chunk("b", "b", "dog")]
    encoder, path = FakeEncoder(), tmp_path / "chunks.npz"
    vectors, first = prepare_embeddings(chunks, encoder, path)
    cached, second = prepare_embeddings(chunks, encoder, path)
    np.testing.assert_array_equal(vectors, cached)
    assert not first["cache_hit"] and second["cache_hit"]
    assert second["encoding_seconds_this_run"] == 0
    assert len(encoder.calls) == 1
    assert second["artifact"]["encoding_seconds"] == first["artifact"]["encoding_seconds"]
    prepare_embeddings(chunks, encoder, path, rebuild=True)
    assert len(encoder.calls) == 2
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("change", ["model", "revision", "dimension", "chunking", "contents", "order", "runtime", "prompt"])
def test_cache_identity_rejects_stale_inputs(tmp_path: Path, change: str) -> None:
    chunks = [Chunk("a", "a", "cat"), Chunk("b", "b", "dog")]
    path = tmp_path / "chunks.npz"
    prepare_embeddings(chunks, FakeEncoder(), path)
    identity = copy.deepcopy(cache_identity(chunks, ChunkingConfig()))
    if change in ("model", "revision", "dimension"):
        identity["encoding"][change] = "different"
    elif change == "chunking":
        identity["chunking"]["overlap"] = 0
    elif change in ("contents", "order"):
        changed = list(reversed(chunks)) if change == "order" else [Chunk("a", "a", "changed"), chunks[1]]
        identity = cache_identity(changed, ChunkingConfig())
    elif change == "runtime":
        identity["runtime_versions"]["torch"] = "different"
    else:
        identity["prompt"] = "different"
    with pytest.raises(ValueError, match="changed"):
        read_embedding_cache(path, identity)


def test_stale_cache_is_regenerated_not_used(tmp_path: Path) -> None:
    path = tmp_path / "chunks.npz"
    encoder = FakeEncoder()
    prepare_embeddings([Chunk("a", "a", "cat")], encoder, path)
    vectors, info = prepare_embeddings([Chunk("a", "a", "dog")], encoder, path)
    assert not info["cache_hit"] and len(encoder.calls) == 2
    assert vectors[0, 1] == 1


@pytest.mark.parametrize("corruption", ["shape", "dtype", "zero", "checksum", "metadata", "bytes"])
def test_corrupt_cache_rejected_and_rebuilt(tmp_path: Path, corruption: str) -> None:
    chunks = [Chunk("a", "a", "cat")]
    path = tmp_path / "chunks.npz"
    encoder = FakeEncoder()
    vectors, info = prepare_embeddings(chunks, encoder, path)
    metadata = info["artifact"]
    if corruption == "shape":
        vectors = vectors[:, :2]
    elif corruption == "dtype":
        vectors = vectors.astype(np.float64)
    elif corruption == "zero":
        vectors = vectors * 0
    elif corruption == "checksum":
        vectors = np.roll(vectors, 1, axis=1)
    elif corruption == "metadata":
        metadata = []
    if corruption == "bytes":
        path.write_bytes(b"invalid npz")
    else:
        np.savez(path, embeddings=vectors, metadata=json.dumps(metadata))
    _, repaired = prepare_embeddings(chunks, encoder, path)
    assert not repaired["cache_hit"] and len(encoder.calls) == 2


def test_chunk_truncation_is_audited_without_changing_text(tmp_path: Path) -> None:
    encoder = FakeEncoder()
    encoder.tokenizer = lambda texts, **kwargs: {"input_ids": [[1] * 300 for _ in texts]}
    _, info = prepare_embeddings([Chunk("a", "a", "cat")], encoder, tmp_path / "chunks.npz")
    assert info["artifact"]["truncated_chunks"] == 1
    assert encoder.calls == [["cat"]]
