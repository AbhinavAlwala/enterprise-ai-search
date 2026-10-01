import math

import pytest

from enterprise_ai_search.bm25 import BM25Index
from enterprise_ai_search.models import Chunk, Document
from enterprise_ai_search.text import ChunkingConfig, chunk_documents, normalize_text, tokenize


def test_normalization_preserves_readable_text_and_tokenization_is_shared() -> None:
    assert normalize_text("  Cafe\u0301\nSTUDY\t ") == "Café STUDY"
    assert tokenize("Café STUDY, IL-6!") == ["café", "study", "il", "6"]


def test_chunk_windows_overlap_ids_and_no_redundant_tail() -> None:
    documents = [Document("paper", "a b c d e f g")]
    chunks = chunk_documents(documents, ChunkingConfig(4, 2))
    assert [chunk.text for chunk in chunks] == ["a b c d", "c d e f", "e f g"]
    assert [chunk.chunk_id for chunk in chunks] == [f"paper::chunk::{i}" for i in range(3)]
    assert all(chunk.document_id == "paper" for chunk in chunks)
    assert chunks == chunk_documents(documents, ChunkingConfig(4, 2))
    assert len(chunk_documents([Document("x", "a b c d")], ChunkingConfig(4, 2))) == 1


def test_chunking_title_empty_text_and_no_overlap() -> None:
    assert chunk_documents([Document("x", "   !!!")]) == []
    assert chunk_documents([]) == []
    chunks = chunk_documents([Document("x", "c d e", "a b")], ChunkingConfig(2, 0))
    assert [chunk.text for chunk in chunks] == ["a b", "c d", "e"]


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (3, -1), (3, 3), (3, 4)])
def test_invalid_chunking_config(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(size, overlap)


def test_duplicate_document_ids_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        chunk_documents([Document("x", "a"), Document("x", "b")])


def test_score_matches_hand_calculated_bm25() -> None:
    index = BM25Index([Chunk("a", "d1", "cat cat dog"), Chunk("b", "d2", "dog")])
    result = index.search("cat")[0]
    # N=2, df(cat)=1, tf=2, length=3, average length=2.
    expected = math.log(2) * (2 * 2.2) / (2 + 1.2 * (0.25 + 0.75 * 3 / 2))
    assert result.score == pytest.approx(expected)
    assert (result.rank, result.chunk_id, result.document_id, result.text) == (
        1, "a", "d1", "cat cat dog"
    )


def test_length_normalization_and_tf_saturation() -> None:
    chunks = [Chunk("short", "s", "cat"), Chunk("long", "l", "cat filler filler filler")]
    assert BM25Index(chunks).search("cat")[0].chunk_id == "short"
    index = BM25Index([Chunk("a", "a", "cat"), Chunk("b", "b", "cat cat cat")], b=0)
    results = index.search("cat")
    assert results[0].chunk_id == "b"
    assert 1 < results[0].score / results[1].score < 3
    assert all(result.score > 0 for result in results)


def test_ties_top_k_repeated_query_and_input_order() -> None:
    chunks = [Chunk("b", "d", "cat"), Chunk("a", "d", "cat")]
    index = BM25Index(chunks)
    assert [result.chunk_id for result in index.search("cat", 99)] == ["a", "b"]
    assert index.search("CAT! cat", 1) == index.search("cat", 1)
    assert index.search("cat") == BM25Index(list(reversed(chunks))).search("cat")


@pytest.mark.parametrize("query", ["", "  ", "!!!", "unseen"])
def test_no_match_returns_no_arbitrary_results(query: str) -> None:
    assert BM25Index([Chunk("a", "a", "cat")]).search(query) == []
    assert BM25Index([]).search(query) == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k(top_k: int) -> None:
    with pytest.raises(ValueError):
        BM25Index([]).search("cat", top_k)


@pytest.mark.parametrize("k1,b", [(0, 0.75), (-1, 0.75), (1.2, -1), (1.2, 2), (math.nan, 0.5), (1, math.inf)])
def test_invalid_bm25_parameters(k1: float, b: float) -> None:
    with pytest.raises(ValueError):
        BM25Index([], k1, b)


def test_invalid_chunks_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        BM25Index([Chunk("a", "x", "cat"), Chunk("a", "y", "dog")])
    with pytest.raises(ValueError, match="token"):
        BM25Index([Chunk("a", "x", "!!!")])


def test_local_document_to_ranked_chunk_pipeline() -> None:
    documents = [Document("heart", "cardiac muscle pumps blood"), Document("bone", "skeletal growth")]
    chunks = chunk_documents(documents, ChunkingConfig(3, 1))
    results = BM25Index(chunks).search("blood", 5)
    assert len(results) == 1
    assert results[0].document_id == "heart"
    assert results[0].text == "pumps blood"
