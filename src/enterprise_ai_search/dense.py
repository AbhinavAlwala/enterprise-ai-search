import hashlib
import json
import logging
import tempfile
from zipfile import BadZipFile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from enterprise_ai_search.models import Chunk, SearchResult
from enterprise_ai_search.text import ChunkingConfig, normalize_text

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    dimension: int = 384
    max_sequence_length: int = 256
    batch_size: int = 32
    cpu_threads: int = 4


def load_encoder(model_cache: Path) -> tuple["SentenceTransformer", float]:
    started = perf_counter()
    # Lazy imports keep model downloads/loading out of BM25 commands and tests.
    import torch
    from sentence_transformers import SentenceTransformer

    config = EmbeddingConfig()
    torch.set_num_threads(config.cpu_threads)
    encoder = SentenceTransformer(
        config.model, revision=config.revision, device="cpu", cache_folder=str(model_cache),
        trust_remote_code=False, model_kwargs={"use_safetensors": True},
    )
    encoder.max_seq_length = config.max_sequence_length
    if encoder.get_sentence_embedding_dimension() != config.dimension:
        raise ValueError("Unexpected embedding model dimension")
    return encoder, perf_counter() - started


def normalize_vectors(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] == 0 or not np.isfinite(array).all():
        raise ValueError("Vectors must be a finite 2D matrix with nonzero dimension")
    norms = np.linalg.norm(array.astype(np.float64), axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero vectors have undefined cosine similarity")
    return np.asarray(array / norms, dtype=np.float32)


def cache_identity(chunks: list[Chunk], chunking: ChunkingConfig) -> dict[str, Any]:
    contents = json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False, separators=(",", ":"))
    return {
        "schema_version": 1,
        "encoding": asdict(EmbeddingConfig()),
        "chunking": asdict(chunking),
        "chunk_count": len(chunks),
        "ordered_chunks_sha256": hashlib.sha256(contents.encode("utf-8")).hexdigest(),
        "dtype": "float32",
        "normalized": True,
        "prompt": "",
        "runtime_versions": {
            package: version(package)
            for package in ("sentence-transformers", "transformers", "torch", "numpy")
        },
    }


def read_embedding_cache(path: Path, identity: dict[str, Any]) -> tuple[NDArray[np.float32], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as artifact:
        metadata = json.loads(str(artifact["metadata"].item()))
        vectors = artifact["embeddings"]
    if not isinstance(metadata, dict) or metadata.get("identity") != identity:
        raise ValueError("Embedding cache inputs/settings changed")
    expected_shape = (identity["chunk_count"], identity["encoding"]["dimension"])
    if vectors.shape != expected_shape or vectors.dtype != np.float32:
        raise ValueError("Embedding cache shape/dtype mismatch")
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError("Embedding cache vectors must be finite and normalized")
    if hashlib.sha256(vectors.tobytes()).hexdigest() != metadata["embedding_sha256"]:
        raise ValueError("Embedding cache checksum mismatch")
    return vectors, metadata


def write_embedding_cache(path: Path, vectors: NDArray[np.float32], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            np.savez(output, embeddings=vectors, metadata=json.dumps(metadata, allow_nan=False))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_embeddings(
    chunks: list[Chunk], encoder: "SentenceTransformer", cache_path: Path,
    chunking: ChunkingConfig = ChunkingConfig(), rebuild: bool = False,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    started = perf_counter()
    identity = cache_identity(chunks, chunking)
    if cache_path.is_file() and not rebuild:
        try:
            vectors, metadata = read_embedding_cache(cache_path, identity)
            logger.info("Verified embedding cache: %s", cache_path)
            return vectors, {
                "cache_hit": True, "preparation_seconds": perf_counter() - started,
                "encoding_seconds_this_run": 0.0, "artifact": metadata,
            }
        except (ValueError, KeyError, OSError, EOFError, BadZipFile) as error:
            logger.warning("Rebuilding invalid embedding cache: %s", error)
    config = EmbeddingConfig()
    tokenization_started = perf_counter()
    truncated_chunks = 0
    for start in range(0, len(chunks), config.batch_size):
        texts = [chunk.text for chunk in chunks[start : start + config.batch_size]]
        token_ids = encoder.tokenizer(texts, truncation=False, padding=False)["input_ids"]
        truncated_chunks += sum(len(ids) > config.max_sequence_length for ids in token_ids)
    tokenization_seconds = perf_counter() - tokenization_started
    encoding_started = perf_counter()
    if chunks:
        vectors = normalize_vectors(encoder.encode(
            [chunk.text for chunk in chunks], batch_size=config.batch_size,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True, prompt="",
        ))
    else:
        vectors = np.empty((0, config.dimension), dtype=np.float32)
    if vectors.shape != (len(chunks), config.dimension):
        raise ValueError("Encoder returned an unexpected chunk embedding shape")
    encoding_seconds = perf_counter() - encoding_started
    metadata = {
        "identity": identity,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "embedding_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
        "encoding_seconds": encoding_seconds,
        "tokenization_audit_seconds": tokenization_seconds,
        "truncated_chunks": truncated_chunks,
    }
    write_embedding_cache(cache_path, vectors, metadata)
    return vectors, {
        "cache_hit": False, "preparation_seconds": perf_counter() - started,
        "encoding_seconds_this_run": encoding_seconds, "artifact": metadata,
    }


class DenseIndex:
    """Exact cosine search over chunk embeddings using one concrete encoder."""

    def __init__(
        self, chunks: list[Chunk], embeddings: NDArray[np.float32], encoder: "SentenceTransformer"
    ) -> None:
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Chunk IDs must be unique")
        if any(not chunk.chunk_id or not chunk.document_id for chunk in chunks):
            raise ValueError("Chunk and document IDs must be nonempty")
        self.embeddings = normalize_vectors(embeddings)
        if self.embeddings.shape[0] != len(chunks):
            raise ValueError("One embedding row is required per chunk")
        self.chunks = tuple(chunks)
        self.encoder = encoder
        self.chunk_ids = np.asarray([chunk.chunk_id for chunk in chunks])

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query = normalize_text(query)
        if not query or not self.chunks:
            return []
        vector = normalize_vectors(self.encoder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False, prompt="",
        ))
        if vector.shape != (1, self.embeddings.shape[1]):
            raise ValueError("Query embedding must have shape (1, chunk embedding dimension)")
        scores = self.embeddings @ vector[0]
        order = np.lexsort((self.chunk_ids, -scores))[:top_k]
        return [
            SearchResult(rank, float(scores[i]), self.chunks[i].chunk_id,
                         self.chunks[i].document_id, self.chunks[i].text)
            for rank, i in enumerate(order, start=1)
        ]
