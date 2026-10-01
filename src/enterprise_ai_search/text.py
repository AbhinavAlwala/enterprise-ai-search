import re
import unicodedata
from dataclasses import dataclass

from enterprise_ai_search.models import Chunk, Document


@dataclass(frozen=True)
class ChunkingConfig:
    size: int = 180
    overlap: int = 30

    def __post_init__(self) -> None:
        if self.size <= 0 or not 0 <= self.overlap < self.size:
            raise ValueError("Require size > 0 and 0 <= overlap < size")


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", normalize_text(text).casefold())


def chunk_documents(
    documents: list[Document], config: ChunkingConfig = ChunkingConfig()
) -> list[Chunk]:
    chunks = []
    seen_ids: set[str] = set()
    for document in documents:
        if not document.document_id or document.document_id in seen_ids:
            raise ValueError("Document IDs must be nonempty and unique")
        seen_ids.add(document.document_id)
        words = normalize_text(f"{document.title} {document.text}").split()
        for number, start in enumerate(range(0, len(words), config.size - config.overlap)):
            text = " ".join(words[start : start + config.size])
            if tokenize(text):
                chunks.append(
                    Chunk(f"{document.document_id}::chunk::{number}", document.document_id, text)
                )
            # Stop at the first window covering the end, avoiding a redundant tail.
            if start + config.size >= len(words):
                break
    return chunks
