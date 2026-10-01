from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    chunk_id: str
    document_id: str
    text: str
