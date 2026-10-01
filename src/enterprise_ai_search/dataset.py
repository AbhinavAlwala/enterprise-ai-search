import gzip
import hashlib
import json
import logging
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

from enterprise_ai_search.models import Document

logger = logging.getLogger(__name__)
CORPUS_REVISION = "dfb5d0e7aa6f2ace386740fdf81d6908a454a636"
QRELS_REVISION = "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"
FILES = {
    "corpus.jsonl.gz": (
        f"https://huggingface.co/datasets/BeIR/scifact/resolve/{CORPUS_REVISION}/corpus.jsonl.gz",
        "2f4a2a264485e5b5b95ae9cd5534817447757423991a4767d6da811539c0853e",
    ),
    "queries.jsonl.gz": (
        f"https://huggingface.co/datasets/BeIR/scifact/resolve/{CORPUS_REVISION}/queries.jsonl.gz",
        "c29574b9cb8fdf043df8494f689de15f4604ac4b585eb34bda9b6a389aa6c4d3",
    ),
    "qrels/test.tsv": (
        f"https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/{QRELS_REVISION}/test.tsv",
        "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c",
    ),
    "qrels/train.tsv": (
        f"https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/{QRELS_REVISION}/train.tsv",
        "a53f2114831916c096b6c37d9e54da68cef4efdcdbd5ed46533601af972acf1d",
    ),
}


def _checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_scifact(data_dir: Path) -> None:
    for name, (url, expected_hash) in FILES.items():
        destination = data_dir / name
        if destination.is_file() and _checksum(destination) == expected_hash:
            logger.info("Verified cached %s", destination)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                temporary = Path(output.name)
                logger.info("Downloading %s", name)
                with urlopen(url, timeout=60) as response:
                    shutil.copyfileobj(response, output)
            if _checksum(temporary) != expected_hash:
                raise ValueError(f"Checksum mismatch for {name}")
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def load_corpus(path: Path) -> list[Document]:
    """Load BEIR JSONL or gzip JSONL without performing network operations."""
    opener = gzip.open if path.suffix == ".gz" else open
    documents = []
    seen_ids: set[str] = set()
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                document_id, text = record["_id"], record["text"]
                title = record.get("title", "")
                if not isinstance(document_id, str) or not document_id.strip():
                    raise ValueError("_id must be a nonempty string")
                if not isinstance(text, str) or not isinstance(title, str):
                    raise ValueError("text and title must be strings")
                if document_id in seen_ids:
                    raise ValueError(f"Duplicate document ID: {document_id}")
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            seen_ids.add(document_id)
            documents.append(Document(document_id, text, title))
    return documents
