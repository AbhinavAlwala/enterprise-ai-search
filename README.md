# Enterprise AI Search & Retrieval Platform

A document retrieval project built in small, tested milestones, with an emphasis on correctness, understandable design, and reproducibility.

**Status: Milestone 1 implemented.** A local CLI loads BEIR SciFact documents, normalizes text, creates deterministic overlapping chunks, and returns ranked BM25 lexical search results with scores and source IDs.

## Current baseline

- Standard-library implementation with no runtime dependencies.
- Pinned SciFact download revisions and SHA-256 verification; corpus, queries, and relevance judgments remain local and are excluded from Git.
- Configurable word windows, shared query/corpus tokenization, and deterministic ranking.
- Offline unit tests for ingestion, chunking, scoring, and failure cases.

The dataset is [BEIR's SciFact](https://github.com/beir-cellar/beir/wiki/Datasets-available), based on [AllenAI SciFact](https://github.com/allenai/scifact). Downloads use BEIR's pinned Hugging Face mirrors because the original archive host could not be resolved in the development environment. See [design decisions](docs/DESIGN_DECISIONS.md) for provenance.

## Local setup

Prerequisites: Python 3.12 and `uv`. Run from the repository root. Sync and the first dataset download require network access.

```powershell
uv sync --locked --cache-dir .uv-cache
uv run --locked --cache-dir .uv-cache pytest
uv run --locked --cache-dir .uv-cache enterprise-search download
uv run --locked --cache-dir .uv-cache enterprise-search search "Vitamin D deficiency causes rickets" --top-k 3
```

Defaults: `data/scifact`, 180 whitespace words per chunk, 30-word overlap, and top-k 5. Search accepts `--data-dir`, `--chunk-size`, `--overlap`, and `--top-k`. It returns a JSON array containing rank, score, chunk ID, document ID, and full chunk text. Logs go to stderr. Search runs offline after download and rebuilds the in-memory index each time.

The executed example loaded 5,183 documents and 8,778 chunks. These are corpus/index counts, not quality metrics. No retrieval benchmark has been run. Queries and document-level relevance judgments are downloaded for later evaluation but are not used by search.

## Engineering documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Data flow](docs/DATA_FLOW.md)
- [Code walkthrough](docs/CODE_WALKTHROUGH.md)
- [Design decisions](docs/DESIGN_DECISIONS.md)
- [Interview preparation](docs/INTERVIEW_PREP.md)

## Limitations and planned scope

Lexical matches require shared tokens; there is no stemming, synonym expansion, or semantic retrieval. Chunk boundaries can split sentences, and several results can share a parent document. BM25 scores are ranking signals, not probabilities or factual verification.

Dense/hybrid retrieval, reranking, evaluation, RAG with citations, FastAPI, and Docker remain future milestones. None is implemented here.
