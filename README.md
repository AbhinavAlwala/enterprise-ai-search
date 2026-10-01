# Enterprise AI Search & Retrieval Platform

A Python engineering project intended to develop an enterprise document search and retrieval platform, with an emphasis on correctness, understandable design, and reproducibility.

**Status:** Project foundation only. The repository contains package configuration and engineering documentation; search and AI functionality are not implemented.

## Planned scope

Document ingestion and chunking, BM25 and dense retrieval, hybrid retrieval, reranking, retrieval evaluation, RAG with citations, and delivery through FastAPI and Docker. These are future milestones, not current capabilities.

## Local setup

Prerequisites: Python 3.12 and `uv`.

```powershell
uv sync
uv run python -c "import enterprise_ai_search; print(enterprise_ai_search.__file__)"
uv build
```

`uv sync` creates the local environment and generates `uv.lock` if absent. Commit that lockfile after generation; use `uv sync --locked` for subsequent reproducible setup. There are no runtime dependencies; Hatchling is used only to build the package.

## Engineering documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Data flow](docs/DATA_FLOW.md)
- [Code walkthrough](docs/CODE_WALKTHROUGH.md)
- [Design decisions](docs/DESIGN_DECISIONS.md)
- [Interview preparation](docs/INTERVIEW_PREP.md)

Milestones will introduce behavior, relevant tests, and documentation together. No performance results are available yet.
