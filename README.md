# Enterprise AI Search & Retrieval Platform

A document retrieval project built in small, tested milestones, with an emphasis on correctness, understandable design, and reproducibility.

**Status: Milestone 2 implemented.** A local CLI provides BM25 chunk search and document-level retrieval evaluation on the full SciFact test split. Evaluation preserves the original baseline settings without tuning.

## Current baseline

- Standard-library implementation with no runtime dependencies.
- Pinned SciFact download revisions and SHA-256 verification; corpus, queries, and relevance judgments remain local and are excluded from Git.
- Configurable word windows, shared query/corpus tokenization, and deterministic ranking.
- Offline unit tests for ingestion, chunking, scoring, and failure cases.
- Explicit Recall@5, Recall@10, MRR@10, and nDCG@10 with a generated JSON report containing per-query rankings, timings, settings, and input/source fingerprints.

The dataset is [BEIR's SciFact](https://github.com/beir-cellar/beir/wiki/Datasets-available), based on [AllenAI SciFact](https://github.com/allenai/scifact). Downloads use BEIR's pinned Hugging Face mirrors because the original archive host could not be resolved in the development environment. See [design decisions](docs/DESIGN_DECISIONS.md) for provenance.

## Local setup

Prerequisites: Python 3.12 and `uv`. Run from the repository root. Sync and the first dataset download require network access.

```powershell
uv sync --locked --cache-dir .uv-cache
uv run --locked --cache-dir .uv-cache pytest
uv run --locked --cache-dir .uv-cache enterprise-search download
uv run --locked --cache-dir .uv-cache enterprise-search search "Vitamin D deficiency causes rickets" --top-k 3
uv run --locked --cache-dir .uv-cache enterprise-search evaluate
```

Defaults: `data/scifact`, 180 whitespace words per chunk, 30-word overlap, and top-k 5. Search accepts `--data-dir`, `--chunk-size`, `--overlap`, and `--top-k`. It returns a JSON array containing rank, score, chunk ID, document ID, and full chunk text. Logs go to stderr. Search runs offline after download and rebuilds the in-memory index each time.

Evaluation uses `qrels/test.tsv` to select queries, retrieves all matching chunks, and keeps each parent document's first occurrence before scoring the top ten documents. It builds the index once and prints quality metrics separately from timings. `--data-dir` and `--output` override file locations; retrieval settings cannot be tuned through this command.

Actual execution results are in [the generated baseline report](results/scifact_bm25_test.json). Metric values are not maintained manually in this README. Queries and relevance judgments are evaluation inputs, never ranking inputs.

## Engineering documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Data flow](docs/DATA_FLOW.md)
- [Code walkthrough](docs/CODE_WALKTHROUGH.md)
- [Design decisions](docs/DESIGN_DECISIONS.md)
- [Interview preparation](docs/INTERVIEW_PREP.md)

## Limitations and planned scope

Lexical matches require shared tokens; there is no stemming, synonym expansion, or semantic retrieval. Chunk boundaries can split sentences, and several results can share a parent document. BM25 scores are ranking signals, not probabilities or factual verification.

Evaluation treats unjudged documents as nonrelevant. Results describe this dataset and configuration; one local timing run is not a production performance benchmark. Retrieving every matching chunk avoids candidate truncation but increases memory and processing costs.

Dense/hybrid retrieval, reranking, RAG with citations, FastAPI, and Docker remain future milestones. None is implemented here.
