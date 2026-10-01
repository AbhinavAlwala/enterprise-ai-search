# Enterprise AI Search & Retrieval Platform

A document retrieval project built in small, tested milestones, with an emphasis on correctness, understandable design, and reproducibility.

**Status: Milestone 3 implemented.** A local CLI provides BM25 and exact dense chunk search, document-level evaluation on the full SciFact test split, and a reproducible comparison. The original BM25 implementation, preprocessing, and result artifact are preserved.

## Current baseline

- Explicit standard-library BM25 plus CPU dense retrieval with sentence-transformers, NumPy, and PyTorch.
- Pinned SciFact download revisions and SHA-256 verification; corpus, queries, and relevance judgments remain local and are excluded from Git.
- Configurable word windows, shared query/corpus tokenization, and deterministic ranking.
- Offline unit tests for ingestion, chunking, scoring, and failure cases.
- Explicit Recall@5, Recall@10, MRR@10, and nDCG@10 with a generated JSON report containing per-query rankings, timings, settings, and input/source fingerprints.
- Revision-pinned `sentence-transformers/all-MiniLM-L6-v2`: 384-dimensional embeddings, exact cosine ranking, and validated local caching.

The dataset is [BEIR's SciFact](https://github.com/beir-cellar/beir/wiki/Datasets-available), based on [AllenAI SciFact](https://github.com/allenai/scifact). Downloads use BEIR's pinned Hugging Face mirrors because the original archive host could not be resolved in the development environment. See [design decisions](docs/DESIGN_DECISIONS.md) for provenance.

## Local setup

Prerequisites: Python 3.12 and `uv`. Run from the repository root. Sync and the first dataset download require network access.

```powershell
uv sync --locked --cache-dir .uv-cache
uv run --locked --cache-dir .uv-cache pytest
uv run --locked --cache-dir .uv-cache enterprise-search download
uv run --locked --cache-dir .uv-cache enterprise-search search "Vitamin D deficiency causes rickets" --top-k 3
uv run --locked --cache-dir .uv-cache enterprise-search prepare-dense
$env:HF_HUB_OFFLINE = "1"
uv run --locked --offline --cache-dir .uv-cache enterprise-search evaluate-dense
uv run --locked --offline --cache-dir .uv-cache enterprise-search compare
```

Defaults: `data/scifact`, 180 whitespace words per chunk, 30-word overlap, and top-k 5. Search accepts `--data-dir`, `--chunk-size`, `--overlap`, and `--top-k`. It returns a JSON array containing rank, score, chunk ID, document ID, and full chunk text. Logs go to stderr. Search runs offline after download and rebuilds the in-memory index each time.

Evaluation uses `qrels/test.tsv` to select the same 300 queries and keeps each parent document's first ranked chunk before scoring the top ten documents. Dense evaluation scans all chunks; BM25 evaluates all positive lexical matches. Both use the existing document mapping and metric definitions without tuning.

Actual execution results are in the [frozen BM25 report](results/scifact_bm25_test.json), [dense report](results/scifact_dense_test.json), and [generated comparison](results/scifact_comparison.json). Dense improved recall at both cutoffs but regressed on MRR and nDCG in this experiment. Query timings come from separate local runs, while dense model-loading/corpus-encoding costs are reported separately. No method is claimed to be universally better.

`search-dense "your query" --top-k 5` inspects dense chunks. Dense commands accept `--data-dir`, `--cache-dir`, and `--rebuild`; evaluation also accepts `--output`. Downloaded models and chunk embeddings live under ignored `data/dense/`. Remove `HF_HUB_OFFLINE` from the shell environment before a missing model must be downloaded. The original `evaluate` command remains available; use a separate `--output` path to preserve the frozen report.

Metric values and the two selected real hit/miss examples are generated from execution rather than hand-maintained in this README. Queries and relevance judgments are evaluation inputs, never ranking inputs.

## Engineering documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Data flow](docs/DATA_FLOW.md)
- [Code walkthrough](docs/CODE_WALKTHROUGH.md)
- [Design decisions](docs/DESIGN_DECISIONS.md)
- [Interview preparation](docs/INTERVIEW_PREP.md)

## Limitations and planned scope

Lexical matches require shared tokens. Dense similarity can capture paraphrases but may miss precise terminology or distinctions. Chunk boundaries can split sentences, and several results can share a parent document. Neither BM25 nor cosine scores are probabilities or factual verification.

Evaluation treats unjudged documents as nonrelevant. MiniLM's 256-wordpiece limit truncates some unchanged chunks; the dense artifact records the count. Results describe this dataset and configuration; one local timing run is not a production performance benchmark. Exact scans and complete candidate rankings are practical here but do not demonstrate scalability.

Hybrid retrieval, reranking, RAG with citations, FastAPI, and Docker remain future milestones. None is implemented here.
