# Design decisions

## Standard metadata with uv

Use `pyproject.toml` for package metadata and `uv` for dependency and environment management. This keeps configuration in one standard file and provides a lockfile workflow when the environment is initialized.

An alternative is `venv` with pip and requirements files. It requires more separate steps for environment setup and dependency locking. Poetry is another option, but its additional project conventions are unnecessary here.

## Python 3.12

Limit the foundation to Python 3.12 to keep the supported interpreter range small and reproducible. Broader version support can be added when it is tested. A lockfile does not itself pin an exact Python patch version.

## src layout and Hatchling

Place the package under `src/` so that ordinary imports from the repository root depend on installing the package. This helps expose packaging mistakes. A flat layout is simpler to import directly but can conceal installation issues.

Hatchling provides a small, declarative build configuration. Setuptools would also work; its additional flexibility is not needed. Hatchling is a build dependency, not a runtime dependency.

## SciFact without the BEIR stack

Use the real [BEIR SciFact dataset](https://github.com/beir-cellar/beir/wiki/Datasets-available), originally from [AllenAI](https://github.com/allenai/scifact). The archive host failed DNS resolution here, so use the same dataset from BEIR's Hugging Face mirrors: [corpus/queries revision](https://huggingface.co/datasets/BeIR/scifact/tree/dfb5d0e7aa6f2ace386740fdf81d6908a454a636) and [qrels revision](https://huggingface.co/datasets/BeIR/scifact-qrels/tree/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029). The older corpus revision exposes gzip JSONL instead of requiring a Parquet reader.

The four SHA-256 values in `dataset.py` were computed from downloaded bytes and pinned with the revisions. Cached downloads are reverified. This detects byte changes; it is not a cryptographic signature from the dataset authors. Search loads local corpus files without rechecking their checksum; rerun download to verify/repair them. Corpus ingestion ignores BEIR metadata, retaining ID, title, and abstract. Query/qrels formats are retained for later evaluation, not interpreted now.

## Word windows and simple tokenization

Use 180 whitespace words with 30-word overlap, configured in `ChunkingConfig`. These are initial choices, not tuned results. Overlap preserves some context across boundaries but repeats content and increases index size. Sentence-aware chunking would improve readability but adds complexity. IDs are stable only for unchanged source text and chunk configuration.

Normalize NFC and whitespace for display, then case-fold and split into Unicode alphanumeric terms for search. No stemming, stopword removal, or NLP dependency is used. Scientific punctuation and hyphenated terms lose structure. Titles are prepended once rather than repeated in every chunk or given a separate weight.

## Explicit BM25

Implement the small scoring formula in the standard library instead of adding rank-bm25 and its NumPy dependency. This makes the calculation visible and avoids variant-specific negative IDF behavior. A library would offer established reuse; our implementation instead requires explicit numeric tests.

For each unique query term occurring in a chunk:

```text
IDF = ln(1 + (N - df + 0.5) / (df + 0.5))
contribution = IDF * tf * (k1 + 1) / (tf + k1 * (1 - b + b * L / avgL))
score = sum of contributions
```

`N` is chunk count, `df` counts chunks containing the term, `tf` is occurrences in this chunk, and `L` is chunk token count. Defaults are `k1=1.2` and `b=0.75`. Positive IDF follows the form documented by [Lucene BM25Similarity](https://lucene.apache.org/core/9_12_3/core/org/apache/lucene/search/similarities/BM25Similarity.html); this is not a claim of identical Lucene scores.

Unique query terms contribute once. Only positive matches are returned; ties use chunk ID. Full scans and sorting are simple enough for this corpus but do not scale like an inverted index. The index is rebuilt on each CLI query. Multiple chunks from one document can occupy top-k; document aggregation and evaluation are deferred because qrels refer to documents, not chunks.

## Verification scope

pytest is the only added development dependency. Offline fixture tests and an executed real-corpus smoke query validate this milestone. They do not establish retrieval quality, scientific correctness, or performance. No environment template, server, or future-feature modules are needed.
