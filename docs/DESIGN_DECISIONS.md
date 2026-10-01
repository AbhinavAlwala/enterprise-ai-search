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

The four SHA-256 values in `dataset.py` were computed from downloaded bytes and pinned with the revisions. Cached downloads are reverified. This detects byte changes; it is not a cryptographic signature from the dataset authors. Search loads local corpus files without rechecking their checksum; rerun download to verify/repair them. Corpus ingestion ignores BEIR metadata, retaining ID, title, and abstract. Evaluation reads the downloaded query JSONL and test qrels TSV; train qrels are not used for this baseline measurement.

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

Unique query terms contribute once. Only positive matches are returned; ties use chunk ID. Full scans and sorting are simple enough for this corpus but do not scale like an inverted index. The index is rebuilt on each CLI search. Multiple chunks from one document can occupy search top-k. Evaluation separately maps chunks back to unique documents because qrels refer to documents.

## Measure the frozen baseline

Milestone 2 preserves normalization, tokenization, 180/30 chunking, `k1=1.2`, and `b=0.75`. Evaluation offers no tuning flags. Observing results must not become a reason to optimize against the test set. The existing BM25, text, and model source files are unchanged; the report records their hashes.

The inspected test qrels contain 300 queries and 339 positive query/document pairs, all grade 1, with 23 queries having multiple relevant documents. The query file contains 1,109 records including train queries; evaluating all of them would incorrectly include queries outside the test split. No missing IDs or duplicate pairs were found. Reject malformed rows, inconsistent IDs, duplicate pairs, and queries without positive judgments instead of silently filtering them.

## Document ranking and candidate completeness

Retrieve all positive-score chunks using the existing search method, then keep each parent's first occurrence. This is equivalent to ordering documents by their best chunk, preserving the existing chunk-ID tie order. A fixed multiple such as 10k candidate chunks has no guarantee of ten distinct parents; full candidate retrieval avoids that bias. It costs more memory and result-construction time, which is included in measured query latency. Only matching documents are returned; fewer than ten matches are allowed without padding.

## Metric definitions

- Recall@K: relevant documents in the first K divided by all positively judged documents for that query.
- Reciprocal rank@K: `1 / first_relevant_rank` if a relevant document appears by K, else zero. MRR@10 is its mean over queries.
- nDCG@K: `sum(grade / log2(rank + 1))` for the first K, divided by the same sum for the best possible ordering of all provided grades, truncated at K. Use raw grades, as in [trec_eval's nDCG at cutoffs](https://github.com/usnistgov/trec_eval/blob/main/m_ndcg_cut.c). SciFact's observed binary grades make linear and exponential gain equivalent here.

Treat grade > 0 as relevant for Recall/MRR, and unjudged documents as gain zero. Macro-average across every test-qrels query, including those with no hits. A query with no positive judgments is rejected because recall's denominator and ideal DCG would be undefined. These policies are explicit and tested; no evaluation framework is needed.

## Reproducibility and interpretation

The generated JSON contains aggregate/per-query metrics and top document IDs without corpus text, keeping it small enough to commit. Settings, input/source hashes, Python/OS details, and timing boundaries describe the run. Rerunning overwrites the default report; `--output` can preserve another run. No numeric quality values are manually written into code or README.

Quality and latency answer different questions. Recall/MRR/nDCG measure agreement with provided judgments; latency measures local work on an already-built index. Total time includes setup and metric/report assembly. This is one local sequential run without warmup, repeated trials, uncertainty estimates, or production load. Qrels may be incomplete, and abstract retrieval on SciFact does not establish general enterprise search quality or factual correctness. This configuration's chunk-based statistics and parent mapping differ from a whole-document BM25 baseline.

## Verification scope

pytest remains the only direct development dependency. Offline fixture tests validate metric logic and failure cases. The full real-corpus evaluation measures this baseline against provided test judgments, but does not establish scientific correctness or production performance. No environment template, server, or future-feature modules are needed.
