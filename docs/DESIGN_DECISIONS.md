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

The generated JSON contains aggregate/per-query metrics and top document IDs without corpus text, keeping it small enough to commit. Settings, input/source hashes, Python/OS details, and timing boundaries describe the run. The original BM25/dense reports are frozen and CLI writers reject their paths; reruns require another `--output`. Numeric quality values in README are generated from executed artifacts rather than typed manually.

Quality and latency answer different questions. Recall/MRR/nDCG measure agreement with provided judgments; latency measures local work on an already-built index. Total time includes setup and metric/report assembly. This is one local sequential run without warmup, repeated trials, uncertainty estimates, or production load. Qrels may be incomplete, and abstract retrieval on SciFact does not establish general enterprise search quality or factual correctness. This configuration's chunk-based statistics and parent mapping differ from a whole-document BM25 baseline.

## Verification scope

pytest remains the only direct development dependency. Offline fixture tests validate metric logic and failure cases. The full real-corpus evaluation measures this baseline against provided test judgments, but does not establish scientific correctness or production performance. No environment template, server, or future-feature modules are needed.

## One small dense baseline

Use [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. It is an Apache-2.0 English sentence/short-paragraph encoder with 384 dimensions, approximately 91 MB of weights, and a documented 256-wordpiece truncation limit. Both passages and queries use plain text with an empty prefix. No model sweep or tuning is performed.

sentence-transformers handles tokenizer, transformer inference, and pooling. NumPy handles normalized vectors and exact ranking; PyTorch provides the inference runtime. CPU-only wheels are selected explicitly for Windows/Linux through uv; macOS uses PyPI's supported wheel. The locked sentence-transformers 5.1 series uses Transformers 4.x. No training extras, API keys, ANN libraries, or vector database are needed. CPU batches of 32 and four Torch threads are operational choices, not quality tuning.

The same chunks are preserved for a fair comparison, although the model may truncate them after 256 wordpieces. Whitespace words and wordpieces are different units. We count truncated chunks and disclose this limitation rather than alter the frozen corpus preprocessing. The model's pretraining is external; dataset overlap cannot be ruled out, and these measurements do not establish contamination-free generalization.

## Cache identity and exact cosine search

Save float32 unit vectors and JSON metadata together in one ignored NPZ file. Identity includes model/revision, sequence limit, prompts, batch/thread settings, ordered chunk IDs/parents/text, chunk configuration, and relevant package versions. Shape, dtype, finite unit norms, and vector-byte checksum are checked before reuse. Rebuild stale/corrupt artifacts without using them; atomic replacement preserves an existing file if encoding fails. Changing code that alters encoding semantics should also change the cache schema/settings identity.

A full matrix-vector dot product scans every chunk. With unit vectors this equals cosine, which compares direction rather than magnitude. Sort all scores with chunk-ID tie-breaking. Exact search is simple and complete for this corpus; ANN would trade accuracy for speed at larger scale and is unnecessary here. Dense ranking includes zero/negative similarities without a tuned threshold. Similarity is neither calibrated relevance probability nor factual confidence.

## Compare artifacts, disclose costs

Keep `results/scifact_bm25_test.json` frozen. Dense evaluation reuses its test data, chunking, document deduplication, metric definitions, and macro averaging. The comparison checks matching inputs and policies and computes metric differences from JSON; no quality metrics are hand-maintained in README. Its two examples use a fixed numeric-ID selection rule at cutoff 10 and are illustrative, not a second benchmark.

Report corpus encoding once and cache/index preparation separately from online query latency, which includes query encoding. A warm cache still requires model loading and validation. BM25's preserved latency and dense latency come from separate local runs with different numerical/runtime stacks, so their ratio is descriptive rather than a controlled hardware benchmark. No method is assumed to be universally better; measured results are linked from README after the full evaluation completes.

## Measured Milestone 3 trade-offs

The [generated comparison](../results/scifact_comparison.json) shows higher dense Recall@5/10 but lower MRR@10 and nDCG@10. Dense finds more judged documents within the cutoffs while its early ranking is weaker on average. Its measured online latency is slightly lower in these separate runs, at the expense of model loading and substantial one-time corpus encoding. The [dense report](../results/scifact_dense_test.json) records these costs and the token-limit audit: 3,478 of 8,778 chunks are truncated. No settings were changed in response to the result.

The fixed example rule selects query 1, where dense retrieves relevant document `31715818` at rank 5 and BM25 misses it in the top ten, and query 70, where BM25 retrieves relevant documents `5956380` and `4414547` at ranks 1 and 2 while dense misses both in the top ten. These illustrate differing failures without establishing their causes or universal superiority.

## Document RRF without score calibration

BM25 captures precise token matches; embeddings can capture related meanings with different words. Their measured differing failures motivate combining rankings. Raw scores have different scales and meanings: BM25 depends on token statistics, while cosine compares vector directions. Adding them would impose an arbitrary relative weight without calibration.

Use equal-weight Reciprocal Rank Fusion: `score(d) = sum(1 / (60 + rank(d)))` over lists containing the document. Fixed `k=60` softens the advantage of rank 1 and lets agreement contribute strongly. This follows the [RRF formulation](https://doi.org/10.1145/1571941.1572114); it is not a claim that 60 is optimal here. Alternatives include calibrated score combination or learned fusion, but add choices/training this baseline does not need. No new dependency is required.

Fuse documents because SciFact qrels label documents. Each retriever retains its first/best chunk for each parent, then assigns consecutive document ranks. Repeated windows cannot add votes. Keep both representative passages: the retrievers can prefer different chunks of the same source. Exact fused-score ties use ascending document ID, independent of raw scores.

## Fixed candidates, separate final cutoff

Choose **100 unique documents per retriever before seeing hybrid metrics**, ten times the evaluated final cutoff. This provides a modest pool beyond ten while keeping fusion simple. It is an initial choice, not a completeness guarantee or test-set optimum. Full chunk rankings precede deduplication so repeated passages cannot consume document slots. Candidates outside the top 100 contribute no vote.

The union has at most 200 documents; RRF returns final top-k. Complete retrieval/sorting remains the dominant work on this small corpus. Deeper pools or more efficient candidate generation require a separate experiment; no parameter sweep was performed.

## Measured Milestone 4 trade-offs

The [three-way comparison](../results/scifact_hybrid_comparison.json) reports higher hybrid Recall@5/10, MRR@10, and nDCG@10 than both frozen baselines on these 300 queries, with higher average online latency. Recall@10's gain over dense is modest. The model and embedding cache were reused; online hybrid work includes sequential BM25, query encoding/dense scanning, document extraction, and RRF. Setup is separate. These local runs are not a controlled hardware benchmark.

Example selection is automatic: first numeric query ID per category, with no ranking changes. Query 70 has a hybrid top-ten hit despite dense missing; query 75 moves the first relevant hit from BM25 rank 2/dense rank 3 to hybrid rank 1; query 1 loses dense's rank-5 hit in hybrid's top ten. Agreement can outweigh a valuable single-retriever match, so better averages do not mean improvement for every query. Input/source hashes and compatibility checks preserve provenance. No baseline settings or test-set parameters were changed.
