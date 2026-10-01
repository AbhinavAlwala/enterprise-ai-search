# Current data flow

1. `enterprise-search download` fetches compressed corpus/query JSONL and train/test qrels from pinned BEIR mirror revisions. Each file is SHA-256 verified before replacing its destination. Valid cached files are reused. Partial download files are cleaned up.
2. `enterprise-search search` reads only the local corpus. `load_corpus` maps `_id`, `text`, and optional `title` to documents, preserving source IDs. Invalid records and duplicate IDs raise errors with line numbers.
3. `chunk_documents` joins each title and abstract once, normalizes Unicode to NFC, and collapses whitespace. It makes 180-word windows with 30-word overlap by default. Chunk IDs are `<document-id>::chunk::<zero-based-window-number>`.
4. `BM25Index` tokenizes chunk text: case folding plus Unicode alphanumeric sequences, splitting at punctuation and underscores. It stores term frequencies, token lengths, average length, and chunk-level document frequencies.
5. `search` tokenizes the query identically, removes duplicate query tokens, ignores unseen terms, and scores every chunk using BM25. Only positive-score matches are retained. Ties use ascending chunk ID.
6. The first up-to-k results receive one-based ranks and are printed as JSON with score, IDs, and full chunk text. Empty/unmatched queries return `[]`; invalid top-k values are rejected.

Chunk size counts whitespace words; BM25 length counts tokenizer output. They are deliberately different units. The final window stops when it reaches the document end, so no redundant overlap-only tail is emitted. Windows with no searchable tokens are omitted.

Search queries come directly from CLI input; relevance judgments never influence ranking.

## Evaluation flow

1. `evaluate_scifact` loads the same corpus plus `queries.jsonl.gz` and the inspected `qrels/test.tsv`. The query file includes both train and test queries; only IDs in test qrels are evaluated, in sorted ID order.
2. The qrels loader validates the TSV header, nonnegative integer grades, unique query/document pairs, and at least one positive judgment per judged query. Evaluation also checks that referenced query/document IDs exist. It does not silently drop inconsistent rows or unanswered queries.
3. Evaluation calls the unchanged `ChunkingConfig()`, `chunk_documents`, and `BM25Index` defaults and builds the index once. Every test query is searched with a candidate limit equal to the index's entire chunk count, so every positive-score match is available.
4. `ranked_document_ids` walks chunk results in their existing order, skipping already-seen parents, until it has ten unique documents or exhausts matches. Chunk ranks 1, 2, 3 for parents A, A, B become document ranks 1, 2 for A, B. It does not sum chunk scores or rerank documents.
5. Each document ranking is compared with that query's provided qrels. Missing/unjudged documents have gain zero. Per-query Recall@5, Recall@10, reciprocal rank@10, and nDCG@10 are computed, including zero-score queries.
6. Each metric is averaged equally across all evaluated queries. This macro mean is distinct from pooling retrieved/relevant counts across queries.
7. The CLI writes `results/scifact_bm25_test.json` and prints a concise summary. The JSON contains settings, counts, input/source SHA-256 values, environment, aggregate metrics, top document IDs and metrics per query, and timings. No document bodies or query text are copied into the report.

Per-query latency measures search plus deduplication with an already-built index. Total evaluation time includes loading, validation, index construction, retrieval, metrics, and report assembly, but excludes JSON serialization/writing and terminal output. Timestamps/timings vary between runs; unchanged inputs/code produce deterministic rankings and quality metrics.
