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
7. The CLI writes a report and prints a concise summary. The original `results/scifact_bm25_test.json` is frozen; reruns require another `--output` path. The JSON contains settings, counts, input/source SHA-256 values, environment, aggregate metrics, top document IDs and metrics per query, and timings. No document bodies or query text are copied into the report.

Per-query latency measures search plus deduplication with an already-built index. Total evaluation time includes loading, validation, index construction, retrieval, metrics, and report assembly, but excludes JSON serialization/writing and terminal output. Timestamps/timings vary between runs; unchanged inputs/code produce deterministic rankings and quality metrics.

## Dense flow

1. The existing loader and unmodified 180/30 chunking produce the same ordered chunk records. Dense encoding receives readable chunk text directly, preserving punctuation and case; it does not apply BM25 tokenization.
2. `load_encoder` lazily loads revision-pinned `all-MiniLM-L6-v2` on CPU, with four Torch threads, a 256-wordpiece limit, and no prefixes. Model loading includes imports and initial download when needed and is timed separately.
3. `prepare_embeddings` computes a cache identity from model/revision/settings, chunk configuration, ordered IDs/parent IDs/text, and encoding runtime versions. A valid `data/dense/chunks.npz` is reused. Stale/corrupt caches are rebuilt; `--rebuild` forces regeneration.
4. On a miss, batches of 32 texts are encoded into 384-dimensional float32 vectors. A separate tokenizer audit counts chunks exceeding the sequence limit. These inputs are truncated inside the model; the original chunks remain unchanged. Rows are L2-normalized and saved with identity, checksum, original encoding time, and truncation count.
5. Online search normalizes query whitespace/Unicode with the existing function, encodes one query, and L2-normalizes its vector. `DenseIndex.search` computes all chunk/query dot products, equivalent to cosine for unit vectors, then sorts descending by score and ascending by chunk ID for exact ties.
6. Search returns the existing `SearchResult` records. Dense scores can be negative or zero; every chunk is a candidate, unlike BM25's positive lexical matches. Empty queries/corpora return no results. Zero or nonfinite vectors and mismatched shapes are rejected.
7. Dense evaluation requests the complete chunk ranking and calls the original M2 document deduplication and metric functions on the identical test query IDs. The frozen BM25 artifact is untouched. `results/scifact_dense_test.json` separates model loading, cache/index preparation, original corpus encoding, and online query timings.
8. `compare` reads both reports, checks input hashes/counts, chunking, query IDs, metric policies, and aggregate consistency, then writes to a new `--output` path. The original `results/scifact_comparison.json` remains frozen. It selects the first numeric query ID for each method having a top-10 relevant hit when the other method has none. Queries come from the evaluated query file.

Online dense timing includes one query encoding, exact similarity calculation/sorting, full result construction, and document deduplication. It excludes model loading and index preparation. Cache hits avoid repeated corpus encoding but still incur model loading and cache validation. The report retains the original corpus encoding cost rather than presenting a cache hit as free preparation.

## Hybrid flow

1. Load the unchanged corpus/chunks, pinned encoder, and validated embedding cache. Build the existing BM25 and dense indexes over identical ordered chunks once per evaluation.
2. For each query, run BM25 and dense sequentially, requesting complete chunk rankings (all positive matches for BM25). A limit of 100 chunks would not guarantee 100 different documents.
3. `document_candidates` calls the existing parent deduplication, retains up to 100 unique documents per retriever, and preserves each parent's first/best chunk. Reassign consecutive document ranks: chunks A, A, B become documents A at rank 1, B at rank 2.
4. For each candidate document, sum `1 / (60 + document_rank)` from each retriever where it appears. A missing candidate contributes zero; multiple chunks never cast extra votes. Raw BM25/cosine scores remain passage information only.
5. Sort the candidate union by descending RRF score, then ascending document ID for exact ties. Return the final top-k unique documents, carrying separate BM25/dense representative passages when available.
6. `evaluate_hybrid_scifact` evaluates the same 300 sorted test-qrels queries using the unchanged metric functions and macro means. It saves top-ten IDs plus component ranks/chunk IDs/RRF scores in `results/scifact_hybrid_test.json` without corpus bodies.
7. `compare --hybrid` checks the three artifacts, computes differences and latency from stored values, and selects the first numeric query ID in each defined example category. It does not retrieve again or adjust settings.

Online hybrid timing includes both retrieval paths, query encoding, document extraction, fusion, and final ranking. Model loading, cache validation, and index construction are separate preparation costs; cached corpus encoding is not repeated. The 100-document candidate cutoff precedes final top-10 selection, so candidates beyond it cannot receive that retriever's vote.

## Reranking flow

1. Load the same corpus/chunks, pinned dense encoder, and verified embedding cache. Build the unchanged `HybridIndex`. Load the separately pinned cross-encoder once.
2. Call `HybridIndex.search(query, 50)`: this preserves its 100-document component pools and k=60 RRF, selecting the first 50 fused documents.
3. Collect the available BM25/dense representatives for each document. Deduplicate by chunk ID within that parent, preserving both source names when a chunk is shared. No other chunks are fetched.
4. Pass normalized query text and readable passage text together to `CrossEncoder.predict`, in CPU batches of 16. Tokenization has a combined 512-token limit including special tokens, using longest-first truncation. Score activation is identity: scores are raw logits.
5. Use the maximum passage score as the document score; retain that passage. Passage-score ties use ascending chunk ID. Sort documents by descending cross-encoder score then ascending document ID. RRF and component scores remain provenance, never numeric contributions to final scores.
6. Evaluation retains the full 50 candidate IDs and reranked order; top ten additionally record hybrid/component ranks, passage IDs, passage scores, and selected passage. Compute existing top-5/10 metrics and candidate Recall@50 from the provided qrels, macro-averaging all test queries. Labels do not affect candidate generation or reranking.
7. Save `results/scifact_reranked_test.json`; four-way comparison writes a new artifact. Compatibility checks include identical inputs/policies, fixed settings, unchanged hybrid top-ten prefix, and a reranked permutation of the candidate pool. Examples are selected automatically using a fixed five-rank movement rule or absence from the pool.

Online time covers hybrid retrieval, pair preparation, inference, passage aggregation, and final document construction. Candidate generation and prediction are separately timed; prediction includes tokenizer and model work. Model loads and corpus preparation are excluded from online time. Candidate Recall@50 is the fraction of judged relevant documents available for reranking, an upper bound on recoverable recall; a final top-ten cutoff can impose a tighter bound.
