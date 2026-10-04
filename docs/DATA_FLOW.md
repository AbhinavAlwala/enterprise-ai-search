# Current data flow

## Final end-to-end map

See [the architecture diagram](ARCHITECTURE.md) for the complete current request path: request ID/body validation -> trusted upstream identity -> BM25+dense -> document RRF -> authorization filter -> cross-encoder -> search response OR numbered RAG evidence -> external generator -> citation-reference validation -> structured logs/metrics. No protected evidence enters reranking or generation before authorization. Health and metrics bypass inference; the privileged scientific CLI bypasses the HTTP identity boundary intentionally.

CI is separate from this runtime flow: checkout -> Python/uv -> locked dependency sync -> offline fixture tests -> source/wheel build -> whitespace checks -> image build only. It never downloads SciFact/weights, starts the API/Ollama, or runs benchmarks. Initial package and image downloads remain necessary on a cold runner. See [the runbook](RUNBOOK.md) for clone setup and manual runtime checks.

## Request observation (M11)

1. Before routing, middleware validates a single canonical lowercase UUIDv4 `X-Request-ID`; invalid/missing/duplicate IDs receive `uuid4()`. Duration starts with monotonic `perf_counter`, not a wall-clock timestamp.
2. A request-local context carries only ID, bounded method/route, timing values, and failure flags into handler threads. Query/evidence/identity values never enter it. Existing M10 checks run unchanged; rejected identity (400/403) sets a denial flag, without naming documents.
3. Search success contributes its existing combined retrieval/reranking time. Ask success contributes existing retrieval/reranking, generation-request, and online durations. A delegating generator observer also detects failed calls or invalid empty/non-string outputs. Failure flags do not change pipeline behavior.
4. Middleware injects `X-Request-ID` in response headers, then records ASGI handling duration through response-body sending/application completion, including validation, serialization, and service-lock wait. This is not client network latency. Before headers, unexpected errors become the existing generic 500; after headers, a sanitized failure propagates without attempting a second response.
5. One short metrics lock updates fixed counters/route-status buckets and duration sums/counts. JSON request/error logs include correlation and safe metadata, not exception messages or protected text. Context is reset after the request.
6. `/metrics` copies a snapshot under the lock. Its current polling request joins counters after the snapshot is sent. Snapshot data is process-local and disappears on restart; no monitoring server, persistence, or worker aggregation is added.

## Permission checks (M10)

1. Startup reads a strict JSON policy file into an immutable snapshot. Duplicate fields/IDs, unknown tenants, invalid identifiers or modes, and unexpected fields reject the whole file. There is no allow-all fallback.
2. Search/ask parse exactly one `X-Tenant-ID` and `X-Principal-ID`, plus at most one optional comma-separated `X-Groups` header. IDs are case-sensitive ASCII tokens, at most 64 characters; groups are limited to 32 entries and 2,048 header characters. Identity is kept out of the natural-language prompt.
3. Missing/malformed/duplicate identity headers return generic 400. An unknown tenant returns generic 403 when the service is ready; unavailable retrieval/policy returns 503. No error names a forbidden document or another identity.
4. Under the existing lock, global hybrid retrieval selects its unchanged top 50. `AuthorizedIndex.search` drops missing/denied policies and inconsistent passage parents, then compacts ranks. Only matching-tenant grants survive; there is no backfill or expanded retrieval.
5. The cross-encoder receives only surviving representative texts. Search returns up to the requested authorized top-k. Ask feeds up to five authorized winners to unchanged numbered context, prompt, and generator. Existing citation mappings can resolve only those supplied sources; missing/unknown markers remain flagged.
6. A denied, missing-policy, or nonexistent candidate has no public result record. All-denied searches return an empty list; ask with no candidates supplies the existing no-evidence context. This hides document-existence details in response content, not timing or global ranking side channels.

The local real-corpus smoke returned disjoint A/B source sets (3/2 documents) for the same query. One qwen2.5:3b ask used A's three authorized documents, took 34.33 s online, and returned an uncited insufficiency response with validation failed. No extra generation or scientific evaluation was run.

## Container startup and shutdown (M9)

1. Prepare missing corpus/encoder/reranker artifacts using the existing host CLI commands in README. Image building downloads Python dependency wheels, not SciFact or model weights. `.dockerignore` limits build inputs to application source and packaging files.
2. `docker compose build` builds the dependency layer from `pyproject.toml`/`uv.lock`, then installs the package. Only the resulting runtime environment is copied into the final image.
3. Compose mounts the existing local directories plus M10's read-only demo policy file, supplies generation variables and `AUTHORIZATION_POLICY_PATH`, and publishes host `127.0.0.1:8000` to container port 8000. Missing mount sources are rejected instead of silently created.
4. Uvicorn starts one worker in `/app`; lifespan reads policy/data/model snapshots, verifies/reuses the dense cache, and prepares resources once. A stale cache can be regenerated on the writable parent mount; model weights and policies remain read-only. Offline flags prohibit fetching missing weights.
5. Host HTTP requests follow the identity/authorization flow above. `/ask` calls configured host Ollama through Docker Desktop's host address; container loopback cannot address the host. Healthchecks read retrieval/policy readiness only and do not validate generator connectivity.
6. `docker compose down --timeout 90` signals Uvicorn, allowing in-flight work and lifespan cleanup before removing the container/network. Bind-mounted data survives; temporary container cache state does not. The owner reported later M9 Docker verification; M10 was verified locally and its updated container was not run here.

## HTTP requests (M8)

1. Uvicorn invokes `api.create_app`; lifespan loads the local corpus, unchanged chunks/indexes, verified dense cache, encoder, reranker, and optional generator once. Run from the repository root; no dataset download is performed by startup. Model/cache misses follow the existing preparation behavior.
2. `/health` reads the retained initialization/configuration flags and preparation time. It performs no retrieval or generator request; configured does not mean reachable.
3. `/search` validates trimmed query text (1–2,000 characters), integer `top_k` (1–10), and M10 identity. Under the service lock, existing BM25+dense -> document RRF top 50 -> authorization -> cross-encoder produces winning passages. The API maps only public rank, IDs, text, and raw reranker score to JSON.
4. `/ask` validates the question/identity and requires generation configuration. Under the same lock, unchanged M6 RAG with the authorized index selects up to five passages, builds its existing numbered prompt, generates free-form text, and validates inline source references. JSON preserves answer, evidence, mappings, failures, and existing online-stage timings. It never invokes claim evaluation.
5. HTTP handlers add `handler_seconds`, including time waiting for the lock but excluding request validation, response serialization, and network transfer. Retrieval/generation stage timings exclude that wait; one-time preparation appears only in health. Shutdown releases retained service references.

Invalid body input is 422; identity errors are 400/403 as described above; missing retrieval/policy/generation configuration is 503; search failures are 500 and expected answer-pipeline failures are 502. Responses and application logs omit exception details, identity/ACL details, keys, and large evidence payloads. The privileged CLI still prepares unscoped resources per invocation and prints JSON; the HTTP service keeps them across requests.

## Local ingestion and lexical search

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

## RAG flow

1. `ask` checks generation endpoint/model configuration before loading corpus or models. The CLI then prepares the same chunks, encoder/cache, hybrid index, and reranker as M5.
2. `rag.ask` calls unchanged hybrid top-50 retrieval and `rerank_candidates(..., top_k=5)`. No score threshold, tuning, extra passage search, or new model is introduced.
3. `select_evidence` preserves the five winning passages' parent/chunk IDs, reranked document ranks, and exact text. Source numbers 1 through N follow that order, independent of original IDs.
4. `build_context` emits `[n] Document: ...; Chunk: ...; Rank: ...` followed by passage text, separating blocks with blank lines. `build_messages` adds one system prompt and one user message containing the question/context. Empty retrieval is represented explicitly as no evidence; the model still decides insufficiency.
5. `HttpGenerator.generate` sends model/messages to the configured full chat-completions URL using a non-streaming JSON POST, optional bearer key, timeout 60 seconds, and output cap 512 tokens. It reads `choices[0].message.content`; malformed/empty responses and HTTP/network failures become concise errors without server bodies or keys.
6. `validate_citations` extracts numeric bracket markers, deduplicates repeated numbers in first-appearance order, maps valid numbers to document/chunk IDs, and records unknown numbers. Missing markers or any unknown number fails validation. The answer is preserved, including an uncited insufficient-evidence response.
7. The CLI prints answer/evidence/validation/timing JSON. No previous result artifact is written. Preparation and total end-to-end time include CLI model/cache/index setup; online time excludes setup but includes context construction and citation checking. Retrieval/reranking and generation-request time are measured separately. Frozen retrieval benchmarks remain unchanged.

Context is deterministic for an unchanged reranked list; generated text need not be. The configured model must fit the complete five-passage prompt plus output within its context window. No model-specific token counting or passage trimming is implemented.

## M7 evaluation flow

1. Load test qrels only to select query IDs. Read their query metadata; exclude empty metadata and reject inconsistent/unknown stance labels. Validate the frozen subset: 188 claims, 124 SUPPORT and 64 CONTRADICT. Training annotations do not guide the prompt.
2. Prepare existing corpus/chunks, cached dense embeddings, hybrid index, and reranker once. For each claim, retrieve hybrid top 50 and supply the same five winning passages as M6, without gold labels or extra passage searches.
3. Send the fixed claim-verification prompt through the existing HTTP client: qwen2.5:3b, temperature 0, max_tokens 128, JSON-schema response_format, timeout 180 seconds, sequential requests, no retries. The endpoint/model still come from environment configuration.
4. Parse exactly verdict/explanation/citations JSON. Keep SUPPORT, CONTRADICT, and ABSTAIN distinct from malformed output and HTTP failure. Validate the authoritative citation array independently, preserving parseable predictions with invalid citations.
5. Record raw output, explanation, verdict, evidence text/IDs, citations, gold-document presence, and retrieval/generation/online timings. Atomically replace the checkpoint after each claim.
6. Run five claims in sorted string-ID order first. Estimate preparation plus 188 times mean online duration; refuse a full run above 90 minutes. Resume compatible smoke predictions into a new full artifact, or continue an interrupted artifact with --resume.
7. Score every completed claim, including failures; only a complete 188-claim artifact supplies benchmark results. Conditional gold-document diagnostics report association, not causality or sentence-level evidence coverage.

M7.3: `citation_numbers` preserves the generated array; `citations` stores deduplicated valid source-to-document/chunk mappings. Arrays are authoritative regardless of explanation markers. SUPPORT/CONTRADICT require a nonempty valid array; ABSTAIN accepts an empty one. Invalid types or unknown numbers fail citation validation without silently repairing the output.

The completed full artifact contains all 188 selected claims (124 SUPPORT, 64 CONTRADICT), with 34 asserted verdicts and 154 abstentions. All requests and parses succeeded; 184 citation arrays passed. An annotated document appeared in 179 contexts; the remaining 9 lacked one. These parent-document diagnostics do not establish whether the exact rationale was supplied.
