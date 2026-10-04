# Code walkthrough

Final audit: the API distinguishes internal ask failures (500/unexpected error) from detected generator failures (502). The CLI guard protects all 13 result paths. `tests/test_result_artifacts.py` checks writer protection and the LF-normalized manifest across checkout line endings.

## CI and final repository boundaries (M12)

- `.github/workflows/ci.yml`: one Ubuntu 24.04/Python 3.12 job on push/PR, SHA-pinned checkout/setup actions, uv 0.11.26, read-only permissions, locked sync, offline tests/build, commit/worktree whitespace checks, and Docker build-only. No asset mounts, benchmark run, runtime model call, registry/PyPI publication, or new linting tool.
- `tests/conftest.py`: rejects HTTP connection creation so forgotten mocks fail without interfering with Windows' internal async socket pairs. Existing tests use fixture files/fake backends and remain independent of local corpus/models/secrets.
- `docs/RUNBOOK.md`: complete Linux/PowerShell preparation, header transport, generation configuration, metrics, Docker mounts/networking, and shutdown commands. `results/README.md` documents immutable artifacts, their sizes/checksums, complete M7 metrics, and distinctions from operational smokes.
- `pyproject.toml`: accurate project description, unchanged Python 3.12 requirement/dependency versions. `uv.lock` stays authoritative. `.gitignore` also protects stray model/checkpoint/cache/key files; `.dockerignore` already restricts image inputs.

No application source or result JSON changes accompany M12. Hosted CI success is separate from locally executed tests/builds. Source/wheel building may use unpinned-within-range Hatchling build dependencies; lockfiles do not make the Docker OS/base tag bit-identical. The project uses the [MIT License](../LICENSE).

## Observability code (M11)

- `observability.py`: `request_id` validates/generates UUIDs; `RequestTracingMiddleware` observes all HTTP paths and sanitizes errors; `current_trace` carries request correlation through handler threads. `log_event` uses Python logging with JSON fields, never caller payloads. `Metrics.record/snapshot` lock fixed counters and duration aggregates. `ObservedGenerator.generate` records call timing/failure and delegates without modifying messages, outputs, or exceptions.
- `api.py`: owns one metrics instance, adds `GET /metrics`, attaches denial/error flags, and records existing successful stage timings. Lifespan emits safe readiness/preparation/shutdown events and suppresses raw Uvicorn access logs. HTTP bodies and M10 status semantics remain unchanged.
- `service.py`: uses the generation observer around unchanged M6 `rag.ask`; initialization logs contain only booleans, duration, and exception types. It does not alter retrieval, authorization, or generation settings.
- `tests/test_observability.py`: offline tests cover ID replacement/propagation, payload-safe correlated logs, counters/error distinctions, fixed snapshot shape, resets, concurrent updates/requests, and response-body duration. Existing authorization tests still inspect actual model inputs.

Request observation adds no dependencies. Existing generation and scientific modules remain frozen; no real generation or expensive benchmark is needed to verify tracing. The runbook shows request/response ID and metrics commands. Duration aggregates are totals/counts/means, not percentiles or quality metrics.

Executed verification: locked offline sync, 431 offline tests (31 new observability cases), package build, and a real local health/metrics/search/403-denial smoke. The caller UUID appeared unchanged in the response and request log; counters changed as expected and no protected values appeared in the six request records. No real generator call was made. Preparation was 25.32 s; retrieval/reranking 0.409 s. The server was stopped afterward.

## Authorization code (M10)

- `authorization.py`: `parse_principal` validates transport identity separately from access rules. `DocumentAccessPolicy` validates one tenant-owned ACL. `load_policy_store` rejects malformed/duplicate JSON and builds an immutable `PolicyStore`; `allows` checks tenant equality before tenant/principal/group grants. `AuthorizedIndex.search` filters the fixed hybrid top 50 and validates passage parent IDs before compacting ranks for frozen reranking.
- `config/demo_access.json`: a deterministic, clearly labelled synthetic overlay for five real SciFact IDs across two tenants. Unlisted documents are denied; the corpus/qrels are untouched.
- `service.py`: loads the required policy path once at startup, retains an unavailable state on configuration failure, and requires `PrincipalContext` in both `search` and `answer`. Both use the adapter; `answer` still calls unchanged `rag.ask`.
- `api.py`: parses headers, rejects ambiguous identity, checks store readiness/known tenant, and passes the context to the service. Health adds only `authorization_initialized`; normal responses expose no ACL internals.
- `tests/test_authorization.py`: 73 offline cases cover rules, malformed policies, immutable snapshots, tenant collisions with identical text, bounded top-k, default deny, sanitized errors, and real fake-predictor/generator inputs. Existing API tests now supply explicit identity/policies. The full suite passed 400 tests with one upstream Starlette deprecation warning.

Trace one request: headers -> validated principal -> ready policy store -> global hybrid 50 -> allowed candidates -> existing cross-encoder -> search results or M6 numbered evidence -> generator -> inline source mapping. Final-response-only filtering would leave protected text in model inputs; the adapter prevents that earlier. The CLI/evaluation routes do not use this adapter.

The runbook contains startup/header examples. Missing/malformed policy configuration returns 503 without broad access. Policy changes require restart. Build/check commands: `uv build --offline --cache-dir .uv-cache` and `git diff --check`. No new dependencies, model downloads, or benchmark reruns were introduced.

## Docker files (M9)

- `Dockerfile`: pinned Python 3.12.15/uv 0.11.26 release tags, dependency-first BuildKit caching, `uv sync --locked --no-dev`, and a final non-editable install. A second stage copies only the installed environment, creates UID/GID 10001, sets offline model flags, and starts one Uvicorn worker. No application source changed.
- `.dockerignore`: allowlists packaging inputs/source, then excludes generated Python files and secret formats. Git history, local data/model caches, results, tests, editor files, and host environments never enter the build context.
- `compose.yaml`: one application service, Linux amd64, loopback-only host port, explicit generation-variable forwarding, four directory mounts plus M10's read-only demo policy file/path, and a 90-second shutdown grace period. Container UID/GID can be overridden for Unix cache ownership.
- `.env.example`: distinguishes host versus Docker Desktop endpoint addresses and documents the required host policy path. Compose forwards the three generation fields and sets its mounted policy path; Python still reads only its process environment.

Build/start/smoke/shutdown commands are in the runbook. To build without Compose: `docker build --platform linux/amd64 -t enterprise-ai-search:latest .`. To inspect size: `docker image inspect enterprise-ai-search:latest --format '{{.Size}}'` (bytes). `docker compose logs search` shows startup/preparation, while `GET /health` reports preparation seconds once ready. Use `docker compose config --quiet` to validate configuration without printing interpolated secrets.

The healthcheck uses installed Python's `urllib.request`, with a 3-second request timeout, 30-second interval, 180-second startup grace, and three retries. HTTP 503 or connection failure marks the service unhealthy; health status alone does not restart it. The owner reported successful M9 build/mount/HTTP verification after Docker was unavailable in the initial pass. M10's updated container was not run here; rebuild it for the new code. No Docker timings or image size are claimed.

## HTTP boundary (M8)

- `api.py`: Pydantic request/response models, bounded text/top-k validation, sanitized HTTP errors, and `create_app(service_factory=load_service)`. Its lifespan prepares once and calls `close` at shutdown. M11 adds an operational metrics route alongside health/search/ask.
- `service.py`: `load_service` composes existing loaders/indexes/cache/reranker without changing their settings. `SearchService.search` requests 50 hybrid candidates through the authorization adapter and reranks; `answer` calls unchanged `rag.ask` through the same adapter. Both require a principal and reuse a shared lock and the same initialized objects. Generation configuration is optional; logs use standard-library logging and omit payloads/secrets/identity.
- `tests/test_api.py`: factory injection, tiny fake candidate/predictor/generator objects, lifecycle/resource reuse, request limits, ranking/source mapping, failed citations, missing configuration, and sanitized failure responses. A mocked default-loader test verifies preparation without models or corpus files.

Start with `uv run --locked --offline --cache-dir .uv-cache uvicorn enterprise_ai_search.api:create_app --factory --host 127.0.0.1 --port 8000` from the repository root after setup and setting `AUTHORIZATION_POLICY_PATH`. Lifespan -> retained service -> validated HTTP identity/request -> authorized retrieval or M6 RAG -> public response model -> JSON. The CLI remains the privileged evaluation/preparation entry point; the API does not expose benchmark execution. See the runbook for the exact PowerShell requests.

- `models.py`: `Document` represents a source record; `Chunk` represents a searchable passage with a parent ID; `SearchResult` adds rank and score. Frozen dataclasses discourage accidental record mutation.
- `dataset.py`: `FILES` fixes download provenance and SHA-256 values. `download_scifact` verifies cached/downloaded bytes and replaces files only after verification. `load_corpus` separately handles plain or compressed JSONL without network access. `load_queries` reuses its ID/text validation; `load_qrels` parses provided TSV grades without inventing labels.
- `text.py`: `ChunkingConfig` centralizes size/overlap and validates them. `normalize_text` keeps readable case/punctuation. `tokenize` creates case-insensitive search terms. `chunk_documents` creates deterministic windows and validates document IDs.
- `bm25.py`: `BM25Index.__init__` computes corpus statistics once. `search` computes each chunk's score, filters nonmatches, sorts, and returns result records. This is a concrete index, not a framework for future retrievers.
- `evaluation.py`: `ranked_document_ids` retains the first occurrence of each parent. `recall_at_k`, `reciprocal_rank_at_k`, and `ndcg_at_k` calculate individual-query metrics. `evaluate_scifact` validates IDs, builds the default index, evaluates every test query, macro-averages metrics, and returns the report. File hashes identify the data and implementation used.
- `cli.py`: `main` parses commands. Download does not build an index. Search loads documents, creates chunks/index, executes the query, and serializes dataclasses to JSON. Evaluate runs `evaluate_scifact`, writes its report, and prints quality/timing fields. Errors exit with status 2; logging uses stderr.
- `pyproject.toml` declares the CLI entry point and pytest development dependency; `uv.lock` records resolved versions. `.gitignore` excludes downloaded data, environments, and generated files. `AGENTS.md` preserves milestone boundaries.

## Execution and checks

`uv sync --locked --cache-dir .uv-cache` installs the package in editable mode and development dependencies. The installed `enterprise-search` entry point calls `cli.main`; `python -m enterprise_ai_search.cli` reaches the same function.

`uv run --locked --cache-dir .uv-cache pytest` runs local fixtures. Retrieval tests include a manually calculated score, overlap boundaries, no matches, deterministic ties, and invalid parameters. Dataset tests cover malformed records, checksums, cache repair, and failed-download cleanup using mocked responses.

`tests/test_evaluation.py` checks document deduplication, candidate exhaustion, deterministic ties, multiple positives, no hits, cutoff boundaries, linear graded nDCG with a full ideal ranking, malformed qrels, inconsistent IDs, macro aggregation, and CLI-generated JSON. No evaluation test downloads data.

`uv run --locked --cache-dir .uv-cache enterprise-search evaluate --output results/scifact_bm25_rerun.json` reruns the entire local test split while preserving the frozen baseline artifact. Quality values come from execution, not hand-maintained constants. No new dependency was introduced for evaluation.

The real-corpus smoke query is in the runbook. `uv build --offline --cache-dir .uv-cache` works after build dependencies are cached and produces source/wheel artifacts under `dist/`. No formatter or linter is configured.

Understand the distribution name (`enterprise-ai-search`) versus import name (`enterprise_ai_search`), and why tests exercise installed source through the src layout.

## Dense implementation

- `dense.EmbeddingConfig`: one model/revision, dimension, batch size, maximum sequence length, and CPU thread count in one place.
- `load_encoder`: lazy model/runtime imports and CPU-only construction. BM25 commands do not load a transformer.
- `normalize_vectors`: validates a finite 2D matrix and divides every row by its L2 norm. Zero rows are rejected because their direction is undefined.
- `cache_identity`, `read_embedding_cache`, and `write_embedding_cache`: fingerprint ordered chunk content/configuration, validate metadata/shape/norms/checksum, and atomically replace the NPZ. Loading uses `allow_pickle=False`.
- `prepare_embeddings`: reuse a validated cache or audit token lengths, encode batches, normalize, and save. It returns vectors plus measured cache/encoding information.
- `DenseIndex.search`: encode a single query, compute `embeddings @ query_vector`, lexicographically sort by negative score then chunk ID, and return ranked chunks.
- `dense_evaluation.evaluate_dense_scifact`: prepare the model/index once, retrieve each test query, and call existing `ranked_document_ids`, recall, reciprocal-rank, and nDCG functions. No retriever interface or factory is introduced.
- `comparison.compare_reports`: validate two artifacts, subtract their measured metrics, and select two opposing hit/miss examples when available. It does not run retrieval or choose a model.

Dense CLI commands are `prepare-dense`, `search-dense`, and `evaluate-dense`; `compare` uses saved reports. `--cache-dir` changes the local dense cache, and `--rebuild` regenerates vectors. After the first model download, setting `HF_HUB_OFFLINE=1` prevents Hub checks for cached model files. `uv --offline` controls package access separately.

`tests/test_dense.py` uses tiny vectors and fake encoders to check shapes, zero/nonfinite vectors, cosine order, ties, top-k, stale/corrupt caches, and truncation auditing. `tests/test_dense_evaluation.py` checks complete integration with existing metrics, cache reuse, artifact comparison, and BM25 report overwrite protection. Neither test file downloads a model.

## Hybrid implementation

- `hybrid.DocumentCandidate` stores a compact document rank, parent ID, and the retriever's best passage. `HybridResult` adds fused rank/score and optional representatives from each component.
- `document_candidates` reuses `evaluation.ranked_document_ids` and looks up each parent's first chunk. Duplicate chunks cannot inflate rank contributions.
- `reciprocal_rank_fusion` validates unique, consecutive document rankings, adds fixed reciprocal contributions, and sorts the union by score then document ID. Missing candidates contribute nothing; raw component scores are never compared.
- `HybridIndex.search` requests complete rankings from the existing indexes, extracts the fixed 100-document candidate lists, and returns final top-k RRF documents. Its constructor checks identical ordered chunks. No retriever interface/factory is added.
- `hybrid_evaluation.evaluate_hybrid_scifact` loads once, reuses the validated dense cache, times online searches, calls existing metrics, and records per-query fusion details and provenance.
- `comparison.compare_reports(..., hybrid_path=...)` validates a third report and selects deterministic examples from saved rankings: hybrid hits despite a component miss, earlier first hits than both components, and hybrid misses despite a component hit, all at cutoff ten.
- `cli` exposes `search-hybrid`, `evaluate-hybrid`, and optional `compare --hybrid`. Output validation protects both frozen baseline paths before expensive evaluations.

Execution: CLI -> corpus/chunks/cache -> existing indexes -> complete chunk rankings -> unique document candidates -> RRF -> top-k -> JSON. Evaluation then compares parent IDs with qrels; labels never influence retrieval.

`tests/test_hybrid.py` checks hand-calculated RRF scores, shared/single-source documents, raw-score independence, representative passages, duplicate chunks, ties, candidate depth, top-k, validation, fixture evaluation/cache reuse, example selection, and report compatibility/protection. Encoders are fake; tests have no network dependency. Runtime dependencies and baseline source files remain unchanged.

## Reranker implementation

- `reranker.RerankerConfig`: model/revision, combined token limit 512, CPU threads 4, and batch size 16. The reranker candidate depth is a separate fixed constant, 50, from hybrid's 100-document component pools.
- `load_reranker`: lazy imports and revision-pinned CPU `CrossEncoder` with safetensors and identity score activation. `prepare-reranker` downloads/loads it without evaluating or encoding the corpus.
- `representative_passages`: groups the two existing representatives by chunk ID and preserves their source names. Conflicting text/parent IDs and absent passages are rejected.
- `rerank_candidates`: caps the supplied ranking at 50, builds pairs, calls one batched prediction, validates finite scalar outputs, takes each document's max passage score, and returns ranked results plus inference timing/pair count. It preserves the original `HybridResult`.
- `reranked_evaluation.evaluate_reranked_scifact`: prepares once, times each stage over all test queries, computes existing metrics and candidate Recall@50, and records candidate/full reranked IDs and final-ten passage details without text.
- `comparison.compare_reports(..., reranked_path=...)`: checks compatibility with the hybrid report and creates four-way metrics/latency plus automatic examples. Five places is the fixed threshold for a substantial movement, comparing the same judged document before/after; absence means it never entered the 50-document pool.

Execution: CLI -> existing chunks/cache/indexes -> hybrid top 50 -> distinct representatives -> query/passage prediction -> max per parent -> final top-k -> metrics/report. `search-reranked` exposes readable winning passages; `evaluate-reranked` saves measurements. Previous baseline/comparison paths are protected; historical reruns need new output paths.

`tests/test_reranker.py` uses fake CPU encoders/predictors. Cases cover max aggregation, duplicate representatives, single-source passages, ties, cutoff/batching, score shape/nonfinite values, provenance, integration/cache reuse, compatible/incompatible reports, example selection, and frozen outputs. The model-loading test mocks both transformer/runtime imports, never initializing a real model.

## Thin RAG implementation

- `generation.Generator`: one `generate(messages) -> str` boundary, allowing an HTTP implementation or test fake without provider logic in retrieval.
- `GenerationConfig.from_env`: requires the full endpoint URL and served-model ID; the optional key is excluded from dataclass repr. `.env.example` is a reference, not an automatically loaded configuration file.
- `HttpGenerator.generate`: JSON POST/response parsing with explicit timeout/output limit and sanitized failure messages. It uses no SDK or retry chain.
- `rag.EvidenceItem`, `Citation`, `CitationValidation`, and `GeneratedAnswer`: small frozen records for selected passages, source mappings, validation status, and output/timings.
- `select_evidence` and `build_context`: top-five cutoff, exact winning text/provenance, and deterministic numbering/blocks.
- `SYSTEM_PROMPT` and `build_messages`: the single location for evidence-only answering, insufficiency, citation syntax, and concise output instructions.
- `validate_citations`: recognizes integer markers such as `[1]`, checks membership in supplied source numbers, and preserves valid mappings when another reference is invalid. Grouped syntax such as `[1, 2]` is unsupported. This function does not evaluate entailment or whether every claim has a citation.
- `rag.ask`: calls existing hybrid/reranking functions, builds evidence/messages, invokes one generator, validates references, and measures each online stage.
- `cli._run_ask`: checks configuration first, prepares existing models/index/cache, and adds preparation/total end-to-end timing to JSON output. Report writers protect all 13 frozen artifacts, including claim-verification reports and smokes.

Tests in `test_rag.py` use existing BM25/dense/hybrid/reranking functions with tiny vectors and fake models/generators, including the full CLI path. `test_generation.py` mocks HTTP to verify payload/authentication, parsing, timeout/configuration errors, and secret-safe failures. No test calls a real LLM, downloads models, or needs network access.

## M7 code

- `dataset.StanceClaim` and `load_stance_claims`: expose claim text, one validated gold stance, and annotated source document IDs. Empty metadata stays unlabeled. Original sentence indices are validated but not aligned or scored.
- `claim_verification.VERIFICATION_PROMPT` and `build_verification_messages`: one prompt and deterministic numbered context; no gold information enters messages.
- `parse_verdict`: strict JSON schema, allowed verdicts, nonempty explanation, duplicate-field rejection, and existing citation validation. Markdown fences/free-form responses are not repaired.
- `verify_claim`: unchanged hybrid/reranker composition, one generator request, separate parse/generation/reference statuses, and timings.
- `claim_evaluation.summarize_predictions`: explicit accuracy, class precision/recall/F1, abstention/coverage, confusion matrix, and document-presence diagnostics.
- `load_test_claims`, `prepare_retrieval`, `evaluation_identity`: fixed subset validation, one model/index setup, and input/code/settings fingerprints.
- `save_checkpoint`, `load_checkpoint`, `run_evaluation`: atomic report replacement, record checksum validation, compatible resume, smoke gate, and per-claim progress logging. Completed failures are retained rather than retried.

Commands (after setting the existing local endpoint/model and HF_HUB_OFFLINE=1):

```powershell
uv run --locked --offline --cache-dir .uv-cache enterprise-search evaluate-claims --smoke --output results/scifact_claim_verification_smoke_rerun.json
uv run --locked --offline --cache-dir .uv-cache enterprise-search evaluate-claims --resume-from results/scifact_claim_verification_smoke_rerun.json --output results/scifact_claim_verification_rerun.json
# Continue an interrupted full run with the same source, inputs, and settings:
uv run --locked --offline --cache-dir .uv-cache enterprise-search evaluate-claims --resume --output results/scifact_claim_verification_rerun.json
```

Tests use fixture annotations, hand-calculated metrics, fake generation with real tiny retrieval indexes, mocked HTTP, and interruption/resume simulations. No unit test calls the real LLM.

M7.1: `VERIFICATION_RESPONSE_FORMAT` defines the two required fields and verdict enum; `VERIFICATION_MAX_OUTPUT_TOKENS` fixes the claim cap at 128. The CLI enables these only for verification, and checkpoint identity includes the schema. The strict parser and fake-generator method signature are unchanged.

M7.3: `validate_citation_array` checks integer types and source membership, deduplicates mappings in first-seen order, and applies verdict-specific empty-array rules. `VerdictParseError` carries parse/verdict stage information. Records expose `structured_parse_success`, `verdict_valid`, `citation_array_valid`, the raw array, and provenance independently. Existing fake-generator signatures and free-form ask are unchanged.

The completed benchmark artifact is read-only historical evidence. Finalization recomputed its aggregate metrics from all 188 records and checked its record checksum and local gold annotations. Restoring the 90-minute gate changes the source fingerprint; preserve the artifact?s recorded execution identity rather than rewriting it to match current source. Use new output paths for future runs.
