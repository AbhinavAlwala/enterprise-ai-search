# Current architecture

## Authorization boundary (M10)

`authorization.py` contains immutable `PrincipalContext`, `DocumentAccessPolicy`, and `PolicyStore` records, separate header parsing, and an `AuthorizedIndex` adapter. `service.load_service` loads one policy snapshot from required `AUTHORIZATION_POLICY_PATH`. A missing/invalid file leaves authorization unavailable; protected routes and health return 503. No policy framework or dependency was added.

```text
trusted upstream identity headers -> API validation -> PrincipalContext
                                    + retained PolicyStore
question -> frozen global hybrid top 50 -> authorized candidates
         -> frozen cross-encoder -> authorized winning passages
         -> search JSON OR frozen M6 context/generator/citation mapping
```

Tenant equality is mandatory. Within that tenant, visibility `tenant` grants all members; `restricted` grants an explicitly listed principal OR a member of a listed group. Empty restricted lists deny everyone. Missing entries deny access. One document ID has one owning tenant; identical text does not merge permissions. The adapter also rejects inconsistent representative parent IDs and assigns consecutive candidate ranks for the unchanged reranker contract.

The privileged index still contains all SciFact documents; this is a service boundary, not physical tenant separation. Scientific CLI/evaluation paths remain unscoped and frozen. The separate five-document/two-tenant JSON overlay is synthetic deployment metadata, not benchmark annotations. Headers are an assumed upstream identity, not authentication; direct clients can spoof them. Policies require restart to change. Filtering the fixed 50 without backfill can reduce recall. No identity/policy details are included in normal responses or application logs.

## Docker boundary (M9)

One Compose service wraps the existing FastAPI application in a Linux amd64 Python 3.12 image. The build stage installs runtime dependencies from unchanged `uv.lock` and a non-editable application package; the final stage retains that environment, without uv, development dependencies, host data, tests, or result artifacts. It runs as a non-root user and starts one Uvicorn worker on container port 8000.

Four directory mounts supply existing SciFact data, the writable dense embedding cache, and read-only encoder/reranker model directories. M10 adds a fifth, read-only demo policy file mount and its explicit environment path. The encoder mount is nested inside `data/dense` to protect weights while allowing NPZ regeneration. Auxiliary Hugging Face cache state lives in ephemeral `/tmp`; the host global cache is not mounted. Ollama remains a host service accessed through `GENERATION_ENDPOINT`. The container healthcheck reads `/health`, never running inference or contacting Ollama. Model lifecycle, locking, prompts, and benchmark artifacts are unchanged.

The project owner subsequently reported successful M9 Docker build/mount/health/search/host-Ollama verification after the initial pass lacked Docker. M10's authorization changes were verified with the local API; the updated container was not run in this pass.

The project is a single Python 3.12 package with CLI and FastAPI entry points, lexical/exact dense indexes, document RRF, a cross-encoder reranker, and an independent HTTP answer-generation boundary.

```text
download command -> pinned BEIR mirror files -> local data/scifact/

search command -> corpus loader -> normalization/chunking -> BM25 index
               -> query tokenization -> scoring/ranking -> JSON results

evaluate command -> corpus + queries + test qrels -> validate source IDs
                 -> same chunking/BM25 index -> all matching chunk results
                 -> ranked unique documents -> metrics -> results/*.json

prepare-dense -> same chunks -> pinned CPU encoder -> ignored embedding cache
search-dense -> query encoder -> normalized matrix/vector dot products -> ranked chunks
evaluate-dense -> same test queries/qrels -> dense chunks -> existing document mapping/metrics
compare -> preserved BM25 report + dense report -> validation -> comparison JSON/examples
search-hybrid -> BM25 chunks + dense chunks -> unique document candidates -> RRF -> documents
evaluate-hybrid -> same test queries/qrels -> hybrid documents -> existing metrics -> JSON
compare --hybrid -> three saved reports -> compatibility checks -> comparison/examples
prepare-reranker -> pinned cross-encoder -> ignored local model cache
search-reranked -> unchanged hybrid top 50 -> representative pairs -> cross-encoder
                -> max passage score per document -> sorted top-k documents
evaluate-reranked -> same test split/metrics + candidate Recall@50 -> new JSON
compare --hybrid --reranked -> four saved reports -> validation -> comparison/examples
ask -> unchanged hybrid/reranker -> five winning passages -> numbered context/prompt
    -> configured chat-completions HTTP endpoint -> answer + citation/source validation
```

- `models.py`: frozen `Document`, `Chunk`, and `SearchResult` dataclasses.
- `dataset.py`: network download with checksums, and a separate offline JSONL loader.
- `text.py`: normalization, shared tokenization, and configurable word windows.
- `bm25.py`: chunk term counts, IDF statistics, BM25 scoring, and top-k ranking.
- `cli.py`: commands, argument validation, logging, and JSON output.
- `evaluation.py`: chunk-to-document ranking, explicit metrics, and the complete test-split run.
- `dense.py`: revision-pinned model loading, batched encoding, cache identity/integrity checks, and exact cosine ranking.
- `dense_evaluation.py`: dense preparation timings and the same test-query evaluation policies, calling M2's existing mapping/metric functions.
- `comparison.py`: checks artifact compatibility, calculates measured differences, and selects deterministic real examples.
- `hybrid.py`: retains representative passages, assigns compact document ranks, and fuses two rankings with fixed RRF settings. `HybridIndex` directly composes the existing indexes.
- `hybrid_evaluation.py`: reuses dense preparation/cache and existing metrics for the full hybrid test run.
- `reranker.py`: fixed model configuration, representative deduplication, batched pair scoring, and max passage aggregation with provenance.
- `reranked_evaluation.py`: full test-split evaluation with candidate recall and separate candidate/inference/online timings.
- `rag.py`: evidence/context models, the grounding prompt, source-reference validation, and thin query-to-answer composition.
- `generation.py`: a small `Generator` protocol and standard-library `HttpGenerator`, configured through environment variables. Tests inject a fake implementation.

BM25 and HTTP generation use the standard library. Dense retrieval needs NumPy, sentence-transformers, and CPU PyTorch plus their required dependencies. FastAPI/Pydantic define the HTTP boundary and Uvicorn serves it; pytest/HTTPX are development dependencies. Hatchling builds the distribution. `uv.lock` records environment resolution. Each dense CLI command loads the encoder and verifies/reuses or regenerates chunk embeddings. Evaluation builds one index for all queries. Data, model weights, and binary caches remain ignored. No SDK, orchestration framework, or vector database was added.

Reranking reuses the installed sentence-transformers `CrossEncoder` API without new dependencies. Its weights live in `data/reranker/models/`. Pair scoring runs online; passage representations cannot be cached independently of the query as dense embeddings can. Frozen retriever modules remain unchanged.

Generation runs in one non-streaming request after retrieval; it cannot change candidate selection or request more passages. The M6 implementation was tested with fakes/mocked HTTP; M7 adds a bounded real-model stance evaluation with separate artifacts.

## Bounded claim verification (M7)

`evaluate-claims` uses the unchanged M6 retrieval path and HTTP client with a separate fixed verification prompt. `dataset.load_stance_claims` exposes explicit test metadata without changing `load_queries`. `claim_verification.py` constructs messages and validates exact JSON verdicts and source references. `claim_evaluation.py` selects the 188 annotated test claims, prepares retrieval once, computes metrics, and saves resumable JSON after every prediction.

Gold labels and annotated document IDs are evaluator inputs only; the generator receives the claim and five winning passages. No sentence-level scoring, external judge, retrieval tuning, or new dependency is involved.

M7.1 adds an optional JSON-schema `response_format` to the existing HTTP configuration, enabled only for claim verification. No SDK or parser repair is added; free-form `ask` remains unchanged.

M7.3 keeps citations as a required integer array in verification output. Claim verification validates source membership and builds provenance separately from JSON/verdict validation; normal ask still uses the frozen inline-marker validator.

The full 188-claim qwen2.5:3b benchmark is preserved in [scifact_claim_verification_test.json](../results/scifact_claim_verification_test.json). Its classification results are separate from retrieval metrics and do not measure general answer correctness. The runtime gate is restored to 90 minutes.

## HTTP application (M8)

`api.create_app` owns request/response models and three business routes. Its lifespan calls `service.load_service` once per process, stores one concrete `SearchService` on application state, and releases references at shutdown. That service retains both indexes (including chunks, vectors, and encoder), the reranker, and optional HTTP generator.

`/health` reads readiness/configuration only, including the policy-initialization flag without tenant details. `/search` calls the frozen hybrid-50/reranking path through the authorization adapter; `/ask` calls frozen M6 `rag.ask` through the same adapter. A shared lock serializes inference; synchronous handlers run in FastAPI's worker threads, leaving the cheap asynchronous health handler available. Additional processes have independent models, policy snapshots, and locks. Tests inject a service factory without transformer initialization. Missing generation configuration disables only `/ask`; failed retrieval or authorization preparation yields explicit not-ready/503 responses. No benchmark or domain source changes accompany this layer.
