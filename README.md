# Enterprise AI Search & Retrieval Platform

A retrieval engineering project built in tested milestones, with explicit algorithms, reproducible evaluation, and documented trade-offs.

**Status: Milestone 11 adds request tracing, structured logs, and bounded in-process metrics to the permission-aware HTTP API.** CLI and HTTP access share BM25, exact dense search, document RRF, cross-encoder reranking, and a thin retrieval-augmented generation (RAG) layer with source-reference validation. Retrieval is evaluated on the same 300-query SciFact test split. API and generation tests run offline with fakes. A separate bounded SciFact stance evaluation adds explicit SUPPORT/CONTRADICT/ABSTAIN output and resumable reports. Frozen retrieval source and artifacts are preserved.

- Verified, revision-pinned [BEIR SciFact](https://github.com/beir-cellar/beir/wiki/Datasets-available) ingestion and deterministic overlapping chunks.
- Explicit BM25 and revision-pinned `sentence-transformers/all-MiniLM-L6-v2` with cached 384-dimensional CPU embeddings.
- Equal-weight document RRF: `k=60`, 100 unique document candidates per retriever, chosen before hybrid evaluation.
- Revision-pinned `cross-encoder/ms-marco-MiniLM-L6-v2`: reranks the top 50 hybrid documents using the maximum score across their existing representative passages, in CPU batches of 16.
- Top-five winning passages form numbered evidence blocks for an independently configured OpenAI-compatible generator. Answers retain source mappings and explicit citation-validation status.
- Offline fixture tests, deterministic ranking, and reports with per-query results, input/source fingerprints, and separate preparation costs.

## Measured results

<!-- BEGIN GENERATED COMPARISON -->

| Metric | BM25 | Dense | Hybrid | Hybrid + Reranker |
|---|---:|---:|---:|---:|
| Recall@5 | 0.709278 | 0.731111 | 0.755167 | 0.766667 |
| Recall@10 | 0.774667 | 0.806222 | 0.807889 | 0.837722 |
| MRR@10 | 0.621536 | 0.597878 | 0.643040 | 0.675521 |
| nDCG@10 | 0.653548 | 0.645294 | 0.680262 | 0.708730 |
| Average online query (ms) | 36.972 | 34.207 | 93.155 | 3378.577 |

Hybrid candidate Recall@50: **0.933000**. This measures relevant documents available before reranking, not final retrieval quality.

<!-- END GENERATED COMPARISON -->

Generated from the [four-way comparison](results/scifact_reranked_comparison.json). Reranking improved all four quality means over hybrid in this experiment, with substantially higher CPU latency. Timings come from separate local runs, not a controlled hardware benchmark. Online reranked time includes candidate generation, pair preparation/inference, aggregation, and final ranking; model loading and corpus preparation are separate.

Artifacts: [BM25](results/scifact_bm25_test.json), [dense](results/scifact_dense_test.json), [hybrid](results/scifact_hybrid_test.json), [reranked](results/scifact_reranked_test.json). The original [two-way](results/scifact_comparison.json) and [three-way](results/scifact_hybrid_comparison.json) comparisons are retained. Automatically selected examples show upward/downward relevant-document movements and a document absent from the candidate pool.

## Local setup

Prerequisites: Python 3.12 and `uv`. Run from the repository root. Initial package, dataset, and model downloads require network access.

```powershell
uv sync --locked --cache-dir .uv-cache
uv run --locked --offline --cache-dir .uv-cache pytest
uv run --locked --cache-dir .uv-cache enterprise-search download
uv run --locked --cache-dir .uv-cache enterprise-search prepare-dense
uv run --locked --offline --cache-dir .uv-cache enterprise-search prepare-reranker
$env:HF_HUB_OFFLINE = "1"
uv run --locked --offline --cache-dir .uv-cache enterprise-search search-reranked "Arterioles have a larger lumen diameter than venules." --top-k 3
uv run --locked --offline --cache-dir .uv-cache enterprise-search compare --hybrid results/scifact_hybrid_test.json --reranked results/scifact_reranked_test.json --output results/scifact_comparison_rerun.json
```

`search` and `search-dense` return chunks; `search-hybrid` returns RRF documents; `search-reranked` returns documents with cross-encoder scores, winning passages, and original hybrid provenance. Defaults are `data/scifact`, `data/dense`, `data/reranker/models`, 180-word chunks with 30-word overlap, and search top-k 5. Evaluations use fixed settings and top ten documents without tuning flags.

Downloaded data, models, and NPZ caches are ignored. After initial downloads, cached retrieval operations work offline. Unset `HF_HUB_OFFLINE` before downloading a missing model; `uv --offline` only controls package access. To rerun frozen evaluations/comparisons, specify a new `--output` path, such as `evaluate-reranked --output results/scifact_reranked_rerun.json`; CLI writers protect all seven previous result files.

## Answer generation

Start an existing OpenAI-compatible chat endpoint serving a model, then set `GENERATION_ENDPOINT` to its full chat-completions URL and `GENERATION_MODEL` to that server's model identifier. Set `GENERATION_API_KEY` only if authentication is required. [.env.example](.env.example) documents these variables; the application reads the process environment and does not automatically load `.env`.

```powershell
$env:GENERATION_ENDPOINT = "http://localhost:11434/v1/chat/completions"
$env:GENERATION_MODEL = "qwen2.5:3b"
uv run --locked --offline --cache-dir .uv-cache enterprise-search ask "What does the retrieved evidence say about PPM1D and p53?"
```

The URL/model above assume Ollama already serves qwen2.5:3b locally; configure another served endpoint as needed. The completed M7 benchmark used local qwen2.5:3b through its OpenAI-compatible endpoint. `ask` prints JSON containing the answer, selected passages, citation-to-document/chunk mappings, validation status, and separate preparation/retrieval/generation/total timings. The client uses standard-library HTTP with optional bearer authentication, a 60-second timeout, and `max_tokens=512`. Generation requires endpoint connectivity even when package/model caches are offline. No local LLM is downloaded by this project.

## Local HTTP API

Run from the repository root after preparing the existing corpus and model caches. Lifespan startup loads retrieval resources once per process. To enable `/ask`, set the generation environment variables above to an available endpoint; missing configuration leaves `/search` usable.

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:AUTHORIZATION_POLICY_PATH = "config/demo_access.json"
uv run --locked --offline --cache-dir .uv-cache uvicorn enterprise_ai_search.api:create_app --factory --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
$identity = @{ "X-Tenant-ID" = "tenant-a"; "X-Principal-ID" = "alice"; "X-Groups" = "researchers" }
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/search -Method Post -Headers $identity -ContentType "application/json" -Body '{"query":"Radioiodine treatment of non-toxic multinodular goitre reduces thyroid volume.","top_k":3}'
Invoke-RestMethod http://127.0.0.1:8000/ask -Method Post -Headers $identity -ContentType "application/json" -Body '{"question":"Does radioiodine treatment reduce thyroid volume in non-toxic multinodular goitre?"}'
```

`GET /health` reports retrieval/policy readiness, generation configuration, and preparation time without running inference or probing the generator. `POST /search` returns authorized reranked document/passages and raw scores; `top_k` defaults to 5 and must be an integer from 1–10. Query/question strings must be nonempty after trimming and at most 2,000 characters. `POST /ask` returns the unchanged M6 free-form answer, up to five authorized evidence passages, inline-citation mappings/status, and online timings; it does not run M7 claim verification.

Invalid body input returns 422; missing/malformed identity returns 400, unknown tenants return generic 403, and unavailable retrieval/policy or generation configuration returns 503. Search failures return sanitized 500 responses, and expected answer-pipeline failures return sanitized 502 responses. CPU inference is serialized within one process; preparation is separate from request timings. Use one worker locally; reloads or additional workers load another model set. No authentication or production deployment is provided.

The local M8 smoke returned 200 for health, real-corpus search, and qwen2.5:3b ask. Preparation took 26.60 s, search 2.60 s, and ask 38.81 s (36.45 s generation). The answer omitted inline markers, and the unchanged validator visibly reported missing citations. These are one-run operational observations, not an answer-quality benchmark.

## Permission-aware API (M10)

Headers represent identity already verified by a trusted upstream gateway: required `X-Tenant-ID`/`X-Principal-ID`, optional comma-separated `X-Groups`. **These headers are not authentication and are spoofable on a directly exposed service.** Authorization requires a matching tenant, then either tenant-wide visibility, an explicit principal grant, or a matching group. Missing ACL entries deny access; a missing or malformed policy file disables protected routes.

The service removes denied documents from the existing hybrid top 50 before cross-encoder inference, numbered RAG context, generation, and source serialization. It does not fetch replacements, so fewer results/evidence passages can remain. The privileged retrieval process still holds the whole corpus. CLI scientific runs remain unscoped and unchanged.

[demo_access.json](config/demo_access.json) is **SYNTHETIC DEMO AUTHORIZATION METADATA**, assigning five existing documents to two tenants; all other documents are denied. It changes neither SciFact nor benchmark judgments. The same locally executed query, `Radioiodine treatment of non-toxic multinodular goitre reduces thyroid volume.`, with `top_k=5` produced:

| Identity | Authorized document IDs, ranked | Search time (seconds) |
|---|---|---:|
| tenant-a / alice / researchers | 9745001, 26026009, 37912677 | 0.761 |
| tenant-b / bob / researchers | 6751418, 43122426 | 0.375 |

The sets were disjoint. One real tenant-a `/ask` used only its three authorized documents and returned an insufficient-evidence answer; missing citations were visibly flagged. Generation took 34.10 s and online processing 34.33 s, with 32.45 s one-time preparation. These are deployment checks with synthetic permissions, not retrieval or answer-quality benchmarks. The verification server was stopped. The full offline suite passed 400 tests, including 73 authorization cases.

## Request tracing and metrics (M11)

Every HTTP response includes `X-Request-ID`. A single canonical lowercase UUIDv4 supplied by the caller is preserved; absent, duplicate, malformed, or oversized IDs are replaced with a new UUIDv4. Use opaque IDs, never sensitive data. JSON request/error logs correlate that ID with bounded method/route, status, duration, and available timings. Query/answer/evidence, identities, keys, ACLs, and exception messages are not logged. Raw Uvicorn access logs are suppressed to avoid untrusted URL leakage.

```powershell
$identity["X-Request-ID"] = "5996a386-cbd8-40bd-a8b8-2e86559fe286"
$response = Invoke-WebRequest http://127.0.0.1:8000/search -Method Post -Headers $identity -ContentType "application/json" -Body '{"query":"Radioiodine treatment of non-toxic multinodular goitre reduces thyroid volume.","top_k":3}'
$response.Headers["X-Request-ID"]
Invoke-RestMethod http://127.0.0.1:8000/metrics
```

`GET /metrics` returns JSON counters for total/search/ask requests, identity denials, unexpected errors, and detectable generation failures. Fixed route/status buckets and `{count,total,mean}` request/stage duration summaries keep label cardinality bounded. Timings use `perf_counter`; existing search/RAG timing fields and scientific benchmark methodology are unchanged. Retrieval and reranking remain a combined stage. Snapshot polling excludes its own request until that response finishes.

Counters are lock-protected, process-local, reset on restart, non-durable, and independent across workers/replicas. Health and metrics perform no inference. Metrics is an unauthenticated operational endpoint and should remain behind the same trusted service boundary. No monitoring containers or new dependencies are needed.

The real M11 smoke preserved the supplied UUID on an authorized search and counted one subsequent 403 identity denial. Snapshots moved from 1 to 3 to 5 total requests (polls count after their snapshots), with 2 search attempts and 1 denial. All six request logs were correlated without query/identity/document/evidence leakage. Preparation took 25.32 s and retrieval/reranking 0.409 s; no real generation ran. All 431 offline tests passed, including 31 observability cases. The verification server was stopped. These are operational checks, not new scientific benchmarks.

## Docker deployment

Requires Docker Desktop in Linux-container mode, the prepared local data/model caches from Local setup, and host Ollama serving qwen2.5:3b for `/ask`. The image targets Linux amd64. It contains the application and locked runtime dependencies, not datasets or model weights; this deployment relies on mounts and is not standalone.

If artifacts are missing, prepare them on the host with Python/uv before starting Docker (unset Hugging Face offline flags for initial downloads):

```powershell
uv run --locked --cache-dir .uv-cache enterprise-search download
uv run --locked --cache-dir .uv-cache enterprise-search prepare-dense
uv run --locked --cache-dir .uv-cache enterprise-search prepare-reranker
```

| Host directory | Container directory | Access |
|---|---|---|
| `data/scifact` | `/app/data/scifact` | Read-only corpus/query files |
| `data/dense` | `/app/data/dense` | Writable, regenerable embedding cache |
| `data/dense/models` | `/app/data/dense/models` | Read-only nested encoder cache |
| `data/reranker/models` | `/app/data/reranker/models` | Read-only reranker cache |

Existing pinned models are reused directly; the global Hugging Face cache is not needed. Auxiliary cache files use container `/tmp` and are discarded with the container. Offline model flags prevent runtime weight downloads. Mount sources must already exist. Default container UID/GID is 10001; on Unix, set `CONTAINER_UID`/`CONTAINER_GID` to the cache owner's non-root IDs and ensure `data/dense` is writable.

From the repository root:

```powershell
$env:GENERATION_ENDPOINT = "http://host.docker.internal:11434/v1/chat/completions"
$env:GENERATION_MODEL = "qwen2.5:3b"
docker compose build
docker compose up -d --wait --wait-timeout 300
Invoke-RestMethod http://127.0.0.1:8000/health
$identity = @{ "X-Tenant-ID" = "tenant-a"; "X-Principal-ID" = "alice"; "X-Groups" = "researchers" }
Invoke-RestMethod http://127.0.0.1:8000/search -Method Post -Headers $identity -ContentType "application/json" -Body '{"query":"Radioiodine treatment of non-toxic multinodular goitre reduces thyroid volume.","top_k":3}'
Invoke-RestMethod http://127.0.0.1:8000/ask -Method Post -Headers $identity -ContentType "application/json" -Body '{"question":"Does radioiodine treatment reduce thyroid volume in non-toxic multinodular goitre?"}'
docker compose down --timeout 90
```

Compose explicitly forwards generation settings from shell variables or its `.env` interpolation file. Shell values take precedence; `.env.example` is a reference, not runtime loading logic. Unset endpoint/model values default to the host-Ollama example; explicitly blank values disable generation. `localhost` inside the container refers to that container, so host Ollama needs the configured host address. Host port 8000 is bound only to `127.0.0.1`.

One Uvicorn worker reuses the existing lifespan/resources and inference lock. Compose also mounts the demo policy read-only at `/app/config/demo_access.json` and explicitly sets `AUTHORIZATION_POLICY_PATH`; health requires successful policy initialization. Rebuild the image for M10 code. The standard-library healthcheck calls only `/health`; generation configuration is not a connectivity check. Shutdown releases application resources and removes the container/network, preserving bind-mounted files. After the initial M9 pass, the project owner reported successful Docker build, health/search, mounted caches, and host-Ollama ask verification. M10 was verified locally; its updated container was not run in this pass. Image size and container timings are not reported. See [Docker decisions](docs/DESIGN_DECISIONS.md) for reproducibility limits.

## Engineering documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Data flow](docs/DATA_FLOW.md)
- [Code walkthrough](docs/CODE_WALKTHROUGH.md)
- [Design decisions](docs/DESIGN_DECISIONS.md)
- [Interview preparation](docs/INTERVIEW_PREP.md)

## Limitations

The reranker cannot recover documents outside the 50 candidates, and only sees existing representative passages. Max passage aggregation can amplify false positives; some judged documents move downward despite better mean metrics. BM25, cosine, RRF, and cross-encoder scores are ranking signals, not probabilities or factual verification.

Authorization uses a static policy snapshot loaded at startup, globally unique document IDs, and trusted upstream identity. It provides no login, gateway, hot policy reload, per-tenant index, timing-side-channel defense, or prompt-injection defense. Filtering can reduce recall. Model-generated text is not guaranteed truthful; permission checks govern supplied evidence/provenance, not the model's prior knowledge.

The dense encoder truncates 3,478 of 8,778 unchanged chunks. Cross-encoder pairs share a 512-token limit and may also be truncated; that count is not audited. Qrels can be incomplete. Generation sees only five passages; their combined prompt must fit the configured model's context window. Citation validation checks references, not whether claims are supported or true. General free-form answer correctness and explanation entailment remain unmeasured. The M7 benchmark measures only explicit SciFact stances. The local API adds no production deployment; CPU reranking and generation remain expensive, and concurrent inference requests wait for the shared service lock.

## Bounded claim-verification evaluation

`evaluate-claims` evaluates only the 188 test claims with explicit metadata stances (124 SUPPORT, 64 CONTRADICT), using the unchanged top-five reranked evidence. It validates structured verdicts, records abstentions/failures separately, and checkpoints each prediction. It does not score general RAG correctness or sentence-level citation support. Run the five-claim smoke first; a full run is refused when its estimated duration exceeds 90 minutes. See [execution flow](docs/CODE_WALKTHROUGH.md) and [metric definitions](docs/DESIGN_DECISIONS.md).

The full [188-claim report](results/scifact_claim_verification_test.json) records the executed qwen2.5:3b benchmark: temperature 0, 128-token output limit, structured verdict/explanation/citations, and the unchanged top-five reranked passages. These results measure bounded SciFact claim verification, not general RAG or free-form answer accuracy.

| Metric | Measured value |
|---|---:|
| Evaluated claims | 188 |
| Overall accuracy | 0.148936 |
| Non-abstained accuracy | 0.823529 |
| Macro F1 | 0.188228 |
| Abstention rate | 0.819149 |
| Coverage | 0.180851 |
| Parsing failure rate | 0.000000 |
| Generation failure rate | 0.000000 |
| Citation validation pass rate | 0.978723 |
| Gold-document-present rate | 0.952128 |
| Accuracy with gold document present | 0.150838 |
| Accuracy with gold document absent | 0.111111 |
| Total evaluation runtime (seconds) | 4981.39 |

| Gold class | Precision | Recall | F1 |
|---|---:|---:|---:|
| SUPPORT | 0.843750 | 0.217742 | 0.346154 |
| CONTRADICT | 0.500000 | 0.015625 | 0.030303 |

The qwen2.5:3b generator was highly conservative: it issued SUPPORT/CONTRADICT verdicts for only 34 of 188 claims (about 18% coverage), with 28 correct among those 34 (about 82% non-abstained accuracy). It abstained on 154 claims, leaving overall accuracy at about 15% when abstentions count as incorrect. CONTRADICT recall was very low: only 1 of 64 contradicting claims received a correct CONTRADICT verdict. Citation validity checks supplied source references and does not imply evidence entailment or explanation correctness.

The full run completed under an explicitly authorized temporary 100-minute runtime gate; the default 90-minute gate is restored. Earlier smoke artifacts are preserved.
