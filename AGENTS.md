# Project rules

## Scope and priorities

- Implement only the requested milestone. Do not begin the next milestone automatically.
- Prioritize correctness, readability, simplicity, testability, and reproducibility, in that order.
- Prefer simple, explicit code and clear names. Avoid unnecessary classes and abstractions.
- Use Python type hints and small functions when application code is introduced.
- Add dependencies only when required by the current milestone. Use `uv` and `pyproject.toml`; maintain `uv.lock` once generated.
- Do not create empty folders or placeholder modules for future features.
- Use proper logging in application code. Comments should explain why a choice was made.
- Never fabricate metrics, benchmark results, capabilities, or verification results.
- Keep documentation concise. Avoid filler comments and excessive documentation.

## Milestone completion

1. Implement only that milestone.
2. Run relevant tests. Add tests for implemented behavior when appropriate.
3. Run formatting and linting if configured; report checks that could not run.
4. Update the relevant files under `docs/`.
5. Summarize changed files and explain the implementation simply.
6. State limitations honestly and stop at the requested milestone boundary.

## Documentation responsibilities

- `README.md`: concise, professional, recruiter-facing overview and accurate project status.
- `docs/ARCHITECTURE.md`: current architecture only.
- `docs/DATA_FLOW.md`: data flow through implemented behavior only.
- `docs/CODE_WALKTHROUGH.md`: important files, functions, execution flow, and concepts to understand.
- `docs/DESIGN_DECISIONS.md`: choices, rationale, alternatives, and trade-offs.
- `docs/INTERVIEW_PREP.md`: questions and answers grounded only in implemented features.

The implemented scope includes SciFact ingestion/chunking, BM25, exact dense retrieval/caching, document RRF, cross-encoder reranking, retrieval evaluation/comparison, and a thin RAG context/generation boundary. Preserve frozen retrieval/reranking/preprocessing source and all previous result artifacts. Hybrid uses equal-weight RRF with k=60 and 100 unique candidates per retriever. Rerank exactly 50 hybrid documents with the pinned MiniLM cross-encoder; deduplicate existing representatives by chunk ID and use their maximum raw score. Do not search extra chunks, blend reranker/retrieval scores, or tune on test metrics. RAG uses five winning reranked passages, deterministic numbered context, one grounding prompt, and an independent OpenAI-compatible HTTP client. Citation validation checks source references only; never claim it proves entailment/truth or solves hallucination. Missing/invalid citations must be visibly flagged while preserving the answer. Keep generation endpoint/model/key in environment configuration; never commit/log keys or download a large LLM without explicit approval. No real-generation claim without an executed configured endpoint. Unit tests use fake models/generators and mocked HTTP without network access. Separate preparation, retrieval/reranking, generation, and end-to-end timings; keep frozen retrieval benchmarks separate. Ignore local data/models/caches/secrets, generate README metric tables from executed artifacts, and stop before answer-quality evaluation or future features without a requested milestone.


M7 adds only bounded SciFact claim verification on the 188 explicitly stance-annotated test claims. Preserve normal ask behavior and all frozen retrieval source/artifacts. Gold stance must come from query metadata, never qrels or empty metadata. Use the same five evidence passages, one fixed verification prompt, qwen2.5:7b at temperature 0/max_tokens 128 with the claim-verification JSON schema response_format, and no test-driven prompt tuning. Keep verdict, abstention, schema failure, generation failure, and citation-reference validity separate. Never equate source-document presence or valid references with entailment. No sentence-level evidence scoring or external judges. Full evaluation requires a five-claim real smoke estimate of at most 90 minutes; use sequential requests and fingerprinted atomic checkpoints. README stance metrics require a complete executed 188-claim artifact. Stop after M7.

M7.1 is a protocol/runtime correction only: preserve semantic prompt text, strict parsing, and free-form ask defaults. Rerun only the same five smoke claims into a new artifact; do not launch a full evaluation without a subsequent explicit request, regardless of runtime estimate.

M7.3 claim verification uses qwen2.5:3b with required verdict/explanation/citations JSON. The integer citation array is authoritative; SUPPORT/CONTRADICT require valid supplied sources, while ABSTAIN may use an empty array. Record parse success, verdict validity, citation validity, and source mappings separately. Free-form ask retains its existing inline-marker policy. Preserve verdict definitions and all retrieval/settings; run only the same five smoke claims, never the full benchmark without a new request.

M8 exposes only health, search, and normal M6 ask through FastAPI. Preserve all M1–M7 source, prompts, settings, benchmarks, and artifacts. Keep request/response models in the API layer, initialize models/indexes once per lifespan/process, and inject a concrete service factory for offline tests. Search top-k is bounded to 1–10 and query/question text to 2,000 characters. Keep missing generation configuration separate from retrieval readiness; do not probe the generator in health. Use sanitized errors and standard logging without keys or evidence payloads. Serialize shared CPU inference and separate preparation/stage/handler timings. No Docker, authentication, multi-tenancy, caching, routing, monitoring, or deployment milestone without a new request. Stop the development server after verification.

M9 containerizes only the existing application; keep all M1-M8 Python source, dependency lock/settings, prompts, metrics, and artifacts frozen. Build runtime dependencies with uv from the lockfile, run one non-root Uvicorn worker, exclude host data/models/secrets/results/tests from the image, and mount existing artifacts without duplicating weights. Protect corpus/model directories with read-only mounts while allowing dense cache regeneration. Keep host Ollama external and its endpoint configurable; explicitly forward runtime variables rather than assuming Python loads .env. Healthchecks must stay cheap. Do not chown host files automatically or add unrelated services. Report Docker verification gaps honestly and remove verification containers when executed. No permission-aware retrieval or later milestone without a new request; do not commit M9.

M10 adds authorization only to HTTP search/ask. Identity headers represent a trusted upstream gateway; never describe them as authentication. Require a matching tenant before tenant-wide, explicit-principal, or group access. Missing ACL entries deny access; malformed/missing policy configuration fails closed. Load one immutable policy snapshot per lifespan from AUTHORIZATION_POLICY_PATH, reject ambiguous/duplicate policy fields, and keep errors/logs free of identity and ACL details. Filter the existing hybrid top 50 before cross-encoder inference and unchanged M6 RAG; do not backfill candidates or change scientific retrievers, prompts, settings, metrics, evaluations, corpus, or artifacts. The synthetic demo overlay is separate from SciFact judgments. Preserve unscoped scientific CLI behavior as a privileged local path. Test actual predictor/generator inputs, source mappings, identical cross-tenant text, and denial/error cases offline. Keep one Docker worker and mount policies read-only. No authentication or later milestone without a new request; stop verification servers and do not commit M10.

M11 adds only lightweight HTTP observability. Preserve scientific/authorization behavior, prompts, settings, timing methodology, and artifacts. Correlate requests with opaque UUIDv4 IDs; replace invalid/duplicate headers and return the effective ID. Use structured request/lifecycle/error logs without queries, answers, evidence, document/identity/ACL values, keys, raw URLs, or exception messages. Suppress raw Uvicorn access logs. Metrics use fixed route/status buckets and one short lock, never request/tenant/user/document/query labels. Keep identity denials, availability, unexpected errors, and detectable generator failures distinct. Reuse existing stage timings and measure durations with perf_counter. GET /metrics stays cheap, process-local, non-durable, and independent across workers; document that the current poll appears only after its snapshot. Keep one Docker worker and add no monitoring services/dependencies. Tests use offline fake backends. Stop the verification server, do not commit, and stop after M11 without beginning CI/deployment or another milestone.
