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


M7 adds only bounded SciFact claim verification on the 188 explicitly stance-annotated test claims. Preserve normal ask behavior and all frozen retrieval source/artifacts. Gold stance must come from query metadata, never qrels or empty metadata. Use the same five evidence passages, one fixed verification prompt, qwen2.5:7b at temperature 0/max_tokens 256, and no test-driven prompt tuning. Keep verdict, abstention, schema failure, generation failure, and citation-reference validity separate. Never equate source-document presence or valid references with entailment. No sentence-level evidence scoring or external judges. Full evaluation requires a five-claim real smoke estimate of at most 90 minutes; use sequential requests and fingerprinted atomic checkpoints. README stance metrics require a complete executed 188-claim artifact. Stop after M7.
