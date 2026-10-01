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

The implemented scope includes SciFact ingestion/chunking, BM25, exact CPU dense retrieval, embedding caching, document RRF, cross-encoder reranking, test evaluation, and artifact comparison. Preserve frozen BM25/dense/hybrid source, normalization/chunking defaults, and all previous result artifacts. Hybrid uses equal-weight RRF with k=60 and 100 unique candidates per retriever. Rerank exactly the top 50 hybrid documents with pinned `cross-encoder/ms-marco-MiniLM-L6-v2`; deduplicate existing representatives by chunk ID, use the maximum raw passage score, and retain its passage. Do not search additional chunks or blend retrieval scores with reranker scores. Model/settings and candidate depths are baseline choices, not tuned on test metrics. Reuse metrics and the embedding cache. Measure candidate Recall@50 alongside quality metrics; separate model loads, candidate generation, inference, and end-to-end online latency. Generate comparison values/README tables from executed artifacts. Unit tests use fixtures/fake encoders without downloads or real transformer initialization. Ignore data/models/caches; omit corpus text from evaluation JSON. Stop before RAG or other future features without a requested milestone.
