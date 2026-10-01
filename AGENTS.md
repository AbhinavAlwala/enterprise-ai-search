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

The implemented scope includes SciFact ingestion, normalization, deterministic chunking, BM25 lexical retrieval, a local CLI, and document-level test-split evaluation. Milestone 2 measures the original Milestone 1 defaults without tuning. Preserve that baseline; changes to retrieval settings require a separately requested milestone. Keep generated evaluation JSON small and reproducible, and report quality separately from latency. Unit tests must use local fixtures and avoid network access. Keep downloaded data and caches out of Git. Do not begin future features without a requested milestone.
