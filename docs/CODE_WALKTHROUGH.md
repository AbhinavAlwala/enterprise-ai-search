# Code walkthrough

- `pyproject.toml` defines the distribution, Python 3.12 requirement, and build backend. The wheel configuration includes the package under `src/`.
- `src/enterprise_ai_search/__init__.py` establishes the importable package. It contains no functions or application logic.
- `.gitignore` excludes local environments, generated artifacts, caches, and local secret files.
- `AGENTS.md` preserves scope, engineering, and documentation rules for future tasks.

## Setup flow

`uv sync` resolves the project, creates `.venv`, and installs the package in editable mode. The import check verifies that the package is discoverable. `uv build` produces a source distribution and wheel under `dist/`.

Understand the distinction between the distribution name (`enterprise-ai-search`) and the Python import name (`enterprise_ai_search`), and between a build dependency and a runtime dependency.

There are no application tests or configured formatting/linting tools yet.
