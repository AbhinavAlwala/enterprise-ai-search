# Current architecture

The project is a single Python distribution named `enterprise-ai-search`, with its importable package under `src/enterprise_ai_search/`.

`pyproject.toml` defines package metadata, Python compatibility, and the Hatchling build backend. `uv` manages the local environment and dependency resolution.

The package currently contains only its initializer. There are no services, application components, external integrations, or runtime dependencies.
