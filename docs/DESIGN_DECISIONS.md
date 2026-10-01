# Design decisions

## Standard metadata with uv

Use `pyproject.toml` for package metadata and `uv` for dependency and environment management. This keeps configuration in one standard file and provides a lockfile workflow when the environment is initialized.

An alternative is `venv` with pip and requirements files. It requires more separate steps for environment setup and dependency locking. Poetry is another option, but its additional project conventions are unnecessary here.

## Python 3.12

Limit the foundation to Python 3.12 to keep the supported interpreter range small and reproducible. Broader version support can be added when it is tested. A lockfile does not itself pin an exact Python patch version.

## src layout and Hatchling

Place the package under `src/` so that ordinary imports from the repository root depend on installing the package. This helps expose packaging mistakes. A flat layout is simpler to import directly but can conceal installation issues.

Hatchling provides a small, declarative build configuration. Setuptools would also work; its additional flexibility is not needed. Hatchling is a build dependency, not a runtime dependency.

## Minimal foundation

Add no runtime dependencies, feature placeholders, application entry point, or environment template. No current behavior needs them. Tests and formatting/linting tools will be added when a milestone calls for them.
