# Interview preparation

**What does the project do today?**

It provides an installable Python package and documented engineering conventions. It does not yet perform search or AI tasks.

**Why use a src layout?**

It separates the package from the repository root and makes the normal import check exercise installation rather than an accidental root-level import.

**What is the role of pyproject.toml?**

It declares package metadata, supported Python versions, runtime dependencies, and the build backend.

**Why is Hatchling present if there are no runtime dependencies?**

It builds the wheel and source distribution. The package itself does not require Hatchling at runtime.

**How is reproducibility supported?**

The project specifies Python 3.12 and uses uv's lockfile workflow. Once generated and committed, `uv.lock` supports locked environment setup. Exact Python patch versions and build dependency resolution are not fully pinned by this foundation.

**What has not been validated yet?**

There is no implemented retrieval behavior, evaluation dataset, benchmark, or application test suite. Package import and build checks are the appropriate foundation checks.
