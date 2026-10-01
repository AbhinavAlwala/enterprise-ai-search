# Code walkthrough

- `models.py`: `Document` represents a source record; `Chunk` represents a searchable passage with a parent ID; `SearchResult` adds rank and score. Frozen dataclasses discourage accidental record mutation.
- `dataset.py`: `FILES` fixes download provenance and SHA-256 values. `download_scifact` verifies cached/downloaded bytes and replaces files only after verification. `load_corpus` separately handles plain or compressed JSONL without network access. `load_queries` reuses its ID/text validation; `load_qrels` parses provided TSV grades without inventing labels.
- `text.py`: `ChunkingConfig` centralizes size/overlap and validates them. `normalize_text` keeps readable case/punctuation. `tokenize` creates case-insensitive search terms. `chunk_documents` creates deterministic windows and validates document IDs.
- `bm25.py`: `BM25Index.__init__` computes corpus statistics once. `search` computes each chunk's score, filters nonmatches, sorts, and returns result records. This is a concrete index, not a framework for future retrievers.
- `evaluation.py`: `ranked_document_ids` retains the first occurrence of each parent. `recall_at_k`, `reciprocal_rank_at_k`, and `ndcg_at_k` calculate individual-query metrics. `evaluate_scifact` validates IDs, builds the default index, evaluates every test query, macro-averages metrics, and returns the report. File hashes identify the data and implementation used.
- `cli.py`: `main` parses commands. Download does not build an index. Search loads documents, creates chunks/index, executes the query, and serializes dataclasses to JSON. Evaluate runs `evaluate_scifact`, writes its report, and prints quality/timing fields. Errors exit with status 2; logging uses stderr.
- `pyproject.toml` declares the CLI entry point and pytest development dependency; `uv.lock` records resolved versions. `.gitignore` excludes downloaded data, environments, and generated files. `AGENTS.md` preserves milestone boundaries.

## Execution and checks

`uv sync --locked --cache-dir .uv-cache` installs the package in editable mode and development dependencies. The installed `enterprise-search` entry point calls `cli.main`; `python -m enterprise_ai_search.cli` reaches the same function.

`uv run --locked --cache-dir .uv-cache pytest` runs local fixtures. Retrieval tests include a manually calculated score, overlap boundaries, no matches, deterministic ties, and invalid parameters. Dataset tests cover malformed records, checksums, cache repair, and failed-download cleanup using mocked responses.

`tests/test_evaluation.py` checks document deduplication, candidate exhaustion, deterministic ties, multiple positives, no hits, cutoff boundaries, linear graded nDCG with a full ideal ranking, malformed qrels, inconsistent IDs, macro aggregation, and CLI-generated JSON. No evaluation test downloads data.

`uv run --locked --cache-dir .uv-cache enterprise-search evaluate` runs the entire local test split and produces the committed baseline artifact. Quality values in that artifact come from execution, not hand-maintained constants. No new dependency was introduced for evaluation.

The real-corpus smoke query is in README. `uv build --offline --cache-dir .uv-cache` works after build dependencies are cached and produces source/wheel artifacts under `dist/`. No formatter or linter is configured.

Understand the distribution name (`enterprise-ai-search`) versus import name (`enterprise_ai_search`), and why tests exercise installed source through the src layout.

## Dense implementation

- `dense.EmbeddingConfig`: one model/revision, dimension, batch size, maximum sequence length, and CPU thread count in one place.
- `load_encoder`: lazy model/runtime imports and CPU-only construction. BM25 commands do not load a transformer.
- `normalize_vectors`: validates a finite 2D matrix and divides every row by its L2 norm. Zero rows are rejected because their direction is undefined.
- `cache_identity`, `read_embedding_cache`, and `write_embedding_cache`: fingerprint ordered chunk content/configuration, validate metadata/shape/norms/checksum, and atomically replace the NPZ. Loading uses `allow_pickle=False`.
- `prepare_embeddings`: reuse a validated cache or audit token lengths, encode batches, normalize, and save. It returns vectors plus measured cache/encoding information.
- `DenseIndex.search`: encode a single query, compute `embeddings @ query_vector`, lexicographically sort by negative score then chunk ID, and return ranked chunks.
- `dense_evaluation.evaluate_dense_scifact`: prepare the model/index once, retrieve each test query, and call existing `ranked_document_ids`, recall, reciprocal-rank, and nDCG functions. No retriever interface or factory is introduced.
- `comparison.compare_reports`: validate two artifacts, subtract their measured metrics, and select two opposing hit/miss examples when available. It does not run retrieval or choose a model.

Dense CLI commands are `prepare-dense`, `search-dense`, and `evaluate-dense`; `compare` uses saved reports. `--cache-dir` changes the local dense cache, and `--rebuild` regenerates vectors. After the first model download, setting `HF_HUB_OFFLINE=1` prevents Hub checks for cached model files. `uv --offline` controls package access separately.

`tests/test_dense.py` uses tiny vectors and fake encoders to check shapes, zero/nonfinite vectors, cosine order, ties, top-k, stale/corrupt caches, and truncation auditing. `tests/test_dense_evaluation.py` checks complete integration with existing metrics, cache reuse, artifact comparison, and BM25 report overwrite protection. Neither test file downloads a model.
