# Current architecture

The project is a single Python 3.12 package with a local CLI and an in-memory lexical index.

```text
download command -> pinned BEIR mirror files -> local data/scifact/

search command -> corpus loader -> normalization/chunking -> BM25 index
               -> query tokenization -> scoring/ranking -> JSON results
```

- `models.py`: frozen `Document`, `Chunk`, and `SearchResult` dataclasses.
- `dataset.py`: network download with checksums, and a separate offline JSONL loader.
- `text.py`: normalization, shared tokenization, and configurable word windows.
- `bm25.py`: chunk term counts, IDF statistics, BM25 scoring, and top-k ranking.
- `cli.py`: commands, argument validation, logging, and JSON output.

There are no runtime dependencies. pytest is a development dependency; Hatchling builds the distribution. `uv.lock` records the environment dependency resolution. The index is rebuilt per search invocation; nothing is persisted beyond dataset files. There are no APIs, databases, embeddings, or model calls.
