# Current architecture

The project is a single Python 3.12 package with a local CLI and an in-memory lexical index.

```text
download command -> pinned BEIR mirror files -> local data/scifact/

search command -> corpus loader -> normalization/chunking -> BM25 index
               -> query tokenization -> scoring/ranking -> JSON results

evaluate command -> corpus + queries + test qrels -> validate source IDs
                 -> same chunking/BM25 index -> all matching chunk results
                 -> ranked unique documents -> metrics -> results/*.json
```

- `models.py`: frozen `Document`, `Chunk`, and `SearchResult` dataclasses.
- `dataset.py`: network download with checksums, and a separate offline JSONL loader.
- `text.py`: normalization, shared tokenization, and configurable word windows.
- `bm25.py`: chunk term counts, IDF statistics, BM25 scoring, and top-k ranking.
- `cli.py`: commands, argument validation, logging, and JSON output.
- `evaluation.py`: chunk-to-document ranking, explicit metrics, and the complete test-split run.

There are no runtime dependencies. pytest is a development dependency; Hatchling builds the distribution. `uv.lock` records the environment dependency resolution. Search builds an index per invocation; evaluation builds one index for all test queries. Dataset files remain ignored, while the small generated evaluation report is appropriate to commit. There are no APIs, databases, embeddings, or model calls.
