# Current architecture

The project is a single Python 3.12 package with a local CLI, lexical/exact dense indexes, document RRF, a cross-encoder reranker, and an independent HTTP answer-generation boundary.

```text
download command -> pinned BEIR mirror files -> local data/scifact/

search command -> corpus loader -> normalization/chunking -> BM25 index
               -> query tokenization -> scoring/ranking -> JSON results

evaluate command -> corpus + queries + test qrels -> validate source IDs
                 -> same chunking/BM25 index -> all matching chunk results
                 -> ranked unique documents -> metrics -> results/*.json

prepare-dense -> same chunks -> pinned CPU encoder -> ignored embedding cache
search-dense -> query encoder -> normalized matrix/vector dot products -> ranked chunks
evaluate-dense -> same test queries/qrels -> dense chunks -> existing document mapping/metrics
compare -> preserved BM25 report + dense report -> validation -> comparison JSON/examples
search-hybrid -> BM25 chunks + dense chunks -> unique document candidates -> RRF -> documents
evaluate-hybrid -> same test queries/qrels -> hybrid documents -> existing metrics -> JSON
compare --hybrid -> three saved reports -> compatibility checks -> comparison/examples
prepare-reranker -> pinned cross-encoder -> ignored local model cache
search-reranked -> unchanged hybrid top 50 -> representative pairs -> cross-encoder
                -> max passage score per document -> sorted top-k documents
evaluate-reranked -> same test split/metrics + candidate Recall@50 -> new JSON
compare --hybrid --reranked -> four saved reports -> validation -> comparison/examples
ask -> unchanged hybrid/reranker -> five winning passages -> numbered context/prompt
    -> configured chat-completions HTTP endpoint -> answer + citation/source validation
```

- `models.py`: frozen `Document`, `Chunk`, and `SearchResult` dataclasses.
- `dataset.py`: network download with checksums, and a separate offline JSONL loader.
- `text.py`: normalization, shared tokenization, and configurable word windows.
- `bm25.py`: chunk term counts, IDF statistics, BM25 scoring, and top-k ranking.
- `cli.py`: commands, argument validation, logging, and JSON output.
- `evaluation.py`: chunk-to-document ranking, explicit metrics, and the complete test-split run.
- `dense.py`: revision-pinned model loading, batched encoding, cache identity/integrity checks, and exact cosine ranking.
- `dense_evaluation.py`: dense preparation timings and the same test-query evaluation policies, calling M2's existing mapping/metric functions.
- `comparison.py`: checks artifact compatibility, calculates measured differences, and selects deterministic real examples.
- `hybrid.py`: retains representative passages, assigns compact document ranks, and fuses two rankings with fixed RRF settings. `HybridIndex` directly composes the existing indexes.
- `hybrid_evaluation.py`: reuses dense preparation/cache and existing metrics for the full hybrid test run.
- `reranker.py`: fixed model configuration, representative deduplication, batched pair scoring, and max passage aggregation with provenance.
- `reranked_evaluation.py`: full test-split evaluation with candidate recall and separate candidate/inference/online timings.
- `rag.py`: evidence/context models, the grounding prompt, source-reference validation, and thin query-to-answer composition.
- `generation.py`: a small `Generator` protocol and standard-library `HttpGenerator`, configured through environment variables. Tests inject a fake implementation.

BM25 and HTTP generation use the standard library. Dense retrieval needs NumPy, sentence-transformers, and CPU PyTorch plus their required dependencies. pytest is a development dependency; Hatchling builds the distribution. `uv.lock` records environment resolution. Each dense command loads the encoder and verifies/reuses or regenerates chunk embeddings. Evaluation builds one index for all queries. Data, model weights, and binary caches remain ignored. No SDK, orchestration framework, server API, or vector database was added.

Reranking reuses the installed sentence-transformers `CrossEncoder` API without new dependencies. Its weights live in `data/reranker/models/`. Pair scoring runs online; passage representations cannot be cached independently of the query as dense embeddings can. Frozen retriever modules remain unchanged.

Generation runs in one non-streaming request after retrieval; it cannot change candidate selection or request more passages. No real endpoint/model was configured during M6, so generation integration is verified with fakes/mocked HTTP only. Answer evaluation is not implemented.
