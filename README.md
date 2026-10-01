# Enterprise AI Search & Retrieval Platform

A retrieval engineering project built in tested milestones, with explicit algorithms, reproducible evaluation, and documented trade-offs.

**Status: Milestone 4 implemented.** A local CLI supports BM25 lexical search, exact dense retrieval, and document-level hybrid search with Reciprocal Rank Fusion (RRF). All three are evaluated on the same 300-query SciFact test split. The original BM25/dense implementations and result artifacts are preserved.

- Verified, revision-pinned [BEIR SciFact](https://github.com/beir-cellar/beir/wiki/Datasets-available) ingestion and deterministic overlapping chunks.
- Explicit BM25 and revision-pinned `sentence-transformers/all-MiniLM-L6-v2` with cached 384-dimensional CPU embeddings.
- Equal-weight document RRF: `k=60`, 100 unique document candidates per retriever, chosen before hybrid evaluation.
- Offline fixture tests, deterministic ranking, and reports with per-query results, input/source fingerprints, and separate preparation costs.

## Measured results

<!-- BEGIN GENERATED COMPARISON -->

| Metric | BM25 | Dense | Hybrid |
|---|---:|---:|---:|
| Recall@5 | 0.709278 | 0.731111 | 0.755167 |
| Recall@10 | 0.774667 | 0.806222 | 0.807889 |
| MRR@10 | 0.621536 | 0.597878 | 0.643040 |
| nDCG@10 | 0.653548 | 0.645294 | 0.680262 |
| Average online query (ms) | 36.972 | 34.207 | 93.155 |

<!-- END GENERATED COMPARISON -->

Generated from the [three-way comparison](results/scifact_hybrid_comparison.json). Hybrid improved all four quality metrics over both frozen baselines in this experiment, at higher online latency. Recall@10's gain over dense is small. Timings are from separate local runs, not a controlled hardware benchmark; online hybrid timing includes both retrieval paths, document extraction, and fusion, excluding model loading and corpus preparation.

Artifacts: [BM25](results/scifact_bm25_test.json), [dense](results/scifact_dense_test.json), [hybrid](results/scifact_hybrid_test.json). The [original two-way comparison](results/scifact_comparison.json) is retained. Automatically selected examples include an improved early hit and a hybrid miss despite a dense hit; see the three-way artifact.

## Local setup

Prerequisites: Python 3.12 and `uv`. Run from the repository root. Initial package, dataset, and model downloads require network access.

```powershell
uv sync --locked --cache-dir .uv-cache
uv run --locked --offline --cache-dir .uv-cache pytest
uv run --locked --cache-dir .uv-cache enterprise-search download
uv run --locked --cache-dir .uv-cache enterprise-search prepare-dense
$env:HF_HUB_OFFLINE = "1"
uv run --locked --offline --cache-dir .uv-cache enterprise-search search-hybrid "Activation of PPM1D suppresses p53 function." --top-k 3
uv run --locked --offline --cache-dir .uv-cache enterprise-search evaluate-hybrid
uv run --locked --offline --cache-dir .uv-cache enterprise-search compare --hybrid results/scifact_hybrid_test.json --output results/scifact_hybrid_comparison.json
```

`search` and `search-dense` return chunks; `search-hybrid` returns unique documents with RRF scores and each retriever's representative passage/rank when present. Defaults are `data/scifact`, `data/dense`, 180-word chunks with 30-word overlap, and search top-k 5. Hybrid evaluation uses the fixed candidate depth and top ten documents without tuning flags.

Downloaded data, models, and NPZ caches are ignored. After initial downloads, cached operations work offline. Unset `HF_HUB_OFFLINE` before downloading a missing model. To rerun either frozen baseline, specify a new path, such as `evaluate --output results/scifact_bm25_rerun.json` or `evaluate-dense --output results/scifact_dense_rerun.json`; CLI report writers reject the two original paths.

## Engineering documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Data flow](docs/DATA_FLOW.md)
- [Code walkthrough](docs/CODE_WALKTHROUGH.md)
- [Design decisions](docs/DESIGN_DECISIONS.md)
- [Interview preparation](docs/INTERVIEW_PREP.md)

## Limitations

RRF rewards agreement and can demote a useful single-retriever hit. The candidate cutoff can exclude relevant documents. BM25 requires shared tokens; dense similarity can blur terminology and negation. Scores are ranking signals, not probabilities or factual verification.

MiniLM truncates 3,478 of 8,778 unchanged chunks at its wordpiece limit. Qrels can be incomplete. Exact scans and full chunk rankings fit this corpus but do not establish enterprise-scale performance. No reranking, generation, server API, or production deployment is implemented.
