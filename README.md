# Enterprise AI Search & Retrieval Platform

A retrieval engineering project built in tested milestones, with explicit algorithms, reproducible evaluation, and documented trade-offs.

**Status: Milestone 7 implemented; full evaluation deferred by the runtime gate.** A local CLI provides BM25, exact dense search, document RRF, cross-encoder reranking, and a thin retrieval-augmented generation (RAG) layer with source-reference validation. Retrieval is evaluated on the same 300-query SciFact test split. RAG and the HTTP boundary are tested offline with fakes. A separate bounded SciFact stance evaluation adds explicit SUPPORT/CONTRADICT/ABSTAIN output and resumable reports. Frozen retrieval source and artifacts are preserved.

- Verified, revision-pinned [BEIR SciFact](https://github.com/beir-cellar/beir/wiki/Datasets-available) ingestion and deterministic overlapping chunks.
- Explicit BM25 and revision-pinned `sentence-transformers/all-MiniLM-L6-v2` with cached 384-dimensional CPU embeddings.
- Equal-weight document RRF: `k=60`, 100 unique document candidates per retriever, chosen before hybrid evaluation.
- Revision-pinned `cross-encoder/ms-marco-MiniLM-L6-v2`: reranks the top 50 hybrid documents using the maximum score across their existing representative passages, in CPU batches of 16.
- Top-five winning passages form numbered evidence blocks for an independently configured OpenAI-compatible generator. Answers retain source mappings and explicit citation-validation status.
- Offline fixture tests, deterministic ranking, and reports with per-query results, input/source fingerprints, and separate preparation costs.

## Measured results

<!-- BEGIN GENERATED COMPARISON -->

| Metric | BM25 | Dense | Hybrid | Hybrid + Reranker |
|---|---:|---:|---:|---:|
| Recall@5 | 0.709278 | 0.731111 | 0.755167 | 0.766667 |
| Recall@10 | 0.774667 | 0.806222 | 0.807889 | 0.837722 |
| MRR@10 | 0.621536 | 0.597878 | 0.643040 | 0.675521 |
| nDCG@10 | 0.653548 | 0.645294 | 0.680262 | 0.708730 |
| Average online query (ms) | 36.972 | 34.207 | 93.155 | 3378.577 |

Hybrid candidate Recall@50: **0.933000**. This measures relevant documents available before reranking, not final retrieval quality.

<!-- END GENERATED COMPARISON -->

Generated from the [four-way comparison](results/scifact_reranked_comparison.json). Reranking improved all four quality means over hybrid in this experiment, with substantially higher CPU latency. Timings come from separate local runs, not a controlled hardware benchmark. Online reranked time includes candidate generation, pair preparation/inference, aggregation, and final ranking; model loading and corpus preparation are separate.

Artifacts: [BM25](results/scifact_bm25_test.json), [dense](results/scifact_dense_test.json), [hybrid](results/scifact_hybrid_test.json), [reranked](results/scifact_reranked_test.json). The original [two-way](results/scifact_comparison.json) and [three-way](results/scifact_hybrid_comparison.json) comparisons are retained. Automatically selected examples show upward/downward relevant-document movements and a document absent from the candidate pool.

## Local setup

Prerequisites: Python 3.12 and `uv`. Run from the repository root. Initial package, dataset, and model downloads require network access.

```powershell
uv sync --locked --cache-dir .uv-cache
uv run --locked --offline --cache-dir .uv-cache pytest
uv run --locked --cache-dir .uv-cache enterprise-search download
uv run --locked --cache-dir .uv-cache enterprise-search prepare-dense
uv run --locked --offline --cache-dir .uv-cache enterprise-search prepare-reranker
$env:HF_HUB_OFFLINE = "1"
uv run --locked --offline --cache-dir .uv-cache enterprise-search search-reranked "Arterioles have a larger lumen diameter than venules." --top-k 3
uv run --locked --offline --cache-dir .uv-cache enterprise-search compare --hybrid results/scifact_hybrid_test.json --reranked results/scifact_reranked_test.json --output results/scifact_comparison_rerun.json
```

`search` and `search-dense` return chunks; `search-hybrid` returns RRF documents; `search-reranked` returns documents with cross-encoder scores, winning passages, and original hybrid provenance. Defaults are `data/scifact`, `data/dense`, `data/reranker/models`, 180-word chunks with 30-word overlap, and search top-k 5. Evaluations use fixed settings and top ten documents without tuning flags.

Downloaded data, models, and NPZ caches are ignored. After initial downloads, cached retrieval operations work offline. Unset `HF_HUB_OFFLINE` before downloading a missing model; `uv --offline` only controls package access. To rerun frozen evaluations/comparisons, specify a new `--output` path, such as `evaluate-reranked --output results/scifact_reranked_rerun.json`; CLI writers protect all seven previous result files.

## Answer generation

Start an existing OpenAI-compatible chat endpoint serving a model, then set `GENERATION_ENDPOINT` to its full chat-completions URL and `GENERATION_MODEL` to that server's model identifier. Set `GENERATION_API_KEY` only if authentication is required. [.env.example](.env.example) documents these variables; the application reads the process environment and does not automatically load `.env`.

```powershell
$env:GENERATION_ENDPOINT = "http://localhost:8000/v1/chat/completions"
$env:GENERATION_MODEL = "your-served-model"
uv run --locked --offline --cache-dir .uv-cache enterprise-search ask "What does the retrieved evidence say about PPM1D and p53?"
```

The URL/model above are configuration examples. M7 uses the existing local qwen2.5:7b service through its OpenAI-compatible endpoint. `ask` prints JSON containing the answer, selected passages, citation-to-document/chunk mappings, validation status, and separate preparation/retrieval/generation/total timings. The client uses standard-library HTTP with optional bearer authentication, a 60-second timeout, and `max_tokens=512`. Generation requires endpoint connectivity even when package/model caches are offline. No local LLM is downloaded by this project.

## Engineering documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Data flow](docs/DATA_FLOW.md)
- [Code walkthrough](docs/CODE_WALKTHROUGH.md)
- [Design decisions](docs/DESIGN_DECISIONS.md)
- [Interview preparation](docs/INTERVIEW_PREP.md)

## Limitations

The reranker cannot recover documents outside the 50 candidates, and only sees existing representative passages. Max passage aggregation can amplify false positives; some judged documents move downward despite better mean metrics. BM25, cosine, RRF, and cross-encoder scores are ranking signals, not probabilities or factual verification.

The dense encoder truncates 3,478 of 8,778 unchanged chunks. Cross-encoder pairs share a 512-token limit and may also be truncated; that count is not audited. Qrels can be incomplete. Generation sees only five passages; their combined prompt must fit the configured model's context window. Citation validation checks references, not whether claims are supported or true. General free-form answer correctness and explanation entailment remain unmeasured. The M7 benchmark measures only explicit SciFact stances. No server API or production deployment is implemented.

## Bounded claim-verification evaluation

`evaluate-claims` evaluates only the 188 test claims with explicit metadata stances (124 SUPPORT, 64 CONTRADICT), using the unchanged top-five reranked evidence. It validates structured verdicts, records abstentions/failures separately, and checkpoints each prediction. It does not score general RAG correctness or sentence-level citation support. Run the five-claim smoke first; a full run is refused when its estimated duration exceeds 90 minutes. See [execution flow](docs/CODE_WALKTHROUGH.md) and [metric definitions](docs/DESIGN_DECISIONS.md).

The [real five-claim smoke artifact](results/scifact_claim_verification_smoke.json) preserves raw model outputs and failures. Its runtime estimate exceeded the full-run limit, so no 188-claim benchmark results are reported.
