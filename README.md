# Enterprise AI Search & Retrieval Platform

An evaluated retrieval and RAG engineering project with explicit ranking algorithms, permission-aware serving, and reproducible reports. It combines classical information retrieval, transformer models, measured evaluation, and tested service boundaries.

**M1-M12 implemented:** BM25, dense retrieval, hybrid fusion, cross-encoder reranking, citation provenance, bounded claim verification, FastAPI, Docker, document authorization, request tracing, and CI configuration. Scientific source and results are frozen; this is a local engineering project, not a claim of production readiness.

## Architecture

```text
HTTP request -> request ID / validation -> trusted upstream identity
             -> BM25 + dense retrieval -> document RRF top 50
             -> authorization filter -> cross-encoder reranking
             -> search response OR numbered evidence -> generator -> citation validation
             -> structured request logs + bounded process-local metrics
```

The API retains models/indexes once per process. The scientific CLI is a separate privileged path without tenant filtering. Details: [architecture](docs/ARCHITECTURE.md), [execution flow](docs/DATA_FLOW.md), and [code walkthrough](docs/CODE_WALKTHROUGH.md).

## Measured retrieval results

All four methods use the same **300-query BEIR SciFact test split**, deterministic 180-word chunks with 30-word overlap, and document-level relevance judgments. Dense retrieval uses revision-pinned MiniLM embeddings and exact cosine search; hybrid uses equal-weight RRF with k=60 and 100 unique candidates per retriever. The pinned MiniLM cross-encoder reranks 50 documents using their existing representative passages.

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

Generated from the frozen [four-way comparison](results/scifact_reranked_comparison.json). Reranking increased nDCG@10 from **0.680262 to 0.708730**, while mean online latency rose from **93.155 to 3378.577 ms**. Timings came from separate local CPU runs, not controlled hardware trials; preparation is excluded. Reranking cannot recover relevant documents absent from its candidate pool.

## Bounded claim verification

A separate executed **188-claim SciFact benchmark** used qwen2.5:3b, temperature 0, a 128-token cap, structured verdict/explanation/citations, and five reranked evidence passages.

| Metric | Measured value |
|---|---:|
| Overall accuracy | 0.148936 |
| Non-abstained accuracy | 0.823529 |
| Macro F1 | 0.188228 |
| Coverage | 0.180851 |
| Abstention rate | 0.819149 |
| Parsing / generation failure rate | 0 / 0 |
| Citation-reference validation pass rate | 0.978723 |

The model answered only **34/188 claims**, with **28/34 correct**; 154 abstentions leave overall accuracy around 15%. CONTRADICT recall was **0.015625 (1/64)**. High conditional accuracy at low coverage is not high general RAG accuracy. Valid source references do not prove semantic entailment or explanation correctness. [Full report](results/scifact_claim_verification_test.json), [all metrics and artifact provenance](results/README.md).

## Serving and safety boundaries

- FastAPI exposes `/health`, `/metrics`, `/search`, and normal free-form `/ask`. Search top-k is bounded to 1-10; query/question length to 2,000 characters.
- `X-Tenant-ID`, `X-Principal-ID`, and optional `X-Groups` assume identity verified by a trusted upstream gateway. **Headers alone are spoofable, not authentication.** Tenant equality precedes tenant-wide, principal, or group grants. Missing ACLs deny access; invalid policy configuration fails closed. Filtering occurs before reranking, RAG context, generation, and source serialization.
- [The five-document ACL overlay](config/demo_access.json) is **synthetic demo authorization metadata**, separate from SciFact judgments. The same query returned disjoint A/B document sets in the local smoke; this is an operational check, not a tenant retrieval benchmark.
- Request IDs are opaque UUIDv4 values, returned in `X-Request-ID`. Structured logs omit queries, answers, evidence, identities, ACLs, keys, and raw URLs. Metrics use fixed route/status buckets and lock-protected duration aggregates; they reset on restart and are independent across workers. `/metrics` belongs inside the trusted boundary.
- Docker runs one non-root Uvicorn worker. Corpus/model/policy mounts are read-only; dense cache storage can regenerate. Host Ollama stays external and configurable. The image contains locked runtime dependencies, not corpus/weights or secrets.

## Quick start

Use Python **3.12** and uv (**0.11.26** aligns with CI/Docker), from the repository root:

```sh
uv sync --locked
uv run --locked --offline pytest -q
uv run --locked enterprise-search download
uv run --locked enterprise-search prepare-dense
uv run --locked enterprise-search prepare-reranker
```

Only initial preparation requires dataset/model downloads. Tests use small fixtures and fake models/generators; they need no corpus, weights, Ollama, or secrets. For a prepared deployment, `docker compose up --build -d`; query and shutdown commands, Linux/PowerShell identity headers, generation configuration, and request-ID/metrics examples are in the [runbook](docs/RUNBOOK.md).

## Verification and CI

**440 offline tests** cover ranking/metrics, cache integrity, protocol parsing, authorization leakage, tracing, and concurrency. The [GitHub Actions workflow](.github/workflows/ci.yml) uses SHA-pinned actions, Python 3.12, uv.lock, offline tests, distribution builds, whitespace checks, and a Docker **build-only** step. CI does not download SciFact/models, start Ollama/the application, or run scientific benchmarks; initial Python dependency/image pulls still need network access. No formatter/linter is configured, and no packages/images are published. Hosted CI execution remains pending the first push/PR.

## Limitations and further reading

Exact scans and serialized CPU inference limit scale/throughput. Model truncation, incomplete qrels, fixed candidate/evidence cutoffs, and ACL filtering constrain recall. Static policies require restart; the global index/CLI retain privileged corpus access. Authentication, physical tenant isolation, timing-side-channel defenses, prompt-injection defenses, durable metrics, and general answer correctness are not implemented. Locked Python dependencies do not guarantee bit-identical Docker images: base tags, OS packages, and build tools have separate reproducibility limits.

[Design decisions](docs/DESIGN_DECISIONS.md) explain trade-offs and historical smokes; [interview preparation](docs/INTERVIEW_PREP.md) covers implemented behavior and measured results. All 13 frozen JSON reports are intentionally retained (about 5.95 MB with LF line endings); operational smoke timings and synthetic ACLs are distinct from scientific benchmarks. No LICENSE has been selected.
