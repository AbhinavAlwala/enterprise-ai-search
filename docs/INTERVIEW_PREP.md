# Interview preparation

Use these questions to explain the code, the executed evidence, and the limits. Follow [the file walkthrough](CODE_WALKTHROUGH.md) for implementation locations and [result provenance](../results/README.md) for complete measurements.

## Retrieval and evaluation

**1. Why start with BM25?** It is an interpretable lexical baseline with explicit corpus statistics and no model download. Term frequency rewards matches, IDF rewards rarer terms, saturation limits repetition, and length normalization reduces long-passage advantage. BM25 scores are ranking signals, not confidence or truth. Code: `bm25.py`.

**2. What is a document versus a chunk?** A document is a stable source record; a chunk is a searchable passage with a parent ID. We combine title/abstract, normalize NFC/whitespace, and form deterministic 180-word windows with 30-word overlap. Chunk IDs depend on parent and window number. Chunking localizes matching but overlap duplicates content; it is not assumed to improve short abstracts. Code: `models.py`, `text.py`.

**3. Why dense retrieval?** MiniLM embeddings can retrieve paraphrases without shared words. The bi-encoder separately embeds query/passages, enabling corpus caching. We normalize 384-dimensional vectors so dot products equal cosine; exact scanning is simple for this corpus. Semantic similarity can miss precise terms/negation and does not prove agreement. Code: `dense.py`.

**4. How is the embedding cache reproducible?** Its identity includes pinned model/configuration, ordered chunk content/IDs, preprocessing, and encoding runtime versions. Shape, norms, finite values, and checksum are checked; stale/corrupt caches rebuild atomically. NPZ loading disables pickle. Weight revisions, dependency versions, data checksums, and cache identity are different reproducibility layers.

**5. Why RRF instead of adding scores?** BM25 and cosine have incompatible scales. A document gets `1/(60+rank)` per component, with equal weights and 100 unique candidates each. Rank 1 in both contributes `2/61`; missing contributes zero. Parent deduplication prevents overlapping chunks giving extra votes. k/weights were fixed before evaluation. Code: `hybrid.py`.

**6. Why rerank, and what is the cost?** A cross-encoder jointly processes query/passage for richer interaction, unlike independently cached bi-encoder vectors. It scores only the top 50 hybrid documents' existing distinct representatives, taking each parent's maximum raw logit. No extra chunks or score blending. nDCG@10 improved 0.680262 -> 0.708730, while mean online latency rose 93.155 -> 3378.577 ms in separate local CPU runs. Code: `reranker.py`.

**7. Why candidate Recall@50?** It measures how much judged relevant content is available before reranking. The mean was 0.933000; missed candidates cannot be recovered by any reranker. This is a per-query opportunity ceiling, not achieved top-ten quality. Permission filtering can reduce it further without a new measured tenant benchmark.

**8. Explain Recall, MRR, and nDCG.** Recall measures the fraction of judged positives retrieved by a cutoff. MRR emphasizes the first positive's reciprocal rank. nDCG rewards gains across the ranking, discounts later positions, and normalizes by an ideal ranking. We deduplicate chunks to parents because qrels label documents, macro-average all 300 test queries, and retain zero-hit queries. Code: `evaluation.py` and the dense/hybrid/reranked evaluation modules.

**9. What makes the measurements defensible, and limited?** Same test split/policies, frozen defaults, pinned revisions/checksums, per-query records, source/input hashes, deterministic ties, and no test-driven tuning. Incomplete qrels, one small scientific corpus, token truncation, representative selection, and separate-run CPU timings limit generalization. The dense model truncated 3,478 of 8,778 chunks; cross-encoder truncation is not separately audited.

## RAG and claim verification

**10. Why citation validity is not correctness?** `[n]` can map to a supplied document/chunk while the answer misreads or contradicts it. Validation checks source references, not entailment, claim coverage, truth, or prompt-injection resistance. Missing/invalid markers remain visibly flagged while the answer is preserved. Normal ask and M7 use different citation representations. Code: `rag.py`, `claim_verification.py`.

**11. Why separate free-form RAG from claim verification?** Free-form answers need reference facts/rubrics for semantic scoring. M7 instead evaluates 188 explicit SciFact SUPPORT/CONTRADICT metadata labels, with ABSTAIN as a model action. Qrels indicate relevance, not stance; empty metadata is not gold abstention. No sentence-level rationale scoring or external judge is implemented.

**12. Why high conditional accuracy but low coverage?** qwen2.5:3b answered only 34/188 claims, with 28 correct: 0.823529 non-abstained accuracy at 0.180851 coverage. Its 154 abstentions reduce overall accuracy to 0.148936; macro F1 is 0.188228. CONTRADICT recall was 1/64 (0.015625). This measures conservative bounded classification, not high general answer accuracy.

**13. Did structured output solve grounding?** No. JSON schema corrected serialization and a required integer citation array corrected reference protocol, without changing verdict definitions. Request/parsing failures were zero in the full benchmark, but citation validity 0.978723 still does not prove entailment. Gold-document presence 0.952128 does not establish that its winning passage contains the rationale. Code: `generation.py`, `claim_verification.py`.

**14. How are long runs controlled?** Sequential generation, a five-claim runtime gate, input/settings/source fingerprints, atomic checkpoints, and strict resume compatibility. Failures/abstentions remain separate. The completed report took 4981.39 s; an explicitly authorized temporary 100-minute gate enabled it, then 90 minutes was restored. Historical identities/results are never rewritten to match later code. Code: `claim_evaluation.py`.

## Serving, security, and operations

**15. Why authorization before reranking/generation?** Hiding final results is too late once protected text reached a model. The service filters hybrid top 50 before predictor pairs, context, generator, and provenance. Fake backends record their actual inputs; tests cover identical cross-tenant text, missing policies, and source mappings. No backfill means fewer results and possible recall loss. Code: `authorization.py`, `service.py`.

**16. Authentication versus authorization?** Authentication establishes identity; authorization determines access. Here headers assume a trusted gateway verifies/replaces them; direct callers can spoof them. Tenant equality precedes tenant-wide/principal/group grants. Missing ACLs deny access; malformed/missing policies fail closed. The SciFact ACL overlay is synthetic demo metadata, not benchmark ground truth.

**17. Why one Uvicorn worker?** Lifespan loads models/indexes once per process, not across processes. More workers duplicate memory and counters. One worker shares a lock that serializes expensive search/ask, while async health/metrics remain cheap. This improves reuse, not throughput. API request limits/sanitized errors do not amount to production hardening. Code: `api.py`, `service.py`.

**18. Why keep Ollama outside Docker?** The application image contains Python runtime/application dependencies, not a downloaded LLM or dataset. Existing read-only mounts reuse pinned weights; the dense cache can regenerate. `host.docker.internal` reaches host Ollama on Docker Desktop, while container localhost refers to itself. Environment forwarding is explicit; Python does not load `.env`. Code: Dockerfile/Compose.

**19. Why request IDs and structured logs?** A canonical opaque UUIDv4 correlates response/request/error events without deriving it from identity/query. JSON fields are machine-readable. IDs can repeat when caller supplied and are not authentication or metric labels. Logs omit payloads, evidence, identities, ACLs, keys, raw URLs, and exception messages. Code: `observability.py`.

**20. Why bounded process-local metrics?** Fixed route/status buckets prevent unbounded labels and sensitive identity/query leaks. A short lock protects counters/means; restart resets them and workers/replicas are independent. There is no durability, histogram, percentile, alerting, or external collector. `perf_counter` measures durations without wall-clock jumps; request timing includes lock wait/body handling, unlike stage timings. `/metrics` stays inside the trusted boundary.

**21. What does CI prove?** Offline fakes/mocks validate behavior without local assets or generation. Locked sync fixes resolved Python versions; package builds validate packaging; Docker build-only validates construction, not mounts/readiness/host connectivity. CI does no benchmark tuning/evaluation or publication. Hosted workflow success must be observed separately from local command success. Code: `.github/workflows/ci.yml`.

**22. Biggest production limitations and technical debt?** Spoofable headers without a gateway, static ACL snapshots, privileged global indexes/CLI, no timing-side-channel or prompt-injection defense, exact-scan scale limits, serialized inference, incomplete/truncated evidence, poor contradiction recall, process-local telemetry, and no cloud deployment/load testing. Python 3.12 is constrained to one minor version; Docker base tags/OS/build tools are not digest/bit-identical guarantees. One upstream Starlette/HTTPX deprecation warning remains; no broad dependency upgrade was made.
