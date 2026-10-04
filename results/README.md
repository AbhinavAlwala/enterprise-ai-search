# Frozen SciFact results

All 13 JSON artifacts below are intentionally committed historical measurements, not runtime caches. Total size is 6,129,866 bytes (about 6.13 MB); the largest is 2,914,691 bytes (reranked retrieval). Raw corpus/model files remain ignored. No artifact was rerun or modified during M12.

## Scope and provenance

- Four retrieval reports evaluate the same 300 SciFact test queries and document-level qrels with frozen chunking/metric policies; preparation is separate from online timing.
- Two-, three-, and four-way comparisons retain compatible artifact hashes, deterministic examples, and quality/latency differences. README's retrieval table is generated from `scifact_reranked_comparison.json` and was checked against the underlying reports.
- The complete claim-verification report covers only 188 explicitly stance-annotated test claims (124 SUPPORT, 64 CONTRADICT). Five separate smoke artifacts preserve protocol/runtime experiments, including failures; they are not quality benchmarks.
- Claim reports intentionally include public SciFact claim/evidence text, predictions, citation mappings, raw generator outputs, and checkpoint identities. They contain no enterprise tenant ACLs or credentials. Historical source/input/settings identities are preserved even when later serving source changes.

Retrieval Recall/MRR/nDCG measure document relevance; candidate Recall@50 measures the relevant content available to rerank. Citation-reference validation measures supplied source membership, not semantic entailment. Claim accuracy measures explicit bounded stance predictions, not general free-form answer quality. Qrels cannot supply SUPPORT/CONTRADICT stances, and empty metadata is not gold ABSTAIN.

## Complete claim benchmark

qwen2.5:3b, temperature 0, 128-token cap, structured verdict/explanation/citations, unchanged five reranked passages. The temporary authorized 100-minute gate enabled this run; the default 90-minute gate is restored. All values here are read from the executed full artifact, rounded only for display.

| Metric | Measured value |
|---|---:|
| Evaluated claims | 188 |
| Overall accuracy | 0.148936 |
| Non-abstained accuracy | 0.823529 |
| Macro F1 | 0.188228 |
| Abstention rate | 0.819149 |
| Coverage | 0.180851 |
| Parsing failure rate | 0.000000 |
| Generation failure rate | 0.000000 |
| Citation validation pass rate | 0.978723 |
| Gold-document-present rate | 0.952128 |
| Accuracy with gold document present | 0.150838 |
| Accuracy with gold document absent | 0.111111 |
| Total evaluation runtime (seconds) | 4981.39 |

| Gold class | Precision | Recall | F1 |
|---|---:|---:|---:|
| SUPPORT | 0.843750 | 0.217742 | 0.346154 |
| CONTRADICT | 0.500000 | 0.015625 | 0.030303 |

Only 34/188 claims received asserted verdicts, with 28 correct; 154 abstentions yield 18.09% coverage and 82.35% non-abstained accuracy but 14.89% overall accuracy. CONTRADICT recall was only 1/64. Citation validity does not establish entailment, and a gold parent document in context does not prove its selected passage contains the rationale. Present/absent accuracy is diagnostic association, not a causal separation of retrieval and reasoning failure.

## Operational observations are separate

The owner reported M9 Docker runtime verification. M10's synthetic ACL smoke returned disjoint A/B source sets; one real ask used authorized sources but returned an uncited insufficiency answer. M11 verified request-ID propagation, counters, and safe logs without generation. Their one-run timings are recorded in design decisions; they are not replacements for frozen retrieval latency or new answer-quality metrics. Synthetic tenant metadata is separate from SciFact corpus/qrels.

## File sizes and SHA-256

These checksums describe exact JSON bytes. They are independent of each artifact's recorded source/input fingerprints. Future explicitly requested experiments must write new output paths; do not overwrite or rewrite these reports.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| [scifact_bm25_test.json](scifact_bm25_test.json) | 146264 | `54c9fe04580db80d5dce11559ac68fffd6225353490ccdce4a5c268df189c1ca` |
| [scifact_claim_verification_smoke.json](scifact_claim_verification_smoke.json) | 45107 | `0892916eae4c484a46aecc2ede0ad3257f8036a7770be67d09a4b6874e9ff375` |
| [scifact_claim_verification_smoke_m71.json](scifact_claim_verification_smoke_m71.json) | 47723 | `eb1c2630ffc22766a64135df0ac676592dcb5efc28ef63d8cfb424dd823f4267` |
| [scifact_claim_verification_smoke_m72.json](scifact_claim_verification_smoke_m72.json) | 48732 | `1efb96f0e3b674a456ba9f36e3e8f388b06e91bed220929a826d06ac41080be1` |
| [scifact_claim_verification_smoke_m73.json](scifact_claim_verification_smoke_m73.json) | 52328 | `9dc9c88a32cf5d492f6b8207ad57d4b39d1ab37e01f1cc333debb6dc1d8680a7` |
| [scifact_claim_verification_smoke_m74.json](scifact_claim_verification_smoke_m74.json) | 52743 | `d0e51d2b28139af22dc1270ac78885d51bc4d8cb704c2c9a8a3baabc8bc0427c` |
| [scifact_claim_verification_test.json](scifact_claim_verification_test.json) | 1718203 | `b60fa50f3c510c297d85d1c959fd32f62116d5adcad5aa1880db9f8a60e9907c` |
| [scifact_comparison.json](scifact_comparison.json) | 2770 | `1d85ca8c7441e295ab2f6c4732f504dce6fe3199ae7cc74f9745a0604c27d36a` |
| [scifact_dense_test.json](scifact_dense_test.json) | 148658 | `3b08270b868918c99fc021d1fcf9831f522515af78e203ea5db74b3e1c17f4e9` |
| [scifact_hybrid_comparison.json](scifact_hybrid_comparison.json) | 5439 | `fad9515ef5e9bd2f01984b7f3ebe17d974060eb5a8e24719f9de935564ffc22c` |
| [scifact_hybrid_test.json](scifact_hybrid_test.json) | 941861 | `dc1b82e1c0b5c08935f35373bc8b90b04072627dee6fe2eab32a9920e2418365` |
| [scifact_reranked_comparison.json](scifact_reranked_comparison.json) | 5347 | `ebcb1c15cadc0c1fc11694c09131471a41fdec226906890c628a352c4ebe45d7` |
| [scifact_reranked_test.json](scifact_reranked_test.json) | 2914691 | `3ce0263ebc79fcba244469a5013d1b2867d75c963b77819d3d7a164767fb5cf5` |
