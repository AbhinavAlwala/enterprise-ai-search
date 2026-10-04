# Frozen SciFact results

All 13 JSON artifacts below are intentionally committed historical measurements, not runtime caches. With LF line endings, total size is 5,953,178 bytes (about 5.95 MB); the largest is 2,809,174 bytes (reranked retrieval). Raw corpus/model files remain ignored. No artifact was rerun or modified during M12.

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

## File sizes and SHA-256 (LF-normalized)

These sizes and checksums use UTF-8 JSON bytes with CRLF replaced by LF, matching Git's committed blobs. Windows checkout conversion can change raw file hashes without changing JSON. Normalize only line endings before verifying; do not reserialize JSON. Historical raw hashes inside comparison/checkpoint artifacts retain their original execution meaning and are not rewritten. Future explicitly requested experiments must write new output paths; do not overwrite or rewrite these reports.

| Artifact | LF bytes | LF SHA-256 |
|---|---:|---|
| [scifact_bm25_test.json](scifact_bm25_test.json) | 139603 | `ebf7ae8c5582b1aad764c6dfd9259ad5099af1d09ac1ceb80604a5fd74966d5f` |
| [scifact_claim_verification_smoke.json](scifact_claim_verification_smoke.json) | 44459 | `963460c90a7efd05a54c76928352606b03dca87fb5a1be6b45d616e3584c79a3` |
| [scifact_claim_verification_smoke_m71.json](scifact_claim_verification_smoke_m71.json) | 47003 | `abe0fa8eb8c0b1c3630b84d9117c8e5aafc393d4dbdcf95d195c6a5225d4a42b` |
| [scifact_claim_verification_smoke_m72.json](scifact_claim_verification_smoke_m72.json) | 48036 | `d2be767a81a120e2550efb2ab53f834043c61aef63c7a3d31b648b5f37d2c51d` |
| [scifact_claim_verification_smoke_m73.json](scifact_claim_verification_smoke_m73.json) | 51436 | `de7ef3bfb29e3ff27644a04eefad254ecf91bb763b64632a02fc4dda14e6389d` |
| [scifact_claim_verification_smoke_m74.json](scifact_claim_verification_smoke_m74.json) | 51851 | `fcd62d515b4cb10c6b9f6bb5e5919915e342a03b563978b3d3ec3f833d3fbacd` |
| [scifact_claim_verification_test.json](scifact_claim_verification_test.json) | 1696072 | `7183ba3ed4a5410eaacbcc3b23d5d38233b5afa620a613b33dee27f1f96e329f` |
| [scifact_comparison.json](scifact_comparison.json) | 2662 | `8d736569402c1d6a5d4d3a23975e07be696da05b022c6599c9b221435604e37e` |
| [scifact_dense_test.json](scifact_dense_test.json) | 141939 | `52e2a9273b58b35927f46582f98a26d55b9fa17cb401ed975c71556117ba03c7` |
| [scifact_hybrid_comparison.json](scifact_hybrid_comparison.json) | 5228 | `5d3b5456485c426a04ecc5d974ec5a04695831fa6ef4190d28cb61825ba2048f` |
| [scifact_hybrid_test.json](scifact_hybrid_test.json) | 910532 | `7239dba538c5cf5991ceb9a2237ca1724fc8160707de31f59492dc879ada9bb2` |
| [scifact_reranked_comparison.json](scifact_reranked_comparison.json) | 5183 | `2b74fbd6eb85b0e41be28d6caa7cd83503266659ee51d483ac81ce28be830711` |
| [scifact_reranked_test.json](scifact_reranked_test.json) | 2809174 | `978c191c25ac35558b362591f187fca39039dfc5c292312659bfef18a76ea99c` |
