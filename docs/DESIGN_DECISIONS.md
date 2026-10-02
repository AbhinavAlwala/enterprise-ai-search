# Design decisions

## Standard metadata with uv

Use `pyproject.toml` for package metadata and `uv` for dependency and environment management. This keeps configuration in one standard file and provides a lockfile workflow when the environment is initialized.

An alternative is `venv` with pip and requirements files. It requires more separate steps for environment setup and dependency locking. Poetry is another option, but its additional project conventions are unnecessary here.

## Python 3.12

Limit the foundation to Python 3.12 to keep the supported interpreter range small and reproducible. Broader version support can be added when it is tested. A lockfile does not itself pin an exact Python patch version.

## src layout and Hatchling

Place the package under `src/` so that ordinary imports from the repository root depend on installing the package. This helps expose packaging mistakes. A flat layout is simpler to import directly but can conceal installation issues.

Hatchling provides a small, declarative build configuration. Setuptools would also work; its additional flexibility is not needed. Hatchling is a build dependency, not a runtime dependency.

## SciFact without the BEIR stack

Use the real [BEIR SciFact dataset](https://github.com/beir-cellar/beir/wiki/Datasets-available), originally from [AllenAI](https://github.com/allenai/scifact). The archive host failed DNS resolution here, so use the same dataset from BEIR's Hugging Face mirrors: [corpus/queries revision](https://huggingface.co/datasets/BeIR/scifact/tree/dfb5d0e7aa6f2ace386740fdf81d6908a454a636) and [qrels revision](https://huggingface.co/datasets/BeIR/scifact-qrels/tree/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029). The older corpus revision exposes gzip JSONL instead of requiring a Parquet reader.

The four SHA-256 values in `dataset.py` were computed from downloaded bytes and pinned with the revisions. Cached downloads are reverified. This detects byte changes; it is not a cryptographic signature from the dataset authors. Search loads local corpus files without rechecking their checksum; rerun download to verify/repair them. Corpus ingestion ignores BEIR metadata, retaining ID, title, and abstract. Evaluation reads the downloaded query JSONL and test qrels TSV; train qrels are not used for this baseline measurement.

## Word windows and simple tokenization

Use 180 whitespace words with 30-word overlap, configured in `ChunkingConfig`. These are initial choices, not tuned results. Overlap preserves some context across boundaries but repeats content and increases index size. Sentence-aware chunking would improve readability but adds complexity. IDs are stable only for unchanged source text and chunk configuration.

Normalize NFC and whitespace for display, then case-fold and split into Unicode alphanumeric terms for search. No stemming, stopword removal, or NLP dependency is used. Scientific punctuation and hyphenated terms lose structure. Titles are prepended once rather than repeated in every chunk or given a separate weight.

## Explicit BM25

Implement the small scoring formula in the standard library instead of adding rank-bm25 and its NumPy dependency. This makes the calculation visible and avoids variant-specific negative IDF behavior. A library would offer established reuse; our implementation instead requires explicit numeric tests.

For each unique query term occurring in a chunk:

```text
IDF = ln(1 + (N - df + 0.5) / (df + 0.5))
contribution = IDF * tf * (k1 + 1) / (tf + k1 * (1 - b + b * L / avgL))
score = sum of contributions
```

`N` is chunk count, `df` counts chunks containing the term, `tf` is occurrences in this chunk, and `L` is chunk token count. Defaults are `k1=1.2` and `b=0.75`. Positive IDF follows the form documented by [Lucene BM25Similarity](https://lucene.apache.org/core/9_12_3/core/org/apache/lucene/search/similarities/BM25Similarity.html); this is not a claim of identical Lucene scores.

Unique query terms contribute once. Only positive matches are returned; ties use chunk ID. Full scans and sorting are simple enough for this corpus but do not scale like an inverted index. The index is rebuilt on each CLI search. Multiple chunks from one document can occupy search top-k. Evaluation separately maps chunks back to unique documents because qrels refer to documents.

## Measure the frozen baseline

Milestone 2 preserves normalization, tokenization, 180/30 chunking, `k1=1.2`, and `b=0.75`. Evaluation offers no tuning flags. Observing results must not become a reason to optimize against the test set. The existing BM25, text, and model source files are unchanged; the report records their hashes.

The inspected test qrels contain 300 queries and 339 positive query/document pairs, all grade 1, with 23 queries having multiple relevant documents. The query file contains 1,109 records including train queries; evaluating all of them would incorrectly include queries outside the test split. No missing IDs or duplicate pairs were found. Reject malformed rows, inconsistent IDs, duplicate pairs, and queries without positive judgments instead of silently filtering them.

## Document ranking and candidate completeness

Retrieve all positive-score chunks using the existing search method, then keep each parent's first occurrence. This is equivalent to ordering documents by their best chunk, preserving the existing chunk-ID tie order. A fixed multiple such as 10k candidate chunks has no guarantee of ten distinct parents; full candidate retrieval avoids that bias. It costs more memory and result-construction time, which is included in measured query latency. Only matching documents are returned; fewer than ten matches are allowed without padding.

## Metric definitions

- Recall@K: relevant documents in the first K divided by all positively judged documents for that query.
- Reciprocal rank@K: `1 / first_relevant_rank` if a relevant document appears by K, else zero. MRR@10 is its mean over queries.
- nDCG@K: `sum(grade / log2(rank + 1))` for the first K, divided by the same sum for the best possible ordering of all provided grades, truncated at K. Use raw grades, as in [trec_eval's nDCG at cutoffs](https://github.com/usnistgov/trec_eval/blob/main/m_ndcg_cut.c). SciFact's observed binary grades make linear and exponential gain equivalent here.

Treat grade > 0 as relevant for Recall/MRR, and unjudged documents as gain zero. Macro-average across every test-qrels query, including those with no hits. A query with no positive judgments is rejected because recall's denominator and ideal DCG would be undefined. These policies are explicit and tested; no evaluation framework is needed.

## Reproducibility and interpretation

The generated JSON contains aggregate/per-query metrics and top document IDs without corpus text, keeping it small enough to commit. Settings, input/source hashes, Python/OS details, and timing boundaries describe the run. The original BM25/dense reports are frozen and CLI writers reject their paths; reruns require another `--output`. Numeric quality values in README are generated from executed artifacts rather than typed manually.

Quality and latency answer different questions. Recall/MRR/nDCG measure agreement with provided judgments; latency measures local work on an already-built index. Total time includes setup and metric/report assembly. This is one local sequential run without warmup, repeated trials, uncertainty estimates, or production load. Qrels may be incomplete, and abstract retrieval on SciFact does not establish general enterprise search quality or factual correctness. This configuration's chunk-based statistics and parent mapping differ from a whole-document BM25 baseline.

## Verification scope

pytest remains the only direct development dependency. Offline fixture tests validate metric logic and failure cases. The full real-corpus evaluation measures this baseline against provided test judgments, but does not establish scientific correctness or production performance. No environment template, server, or future-feature modules are needed.

## One small dense baseline

Use [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. It is an Apache-2.0 English sentence/short-paragraph encoder with 384 dimensions, approximately 91 MB of weights, and a documented 256-wordpiece truncation limit. Both passages and queries use plain text with an empty prefix. No model sweep or tuning is performed.

sentence-transformers handles tokenizer, transformer inference, and pooling. NumPy handles normalized vectors and exact ranking; PyTorch provides the inference runtime. CPU-only wheels are selected explicitly for Windows/Linux through uv; macOS uses PyPI's supported wheel. The locked sentence-transformers 5.1 series uses Transformers 4.x. No training extras, API keys, ANN libraries, or vector database are needed. CPU batches of 32 and four Torch threads are operational choices, not quality tuning.

The same chunks are preserved for a fair comparison, although the model may truncate them after 256 wordpieces. Whitespace words and wordpieces are different units. We count truncated chunks and disclose this limitation rather than alter the frozen corpus preprocessing. The model's pretraining is external; dataset overlap cannot be ruled out, and these measurements do not establish contamination-free generalization.

## Cache identity and exact cosine search

Save float32 unit vectors and JSON metadata together in one ignored NPZ file. Identity includes model/revision, sequence limit, prompts, batch/thread settings, ordered chunk IDs/parents/text, chunk configuration, and relevant package versions. Shape, dtype, finite unit norms, and vector-byte checksum are checked before reuse. Rebuild stale/corrupt artifacts without using them; atomic replacement preserves an existing file if encoding fails. Changing code that alters encoding semantics should also change the cache schema/settings identity.

A full matrix-vector dot product scans every chunk. With unit vectors this equals cosine, which compares direction rather than magnitude. Sort all scores with chunk-ID tie-breaking. Exact search is simple and complete for this corpus; ANN would trade accuracy for speed at larger scale and is unnecessary here. Dense ranking includes zero/negative similarities without a tuned threshold. Similarity is neither calibrated relevance probability nor factual confidence.

## Compare artifacts, disclose costs

Keep `results/scifact_bm25_test.json` frozen. Dense evaluation reuses its test data, chunking, document deduplication, metric definitions, and macro averaging. The comparison checks matching inputs and policies and computes metric differences from JSON; no quality metrics are hand-maintained in README. Its two examples use a fixed numeric-ID selection rule at cutoff 10 and are illustrative, not a second benchmark.

Report corpus encoding once and cache/index preparation separately from online query latency, which includes query encoding. A warm cache still requires model loading and validation. BM25's preserved latency and dense latency come from separate local runs with different numerical/runtime stacks, so their ratio is descriptive rather than a controlled hardware benchmark. No method is assumed to be universally better; measured results are linked from README after the full evaluation completes.

## Measured Milestone 3 trade-offs

The [generated comparison](../results/scifact_comparison.json) shows higher dense Recall@5/10 but lower MRR@10 and nDCG@10. Dense finds more judged documents within the cutoffs while its early ranking is weaker on average. Its measured online latency is slightly lower in these separate runs, at the expense of model loading and substantial one-time corpus encoding. The [dense report](../results/scifact_dense_test.json) records these costs and the token-limit audit: 3,478 of 8,778 chunks are truncated. No settings were changed in response to the result.

The fixed example rule selects query 1, where dense retrieves relevant document `31715818` at rank 5 and BM25 misses it in the top ten, and query 70, where BM25 retrieves relevant documents `5956380` and `4414547` at ranks 1 and 2 while dense misses both in the top ten. These illustrate differing failures without establishing their causes or universal superiority.

## Document RRF without score calibration

BM25 captures precise token matches; embeddings can capture related meanings with different words. Their measured differing failures motivate combining rankings. Raw scores have different scales and meanings: BM25 depends on token statistics, while cosine compares vector directions. Adding them would impose an arbitrary relative weight without calibration.

Use equal-weight Reciprocal Rank Fusion: `score(d) = sum(1 / (60 + rank(d)))` over lists containing the document. Fixed `k=60` softens the advantage of rank 1 and lets agreement contribute strongly. This follows the [RRF formulation](https://doi.org/10.1145/1571941.1572114); it is not a claim that 60 is optimal here. Alternatives include calibrated score combination or learned fusion, but add choices/training this baseline does not need. No new dependency is required.

Fuse documents because SciFact qrels label documents. Each retriever retains its first/best chunk for each parent, then assigns consecutive document ranks. Repeated windows cannot add votes. Keep both representative passages: the retrievers can prefer different chunks of the same source. Exact fused-score ties use ascending document ID, independent of raw scores.

## Fixed candidates, separate final cutoff

Choose **100 unique documents per retriever before seeing hybrid metrics**, ten times the evaluated final cutoff. This provides a modest pool beyond ten while keeping fusion simple. It is an initial choice, not a completeness guarantee or test-set optimum. Full chunk rankings precede deduplication so repeated passages cannot consume document slots. Candidates outside the top 100 contribute no vote.

The union has at most 200 documents; RRF returns final top-k. Complete retrieval/sorting remains the dominant work on this small corpus. Deeper pools or more efficient candidate generation require a separate experiment; no parameter sweep was performed.

## Measured Milestone 4 trade-offs

The [three-way comparison](../results/scifact_hybrid_comparison.json) reports higher hybrid Recall@5/10, MRR@10, and nDCG@10 than both frozen baselines on these 300 queries, with higher average online latency. Recall@10's gain over dense is modest. The model and embedding cache were reused; online hybrid work includes sequential BM25, query encoding/dense scanning, document extraction, and RRF. Setup is separate. These local runs are not a controlled hardware benchmark.

Example selection is automatic: first numeric query ID per category, with no ranking changes. Query 70 has a hybrid top-ten hit despite dense missing; query 75 moves the first relevant hit from BM25 rank 2/dense rank 3 to hybrid rank 1; query 1 loses dense's rank-5 hit in hybrid's top ten. Agreement can outweigh a valuable single-retriever match, so better averages do not mean improvement for every query. Input/source hashes and compatibility checks preserve provenance. No baseline settings or test-set parameters were changed.

## One cross-encoder after fixed candidate generation

Use [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`: six layers, roughly 22.7 million parameters and 90.9 MB of safetensors weights. The published repository name omits the extra hyphen before 6 in the requested spelling. It is a passage-ranking model trained on MS MARCO, not tuned here for SciFact. Existing sentence-transformers provides the API; dependencies/lockfile remain unchanged.

A bi-encoder processes query/passages independently, enabling cached passage vectors. A cross-encoder processes each pair jointly so tokens can interact across query and passage; this richer signal requires a transformer inference for every pair. Rerank only the first 50 unchanged hybrid documents, as fixed in the milestone request before evaluation. Scoring the corpus would be expensive, and no reranker can retrieve an excluded document. Candidate Recall@50 diagnoses that opportunity separately from final ranking quality.

## Maximum of existing representative passages

Use at most two distinct chunk IDs per document, already selected by the retrievers. A shared chunk is scored once and records both sources. Different chunk IDs remain distinct even if their text happens to match. Choose the max raw cross-encoder score and retain its passage; any strong passage can provide the document's ranking signal. Averaging would penalize one good passage paired with a weaker representative; searching all chunks would change the requested candidate/passage scope and cost.

Max aggregation can favor documents with two opportunities and can amplify a falsely high passage score. Representatives may still omit useful evidence elsewhere. Raw logits are query-specific ranking values, not probabilities or factual confidence, and are not blended with retrieval scores. Exact document ties use document ID; passage ties use chunk ID.

## Conservative CPU inference and honest timing

Use batch size 16 and four Torch threads without optimization/model sweeps. The pair's query, passage, and special tokens share a 512-token budget; longest-first truncation can discard useful content. Dense's original 256-wordpiece truncation remains unchanged. No cross-encoder truncation count is measured in this milestone.

Model loading is one-time preparation. Online timing separates unchanged hybrid top-50 generation and `CrossEncoder.predict` (including tokenization/inference); total online time additionally includes pair construction, validation, max aggregation, and sorting. Dense corpus encoding stays cached. Timings are local sequential observations, not controlled hardware benchmarks. Keep all prior JSON artifacts frozen and use new outputs for reranking/comparison. Store full ID orders plus final-ten passage details to support analysis without copying corpus bodies.

## Measured Milestone 5 trade-off

The [four-way comparison](../results/scifact_reranked_comparison.json) shows higher reranked Recall@5/10, MRR@10, and nDCG@10 than hybrid on this fixed split, with much higher CPU online latency dominated by cross-encoder inference. Candidate Recall@50 leaves some judged relevant documents unavailable; the final ranking recovers less than this candidate ceiling. Exact quality/timing values are generated in README and stored in the [reranked report](../results/scifact_reranked_test.json). No parameters changed after observing results.

Automatic examples select query 128/document `8290953` moving from hybrid rank 9 to reranked rank 1, query 70/document `4414547` falling from 2 to 10, and query 13/document `1606628` absent from the 50 candidates. These demonstrate mixed per-query behavior and candidate limits; they do not explain the model's internal reasoning or prove claims in the passages.

## Thin generation after frozen retrieval

RAG means retrieval-augmented generation: selected source text is supplied to a generator so an answer can use corpus evidence. Retrieval finds passages; generation synthesizes text. Keep these responsibilities separate so an endpoint can change without altering the measured retrieval baseline.

Use a tiny `Generator` protocol and standard-library HTTP client, not a vendor SDK or orchestration framework. The text chat-completions request/response shape follows the [official API reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create). A full URL avoids provider-specific base-path inference. Endpoint/model/key are process environment variables; no dotenv dependency or automatic file loading is needed. The client supports a small non-streaming subset with `max_tokens=512`; some model/provider variants require different parameters and are not verified here. M6 initially had no configured endpoint. M7 executed the existing local qwen2.5:7b endpoint without downloading another model; normal ask settings remain unchanged.

## Five winning passages and one prompt

Use five reranked documents, fixed before any generated examples. Preserve only their already-selected representative passages. More context can increase useful evidence but also adds cost, distractors, and context-window pressure. No extra chunk search, score threshold, automatic token-budget trimming, or model routing is justified in this milestone.

One prompt asks for evidence-only answers, source markers, and an explicit insufficient-evidence response. This is a grounding instruction, not a guarantee; retrieved text can be irrelevant, incomplete, contradictory, or contain misleading instructions. An empty context is sent explicitly, with no fabricated evidence. The model determines insufficiency rather than a retrieval-score threshold. Output cap/timeout are conservative operational defaults, not quality-tuned values.

## Reference validation without claim verification

Number context blocks deterministically and map bracketed integers back to document/chunk IDs. Missing markers or unknown numbers fails validation; keep the answer and report that failure. An uncited insufficiency response also fails citation validation, even though abstaining can be appropriate. Do not silently invent citations or regenerate until validation passes.

Valid reference syntax only establishes that a cited source was supplied. It does not establish that the source supports the nearby claim, that all claims are cited, or that the answer is true. Answer-quality evaluation is deferred. Generation can be nondeterministic despite deterministic context; backend compatibility and context-window capacity must be checked when a real server/model is configured.

## M7: evaluate explicit stances, not arbitrary prose

Free-form answers can express many facts with different wording; objective correctness needs reference facts and an interpretation rubric. SciFact instead gives explicit SUPPORT/CONTRADICT annotations for 188 test claims. We constrain predictions to those labels plus ABSTAIN (insufficient/ambiguous supplied evidence). There is no gold ABSTAIN class in this benchmark. Empty metadata is excluded, not relabeled. Qrels supply relevance, not stance: a relevant source may contradict a claim.

The fixed prompt/settings are selected before real evaluation and are not tuned on individual failures or training annotations. Evidence remains M6's top five representative passages. Temperature zero reduces sampling variability but does not promise identical local-backend outputs.

Metric definitions: overall accuracy is correct/ALL completed claims, including abstentions and failures as incorrect. Non-abstained accuracy is correct/valid SUPPORT-or-CONTRADICT predictions (null if none). Class precision is TP/(TP+FP); recall is TP/(TP+FN), including abstentions/failures in FN; F1 is 2TP/(2TP+FP+FN). Undefined class precision/F1 is zero. Macro F1 weights the two classes equally, while accuracy weights individual claims.

Coverage is valid SUPPORT/CONTRADICT predictions / all claims. Abstention, parsing failure, and generation failure have separate rates; their rates plus coverage sum to one. The 2x3 confusion matrix excludes failures, reported separately by gold class. Citation pass rate uses all claims as denominator, with an additional rate among parsed outputs. Existing validation requires at least one valid numeric citation and no unknown reference, even for ABSTAIN. This checks syntax/source existence only, not evidence entailment.

A gold-annotated document in context does not guarantee its winning passage contains the rationale. Present/absent accuracy therefore cannot identify retrieval versus reasoning failure causally. Original sentence arrays are absent, so we do not split flattened text to invent sentence alignment. SciFact's scientific claims and two gold classes are a narrow proxy for enterprise RAG, not a general correctness or safe-abstention benchmark.

Checkpoints preserve completed records, including failures, and reject changed inputs/code/settings rather than mixing experiments. No silent retries or output repair. Preparation/session durations are accumulated separately from summed query durations; total evaluation time includes checkpoint overhead through the last prediction, excluding the final serialization and CLI startup. Five sorted claims give a rough runtime estimate, not a representative quality sample.

The executed five-claim smoke took 327.80 seconds including 25.80 seconds preparation; its 60.36-second mean online time projected 189.56 minutes for 188 claims. The full run was not launched. All five outputs had unquoted verdict values and failed strict JSON parsing; these are protocol failures, not accepted stances or reasoning-quality measurements. The fixed prompt/settings were not revised after observing them.

## M7.1 protocol correction

The first smoke failed serialization, not the HTTP boundary. Installed Ollama 0.34.1 accepted OpenAI-compatible `response_format` with `type=json_schema`, required verdict/explanation fields, an enum, and no additional properties in a separate non-SciFact capability probe. This uses the existing standard-library client; no SDK, vendor-specific fallback, output repair, or semantic prompt tuning is added.

The fixed claim output cap is reduced from 256 to 128 tokens. Earlier explanations reached 74 whitespace words, making 96 tokens tight once JSON and scientific terms are included. This conservative runtime limit leaves room for the unchanged concise explanation requirement; it is not selected using correctness. A token cap can still truncate an output, which strict parsing will flag. Normal ask retains its 512-token cap and no response_format.

The same five-claim M7.1 smoke produced five valid JSON outputs (two SUPPORT, three ABSTAIN), with zero request/parsing failures. Two outputs passed reference validation; three uncited abstentions were flagged for missing citations. Mean generation was 54.95 s, retrieval/reranking 4.84 s, and online time 59.79 s. Total smoke time including preparation was 320.27 s; projected full runtime was 187.70 minutes. No full evaluation was launched, and these five claims are not a quality benchmark. The original smoke artifact is retained; new results are in results/scifact_claim_verification_smoke_m71.json.

## M7.3 structured citation protocol

The 3b smoke parsed reliably but omitted inline markers. Verification now requires an integer citation array alongside verdict/explanation, using the same constrained generation mechanism. This is an output-protocol refinement; SUPPORT/CONTRADICT/ABSTAIN definitions are unchanged. Explanation markers are ignored for verification. Empty arrays pass for ABSTAIN, fail for asserted verdicts; all provided numbers must exist in the supplied context. Duplicates remain in the raw array and map once in first-seen order. Type errors are flagged without coercion. Reference validity still does not establish entailment. Normal ask retains its historical marker policy.

The same five-claim M7.3 smoke completed with five successful requests, five parsing successes, and five citation-array validation passes (two SUPPORT, three ABSTAIN). Mean retrieval/reranking was 4.34 s, generation 26.24 s, and online time 30.58 s; total smoke time including preparation was 167.17 s. Projected 188-claim runtime was 96.04 minutes. No full run or accuracy-based protocol tuning was performed. Results are preserved separately in results/scifact_claim_verification_smoke_m73.json; valid source references are not evidence-entailment judgments.
