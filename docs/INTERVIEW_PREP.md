# Interview preparation

**What is information retrieval, and what is implemented here?**

Information retrieval selects and ranks stored content for a user's information need expressed as a query. Here a local CLI ranks SciFact chunks using lexical BM25 or dense cosine similarity, combines document rankings with RRF, and reranks candidates. A separate RAG layer can request answers from a configured generator; M7 adds bounded SciFact claim verification through the existing local qwen2.5:7b endpoint.

The current pipeline can also rerank the top 50 hybrid documents with a cross-encoder over existing representative passages.

**How does a document differ from a chunk? Why chunk?**

A document is an original source record with a stable source ID, title, and abstract. A chunk is a smaller searchable passage carrying its parent ID. Chunking makes matching more local in long texts. Overlap helps retain boundary context but duplicates evidence; SciFact abstracts are already short, so chunking is not assumed to improve their retrieval quality.

**What does lexical search mean?**

It matches shared tokens rather than inferred meanings. Our query and corpus share case-folded alphanumeric tokenization. Synonyms and morphological variants can fail to match.

**What are TF and IDF?**

TF counts a term's occurrences in a chunk. IDF gives more weight to terms appearing in fewer chunks. Document frequency counts presence once per chunk, not every occurrence. Our index treats chunks as the scoring documents.

**What is BM25's intuition?**

Rare shared terms help more. Repeated occurrences help with diminishing returns, controlled by `k1`. Length normalization, controlled by `b`, reduces the advantage of long chunks that have more opportunities to match. See the exact formula in DESIGN_DECISIONS.

**What does top-k mean?**

Return up to k highest-scoring matching chunks, with one-based ranks. There may be fewer than k matches, or several chunks from one document. Ties are ordered by chunk ID.

**What does a BM25 score mean?**

It is a query-specific lexical ranking signal. It is not a probability, confidence, factual truth, or a quality metric. Scores should not be compared directly across queries, corpora, or chunking choices. A high-scoring passage can contradict the query or be irrelevant.

**Trace a query through the code.**

`cli.main` reads local documents with `dataset.load_corpus`, creates windows with `text.chunk_documents`, builds `bm25.BM25Index`, calls `search`, and prints `SearchResult` records as JSON. Query files and qrels do not influence ordinary CLI search.

**How are edge cases and reproducibility handled?**

Empty corpora/queries and unmatched terms return no results. Invalid parameters, malformed records, duplicate IDs, and empty-token index chunks are rejected. Revision-pinned downloads have checksums; `uv.lock` fixes environment dependency versions. Tests are offline, and ties/IDs are deterministic for unchanged data and configuration. Python patch versions and build dependencies are not fully pinned.

**What evidence supports the implementation, and what remains unknown?**

Tests check a manual numeric score, chunk boundaries, tokenization, download failures, and evaluation formulas. Both a real-corpus CLI query and the full test-split evaluation were executed. The generated report measures quality against SciFact judgments; generalization, statistical uncertainty, and production performance remain unknown.

**Why aren't a few plausible example results sufficient? What are qrels?**

Examples can be cherry-picked and do not show how many relevant documents were missed. Qrels are dataset-provided mappings from query IDs to document IDs and relevance grades. We measure every judged test query, not labels invented from our own results.

**Why map chunks back to documents, and how?**

SciFact labels original documents, while search returns passages. Repeated chunks must not count as repeated relevant documents. Retrieve all matching chunks, retain each parent's first occurrence in rank order, then assign consecutive document ranks. This preserves each parent's best chunk and avoids missing parents after an arbitrary candidate cutoff.

**Explain Recall@5, Recall@10, MRR@10, and nDCG@10.**

Recall asks what fraction of all known relevant documents appears within the cutoff. MRR emphasizes how soon the first relevant document appears. nDCG considers all relevant hits in the top ten, discounts later ranks, and divides by the ideal ranking's gain. With relevant documents A and B and results X, A, B: Recall@2 is 1/2; reciprocal rank@3 is 1/2; nDCG@3 is `(1/log2(3) + 1/log2(4)) / (1 + 1/log2(3))`.

**How do zero hits and multiple relevant documents affect aggregation?**

No-hit queries receive zero, not exclusion. Multiple positives all contribute to recall's denominator and ideal DCG, but reciprocal rank uses only the first hit. Each query receives equal weight in the macro mean. Duplicate document ranks are rejected by metric functions.

**Trace evaluation and distinguish quality from latency.**

`evaluate_scifact` loads corpus/queries/test qrels, validates IDs, builds the unchanged default index, searches each test query, deduplicates documents, computes metrics, and macro-averages them. The CLI saves JSON. Quality is agreement with judgments; average query latency is search plus deduplication with the index already built. Total time additionally includes loading, indexing, validation, metrics, and report assembly, excluding the file write.

**What limits the conclusions? Why not tune now?**

Unjudged documents are treated as nonrelevant, so incomplete judgments can penalize useful results. This small scientific abstract dataset differs from enterprise documents. Full candidate scans are practical here but not scalable. Timing is one local run. Tuning against the test measurements would contaminate an honest baseline; a separately designed experiment should use a suitable development split.

**What is an embedding, and why can similar text have nearby vectors?**

An embedding is a fixed-length numerical representation. MiniLM maps text into 384 coordinates. Contrastive training brings related examples closer and pushes unrelated examples apart, so vector direction can capture relationships beyond shared words. This learned similarity is imperfect and does not prove factual agreement.

**How does dense search differ from BM25 here?**

BM25 explicitly scores shared token frequency, rarity, and length. Dense search encodes both query and chunk with one trained model, then ranks vector similarities. It can retrieve paraphrases without identical words but may blur precise scientific terms or negation. Both methods return the same chunk records and use the same document-level evaluation.

**What is cosine similarity? Why normalize?**

Cosine is `dot(a, b) / (norm(a) * norm(b))`: it compares angle/direction. Dividing vectors by their L2 norms makes the dot product equal cosine and prevents magnitude from dominating. Zero vectors have no defined direction and are rejected. Scores are not probabilities; floating-point rounding can slightly exceed mathematical bounds.

**What happens offline versus online? Why cache?**

Offline work loads the model and batch-encodes all chunks once. Online work encodes one query and compares it against cached chunk vectors. Repeating thousands of document encodings for every query would waste CPU time. The cache must match the model revision, ordered chunks, preprocessing configuration, and encoding settings; the code also checks vector integrity.

**Why use exact search instead of ANN?**

The corpus fits in a small matrix, so scanning every row is understandable and avoids approximation loss. ANN searches a reduced candidate space to improve speed at scale, at the cost of possible missed neighbors and additional index/configuration complexity. This milestone uses no ANN library.

**What limits the BM25-versus-dense conclusions?**

Model truncation may drop chunk tails. The embedding model is general-purpose and not tuned on this test set. Qrels may be incomplete, and document-level evaluation does not prove the selected passage contains evidence. Online timings exclude setup, while model loading/corpus encoding are reported separately. Read the generated comparison for this dataset's actual improvements/regressions; examples cannot establish universal superiority.

**What did this actual comparison show?**

Dense improved recall at both cutoffs, while BM25 retained higher MRR and nDCG. This means finding more relevant documents within ten did not imply placing the first/all relevant documents earlier. Query 1 is a measured dense-only top-ten hit; query 70 is a BM25-only hit. The exact query text, document ranks, and metric differences come from the generated comparison artifact. This is evidence about this fixed experiment, not a reason to tune on test judgments.

**Why combine BM25 and dense retrieval? Why not add their scores?**

They expose different signals and measured failures: shared terminology versus learned semantic similarity. BM25 and cosine have different scales and interpretations, so their raw sum would have an arbitrary balance. Rank fusion uses ordering without assuming comparable score units.

**Explain our RRF calculation with an example.**

Each document receives `1/(60 + rank)` from each candidate list where it occurs. Rank 1 in both gives `2/61`; rank 1 in only one gives `1/61`. Agreement is rewarded and rank advantages are softened. Missing means zero contribution. RRF scores are not probabilities; k=60 and equal weights are fixed, not tuned.

**Why fuse documents instead of chunks? Which passage is retained?**

Qrels refer to parent documents. Overlapping chunks should not give one source multiple votes. Each retriever's first chunk is the parent's representative, then parents receive compact ranks. BM25 and dense may retain different passages; both are included when available.

**What is candidate generation versus final ranking?**

Each component supplies at most 100 unique documents; RRF sorts their union and selects final top-k (ten for evaluation). Only 100 chunks could yield far fewer parents, so we first retrieve complete chunk rankings. The document cutoff was chosen before evaluation and can still exclude useful documents.

**Trace hybrid search and explain deterministic ties.**

`HybridIndex.search` calls existing BM25/dense searches, `document_candidates` deduplicates parents and retains passages, and `reciprocal_rank_fusion` sums rank contributions. Descending RRF score then ascending document ID determines final order. Component ties still follow chunk ID. No labels or model tuning enter this flow.

**What did the three-way experiment show, and where did hybrid fail?**

All four hybrid quality means improved over both baselines in this run, while online latency increased. Query 75's first relevant hit improved from ranks 2/3 to 1. Query 1's dense rank-5 hit fell outside hybrid's top ten. RRF agreement/candidate cutoffs can demote single-source hits; mean improvement is not universal improvement.

**What should you understand before Milestone 5?**

Understand lexical/semantic complementarity; incompatible raw-score scales; reciprocal-rank contributions and k; chunk-to-document deduplication and representatives; candidate depth versus final top-k; deterministic ties; recall versus early-rank metrics; and online latency versus preparation costs. Explain each using this code and its executed reports.

**How does retrieval differ from reranking in this code?**

Retrieval searches the corpus and supplies 50 hybrid candidate documents. Reranking scores their existing query/passage pairs and changes only their order. It never fetches additional documents or passages. `HybridIndex.search` generates candidates; `rerank_candidates` scores and sorts them.

**Bi-encoder versus cross-encoder: why does cost differ?**

The bi-encoder separately embeds query/passages, letting corpus vectors be computed once and cached. The cross-encoder jointly processes both texts so query/passages interact during transformer inference. It must repeat that work per pair per query; CPU batches reduce overhead without removing the cost.

**What does candidate Recall@50 tell us?**

For each query, count judged relevant documents in the 50 candidates and divide by all judged positives, then macro-average. This measures the relevant content available before reranking. If a relevant document is missing, its score is never computed and reranking cannot recover it. Even perfect reranking cannot overcome that candidate ceiling, and the final cutoff can constrain recall further.

**Why score two passages and take the maximum?**

BM25 and dense may select different chunks. Score each distinct chunk ID once; choose the highest score as the parent's signal and retain its passage. This lets either representative supply evidence. It does not establish truth, and max can amplify false positives or favor documents with two passages.

**What determines final scores and deterministic output?**

Raw cross-encoder logits alone determine order. No BM25, cosine, or RRF score is added. Document ties use ascending document ID; passage ties use ascending chunk ID. Original hybrid ranks/component passages remain provenance. Scores are not probabilities.

**Trace M5 and distinguish timing stages.**

Load models/cache/indexes once -> unchanged hybrid top 50 -> deduplicated representatives -> batches of 16 query/passage pairs -> max score per document -> sorted top ten -> unchanged metrics and candidate recall -> JSON/comparison. One-time model loads are separate; online timing includes hybrid work, prediction/tokenization, aggregation, and sorting. No cached corpus encoding is counted online.

**What should you understand before the next milestone?**

Retrieval versus reranking; independent versus joint text encoding; candidate recall and missed-candidate ceilings; query/passage logits; max aggregation and representative bias; batching and token limits; provenance and deterministic ties; and measured quality versus online/preparation cost. All examples should come from saved execution artifacts.

**What did M5's actual experiment show?**

All four reranked quality means increased over hybrid, at much higher CPU online latency dominated by pair inference. Query 128's relevant document moved 9 -> 1, query 70's moved 2 -> 10, and query 13's judged document was absent from candidates. Read the four-way artifact/README for executed measurements. Better averages do not imply improvement for each query; candidate recall is opportunity, not achieved final recall or claim verification.

**What does RAG mean in this implementation?**

Retrieval-augmented generation places the top five winning reranked passages in the prompt before asking a model to answer. It supplies corpus-specific evidence rather than expecting the model to recall it. Retrieval selects stored text; generation produces new text from the question/context.

**What is grounding, and does the prompt guarantee it?**

Grounding asks the answer to rely on supplied evidence, cite supported claims, and admit insufficiency. Our one prompt specifies this behavior, but a model can still use outside knowledge, misread text, or follow misleading passage instructions. No hallucination or answer-quality guarantee is established.

**What is source provenance? Do citations prove truth?**

Evidence records preserve source number, document ID, selected chunk ID, reranked rank, and exact text. Validation maps `[n]` to those IDs and detects unknown/missing references. A valid marker proves only that the referenced block existed; it does not check claim entailment, coverage, or truth.

**Why five passages, and how can context windows limit RAG?**

Five is a fixed initial choice. More passages may add evidence but also distractors, prompt cost, and tokens. The model's input/output context capacity must fit the full prompt and requested answer. M6 does not count provider-specific tokens or trim passages automatically; the endpoint can reject an oversized prompt.

**Why separate the generator from retrieval?**

One `generate(messages) -> str` contract allows a configured HTTP endpoint or offline fake without changing the frozen retrieval functions. The client posts chat messages and parses answer text; corpus retrieval never depends on a provider SDK.

**How is insufficient evidence handled?**

The prompt supplies an explicit abstention sentence. The model decides from the passages; no arbitrary BM25/cosine/reranker threshold is used. With no passages the prompt states that evidence is absent. An uncited abstention is preserved with citation validation failed, as required by the reference-check policy.

**Trace M6 and explain what has actually been verified.**

CLI configuration -> existing corpus/cache/models -> hybrid 50 -> reranked five winning passages -> numbered context + one prompt -> configured generator -> numeric reference validation -> answer/provenance/timing JSON. Offline tests exercise the complete composition and HTTP contract with fakes/mocks. M6 originally used fakes/mocks; a subsequent local qwen2.5:7b smoke test demonstrated that valid references can accompany misinterpreted evidence. Frozen retrieval metrics do not measure answer quality.

## M7 questions

**Why not score free-form answers directly?** Different phrasing and multiple assertions require gold reference facts and a rubric. This benchmark scores one explicit SciFact stance, not all explanation facts.

**What do SUPPORT, CONTRADICT, and ABSTAIN mean?** The supplied evidence supports the claim, contradicts it, or is insufficient/ambiguous. ABSTAIN is a model action, not an invented gold label for empty metadata.

**Why are qrels insufficient for stance?** Relevance identifies useful documents; both supporting and contradicting documents can be relevant. Stance comes from query metadata.

**How do coverage and accuracy interact?** Coverage is the fraction of valid non-abstained predictions. Conditional accuracy may look high when the system answers very few claims, so report coverage and overall accuracy together. Parsing/HTTP failures are not abstentions.

**Accuracy versus macro F1?** Accuracy counts each claim equally. Macro F1 averages the two class F1 scores equally, making minority-class performance visible. Abstentions/failures reduce class recall here.

**What does valid citation syntax establish?** The marker refers to supplied evidence. It does not establish that the passage entails the generated assertion.

**Can document-presence diagnostics separate failures?** They help inspect retrieval and reasoning, but a retrieved parent may have the wrong passage. Association is not causality.

**Trace M7.** Explicit test metadata -> frozen hybrid 50/reranked five -> fixed prompt -> local HTTP generation -> strict verdict parse/reference check -> checkpoint -> classification metrics and document-presence diagnostics.

**Main limitations?** Only 188 annotated scientific claims; no gold abstentions, sentence alignment, general-answer reference facts, or explanation entailment scoring. A correct verdict can accompany an unsupported explanation.

**What did the real M7 smoke establish?** Retrieval and the HTTP client executed for five claims, but every response violated JSON syntax by leaving the verdict unquoted. Strict parsing recorded five failures, not inferred predictions. The full-run estimate exceeded 90 minutes, so there are no 188-claim quality results.
