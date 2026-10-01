# Interview preparation

**What is information retrieval, and what is implemented here?**

Information retrieval selects and ranks stored content for a user's information need expressed as a query. Here a local CLI ranks SciFact chunks using lexical BM25 or dense cosine similarity. It does not generate answers or verify claims.

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
