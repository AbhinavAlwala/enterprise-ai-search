# Interview preparation

**What is information retrieval, and what is implemented here?**

Information retrieval selects and ranks stored content for a user's information need expressed as a query. Here a local CLI ranks SciFact chunks using lexical BM25. It does not generate answers or verify claims.

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

`cli.main` reads local documents with `dataset.load_corpus`, creates windows with `text.chunk_documents`, builds `bm25.BM25Index`, calls `search`, and prints `SearchResult` records as JSON. Query files and qrels do not influence this flow.

**How are edge cases and reproducibility handled?**

Empty corpora/queries and unmatched terms return no results. Invalid parameters, malformed records, duplicate IDs, and empty-token index chunks are rejected. Revision-pinned downloads have checksums; `uv.lock` fixes environment dependency versions. Tests are offline, and ties/IDs are deterministic for unchanged data and configuration. Python patch versions and build dependencies are not fully pinned.

**What evidence supports the implementation, and what remains unknown?**

Tests check a manual numeric score, chunk boundaries, tokenization, and download failures. A real-corpus CLI query was executed successfully. No retrieval quality metrics have been measured. SciFact qrels are document-level, while this baseline returns chunks; later evaluation must explicitly resolve that mismatch.
