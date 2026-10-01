# Current data flow

1. `enterprise-search download` fetches compressed corpus/query JSONL and train/test qrels from pinned BEIR mirror revisions. Each file is SHA-256 verified before replacing its destination. Valid cached files are reused. Partial download files are cleaned up.
2. `enterprise-search search` reads only the local corpus. `load_corpus` maps `_id`, `text`, and optional `title` to documents, preserving source IDs. Invalid records and duplicate IDs raise errors with line numbers.
3. `chunk_documents` joins each title and abstract once, normalizes Unicode to NFC, and collapses whitespace. It makes 180-word windows with 30-word overlap by default. Chunk IDs are `<document-id>::chunk::<zero-based-window-number>`.
4. `BM25Index` tokenizes chunk text: case folding plus Unicode alphanumeric sequences, splitting at punctuation and underscores. It stores term frequencies, token lengths, average length, and chunk-level document frequencies.
5. `search` tokenizes the query identically, removes duplicate query tokens, ignores unseen terms, and scores every chunk using BM25. Only positive-score matches are retained. Ties use ascending chunk ID.
6. The first up-to-k results receive one-based ranks and are printed as JSON with score, IDs, and full chunk text. Empty/unmatched queries return `[]`; invalid top-k values are rejected.

Chunk size counts whitespace words; BM25 length counts tokenizer output. They are deliberately different units. The final window stops when it reaches the document end, so no redundant overlap-only tail is emitted. Windows with no searchable tokens are omitted.

Query files and qrels remain local for a later evaluation milestone. Current queries come directly from CLI input, and no relevance judgments influence ranking.
