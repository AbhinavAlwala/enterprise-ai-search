import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from urllib.error import URLError

from enterprise_ai_search.bm25 import BM25Index
from enterprise_ai_search.dataset import download_scifact, load_corpus
from enterprise_ai_search.evaluation import evaluate_scifact
from enterprise_ai_search.text import ChunkingConfig, chunk_documents

logger = logging.getLogger(__name__)
DEFAULT_DATA_DIR = Path("data/scifact")


def main() -> None:
    parser = argparse.ArgumentParser(description="SciFact BM25 chunk retrieval")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download pinned SciFact files")
    download.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    evaluate = commands.add_parser("evaluate", help="Evaluate the unchanged BM25 baseline on SciFact test")
    evaluate.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    evaluate.add_argument("--output", type=Path, default=Path("results/scifact_bm25_test.json"))
    search = commands.add_parser("search", help="Search the local SciFact corpus")
    search.add_argument("query")
    search.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    search.add_argument("--top-k", type=int, default=5)
    defaults = ChunkingConfig()
    search.add_argument("--chunk-size", type=int, default=defaults.size, help="Whitespace words")
    search.add_argument("--overlap", type=int, default=defaults.overlap, help="Whitespace words")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.command == "download":
            download_scifact(args.data_dir)
            return
        if args.command == "evaluate":
            report = evaluate_scifact(args.data_dir)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"SciFact test: {report['counts']['evaluated_queries']} queries")
            for name, value in report["metrics"].items():
                print(f"{name}: {value:.6f}")
            timing = report["performance"]
            print(f"Total evaluation: {timing['total_evaluation_seconds']:.3f} s")
            print(f"Average query: {timing['average_query_seconds'] * 1000:.3f} ms")
            print(f"Saved: {args.output}")
            return
        config = ChunkingConfig(args.chunk_size, args.overlap)
        if args.top_k <= 0:
            raise ValueError("top_k must be positive")
        documents = load_corpus(args.data_dir / "corpus.jsonl.gz")
        chunks = chunk_documents(documents, config)
        logger.info("Loaded %d documents; indexed %d chunks", len(documents), len(chunks))
        results = BM25Index(chunks).search(args.query, args.top_k)
        print(json.dumps([asdict(result) for result in results], indent=2))
    except (OSError, ValueError, URLError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
