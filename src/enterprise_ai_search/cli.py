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


def _save_report(output: Path, report: dict) -> None:
    if output.resolve() == Path("results/scifact_bm25_test.json").resolve():
        raise ValueError("Output must not overwrite the frozen BM25 report")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _run_dense_search_or_prepare(args: argparse.Namespace) -> None:
    from enterprise_ai_search.dense import DenseIndex, load_encoder, prepare_embeddings

    if args.command == "search-dense" and args.top_k <= 0:
        raise ValueError("top_k must be positive")
    chunks = chunk_documents(load_corpus(args.data_dir / "corpus.jsonl.gz"))
    encoder, model_seconds = load_encoder(args.cache_dir / "models")
    vectors, preparation = prepare_embeddings(chunks, encoder, args.cache_dir / "chunks.npz", rebuild=args.rebuild)
    if args.command == "prepare-dense":
        print(f"Prepared {len(chunks)} chunks, {vectors.shape[1]} dimensions; cache hit={preparation['cache_hit']}")
        print(f"Model loading (including imports/download if needed): {model_seconds:.3f} s")
        print(f"Corpus encoding this run: {preparation['encoding_seconds_this_run']:.3f} s")
        print(f"Cache preparation: {preparation['preparation_seconds']:.3f} s")
    else:
        results = DenseIndex(chunks, vectors, encoder).search(args.query, args.top_k)
        print(json.dumps([asdict(result) for result in results], indent=2))


def _run_dense_evaluation(args: argparse.Namespace) -> None:
    from enterprise_ai_search.dense_evaluation import evaluate_dense_scifact

    if args.output.resolve() == Path("results/scifact_bm25_test.json").resolve():
        raise ValueError("Dense output must not overwrite the frozen BM25 report")
    report = evaluate_dense_scifact(args.data_dir, args.cache_dir, args.rebuild)
    _save_report(args.output, report)
    print(f"SciFact dense test: {report['counts']['evaluated_queries']} queries")
    for name, value in report["metrics"].items():
        print(f"{name}: {value:.6f}")
    preparation = report["preparation"]
    print(f"Model loading: {preparation['model_load_seconds']:.3f} s")
    print(f"Corpus encoding (artifact generation): {preparation['artifact']['encoding_seconds']:.3f} s")
    print(f"Index preparation this run: {preparation['index_preparation_seconds']:.3f} s; cache hit={preparation['cache_hit']}")
    print(f"Average online query: {report['performance']['average_query_seconds'] * 1000:.3f} ms")
    print(f"Total evaluation: {report['performance']['total_evaluation_seconds']:.3f} s")
    print(f"Saved: {args.output}")


def _run_comparison(args: argparse.Namespace) -> None:
    from enterprise_ai_search.comparison import compare_reports

    if args.output.resolve() in (args.bm25.resolve(), args.dense.resolve()):
        raise ValueError("Comparison output must not overwrite its input reports")
    report = compare_reports(args.bm25, args.dense, args.data_dir)
    _save_report(args.output, report)
    print("Metric              BM25       Dense       Dense - BM25")
    for name, values in report["metrics"].items():
        print(f"{name:16} {values['bm25']:10.6f} {values['dense']:10.6f} {values['dense_minus_bm25']:+13.6f}")
    latency = report["query_latency_ms"]
    print(f"Query latency ms {latency['bm25']:10.3f} {latency['dense']:10.3f}")
    print(f"Saved: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SciFact lexical and dense retrieval")
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
    for name in ("prepare-dense", "search-dense", "evaluate-dense"):
        command = commands.add_parser(name, help=name.replace("-", " "))
        command.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
        command.add_argument("--cache-dir", type=Path, default=Path("data/dense"))
        command.add_argument("--rebuild", action="store_true", help="Regenerate chunk embeddings")
        if name == "search-dense":
            command.add_argument("query")
            command.add_argument("--top-k", type=int, default=5)
        if name == "evaluate-dense":
            command.add_argument("--output", type=Path, default=Path("results/scifact_dense_test.json"))
    compare = commands.add_parser("compare", help="Compare saved BM25/dense evaluation artifacts")
    compare.add_argument("--bm25", type=Path, default=Path("results/scifact_bm25_test.json"))
    compare.add_argument("--dense", type=Path, default=Path("results/scifact_dense_test.json"))
    compare.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    compare.add_argument("--output", type=Path, default=Path("results/scifact_comparison.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.command == "download":
            download_scifact(args.data_dir)
            return
        if args.command in ("prepare-dense", "search-dense"):
            _run_dense_search_or_prepare(args)
            return
        if args.command == "compare":
            _run_comparison(args)
            return
        if args.command == "evaluate-dense":
            _run_dense_evaluation(args)
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
