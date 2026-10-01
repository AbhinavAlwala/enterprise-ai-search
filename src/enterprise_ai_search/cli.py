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


def _check_report_output(output: Path) -> None:
    if output.resolve() in (
        Path("results/scifact_bm25_test.json").resolve(), Path("results/scifact_dense_test.json").resolve(),
        Path("results/scifact_hybrid_test.json").resolve(), Path("results/scifact_comparison.json").resolve(),
        Path("results/scifact_hybrid_comparison.json").resolve(),
    ):
        raise ValueError("Output must not overwrite a frozen baseline report")


def _save_report(output: Path, report: dict) -> None:
    _check_report_output(output)
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

    _check_report_output(args.output)
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

    inputs = [args.bm25.resolve(), args.dense.resolve()]
    if args.hybrid is not None:
        inputs.append(args.hybrid.resolve())
    if args.reranked is not None:
        inputs.append(args.reranked.resolve())
    if args.output.resolve() in inputs:
        raise ValueError("Comparison output must not overwrite its input reports")
    report = compare_reports(args.bm25, args.dense, args.data_dir, args.hybrid, args.reranked)
    _save_report(args.output, report)
    methods = ["bm25", "dense"] + (["hybrid"] if args.hybrid is not None else [])
    if args.reranked is not None:
        methods.append("reranked")
    print("Metric          " + "".join(f"{method:>12}" for method in methods))
    for name, values in report["metrics"].items():
        print(f"{name:16}" + "".join(f"{values[method]:12.6f}" for method in methods))
    print("Query latency ms" + "".join(f"{report['query_latency_ms'][method]:12.3f}" for method in methods))
    print(f"Saved: {args.output}")


def _run_hybrid(args: argparse.Namespace) -> None:
    if args.command == "evaluate-hybrid":
        from enterprise_ai_search.hybrid_evaluation import evaluate_hybrid_scifact

        _check_report_output(args.output)
        report = evaluate_hybrid_scifact(args.data_dir, args.cache_dir)
        _save_report(args.output, report)
        settings = report["settings"]
        print(f"SciFact hybrid test: {report['counts']['evaluated_queries']} queries; "
              f"RRF k={settings['rrf_k']}; depth={settings['candidate_depth_documents_per_retriever']} documents/retriever")
        for name, value in report["metrics"].items():
            print(f"{name}: {value:.6f}")
        print(f"Average online hybrid query: {report['performance']['average_query_seconds'] * 1000:.3f} ms")
        print(f"Total evaluation: {report['performance']['total_evaluation_seconds']:.3f} s")
        print(f"Saved: {args.output}")
        return
    from enterprise_ai_search.dense import DenseIndex, load_encoder, prepare_embeddings
    from enterprise_ai_search.hybrid import HybridIndex

    if args.top_k <= 0:
        raise ValueError("top_k must be positive")
    chunks = chunk_documents(load_corpus(args.data_dir / "corpus.jsonl.gz"))
    encoder, _ = load_encoder(args.cache_dir / "models")
    vectors, _ = prepare_embeddings(chunks, encoder, args.cache_dir / "chunks.npz")
    results = HybridIndex(BM25Index(chunks), DenseIndex(chunks, vectors, encoder)).search(args.query, args.top_k)
    print(json.dumps([asdict(result) for result in results], indent=2))


def _run_reranked(args: argparse.Namespace) -> None:
    if args.command == "evaluate-reranked":
        from enterprise_ai_search.reranked_evaluation import evaluate_reranked_scifact

        _check_report_output(args.output)
        report = evaluate_reranked_scifact(args.data_dir, args.cache_dir, args.model_cache_dir)
        _save_report(args.output, report)
        print(f"SciFact reranked test: {report['counts']['evaluated_queries']} queries")
        print(f"Hybrid candidate Recall@50: {report['candidate_metrics']['Recall@50']:.6f}")
        for name, value in report["metrics"].items():
            print(f"{name}: {value:.6f}")
        print(f"Reranker model loading: {report['preparation']['reranker_model_load_seconds']:.3f} s")
        for key in ("average_candidate_generation_seconds", "average_reranker_inference_seconds", "average_query_seconds"):
            print(f"{key}: {report['performance'][key] * 1000:.3f} ms")
        print(f"Saved: {args.output}")
        return
    from enterprise_ai_search.reranker import CANDIDATE_DEPTH, RerankerConfig, load_reranker, rerank_candidates

    if args.command == "search-reranked" and args.top_k <= 0:
        raise ValueError("top_k must be positive")
    model, seconds = load_reranker(args.model_cache_dir)
    if args.command == "prepare-reranker":
        config = RerankerConfig()
        print(f"Prepared {config.model}, revision {config.revision}; model loading {seconds:.3f} s")
        return
    from enterprise_ai_search.dense import DenseIndex, load_encoder, prepare_embeddings
    from enterprise_ai_search.hybrid import HybridIndex

    chunks = chunk_documents(load_corpus(args.data_dir / "corpus.jsonl.gz"))
    encoder, _ = load_encoder(args.cache_dir / "models")
    vectors, _ = prepare_embeddings(chunks, encoder, args.cache_dir / "chunks.npz")
    candidates = HybridIndex(BM25Index(chunks), DenseIndex(chunks, vectors, encoder)).search(args.query, CANDIDATE_DEPTH)
    results, _ = rerank_candidates(args.query, candidates, model, args.top_k)
    print(json.dumps([asdict(result) for result in results], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="SciFact lexical, dense, hybrid, and reranked retrieval")
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
    compare.add_argument("--hybrid", type=Path, help="Include a third hybrid report")
    compare.add_argument("--reranked", type=Path, help="Include a fourth reranked report; requires --hybrid")
    compare.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    compare.add_argument("--output", type=Path, default=Path("results/scifact_comparison.json"))
    for name in ("search-hybrid", "evaluate-hybrid"):
        command = commands.add_parser(name, help=name.replace("-", " "))
        command.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
        command.add_argument("--cache-dir", type=Path, default=Path("data/dense"))
        if name == "search-hybrid":
            command.add_argument("query")
            command.add_argument("--top-k", type=int, default=5)
        else:
            command.add_argument("--output", type=Path, default=Path("results/scifact_hybrid_test.json"))
    for name in ("prepare-reranker", "search-reranked", "evaluate-reranked"):
        command = commands.add_parser(name, help=name.replace("-", " "))
        command.add_argument("--model-cache-dir", type=Path, default=Path("data/reranker/models"))
        if name != "prepare-reranker":
            command.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
            command.add_argument("--cache-dir", type=Path, default=Path("data/dense"))
        if name == "search-reranked":
            command.add_argument("query")
            command.add_argument("--top-k", type=int, default=5)
        if name == "evaluate-reranked":
            command.add_argument("--output", type=Path, default=Path("results/scifact_reranked_test.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.command in ("prepare-reranker", "search-reranked", "evaluate-reranked"):
            _run_reranked(args)
            return
        if args.command == "download":
            download_scifact(args.data_dir)
            return
        if args.command in ("search-hybrid", "evaluate-hybrid"):
            _run_hybrid(args)
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
            _check_report_output(args.output)
            report = evaluate_scifact(args.data_dir)
            _save_report(args.output, report)
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
