import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from enterprise_ai_search import dense_evaluation
from enterprise_ai_search.cli import main
from enterprise_ai_search.comparison import compare_reports
from enterprise_ai_search.dataset import load_queries
from enterprise_ai_search.evaluation import evaluate_scifact


class FixtureEncoder:
    def __init__(self) -> None:
        self.tokenizer = lambda texts, **kwargs: {"input_ids": [[1, 2] for _ in texts]}

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i, 0 if text in ("cat", "kitty") else 1] = 1
        return vectors


@pytest.fixture
def local_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    monkeypatch.setattr(dense_evaluation, "load_encoder", lambda path: (FixtureEncoder(), 1.5))
    documents = [{"_id": "a", "text": "cat"}]
    documents += [{"_id": f"b{i:02}", "text": "background"} for i in range(10)]
    documents += [{"_id": "z", "text": "rare"}]
    queries = [{"_id": "1", "text": "cat"}, {"_id": "2", "text": "kitty"}, {"_id": "3", "text": "rare"}]
    for name, records in [("corpus.jsonl.gz", documents), ("queries.jsonl.gz", queries)]:
        content = "\n".join(json.dumps(record) for record in records)
        (tmp_path / name).write_bytes(gzip.compress(content.encode()))
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels/test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n1\ta\t1\n2\ta\t1\n3\tz\t1\n", encoding="utf-8"
    )
    bm25_path, dense_path = tmp_path / "bm25.json", tmp_path / "dense.json"
    bm25_path.write_text(json.dumps(evaluate_scifact(tmp_path)), encoding="utf-8")
    dense_path.write_text(json.dumps(dense_evaluation.evaluate_dense_scifact(tmp_path, tmp_path / "cache")), encoding="utf-8")
    return tmp_path, bm25_path, dense_path


def test_dense_evaluation_uses_identical_defaults_queries_and_metrics(local_reports: tuple[Path, Path, Path]) -> None:
    data_dir, bm25_path, dense_path = local_reports
    dense = json.loads(dense_path.read_text())
    bm25 = json.loads(bm25_path.read_text())
    assert dense["input_sha256"] == bm25["input_sha256"]
    assert dense["counts"] == bm25["counts"]
    assert dense["settings"]["chunking"] == {"size": 180, "overlap": 30}
    assert set(dense["per_query"]) == {"1", "2", "3"}
    assert all(value == pytest.approx(2 / 3) for value in dense["metrics"].values())
    assert dense["per_query"]["3"]["metrics"]["MRR@10"] == 0
    assert dense["preparation"]["model_load_seconds"] == 1.5
    assert not dense["preparation"]["cache_hit"]
    repeated = dense_evaluation.evaluate_dense_scifact(data_dir, data_dir / "cache")
    assert repeated["preparation"]["cache_hit"]
    assert repeated["metrics"] == dense["metrics"]
    assert repeated["preparation"]["encoding_seconds_this_run"] == 0


def test_comparison_selects_measured_examples_deterministically(local_reports: tuple[Path, Path, Path]) -> None:
    data_dir, bm25_path, dense_path = local_reports
    comparison = compare_reports(bm25_path, dense_path, data_dir)
    assert comparison["evaluated_queries"] == 3
    assert [(example["query_id"], example["winner"]) for example in comparison["examples"]] == [
        ("2", "dense"), ("3", "bm25")
    ]
    assert comparison["examples"][0]["query"] == load_queries(data_dir / "queries.jsonl.gz")["2"]
    assert all(row["dense_minus_bm25"] == 0 for row in comparison["metrics"].values())


@pytest.mark.parametrize("change", ["split", "inputs", "chunking", "query_ids", "policy", "metric"])
def test_comparison_rejects_inconsistent_reports(local_reports: tuple[Path, Path, Path], change: str) -> None:
    data_dir, bm25_path, dense_path = local_reports
    report = json.loads(dense_path.read_text())
    if change == "split":
        report["split"] = "train"
    elif change == "inputs":
        report["input_sha256"]["corpus.jsonl.gz"] = "different"
    elif change == "chunking":
        report["settings"]["chunking"]["overlap"] = 0
    elif change == "query_ids":
        report["per_query"].pop("1")
    elif change == "policy":
        report["settings"]["ndcg_gain"] = "different"
    else:
        report["metrics"]["Recall@5"] = 1.5
    dense_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        compare_reports(bm25_path, dense_path, data_dir)


def test_cli_dense_evaluation_and_comparison_save_reports(
    local_reports: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir, bm25_path, dense_path = local_reports
    monkeypatch.setattr("sys.argv", [
        "enterprise-search", "evaluate-dense", "--data-dir", str(data_dir),
        "--cache-dir", str(data_dir / "cache"), "--output", str(dense_path),
    ])
    main()
    assert "SciFact dense test: 3 queries" in capsys.readouterr().out
    output = data_dir / "comparison.json"
    monkeypatch.setattr("sys.argv", [
        "enterprise-search", "compare", "--data-dir", str(data_dir),
        "--bm25", str(bm25_path), "--dense", str(dense_path), "--output", str(output),
    ])
    main()
    assert len(json.loads(output.read_text())["examples"]) == 2


def test_cli_dense_cannot_overwrite_frozen_bm25(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", [
        "enterprise-search", "evaluate-dense", "--output", "results/scifact_bm25_test.json"
    ])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_cli_comparison_cannot_overwrite_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", [
        "enterprise-search", "compare", "--output", "results/scifact_bm25_test.json"
    ])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
