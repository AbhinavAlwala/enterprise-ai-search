import gzip
import hashlib
import io
import json
from pathlib import Path
from urllib.error import URLError

import pytest

from enterprise_ai_search import dataset
from enterprise_ai_search.cli import main


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unit tests must not access the network")
    monkeypatch.setattr(dataset, "urlopen", forbidden)


@pytest.mark.parametrize("compressed", [False, True])
def test_load_jsonl_preserves_ids_text_and_optional_title(tmp_path: Path, compressed: bool) -> None:
    path = tmp_path / ("corpus.jsonl.gz" if compressed else "corpus.jsonl")
    content = json.dumps({"_id": "42", "text": "Café", "title": "Title"}) + "\n\n"
    content += json.dumps({"_id": "43", "text": ""}) + "\n"
    if compressed:
        path.write_bytes(gzip.compress(content.encode()))
    else:
        path.write_text(content, encoding="utf-8")
    documents = dataset.load_corpus(path)
    assert [document.document_id for document in documents] == ["42", "43"]
    assert documents[0].text == "Café"
    assert documents[0].title == "Title"
    assert documents[1].title == ""


@pytest.mark.parametrize("line", ['not json', '{}', '[]', '{"_id":1,"text":"x"}', '{"_id":"","text":"x"}', '{"_id":"x","text":null}', '{"_id":"x","text":"x","title":null}'])
def test_malformed_records_include_line_number(tmp_path: Path, line: str) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n" + line, encoding="utf-8")
    with pytest.raises(ValueError, match=r":2:"):
        dataset.load_corpus(path)


def test_duplicate_ids_and_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    with pytest.raises(FileNotFoundError):
        dataset.load_corpus(path)
    line = json.dumps({"_id": "x", "text": "text"}) + "\n"
    path.write_text(line * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        dataset.load_corpus(path)


def test_download_checksum_cache_and_repair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"fixture data"
    expected = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(dataset, "FILES", {"qrels/test.tsv": ("https://example.test/file", expected)})
    calls = []
    def response(url: str, timeout: int) -> io.BytesIO:
        calls.append(url)
        return io.BytesIO(payload)
    monkeypatch.setattr(dataset, "urlopen", response)
    dataset.download_scifact(tmp_path)
    dataset.download_scifact(tmp_path)
    assert len(calls) == 1
    destination = tmp_path / "qrels/test.tsv"
    destination.write_bytes(b"corrupted")
    dataset.download_scifact(tmp_path)
    assert len(calls) == 2
    assert destination.read_bytes() == payload
    assert list(destination.parent.iterdir()) == [destination]


def test_bad_download_keeps_existing_file_and_cleans_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dataset, "FILES", {"corpus": ("https://example.test/file", "0" * 64)})
    monkeypatch.setattr(dataset, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong"))
    destination = tmp_path / "corpus"
    destination.write_bytes(b"existing")
    with pytest.raises(ValueError, match="Checksum"):
        dataset.download_scifact(tmp_path)
    assert destination.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [destination]


def test_failed_download_cleans_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dataset, "FILES", {"corpus": ("https://example.test/file", "0" * 64)})
    def fail(*args: object, **kwargs: object) -> None:
        raise URLError("fixture failure")
    monkeypatch.setattr(dataset, "urlopen", fail)
    with pytest.raises(URLError):
        dataset.download_scifact(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_cli_search_outputs_ranked_json_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [{"_id": "a", "text": "heart pumps blood"}, {"_id": "b", "text": "bone growth"}]
    content = "\n".join(json.dumps(record) for record in records)
    (tmp_path / "corpus.jsonl.gz").write_bytes(gzip.compress(content.encode()))
    monkeypatch.setattr("sys.argv", ["enterprise-search", "search", "blood", "--data-dir", str(tmp_path)])
    main()
    results = json.loads(capsys.readouterr().out)
    assert len(results) == 1
    assert results[0]["document_id"] == "a"
    assert results[0]["rank"] == 1
    assert results[0]["score"] > 0


@pytest.mark.parametrize("option,value", [("--top-k", "0"), ("--chunk-size", "0"), ("--overlap", "180")])
def test_cli_reports_invalid_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    option: str, value: str,
) -> None:
    monkeypatch.setattr("sys.argv", ["enterprise-search", "search", "query", "--data-dir", str(tmp_path), option, value])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
