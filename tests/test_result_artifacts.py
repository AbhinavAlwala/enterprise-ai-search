import hashlib
import re
from pathlib import Path

import pytest

from enterprise_ai_search.cli import _save_report


def test_documented_artifact_checksums_are_portable_across_line_endings() -> None:
    manifest = Path("results/README.md").read_text(encoding="utf-8")
    rows = re.findall(r"\| \[([^]]+\.json)\]\([^)]*\) \| (\d+) \| `([a-f0-9]{64})` \|", manifest)
    assert {name for name, _, _ in rows} == {path.name for path in Path("results").glob("*.json")}
    for name, size, checksum in rows:
        contents = (Path("results") / name).read_bytes().replace(b"\r\n", b"\n")
        assert len(contents) == int(size), name
        assert hashlib.sha256(contents).hexdigest() == checksum, name


def test_report_writer_preserves_every_frozen_artifact(tmp_path, monkeypatch) -> None:
    names = [path.name for path in Path("results").glob("*.json")]
    monkeypatch.chdir(tmp_path)
    Path("results").mkdir()
    for name in names:
        path = Path("results") / name
        path.write_bytes(b"frozen fixture")
        with pytest.raises(ValueError, match="frozen"):
            _save_report(path, {"replacement": True})
        assert path.read_bytes() == b"frozen fixture"
