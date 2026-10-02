import json
import sys
from pathlib import Path

import pytest

from enterprise_ai_search import claim_evaluation as evaluation
from enterprise_ai_search.dataset import StanceClaim


def prediction(gold: str, predicted: str | None, *, present: bool = False, generation_failure: bool = False) -> dict:
    return {
        "gold_stance": gold, "predicted_stance": predicted, "gold_document_present": present,
        "parse_status": "not_attempted" if generation_failure else ("ok" if predicted else "failed"),
        "generation_status": "failed" if generation_failure else "ok",
        "citation_validation": {"passed": True} if predicted in ("SUPPORT", "CONTRADICT") else None,
    }


def test_hand_calculated_metrics_include_abstentions_and_failures() -> None:
    records = [
        prediction("SUPPORT", "SUPPORT", present=True), prediction("SUPPORT", "CONTRADICT", present=True),
        prediction("SUPPORT", "ABSTAIN"), prediction("CONTRADICT", "CONTRADICT"),
        prediction("CONTRADICT", None), prediction("CONTRADICT", None, generation_failure=True),
    ]
    metrics = evaluation.summarize_predictions(records)
    assert metrics["overall_accuracy"] == 2 / 6
    assert metrics["non_abstained_accuracy"] == 2 / 3
    assert metrics["macro_f1"] == pytest.approx(0.45)
    assert metrics["per_class"]["SUPPORT"] == {"precision": 1, "recall": 1 / 3, "f1": 0.5, "gold_count": 3}
    assert metrics["per_class"]["CONTRADICT"] == {"precision": 0.5, "recall": 1 / 3, "f1": 0.4, "gold_count": 3}
    assert metrics["coverage"] == 0.5 and metrics["abstention_rate"] == 1 / 6
    assert metrics["parsing_failure_rate"] == metrics["generation_failure_rate"] == 1 / 6
    assert metrics["citation_validation_pass_rate"] == 0.5
    assert metrics["citation_pass_rate_among_parsed"] == 3 / 4
    assert metrics["confusion_matrix"] == {"SUPPORT": {"SUPPORT": 1, "CONTRADICT": 1, "ABSTAIN": 1}, "CONTRADICT": {"SUPPORT": 0, "CONTRADICT": 1, "ABSTAIN": 0}}
    assert metrics["failures_by_gold"] == {"SUPPORT": 0, "CONTRADICT": 2}
    assert metrics["gold_document_present_rate"] == 1 / 3
    assert metrics["diagnostics"]["gold_document_present"] == {"count": 2, "overall_accuracy": 0.5}
    assert metrics["diagnostics"]["gold_document_absent"] == {"count": 4, "overall_accuracy": 0.25}


def test_undefined_conditional_accuracy_is_null_not_zero() -> None:
    metrics = evaluation.summarize_predictions([prediction("SUPPORT", "ABSTAIN")])
    assert metrics["non_abstained_accuracy"] is None and metrics["macro_f1"] == 0
    assert metrics["diagnostics"]["gold_document_present"]["overall_accuracy"] is None
    assert metrics["coverage"] == 0 and metrics["abstention_rate"] == 1
    with pytest.raises(ValueError):
        evaluation.summarize_predictions([])


@pytest.fixture
def claims() -> list[StanceClaim]:
    return [StanceClaim(str(i), "Claim " + str(i), "SUPPORT", ("a",)) for i in range(7)]


def fake_record(claim: StanceClaim, *args: object) -> dict:
    return {
        **prediction(claim.gold_stance, "SUPPORT", present=True), "query_id": claim.query_id,
        "claim": claim.text, "gold_document_ids": list(claim.gold_document_ids),
        "timings": {"online_seconds": 10, "generation_request_seconds": 8, "retrieval_reranking_seconds": 2},
    }


def test_smoke_resume_reuses_predictions_and_rejects_incompatibility(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, claims: list) -> None:
    calls = []
    def verify(claim: StanceClaim, *args: object) -> dict:
        calls.append(claim.query_id)
        return fake_record(claim)
    monkeypatch.setattr(evaluation, "verify_claim", verify)
    smoke, full = tmp_path / "smoke.json", tmp_path / "full.json"
    identity, preparation = {"settings": "fixed"}, {"preparation_seconds": 5}
    report = evaluation.run_evaluation(claims, None, None, None, smoke, identity, preparation, smoke=True)
    assert calls == ["0", "1", "2", "3", "4"] and report["status"] == "smoke_complete"
    assert report["estimated_full_runtime_seconds"] == 75
    original_smoke = smoke.read_bytes()
    report = evaluation.run_evaluation(claims, None, None, None, full, identity, preparation, resume_from=smoke)
    assert calls == [str(i) for i in range(7)] and report["status"] == "complete"
    assert smoke.read_bytes() == original_smoke and report["metrics"]["overall_accuracy"] == 1
    evaluation.run_evaluation(claims, None, None, None, full, identity, preparation, resume_from=full)
    assert len(calls) == 7
    with pytest.raises(ValueError, match="Output exists"):
        evaluation.run_evaluation(claims, None, None, None, full, identity, preparation, smoke=True)
    with pytest.raises(ValueError, match="differ"):
        evaluation.load_checkpoint(full, {"settings": "changed"}, claims)
    corrupted = json.loads(full.read_text())
    corrupted["per_query"]["0"]["predicted_stance"] = "CONTRADICT"
    full.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="checksum"):
        evaluation.load_checkpoint(full, identity, claims)


def test_interrupted_smoke_preserves_completed_claims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, claims: list) -> None:
    output = tmp_path / "checkpoint.json"
    def interrupted(claim: StanceClaim, *args: object) -> dict:
        if claim.query_id == "2":
            raise KeyboardInterrupt
        return fake_record(claim)
    monkeypatch.setattr(evaluation, "verify_claim", interrupted)
    with pytest.raises(KeyboardInterrupt):
        evaluation.run_evaluation(claims, None, None, None, output, {}, {"preparation_seconds": 0}, smoke=True)
    assert list(json.loads(output.read_text())["per_query"]) == ["0", "1"]
    monkeypatch.setattr(evaluation, "verify_claim", fake_record)
    report = evaluation.run_evaluation(claims, None, None, None, output, {}, {"preparation_seconds": 0}, smoke=True, resume_from=output)
    assert len(report["per_query"]) == 5 and report["smoke_completed"]
    assert not output.with_name(output.name + ".tmp").exists()


def test_full_run_gate_and_runtime_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, claims: list) -> None:
    monkeypatch.setattr(evaluation, "verify_claim", fake_record)
    smoke = tmp_path / "smoke.json"
    with pytest.raises(ValueError, match="five-claim smoke"):
        evaluation.run_evaluation(claims, None, None, None, tmp_path / "full.json", {}, {"preparation_seconds": 0})
    report = evaluation.run_evaluation(claims, None, None, None, smoke, {}, {"preparation_seconds": 6000}, smoke=True)
    with pytest.raises(ValueError, match="90 minutes"):
        evaluation.run_evaluation(claims, None, None, None, tmp_path / "full.json", {}, {"preparation_seconds": 0}, resume_from=smoke)
    assert report["estimated_full_runtime_seconds"] == 6070


def test_generation_failure_is_checkpointed_and_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, claims: list) -> None:
    def fail(claim: StanceClaim, *args: object) -> dict:
        record = fake_record(claim)
        record.update(prediction(claim.gold_stance, None, generation_failure=True))
        return record
    monkeypatch.setattr(evaluation, "verify_claim", fail)
    output = tmp_path / "failed.json"
    with pytest.raises(ValueError, match="Generation failed"):
        evaluation.run_evaluation(claims, None, None, None, output, {}, {"preparation_seconds": 0}, smoke=True)
    assert len(json.loads(output.read_text())["per_query"]) == 1


def test_cli_smoke_and_full_resume_with_fake_preparation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], claims: list) -> None:
    from enterprise_ai_search import generation
    from enterprise_ai_search.cli import main

    monkeypatch.setenv("GENERATION_ENDPOINT", "http://localhost/v1")
    monkeypatch.setenv("GENERATION_MODEL", "qwen2.5:7b")
    monkeypatch.setattr(evaluation, "load_test_claims", lambda path: claims)
    class FakeIndex:
        class bm25:
            class chunk:
                document_id = "a"
            chunks = [chunk()]
    monkeypatch.setattr(evaluation, "prepare_retrieval", lambda *args: (FakeIndex(), None, {"preparation_seconds": 0}))
    monkeypatch.setattr(evaluation, "evaluation_identity", lambda *args: {"fixed": True})
    monkeypatch.setattr(evaluation, "verify_claim", fake_record)
    monkeypatch.setattr(generation, "HttpGenerator", lambda config: None)
    smoke, full = tmp_path / "smoke.json", tmp_path / "full.json"
    monkeypatch.setattr(sys, "argv", ["enterprise-search", "evaluate-claims", "--smoke", "--output", str(smoke)])
    main()
    assert json.loads(capsys.readouterr().out)["status"] == "smoke_complete"
    monkeypatch.setattr(sys, "argv", ["enterprise-search", "evaluate-claims", "--resume-from", str(smoke), "--output", str(full)])
    main()
    assert json.loads(capsys.readouterr().out)["metrics"]["evaluated_claims"] == 7


def test_cli_protects_frozen_artifacts_before_model_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    from enterprise_ai_search.cli import main

    monkeypatch.setattr(evaluation, "prepare_retrieval", lambda *args: pytest.fail("Must reject output before preparing models"))
    monkeypatch.setattr(sys, "argv", ["enterprise-search", "evaluate-claims", "--smoke", "--output", "results/scifact_reranked_test.json"])
    with pytest.raises(SystemExit) as failure:
        main()
    assert failure.value.code == 2
