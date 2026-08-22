from __future__ import annotations

import json

import numpy as np
import pytest

import module_a.scripts.evaluate_model as evaluate_script
from module_a.scripts.evaluate_model import (
    _load_phase_manifests,
    _validate_args,
    build_parser,
)
from module_a.src.evaluation import (
    EvaluationError,
    load_split_manifest,
    validate_evaluation_isolation,
)
from module_a.src.evaluation_pipeline import (
    require_frozen_calibrations,
    run_test_protocols,
    run_validation_protocols,
)
from module_a.src.sv_evaluation import build_sv_trials
from module_a.src.training_data import TrainingRecord


def _data(split: str):
    records = [
        TrainingRecord(f"{split}/{speaker}/{index}.wav", f"{split}_{speaker}", split)
        for speaker in range(5)
        for index in range(4)
    ]
    embeddings = {}
    for record in records:
        speaker = int(record.speaker_id.rsplit("_", 1)[1])
        vector = np.zeros(16, dtype=np.float32)
        vector[speaker] = 1.0
        embeddings[record.path] = vector
    return records, embeddings


def test_test_phase_refuses_to_start_without_validation_calibration(tmp_path):
    records, embeddings = _data("test")
    with pytest.raises(EvaluationError, match="does not exist"):
        run_test_protocols(
            records,
            embeddings,
            output_dir=tmp_path,
            seed=42,
            max_sv_positive_per_speaker=2,
            sid_known_ratio=0.8,
            sid_max_enrollment=2,
        )
    assert not (tmp_path / "trials").exists()


def test_test_consumes_frozen_validation_thresholds_without_overwrite(tmp_path):
    val_records, val_embeddings = _data("val")
    validation = run_validation_protocols(
        val_records,
        val_embeddings,
        output_dir=tmp_path,
        seed=42,
        max_sv_positive_per_speaker=2,
        sid_known_ratio=0.8,
        sid_max_enrollment=2,
    )
    sv_path = tmp_path / "calibration" / "sv_calibration.json"
    sid_path = tmp_path / "calibration" / "sid_calibration.json"
    before = (sv_path.read_bytes(), sid_path.read_bytes())
    test_records, test_embeddings = _data("test")
    test = run_test_protocols(
        test_records,
        test_embeddings,
        output_dir=tmp_path,
        seed=42,
        max_sv_positive_per_speaker=2,
        sid_known_ratio=0.8,
        sid_max_enrollment=2,
    )
    assert before == (sv_path.read_bytes(), sid_path.read_bytes())
    assert test["sv"]["metrics"]["frozen_validation_sv_deployment_threshold"] == validation["sv"]["calibration"]["deployment_threshold"]
    assert test["sid"]["metrics"]["frozen_validation_sid_threshold"] == validation["sid"]["calibration"]["selected_threshold"]
    assert test["sid"]["metrics"]["frozen_validation_sid_target_unknown_far"] == 0.05
    assert validation["sid"]["calibration"]["deployment_policy"] == "target_unknown_far"
    assert validation["sv"]["metrics"]["eer_threshold"] == validation["sv"]["calibration"]["eer_threshold"]
    assert validation["sv"]["metrics"]["deployment_target_far"] == 0.05


def test_test_rejects_calibration_from_another_checkpoint(tmp_path):
    val_records, val_embeddings = _data("val")
    run_validation_protocols(
        val_records,
        val_embeddings,
        output_dir=tmp_path,
        seed=42,
        max_sv_positive_per_speaker=2,
        sid_known_ratio=0.8,
        sid_max_enrollment=2,
        calibration_provenance={"checkpoint_sha256": "checkpoint-a"},
    )
    test_records, test_embeddings = _data("test")
    with pytest.raises(EvaluationError, match="provenance mismatch"):
        run_test_protocols(
            test_records,
            test_embeddings,
            output_dir=tmp_path,
            seed=42,
            max_sv_positive_per_speaker=2,
            sid_known_ratio=0.8,
            sid_max_enrollment=2,
            expected_calibration_provenance={"checkpoint_sha256": "checkpoint-b"},
        )


def test_test_rejects_calibration_from_another_protocol(tmp_path):
    val_records, val_embeddings = _data("val")
    run_validation_protocols(
        val_records,
        val_embeddings,
        output_dir=tmp_path,
        seed=42,
        max_sv_positive_per_speaker=2,
        sid_known_ratio=0.8,
        sid_max_enrollment=2,
        calibration_provenance={"sid_known_ratio": 0.8},
    )
    test_records, test_embeddings = _data("test")
    with pytest.raises(EvaluationError, match="provenance mismatch"):
        run_test_protocols(
            test_records,
            test_embeddings,
            output_dir=tmp_path,
            seed=42,
            max_sv_positive_per_speaker=2,
            sid_known_ratio=0.8,
            sid_max_enrollment=2,
            expected_calibration_provenance={"sid_known_ratio": 0.7},
        )


def test_old_calibration_policy_is_rejected(tmp_path):
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    old = {"source_split": "validation", "selected_threshold": 0.5}
    (calibration_dir / "sv_calibration.json").write_text(
        json.dumps(old), encoding="utf-8"
    )
    (calibration_dir / "sid_calibration.json").write_text(
        json.dumps(old), encoding="utf-8"
    )
    with pytest.raises(EvaluationError, match="frozen validation artifact"):
        require_frozen_calibrations(tmp_path)


def test_old_sid_raw_accuracy_policy_is_rejected(tmp_path):
    val_records, val_embeddings = _data("val")
    run_validation_protocols(
        val_records,
        val_embeddings,
        output_dir=tmp_path,
        seed=42,
        max_sv_positive_per_speaker=2,
        sid_known_ratio=0.8,
        sid_max_enrollment=2,
    )
    sid_path = tmp_path / "calibration" / "sid_calibration.json"
    sid = json.loads(sid_path.read_text(encoding="utf-8"))
    sid["calibration_schema_version"] = 2
    sid["calibration_policy_version"] = "sid_balanced_open_set_accuracy_v1"
    sid["objective"] = "maximize validation balanced open-set accuracy"
    sid_path.write_text(json.dumps(sid), encoding="utf-8")
    with pytest.raises(EvaluationError, match="frozen validation artifact"):
        require_frozen_calibrations(tmp_path)


def test_test_rejects_calibration_from_another_validation_manifest(tmp_path):
    val_records, val_embeddings = _data("val")
    run_validation_protocols(
        val_records,
        val_embeddings,
        output_dir=tmp_path,
        seed=42,
        max_sv_positive_per_speaker=2,
        sid_known_ratio=0.8,
        sid_max_enrollment=2,
        calibration_provenance={"validation_manifest_sha256": "manifest-a"},
    )
    test_records, test_embeddings = _data("test")
    with pytest.raises(EvaluationError, match="provenance mismatch"):
        run_test_protocols(
            test_records,
            test_embeddings,
            output_dir=tmp_path,
            seed=42,
            max_sv_positive_per_speaker=2,
            sid_known_ratio=0.8,
            sid_max_enrollment=2,
            expected_calibration_provenance={
                "validation_manifest_sha256": "manifest-b"
            },
        )


def test_validation_phase_does_not_load_test_manifest(monkeypatch):
    args = build_parser().parse_args(
        [
            "--dataset-root", "dataset",
            "--val-manifest", "val.csv",
            "--test-manifest", "must-not-load.csv",
            "--checkpoint", "last.pt",
            "--output-dir", "a4",
            "--phase", "validation",
        ]
    )
    calls = []

    def fake_load(path, split):
        calls.append((path, split))
        return []

    monkeypatch.setattr(evaluate_script, "load_split_manifest", fake_load)
    validation_records, test_records = _load_phase_manifests(args)
    assert validation_records == []
    assert test_records is None
    assert calls == [("val.csv", "val")]


def test_split_isolation_rejects_train_and_validation_test_leakage():
    val = [TrainingRecord("val/a.wav", "speaker_a", "val")]
    test = [TrainingRecord("test/a.wav", "speaker_a", "test")]
    with pytest.raises(EvaluationError, match="Train speaker"):
        validate_evaluation_isolation(train_speakers=["speaker_a"], validation_records=val)
    with pytest.raises(EvaluationError, match="contamination"):
        validate_evaluation_isolation(
            train_speakers=[], validation_records=val, test_records=test
        )


def test_invalid_manifest_split_fails_closed(tmp_path):
    manifest = tmp_path / "val.csv"
    manifest.write_text(
        "path,speaker_id,split\na.wav,a,val\nb.wav,b,test\n",
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="another split"):
        load_split_manifest(manifest, "val")


def test_sv_protocol_rejects_empty_positive_trials():
    records = [
        TrainingRecord("val/a.wav", "a", "val"),
        TrainingRecord("val/b.wav", "b", "val"),
    ]
    with pytest.raises(EvaluationError, match="positive"):
        build_sv_trials(records, seed=42, max_positive_per_speaker=1)


def test_cli_test_phase_requires_test_manifest():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dataset-root",
            "dataset",
            "--val-manifest",
            "val.csv",
            "--checkpoint",
            "last.pt",
            "--output-dir",
            "a4",
            "--phase",
            "test",
        ]
    )
    with pytest.raises(SystemExit):
        _validate_args(parser, args)


def test_evaluate_model_cli_help_works(capsys):
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(["--help"])
    assert raised.value.code == 0
    assert "frozen Stage-1 checkpoint" in capsys.readouterr().out
