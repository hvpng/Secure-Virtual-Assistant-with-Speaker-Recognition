from __future__ import annotations

import numpy as np
import pytest

from module_a.scripts.evaluate_model import _validate_args, build_parser
from module_a.src.evaluation import (
    EvaluationError,
    load_split_manifest,
    validate_evaluation_isolation,
)
from module_a.src.evaluation_pipeline import run_test_protocols, run_validation_protocols
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
    assert test["sv"]["metrics"]["frozen_validation_sv_threshold"] == validation["sv"]["calibration"]["selected_threshold"]
    assert test["sid"]["metrics"]["frozen_validation_sid_threshold"] == validation["sid"]["calibration"]["selected_threshold"]


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
