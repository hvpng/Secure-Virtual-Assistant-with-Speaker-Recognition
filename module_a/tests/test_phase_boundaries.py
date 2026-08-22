from __future__ import annotations

import pytest

import module_a.scripts.evaluate_export as evaluate_script
import module_a.scripts.train_ecapa as train_script
from module_a.src.data import AudioRecord
from module_a.src.evaluation import EvaluationError


def test_phase1_resolves_only_voxvietnam_t(monkeypatch, tiny_config, tmp_path):
    resolved = []

    def fake_resolve(**kwargs):
        resolved.append(kwargs["subset_name"])
        return tmp_path / "VoxVietnam-T"

    train_records = [AudioRecord("a/1.wav", "a", "train")]
    validation_records = [AudioRecord("b/1.wav", "b", "validation")]
    monkeypatch.setattr(train_script, "load_config", lambda _: tiny_config)
    monkeypatch.setattr(train_script, "resolve_dataset_subset", fake_resolve)
    monkeypatch.setattr(
        train_script,
        "prepare_train_validation_manifests",
        lambda *args, **kwargs: (train_records, validation_records),
    )
    monkeypatch.setattr(train_script, "save_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_script, "resolve_device", lambda _: "cpu")
    monkeypatch.setattr(
        train_script,
        "train_phase1",
        lambda *args, **kwargs: {"status": "completed"},
    )
    assert train_script.main(
        [
            "--config", "config.yaml",
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "outputs"),
        ]
    ) == 0
    assert resolved == ["VoxVietnam-T"]


def test_phase2_requires_frozen_calibration_before_touching_voxvietnam_o(
    monkeypatch, tiny_config, tmp_path
):
    touched_test_data = False

    def forbidden_resolve(**kwargs):
        nonlocal touched_test_data
        touched_test_data = True
        raise AssertionError("VoxVietnam-O must not be touched before calibration guard")

    monkeypatch.setattr(
        evaluate_script,
        "load_checkpoint_model",
        lambda *args, **kwargs: (object(), tiny_config, {}),
    )
    monkeypatch.setattr(evaluate_script, "sha256_file", lambda _: "checkpoint-hash")
    monkeypatch.setattr(
        evaluate_script,
        "load_frozen_calibrations",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            EvaluationError("missing calibration")
        ),
    )
    monkeypatch.setattr(evaluate_script, "resolve_dataset_subset", forbidden_resolve)
    monkeypatch.setattr(evaluate_script, "resolve_device", lambda _: "cpu")
    with pytest.raises(EvaluationError, match="missing calibration"):
        evaluate_script.main(
            [
                "--dataset-root", str(tmp_path / "VoxVietnam-O"),
                "--output-dir", str(tmp_path / "outputs"),
            ]
        )
    assert not touched_test_data
