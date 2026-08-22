from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import module_a.scripts.evaluate_export as evaluate_script
from module_a.src.data import AudioRecord
from module_a.src.ecapa import AAMSoftmax, build_embedding_model
from module_a.src.runtime import extract_embedding, load_model
from module_a.src.training import (
    load_checkpoint_model,
    save_checkpoint,
    train_phase1,
)


def test_checkpoint_save_load_roundtrip(tiny_config, tmp_path):
    model = build_embedding_model(tiny_config).eval()
    classifier = AAMSoftmax(192, 4)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()), lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    waveform = torch.randn(2, 3200)
    with torch.no_grad():
        expected = model.extract_embedding(waveform)
    path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        classifier=classifier,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=tiny_config,
        speaker_to_index={f"s{index}": index for index in range(4)},
        epoch=0,
        global_step=1,
        best_validation_eer=0.25,
    )
    restored, restored_config, payload = load_checkpoint_model(
        path, device=torch.device("cpu")
    )
    with torch.no_grad():
        actual = restored.extract_embedding(waveform)
    assert restored_config["model"]["architecture"] == "ecapa_tdnn"
    assert payload["global_step"] == 1
    assert torch.allclose(expected, actual, atol=1e-6)


def test_phase1_calibration_phase2_frozen_thresholds_and_runtime_abi(
    tiny_config, tmp_path, write_wav, monkeypatch
):
    train_root = tmp_path / "VoxVietnam-T"
    train_records = []
    validation_records = []
    for speaker in range(4):
        for index in range(2):
            relative = f"train_{speaker}/{index}.wav"
            write_wav(train_root / relative)
            train_records.append(AudioRecord(relative, f"train_{speaker}", "train"))
    for speaker in range(2):
        for index in range(2):
            relative = f"validation_{speaker}/{index}.wav"
            write_wav(train_root / relative)
            validation_records.append(
                AudioRecord(relative, f"validation_{speaker}", "validation")
            )

    output = tmp_path / "outputs"
    summary = train_phase1(
        train_records,
        validation_records,
        train_root,
        tiny_config,
        output,
        device=torch.device("cpu"),
        max_steps=1,
    )
    assert summary["global_step"] == 1
    assert (output / "checkpoints" / "last.pt").is_file()
    assert (output / "checkpoints" / "best.pt").is_file()
    sv_calibration_path = output / "calibration" / "sv_calibration.json"
    sid_calibration_path = output / "calibration" / "sid_calibration.json"
    sv_calibration = json.loads(sv_calibration_path.read_text(encoding="utf-8"))
    sid_calibration = json.loads(sid_calibration_path.read_text(encoding="utf-8"))
    calibration_bytes = (sv_calibration_path.read_bytes(), sid_calibration_path.read_bytes())

    test_root = tmp_path / "VoxVietnam-O"
    sample_path = None
    for speaker in range(4):
        for index in range(2):
            candidate = write_wav(test_root / f"test_{speaker}" / f"{index}.wav")
            sample_path = sample_path or candidate
    extraction_calls = 0
    original_extract = evaluate_script.extract_embeddings

    def counted_extract(*args, **kwargs):
        nonlocal extraction_calls
        extraction_calls += 1
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(evaluate_script, "extract_embeddings", counted_extract)
    assert evaluate_script.main(
        [
            "--dataset-root", str(test_root),
            "--output-dir", str(output),
            "--device", "cpu",
            "--sv-protocol", "custom",
        ]
    ) == 0
    assert extraction_calls == 1  # One ECAPA embedding map feeds both SV and SID.
    assert calibration_bytes == (
        sv_calibration_path.read_bytes(), sid_calibration_path.read_bytes()
    )
    sv_test = json.loads((output / "metrics" / "sv_test_metrics.json").read_text())
    sid_test = json.loads((output / "metrics" / "sid_test_metrics.json").read_text())
    assert sv_test["frozen_validation_threshold"] == sv_calibration["threshold"]
    assert sid_test["frozen_validation_threshold"] == sid_calibration["threshold"]
    assert not sv_test["threshold_recalibrated_on_test"]
    assert not sid_test["threshold_recalibrated_on_test"]

    export_dir = output / "module_a_export"
    assert {path.name for path in export_dir.iterdir()} == {
        "model.pt", "config.json", "thresholds.json", "metadata.json"
    }
    thresholds = json.loads((export_dir / "thresholds.json").read_text())
    assert thresholds["threshold_source"] == "validation"
    runtime_model = load_model(export_dir, device="cpu")
    embedding = extract_embedding(runtime_model, sample_path)
    assert isinstance(embedding, np.ndarray)
    assert embedding.dtype == np.float32
    assert embedding.shape == (192,)
    assert np.isfinite(embedding).all()
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-4)
