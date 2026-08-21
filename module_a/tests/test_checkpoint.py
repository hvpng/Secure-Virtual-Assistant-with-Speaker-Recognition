from __future__ import annotations

import torch

from module_a.src.checkpoint import CHECKPOINT_FIELDS, load_checkpoint, save_checkpoint
from module_a.src.config import config_to_dict
from module_a.src.model_factory import build_model, build_optimizer
from module_a.src.models.wavlm_frontend import DeterministicFakeWavLM


def test_checkpoint_contains_resume_fields_and_restores_optimizer(
    tmp_path, training_model, waveform_batch, small_model_config
):
    optimizer = build_optimizer(training_model, 1e-3)
    loss = training_model(waveform_batch, torch.tensor([0, 1])).loss
    loss.backward()
    optimizer.step()
    path = save_checkpoint(
        tmp_path / "sanity.pt",
        model=training_model,
        optimizer=optimizer,
        epoch=2,
        step=7,
        config=config_to_dict(small_model_config),
        num_classes=3,
        speaker_to_index={"a": 0, "b": 1, "c": 2},
    )

    payload = load_checkpoint(path, model=training_model, optimizer=optimizer)

    assert path.is_file()
    assert CHECKPOINT_FIELDS.issubset(payload)
    assert payload["epoch"] == 2
    assert payload["step"] == 7
    assert payload["speaker_to_index"] == {"a": 0, "b": 1, "c": 2}
    assert payload["config"]["model"]["architecture"] == "wavlm_base_plus_campp"
    assert payload["config"]["model"]["embedding_dimension"] == 192


def test_checkpoint_roundtrip_is_deterministic_in_eval(
    tmp_path, training_model, waveform_batch, small_model_config
):
    optimizer = build_optimizer(training_model, 1e-3)
    training_model.eval()
    with torch.no_grad():
        expected = training_model.extract_embedding(waveform_batch)
    path = save_checkpoint(
        tmp_path / "roundtrip.pt",
        model=training_model,
        optimizer=optimizer,
        epoch=0,
        step=0,
        config={"embedding_dimension": 192},
        num_classes=3,
        speaker_to_index={"a": 0, "b": 1, "c": 2},
    )
    restored = build_model(
        small_model_config,
        num_classes=3,
        frontend=DeterministicFakeWavLM(768, frame_count=12),
    )
    load_checkpoint(path, model=restored)
    restored.eval()
    with torch.no_grad():
        actual = restored.extract_embedding(waveform_batch)

    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)
