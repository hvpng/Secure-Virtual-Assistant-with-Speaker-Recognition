from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from module_a.src import evaluation as evaluation_module
from module_a.src.checkpoint import save_checkpoint
from module_a.src.config import config_to_dict
from module_a.src.evaluation import (
    EvaluationError,
    EvaluationModelBundle,
    EMBEDDING_PREPROCESSING_VERSION,
    extract_embeddings,
    get_or_create_embedding_cache,
    load_evaluation_model,
    validate_embedding_vector,
)
from module_a.src.model_factory import build_optimizer
from module_a.src.models.wavlm_frontend import DeterministicFakeWavLM
from module_a.src.training_data import TrainingRecord


def _bundle(tmp_path, config, *, fingerprint="checkpoint-a"):
    return EvaluationModelBundle(
        model=nn.Identity(),
        config=config,
        checkpoint_path=tmp_path / "checkpoint.pt",
        checkpoint_sha256=fingerprint,
        num_classes=2,
        speaker_to_index={"train_a": 0, "train_b": 1},
        device=torch.device("cpu"),
    )


def _unit(index: int, dimension: int = 192) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    vector[index] = 1.0
    return vector


def test_embedding_cache_roundtrip_is_deterministic_and_normalized(
    tmp_path, small_model_config
):
    records = [
        TrainingRecord("val/a.wav", "val_a", "val"),
        TrainingRecord("val/b.wav", "val_b", "val"),
    ]
    embeddings = {"val/a.wav": _unit(0), "val/b.wav": _unit(1)}
    bundle = _bundle(tmp_path, small_model_config)
    path = tmp_path / "embeddings.npz"
    first = get_or_create_embedding_cache(
        path,
        bundle=bundle,
        records=records,
        split="val",
        dataset_root=tmp_path,
        extractor=lambda: embeddings,
    )
    second = get_or_create_embedding_cache(
        path,
        bundle=bundle,
        records=list(reversed(records)),
        split="val",
        dataset_root=tmp_path,
    )
    assert first.metadata == second.metadata
    assert np.array_equal(first.embeddings["val/a.wav"], second.embeddings["val/a.wav"])
    assert first.embeddings["val/a.wav"].dtype == np.float32
    assert np.isclose(np.linalg.norm(first.embeddings["val/a.wav"]), 1.0)


def test_embedding_cache_rejects_checkpoint_incompatibility(tmp_path, small_model_config):
    records = [TrainingRecord("val/a.wav", "val_a", "val")]
    path = tmp_path / "embeddings.npz"
    get_or_create_embedding_cache(
        path,
        bundle=_bundle(tmp_path, small_model_config, fingerprint="a"),
        records=records,
        split="val",
        dataset_root=tmp_path,
        extractor=lambda: {"val/a.wav": _unit(0)},
    )
    with pytest.raises(EvaluationError, match="recompute-embeddings"):
        get_or_create_embedding_cache(
            path,
            bundle=_bundle(tmp_path, small_model_config, fingerprint="b"),
            records=records,
            split="val",
            dataset_root=tmp_path,
        )


def test_embedding_cache_fingerprint_changes_with_preprocessing_version(
    tmp_path, small_model_config, monkeypatch
):
    records = [TrainingRecord("val/a.wav", "val_a", "val")]
    path = tmp_path / "embeddings.npz"
    cache = get_or_create_embedding_cache(
        path,
        bundle=_bundle(tmp_path, small_model_config),
        records=records,
        split="val",
        dataset_root=tmp_path,
        extractor=lambda: {"val/a.wav": _unit(0)},
    )
    assert cache.metadata["schema_version"] == 2
    assert cache.metadata["embedding_preprocessing"] == EMBEDDING_PREPROCESSING_VERSION

    monkeypatch.setattr(
        evaluation_module,
        "EMBEDDING_PREPROCESSING_VERSION",
        "deterministic_center_crop_repeat_pad_v2",
    )
    with pytest.raises(EvaluationError, match="recompute-embeddings"):
        get_or_create_embedding_cache(
            path,
            bundle=_bundle(tmp_path, small_model_config),
            records=records,
            split="val",
            dataset_root=tmp_path,
        )


def test_extract_embeddings_applies_fixed_segment_before_model_call(
    tmp_path, small_model_config, monkeypatch
):
    class CapturingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.waveforms = None
            self.attention_mask = None
            self.grad_enabled = None

        def extract_embedding(self, waveforms, attention_mask=None):
            self.waveforms = waveforms.detach().cpu().clone()
            self.attention_mask = attention_mask.detach().cpu().clone()
            self.grad_enabled = torch.is_grad_enabled()
            output = torch.zeros((waveforms.shape[0], 192), device=waveforms.device)
            output[:, 0] = 1.0
            return output

    config = replace(
        small_model_config,
        audio=replace(
            small_model_config.audio,
            target_sample_rate=4,
            segment_seconds=1.0,
        ),
        evaluation=replace(small_model_config.evaluation, embedding_batch_size=1),
    )
    model = CapturingModel()
    bundle = replace(_bundle(tmp_path, config), model=model)
    monkeypatch.setattr(
        evaluation_module,
        "load_waveform",
        lambda *_args, **_kwargs: torch.tensor([1.0, 2.0, 3.0]),
    )

    embeddings = extract_embeddings(
        bundle,
        [TrainingRecord("val/short.wav", "val_a", "val")],
        dataset_root=tmp_path,
    )

    assert torch.equal(model.waveforms, torch.tensor([[1.0, 2.0, 3.0, 1.0]]))
    assert torch.all(model.attention_mask == 1)
    assert model.grad_enabled is False
    assert model.training is False
    assert embeddings["val/short.wav"].shape == (192,)


@pytest.mark.parametrize(
    "vector",
    [np.zeros(192, dtype=np.float32), np.full(192, np.nan, dtype=np.float32)],
)
def test_embedding_validation_rejects_non_normalized_or_non_finite(vector):
    with pytest.raises(EvaluationError):
        validate_embedding_vector(vector, 192)


def test_evaluation_checkpoint_loader_reconstructs_frozen_model(
    tmp_path, training_model, small_model_config
):
    optimizer = build_optimizer(training_model, 1e-3)
    checkpoint = save_checkpoint(
        tmp_path / "last.pt",
        model=training_model,
        optimizer=optimizer,
        epoch=3,
        step=6501,
        config=config_to_dict(small_model_config),
        num_classes=3,
        speaker_to_index={"train_a": 0, "train_b": 1, "train_c": 2},
    )
    bundle = load_evaluation_model(
        checkpoint,
        device=torch.device("cpu"),
        frontend=DeterministicFakeWavLM(768, frame_count=12),
    )
    assert bundle.model.training is False
    assert bundle.num_classes == 3
    assert bundle.config.model.embedding_dimension == 192
    assert all(not parameter.requires_grad for parameter in bundle.model.encoder.frontend.parameters())


def test_evaluation_checkpoint_loader_rejects_malformed_checkpoint(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"model_state_dict": {}}, path)
    with pytest.raises(EvaluationError, match="required A3 fields"):
        load_evaluation_model(path, device=torch.device("cpu"))
