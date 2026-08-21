from __future__ import annotations

import pytest
import torch

from module_a.src.device import resolve_device
from module_a.src.model_factory import build_model, build_optimizer
from module_a.src.models.speaker_model import SpeakerModelError
from module_a.src.models.wavlm_frontend import DeterministicFakeWavLM


def test_fake_wavlm_768_is_accepted_and_wrong_dimension_rejected(small_model_config):
    accepted = build_model(
        small_model_config,
        num_classes=3,
        frontend=DeterministicFakeWavLM(768),
    )
    assert accepted.encoder.wavlm_hidden_dimension == 768

    with pytest.raises(SpeakerModelError, match="dimension mismatch"):
        build_model(
            small_model_config,
            num_classes=3,
            frontend=DeterministicFakeWavLM(767),
        )


def test_adapter_and_model_dimensions_follow_config(
    training_model, small_model_config, waveform_batch
):
    hidden, adapted = training_model.encoder.extract_features(waveform_batch)
    raw = training_model.encoder(waveform_batch)

    assert hidden.shape == (2, 12, 768)
    assert adapted.shape == (2, 12, 80)
    assert training_model.encoder.campp.feature_dim == 80
    assert raw.shape == (2, small_model_config.model.embedding_dimension)


def test_batch_size_one_inference_is_finite_and_l2_normalized(
    training_model, waveform_batch
):
    training_model.eval()
    with torch.no_grad():
        embedding = training_model.extract_embedding(waveform_batch[:1])

    assert embedding.shape == (1, 192)
    assert torch.isfinite(embedding).all()
    assert embedding.norm(dim=1) == pytest.approx(torch.ones(1), abs=1e-5)


def test_training_batch_returns_finite_loss_and_logits(training_model, waveform_batch):
    output = training_model(waveform_batch, torch.tensor([0, 1]))

    assert output.raw_embedding.shape == (2, 192)
    assert output.logits.shape == (2, 3)
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)


def test_backward_gradients_only_follow_trainable_path(training_model, waveform_batch):
    output = training_model(waveform_batch, torch.tensor([0, 1]))
    output.loss.backward()

    assert training_model.encoder.projection.weight.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in training_model.encoder.campp.parameters()
        if parameter.requires_grad
    )
    assert training_model.aam_head.weight.grad is not None
    assert all(
        parameter.grad is None for parameter in training_model.encoder.frontend.parameters()
    )
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in training_model.parameters()
        if parameter.grad is not None
    )


def test_optimizer_excludes_frozen_wavlm_and_step_changes_backend(
    training_model, waveform_batch
):
    optimizer = build_optimizer(training_model, 1e-3)
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    frontend_before = {
        name: parameter.detach().clone()
        for name, parameter in training_model.encoder.frontend.named_parameters()
    }
    backend_before = training_model.encoder.projection.weight.detach().clone()

    loss = training_model(waveform_batch, torch.tensor([0, 1])).loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert all(
        id(parameter) not in optimizer_parameter_ids
        for parameter in training_model.encoder.frontend.parameters()
    )
    assert not torch.equal(backend_before, training_model.encoder.projection.weight)
    assert all(
        torch.equal(frontend_before[name], parameter)
        for name, parameter in training_model.encoder.frontend.named_parameters()
    )


@pytest.mark.parametrize(
    "waveforms, message",
    [
        (torch.randn(1_600), "shape"),
        (torch.randn(2, 399), "at least"),
    ],
)
def test_invalid_or_short_waveform_is_rejected(training_model, waveforms, message):
    with pytest.raises(SpeakerModelError, match=message):
        training_model.encoder(waveforms)


def test_nan_waveform_is_rejected(training_model, waveform_batch):
    waveform_batch[0, 0] = float("nan")
    with pytest.raises(SpeakerModelError, match="NaN"):
        training_model.encoder(waveform_batch)


def test_model_factory_is_cpu_safe_and_stage1_frozen(training_model):
    assert resolve_device("cpu") == torch.device("cpu")
    assert all(
        not parameter.requires_grad for parameter in training_model.encoder.frontend.parameters()
    )
    assert training_model.encoder.frontend.training is False
    assert training_model.encoder.layer_norm.weight.requires_grad
    assert training_model.encoder.projection.weight.requires_grad

