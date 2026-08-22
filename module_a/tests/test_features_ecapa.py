from __future__ import annotations

import torch

from module_a.src.ecapa import AAMSoftmax, build_embedding_model


def test_feature_shape_finite_and_per_utterance_mean_normalized(tiny_config):
    model = build_embedding_model(tiny_config).eval()
    waveforms = torch.randn(2, 3200)
    features = model.features(waveforms)
    assert features.ndim == 3
    assert features.shape[:2] == (2, 80)
    assert torch.isfinite(features).all()
    assert torch.allclose(features.mean(dim=2), torch.zeros(2, 80), atol=1e-5)


def test_ecapa_embedding_shape_finite_and_l2_normalized(tiny_config):
    model = build_embedding_model(tiny_config).eval()
    with torch.no_grad():
        embedding = model.extract_embedding(torch.randn(2, 3200))
    assert embedding.shape == (2, 192)
    assert torch.isfinite(embedding).all()
    assert torch.allclose(embedding.norm(dim=1), torch.ones(2), atol=1e-5)


def test_aam_softmax_is_training_only_and_backward_is_finite(tiny_config):
    model = build_embedding_model(tiny_config).train()
    head = AAMSoftmax(192, 4, margin=0.2, scale=30.0)
    embeddings = model(torch.randn(4, 3200))
    loss = head(embeddings, torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder.embedding.weight.grad is not None
    assert head.weight.grad is not None

