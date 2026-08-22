from __future__ import annotations

import pytest
import torch

from module_a.src.models.campp import (
    CAMPlusPlus,
    CAMPlusPlusError,
    StatisticsPooling,
)


def small_campp() -> CAMPlusPlus:
    return CAMPlusPlus(
        feature_dim=80,
        embedding_dimension=192,
        growth_rate=4,
        block_layers=(1, 1, 1),
        init_channels=8,
        bottleneck_channels=8,
        fcm_channels=4,
        segment_frames=4,
    )


def test_campp_rejects_invalid_input_rank():
    with pytest.raises(CAMPlusPlusError, match="shape"):
        small_campp()(torch.randn(2, 80))


def test_campp_rejects_feature_dimension_other_than_80():
    model = small_campp()
    with pytest.raises(CAMPlusPlusError, match="expected 80"):
        model(torch.randn(2, 79, 12))


def test_campp_returns_fixed_finite_embedding():
    model = small_campp().eval()
    with torch.no_grad():
        output = model(torch.randn(1, 80, 3))

    assert output.shape == (1, 192)
    assert torch.isfinite(output).all()


def test_campp_normal_length_backward_reaches_dense_layers():
    model = small_campp().train()
    output = model(torch.randn(2, 80, 12))
    output.square().mean().backward()

    dense_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "backbone.0.layers.0" in name
    ]
    assert dense_parameters
    assert any(parameter.grad is not None for parameter in dense_parameters)
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in dense_parameters
        if parameter.grad is not None
    )


def test_statistics_pooling_zero_variance_std_and_backward_are_finite():
    pooling = StatisticsPooling()
    inputs = torch.ones(2, 3, 4, requires_grad=True)

    output = pooling(inputs)
    output.sum().backward()

    expected_std = torch.full((2, 3), 1e-5**0.5)
    assert torch.allclose(output[:, 3:], expected_std)
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
