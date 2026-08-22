from __future__ import annotations

import pytest
import torch

from module_a.src.models.campp import (
    CAMDenseLayer,
    CAMPlusPlus,
    CAMPlusPlusError,
    StatisticsPooling,
    batch_norm_running_statistics,
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


def test_campp_batch_norm_modules_and_running_stats_are_not_shared():
    model = small_campp()
    batch_norms = [
        module for module in model.modules() if isinstance(module, torch.nn.BatchNorm1d)
    ]

    assert len(batch_norms) > 1
    assert len({id(module) for module in batch_norms}) == len(batch_norms)
    assert len({module.running_mean.untyped_storage().data_ptr() for module in batch_norms}) == len(
        batch_norms
    )
    assert len({module.running_var.untyped_storage().data_ptr() for module in batch_norms}) == len(
        batch_norms
    )


def test_cam_dense_mask_and_local_tdnn_share_normalized_bottleneck_input():
    layer = CAMDenseLayer(
        in_channels=8,
        growth_rate=4,
        bottleneck_channels=8,
        dilation=1,
        segment_frames=4,
    ).train()
    captured = {}

    def capture_normalized(_module, _inputs, output):
        captured["normalized"] = output.detach().clone()

    def capture_mask_input(_module, inputs):
        captured["mask_input"] = inputs[0].detach().clone()

    normalized_hook = layer.tdnn[1].register_forward_hook(capture_normalized)
    mask_hook = layer.mask.register_forward_pre_hook(capture_mask_input)
    try:
        output = layer(torch.randn(4, 8, 12))
    finally:
        normalized_hook.remove()
        mask_hook.remove()

    assert output.shape == (4, 12, 12)
    assert torch.equal(captured["normalized"], captured["mask_input"])


def test_campp_training_updates_bn_stats_without_systematic_variance_collapse():
    torch.manual_seed(42)
    model = small_campp().train()
    for _ in range(40):
        model(torch.randn(8, 80, 24))

    diagnostics = batch_norm_running_statistics(model)
    bottleneck = [item for item in diagnostics if ".bottleneck.0" in str(item["name"])]
    assert bottleneck
    assert all(item["finite"] for item in diagnostics)
    assert min(float(item["running_var_min"]) for item in bottleneck) > 1e-3

    model.eval()
    with torch.no_grad():
        embedding = model(torch.randn(2, 80, 24))
    assert torch.isfinite(embedding).all()


def test_statistics_pooling_zero_variance_std_and_backward_are_finite():
    pooling = StatisticsPooling()
    inputs = torch.ones(2, 3, 4, requires_grad=True)

    output = pooling(inputs)
    output.sum().backward()

    expected_std = torch.full((2, 3), 1e-5**0.5)
    assert torch.allclose(output[:, 3:], expected_std)
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_statistics_pooling_large_finite_input_is_stable_or_fails_clearly():
    pooling = StatisticsPooling()
    stable = torch.tensor([1e18, -1e18], dtype=torch.float32).repeat(2, 3, 4)
    output = pooling(stable)
    assert torch.isfinite(output).all()

    overflowing = torch.tensor([1e20, -1e20], dtype=torch.float32).repeat(2, 3, 4)
    with pytest.raises(CAMPlusPlusError, match="overflowed"):
        pooling(overflowing)
