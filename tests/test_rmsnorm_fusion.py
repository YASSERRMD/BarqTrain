"""RMSNorm block-fusion helper tests."""

import torch

from barqtrain.ops import (
    fused_residual_rms_norm,
    fused_residual_rms_norm_linear,
    fused_rms_norm,
    fused_rms_norm_linear,
)


def test_fused_residual_rms_norm_matches_composed_ops():
    x = torch.randn(2, 4, 8)
    residual = torch.randn(2, 4, 8)
    weight = torch.ones(8)

    output = fused_residual_rms_norm(x, residual, weight)
    expected = fused_rms_norm(x + residual, weight)

    assert torch.allclose(output, expected, rtol=1e-5, atol=1e-6)


def test_fused_rms_norm_linear_matches_composed_ops():
    x = torch.randn(2, 4, 8)
    norm_weight = torch.ones(8)
    linear_weight = torch.randn(16, 8)
    bias = torch.randn(16)

    output = fused_rms_norm_linear(x, norm_weight, linear_weight, bias)
    expected = torch.nn.functional.linear(fused_rms_norm(x, norm_weight), linear_weight, bias)

    assert torch.allclose(output, expected, rtol=1e-5, atol=1e-6)


def test_fused_residual_rms_norm_linear_matches_composed_ops():
    x = torch.randn(2, 4, 8)
    residual = torch.randn(2, 4, 8)
    norm_weight = torch.ones(8)
    linear_weight = torch.randn(16, 8)

    output = fused_residual_rms_norm_linear(x, residual, norm_weight, linear_weight)
    expected = torch.nn.functional.linear(fused_rms_norm(x + residual, norm_weight), linear_weight)

    assert torch.allclose(output, expected, rtol=1e-5, atol=1e-6)
