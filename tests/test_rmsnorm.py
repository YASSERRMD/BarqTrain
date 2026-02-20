"""
Numerical parity tests for BarqTrain RMSNorm kernel

This module verifies that the fused RMSNorm kernel produces
identical results to the standard PyTorch implementation.
"""

import pytest
import torch

pytest.importorskip("barqtrain_cuda", reason="CUDA extension not built")

from barqtrain.ops import FusedRMSNorm, fused_rms_norm


def test_fused_rmsnorm_forward_parity():
    """Test that forward pass matches PyTorch RMSNorm"""
    torch.manual_seed(42)

    batch_size = 4
    seq_len = 128
    hidden_size = 768

    # Create test data
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)
    weight = torch.ones(hidden_size, device="cuda", dtype=torch.float16)

    # PyTorch implementation
    x_reshape = x.view(-1, hidden_size)
    variance_pt = x_reshape.pow(2).mean(dim=-1, keepdim=True) + 1e-6
    x_normalized_pt = x_reshape * torch.rsqrt(variance_pt)
    output_pt = (x_normalized_pt * weight).view(batch_size, seq_len, hidden_size)

    # BarqTrain implementation
    output_bt = fused_rms_norm(x, weight, eps=1e-6)

    # Check parity (allowing for small numerical differences due to float16)
    assert torch.allclose(output_pt, output_bt, rtol=1e-3, atol=1e-5), \
        f"Forward pass mismatch: max diff = {(output_pt - output_bt).abs().max().item()}"

    print("✓ Forward pass parity test passed")


def test_fused_rmsnorm_backward_parity():
    """Test that backward pass matches PyTorch RMSNorm"""
    torch.manual_seed(42)

    batch_size = 2
    hidden_size = 256

    # Create test data
    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.ones(hidden_size, device="cuda", dtype=torch.float32, requires_grad=True)

    # PyTorch implementation
    x_pt = x.detach().clone().requires_grad_(True)
    weight_pt = weight.detach().clone().requires_grad_(True)

    variance_pt = x_pt.pow(2).mean(dim=-1, keepdim=True) + 1e-6
    x_normalized_pt = x_pt * torch.rsqrt(variance_pt)
    output_pt = x_normalized_pt * weight_pt
    output_pt.sum().backward()

    # BarqTrain implementation
    x_bt = x.detach().clone().requires_grad_(True)
    weight_bt = weight.detach().clone().requires_grad_(True)

    output_bt = fused_rms_norm(x_bt, weight_bt, eps=1e-6)
    output_bt.sum().backward()

    # Check gradients
    assert torch.allclose(x_pt.grad, x_bt.grad, rtol=1e-4, atol=1e-6), \
        f"Input gradient mismatch: max diff = {(x_pt.grad - x_bt.grad).abs().max().item()}"

    assert torch.allclose(weight_pt.grad, weight_bt.grad, rtol=1e-4, atol=1e-6), \
        f"Weight gradient mismatch: max diff = {(weight_pt.grad - weight_bt.grad).abs().max().item()}"

    print("✓ Backward pass parity test passed")


def test_fused_rmsnorm_module():
    """Test FusedRMSNorm as a drop-in replacement"""
    torch.manual_seed(42)

    batch_size = 8
    seq_len = 64
    hidden_size = 512

    # Create layer
    layer = FusedRMSNorm(hidden_size, eps=1e-6).cuda().half()

    # Test forward
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)
    output = layer(x)

    assert output.shape == x.shape, "Output shape mismatch"
    assert output.dtype == x.dtype, "Output dtype mismatch"

    # Test backward
    output.sum().backward()

    assert layer.weight.grad is not None, "Weight gradient is None"

    print("✓ Module interface test passed")


def test_fused_rmsnorm_2d_input():
    """Test with 2D input (batch, hidden)"""
    torch.manual_seed(42)

    batch_size = 16
    hidden_size = 1024

    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float16)
    weight = torch.ones(hidden_size, device="cuda", dtype=torch.float16)

    output = fused_rms_norm(x, weight, eps=1e-6)

    assert output.shape == x.shape
    assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    print("✓ 2D input test passed")


if __name__ == "__main__":
    print("Running RMSNorm parity tests...")
    print()

    test_fused_rmsnorm_forward_parity()
    test_fused_rmsnorm_backward_parity()
    test_fused_rmsnorm_module()
    test_fused_rmsnorm_2d_input()

    print()
    print("All tests passed! ✓")
