"""
Smoke tests for BarqTrain – no CUDA required.

These tests verify the Python-level API (imports, fallback ops, patch_models)
work correctly without any GPU or compiled extensions.
"""

import pytest
import torch


def test_barqtrain_imports():
    """BarqTrain package should import cleanly without CUDA."""
    import barqtrain

    assert hasattr(barqtrain, "__version__")
    assert barqtrain.__version__ == "0.1.0"


def test_barqtrain_api_exports():
    """Core API symbols are exported from the top-level package."""
    from barqtrain import (
        PackedCausalLMDataCollator,
        create_optimizer,
        pack_for_causal_lm,
        patch_llama,
        patch_model,
        patch_qwen,
    )

    assert callable(patch_model)
    assert callable(patch_llama)
    assert callable(patch_qwen)
    assert callable(pack_for_causal_lm)
    assert callable(create_optimizer)
    assert callable(PackedCausalLMDataCollator)


def test_ops_fallback_rmsnorm():
    """fused_rms_norm falls back to PyTorch when CUDA ext is absent."""
    from barqtrain.ops import fused_rms_norm

    x = torch.randn(2, 64, device="cpu")
    w = torch.ones(64, device="cpu")
    out = fused_rms_norm(x, w, eps=1e-6)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert out.dtype == x.dtype


def test_ops_fallback_rmsnorm_3d():
    """fused_rms_norm handles 3-D input (batch, seq, hidden)."""
    from barqtrain.ops import fused_rms_norm

    x = torch.randn(2, 16, 64, device="cpu")
    w = torch.ones(64, device="cpu")
    out = fused_rms_norm(x, w)

    assert out.shape == x.shape


def test_fused_rmsnorm_module_cpu():
    """FusedRMSNorm module works on CPU with fallback."""
    from barqtrain.ops import FusedRMSNorm

    layer = FusedRMSNorm(hidden_size=128)
    x = torch.randn(4, 32, 128)
    out = layer(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_ops_fallback_flash_attention():
    """flash_attention falls back to sdpa when CUDA ext is absent."""
    from barqtrain.ops import flash_attention

    q = torch.randn(1, 4, 16, 32)
    k = torch.randn(1, 4, 16, 32)
    v = torch.randn(1, 4, 16, 32)
    out = flash_attention(q, k, v)

    assert out.shape == q.shape


def test_ops_fallback_chunked_cross_entropy():
    """chunked_cross_entropy_loss falls back to F.cross_entropy on CPU."""
    from barqtrain.ops import chunked_cross_entropy_loss

    hidden = torch.randn(2, 8, 64)
    lm_head = torch.randn(100, 64)
    labels = torch.randint(0, 100, (2, 8))
    loss = chunked_cross_entropy_loss(hidden, lm_head, labels)

    assert loss.shape == ()  # scalar
    assert torch.isfinite(loss)


def test_ops_fallback_chunked_cross_entropy_matches_torch_ignore_index():
    """CPU fallback should honor ignore_index and match PyTorch numerically."""
    from barqtrain.ops import chunked_cross_entropy_loss

    hidden = torch.randn(2, 4, 16, requires_grad=True)
    lm_head = torch.randn(32, 16, requires_grad=True)
    labels = torch.randint(0, 32, (2, 4))
    labels[0, 1] = -100

    bt_loss = chunked_cross_entropy_loss(hidden, lm_head, labels)
    torch_loss = torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(hidden, lm_head).view(-1, lm_head.size(0)),
        labels.view(-1),
        reduction="mean",
        ignore_index=-100,
    )

    assert torch.allclose(bt_loss, torch_loss, rtol=1e-5, atol=1e-6)


def test_ops_fallback_chunked_cross_entropy_backward():
    """CPU fallback should propagate gradients to hidden states and lm_head."""
    from barqtrain.ops import chunked_cross_entropy_loss

    hidden = torch.randn(2, 4, 16, requires_grad=True)
    lm_head = torch.randn(32, 16, requires_grad=True)
    labels = torch.randint(0, 32, (2, 4))
    labels[1, 2] = -100

    loss = chunked_cross_entropy_loss(hidden, lm_head, labels)
    loss.backward()

    assert hidden.grad is not None
    assert lm_head.grad is not None
    assert torch.isfinite(hidden.grad).all()
    assert torch.isfinite(lm_head.grad).all()


def test_chunked_cross_entropy_cuda_wrapper_preserves_input_grad_dtypes(monkeypatch):
    """CUDA wrapper should cast backend gradients back to input dtypes."""
    import barqtrain.ops as ops

    class FakeCudaBackend:
        @staticmethod
        def chunked_cross_entropy(hidden_states, lm_head_weight, labels):
            loss = torch.tensor(1.0, dtype=torch.float32)
            grad_hidden = torch.ones_like(hidden_states, dtype=torch.float32)
            grad_lm_head = torch.ones_like(lm_head_weight, dtype=torch.float32)
            return loss, grad_hidden, grad_lm_head

    monkeypatch.setattr(ops, "_get_cuda_backend", lambda: FakeCudaBackend())

    hidden = torch.randn(2, 3, 4, dtype=torch.float16, requires_grad=True)
    lm_head = torch.randn(8, 4, dtype=torch.float16, requires_grad=True)
    labels = torch.randint(0, 8, (2, 3), dtype=torch.long)

    loss = ops.chunked_cross_entropy_loss(hidden, lm_head, labels)
    loss.backward()

    assert hidden.grad is not None
    assert lm_head.grad is not None
    assert hidden.grad.dtype == hidden.dtype
    assert lm_head.grad.dtype == lm_head.dtype


def test_patch_model_generic():
    """patch_model returns the model unchanged for generic models."""
    from barqtrain import patch_model

    class DummyConfig:
        model_type = "unknown"

    class DummyModel(torch.nn.Module):
        config = DummyConfig()

    model = DummyModel()
    result = patch_model(model)
    assert result is model
