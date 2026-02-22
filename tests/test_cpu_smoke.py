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
    from barqtrain import patch_model, patch_llama, patch_qwen

    assert callable(patch_model)
    assert callable(patch_llama)
    assert callable(patch_qwen)


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
