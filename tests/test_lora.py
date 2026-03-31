"""Tests for BarqTrain fused LoRA helpers."""

from types import SimpleNamespace

import torch

from barqtrain.lora import FusedLoRALinear, fused_lora_linear, patch_lora_modules
from barqtrain.ops import chunked_cross_entropy_loss


def test_fused_lora_linear_matches_reference_on_3d_inputs():
    x = torch.randn(2, 3, 8)
    w_base = torch.randn(16, 8)
    a = torch.randn(4, 8)
    b = torch.randn(16, 4)
    scaling = 0.5

    output = fused_lora_linear(x, w_base, a, b, scaling=scaling)
    expected = torch.nn.functional.linear(x, w_base) + (x @ a.T @ b.T) * scaling

    assert output.shape == expected.shape
    assert torch.allclose(output, expected, rtol=1e-5, atol=1e-6)


def test_fused_lora_linear_supports_mixed_precision_base_weights():
    x = torch.randn(2, 4, 8, dtype=torch.float16)
    w_base = torch.randn(16, 8, dtype=torch.float32)
    a = torch.randn(4, 8, dtype=torch.float32)
    b = torch.randn(16, 4, dtype=torch.float32)

    output = fused_lora_linear(x, w_base, a, b, scaling=0.25)

    assert output.dtype == x.dtype
    assert output.shape == (2, 4, 16)


def test_patch_lora_modules_replaces_targeted_linear_layers():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.out_proj = torch.nn.Linear(8, 8, bias=False)

    model = ToyModel()
    patch_lora_modules(model, target_modules=("q_proj",), rank=2, alpha=4.0)

    assert isinstance(model.q_proj, FusedLoRALinear)
    assert isinstance(model.out_proj, torch.nn.Linear)
    assert model._barqtrain_fused_lora_patched is True
    assert model._barqtrain_fused_lora_modules == ("q_proj",)
    assert model.q_proj.base_weight.requires_grad is False


def test_fused_lora_merge_weights_matches_explicit_delta():
    linear = torch.nn.Linear(8, 8, bias=False)
    lora = FusedLoRALinear.from_linear(linear, rank=2, alpha=4.0)
    with torch.no_grad():
        lora.lora_A.copy_(torch.arange(16, dtype=torch.float32).reshape(2, 8) / 10.0)
        lora.lora_B.copy_(torch.arange(16, dtype=torch.float32).reshape(8, 2) / 20.0)
        baseline = lora.base_weight.detach().clone()

    lora.merge_weights()
    expected = baseline + lora.scaling * (lora.lora_B @ lora.lora_A)

    assert torch.allclose(lora.base_weight, expected, rtol=1e-5, atol=1e-6)


def test_patched_lora_modules_stay_compatible_with_chunked_loss():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8)
            self.proj = torch.nn.Linear(8, 8, bias=False)
            self.lm_head = torch.nn.Linear(8, 16, bias=False)

    model = ToyModel()
    patch_lora_modules(model, target_modules=("proj",), rank=2, alpha=4.0)

    hidden = torch.randn(2, 4, 8, requires_grad=True)
    labels = torch.randint(0, 16, (2, 4))
    projected = model.proj(hidden)
    loss = chunked_cross_entropy_loss(projected, model.lm_head.weight, labels)
    loss.backward()

    assert loss.shape == ()
    assert projected.shape == (2, 4, 8)
    assert model.proj.lora_A.grad is not None
    assert model.proj.lora_B.grad is not None
