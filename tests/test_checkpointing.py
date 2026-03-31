"""Activation-checkpointing preset tests."""

from types import SimpleNamespace

import torch

from barqtrain.checkpointing import (
    activation_checkpoint_presets,
    apply_activation_checkpointing,
    reset_activation_checkpointing,
)


class TinyCheckpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="llama", use_return_dict=True)
        self.attention = torch.nn.Linear(8, 8)
        self.mlp = torch.nn.Linear(8, 8)
        self.output = torch.nn.Linear(8, 8)

    def forward(self, x):
        hidden = self.attention(x)
        hidden = torch.relu(hidden)
        hidden = self.mlp(hidden)
        return self.output(hidden)


def test_activation_checkpoint_presets_expose_expected_names():
    presets = activation_checkpoint_presets()

    assert set(presets) == {"max_throughput", "balanced", "max_memory_saving"}


def test_balanced_preset_wraps_attention_only():
    model = TinyCheckpointModel()

    apply_activation_checkpointing(model, preset="balanced")

    assert getattr(model.attention, "_barqtrain_checkpoint_wrapped", False) is True
    assert getattr(model.mlp, "_barqtrain_checkpoint_wrapped", False) is False
    assert getattr(model, "_barqtrain_activation_checkpoint_preset", "") == "balanced"


def test_max_memory_saving_wraps_attention_and_mlp():
    model = TinyCheckpointModel()

    apply_activation_checkpointing(model, preset="max_memory_saving")

    assert getattr(model.attention, "_barqtrain_checkpoint_wrapped", False) is True
    assert getattr(model.mlp, "_barqtrain_checkpoint_wrapped", False) is True


def test_reset_activation_checkpointing_restores_original_forward():
    model = TinyCheckpointModel()
    original_forward = model.attention.forward

    apply_activation_checkpointing(model, preset="balanced")
    reset_activation_checkpointing(model)

    assert model.attention.forward == original_forward
    assert getattr(model, "_barqtrain_activation_checkpoint_preset", "") == "max_throughput"


def test_checkpointed_forward_preserves_backward_parity():
    torch.manual_seed(0)
    reference_model = TinyCheckpointModel().train()
    checkpointed_model = TinyCheckpointModel().train()
    checkpointed_model.load_state_dict(reference_model.state_dict())

    inputs = torch.randn(2, 4, 8, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    checkpoint_inputs = inputs.detach().clone().requires_grad_(True)

    reference_loss = reference_model(reference_inputs).sum()
    reference_loss.backward()

    apply_activation_checkpointing(checkpointed_model, preset="max_memory_saving")
    checkpoint_loss = checkpointed_model(checkpoint_inputs).sum()
    checkpoint_loss.backward()

    assert torch.allclose(checkpoint_loss, reference_loss, rtol=1e-5, atol=1e-6)
    assert torch.allclose(checkpoint_inputs.grad, reference_inputs.grad, rtol=1e-5, atol=1e-6)
