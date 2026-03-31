"""
Optimizer helper tests for BarqTrain.
"""

import builtins
import copy

import pytest
import torch

from barqtrain.optim import BarqTrainAdamW, create_optimizer, optimizer_state_bytes


def _parameters():
    layer = torch.nn.Linear(4, 4)
    return layer.parameters()


def test_create_optimizer_uses_adamw_by_default():
    optimizer = create_optimizer(_parameters(), lr=1e-4)
    assert isinstance(optimizer, torch.optim.AdamW)


def test_create_optimizer_supports_native_barqtrain_modes():
    optimizer = create_optimizer(_parameters(), optimizer_name="barqtrain_adamw")
    compact_optimizer = create_optimizer(_parameters(), optimizer_name="barqtrain_adamw_compact")
    paged_optimizer = create_optimizer(_parameters(), optimizer_name="barqtrain_adamw_paged")

    assert isinstance(optimizer, BarqTrainAdamW)
    assert optimizer.state_mode == "full"
    assert compact_optimizer.state_mode == "compact"
    assert paged_optimizer.state_mode == "paged"


def test_create_optimizer_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported optimizer_name"):
        create_optimizer(_parameters(), optimizer_name="unknown")


def test_create_optimizer_falls_back_when_bitsandbytes_is_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "bitsandbytes":
            raise ImportError("bitsandbytes intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.warns(UserWarning, match="bitsandbytes is not installed"):
        optimizer = create_optimizer(_parameters(), optimizer_name="paged_adamw_32bit")

    assert isinstance(optimizer, torch.optim.AdamW)


def test_barqtrain_adamw_matches_adamw_on_single_step():
    torch.manual_seed(0)
    reference_model = torch.nn.Linear(4, 4)
    native_model = torch.nn.Linear(4, 4)
    native_model.load_state_dict(copy.deepcopy(reference_model.state_dict()))

    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 4)

    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=1e-3)
    native_optimizer = BarqTrainAdamW(native_model.parameters(), lr=1e-3)

    reference_loss = torch.nn.functional.mse_loss(reference_model(inputs), targets)
    native_loss = torch.nn.functional.mse_loss(native_model(inputs), targets)
    reference_loss.backward()
    native_loss.backward()

    reference_optimizer.step()
    native_optimizer.step()

    for reference_param, native_param in zip(reference_model.parameters(), native_model.parameters()):
        assert torch.allclose(native_param, reference_param, rtol=1e-5, atol=1e-6)


def test_optimizer_state_bytes_reflects_compact_and_paged_layouts():
    full_layer = torch.nn.Linear(4, 4)
    compact_layer = copy.deepcopy(full_layer)
    paged_layer = copy.deepcopy(full_layer)
    optimizer_full = BarqTrainAdamW(full_layer.parameters(), lr=1e-3, state_mode="full")
    optimizer_compact = BarqTrainAdamW(compact_layer.parameters(), lr=1e-3, state_mode="compact")
    optimizer_paged = BarqTrainAdamW(paged_layer.parameters(), lr=1e-3, state_mode="paged", page_size=4)

    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 4)

    for optimizer, module in (
        (optimizer_full, full_layer),
        (optimizer_compact, compact_layer),
        (optimizer_paged, paged_layer),
    ):
        module.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(module(inputs), targets)
        loss.backward()
        optimizer.step()

    full_bytes = optimizer_state_bytes(optimizer_full)
    compact_bytes = optimizer_state_bytes(optimizer_compact)
    paged_bytes = optimizer_state_bytes(optimizer_paged)

    assert full_bytes > 0
    assert compact_bytes > 0
    assert paged_bytes > 0
    assert compact_bytes <= full_bytes


def test_barqtrain_optimizer_converges_close_to_adamw():
    torch.manual_seed(0)
    reference_model = torch.nn.Linear(4, 2)
    compact_model = torch.nn.Linear(4, 2)
    paged_model = torch.nn.Linear(4, 2)
    compact_model.load_state_dict(copy.deepcopy(reference_model.state_dict()))
    paged_model.load_state_dict(copy.deepcopy(reference_model.state_dict()))

    inputs = torch.randn(16, 4)
    targets = torch.randn(16, 2)

    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=1e-2)
    compact_optimizer = BarqTrainAdamW(compact_model.parameters(), lr=1e-2, state_mode="compact")
    paged_optimizer = BarqTrainAdamW(paged_model.parameters(), lr=1e-2, state_mode="paged", page_size=8)

    losses = {}
    for name, model, optimizer in (
        ("reference", reference_model, reference_optimizer),
        ("compact", compact_model, compact_optimizer),
        ("paged", paged_model, paged_optimizer),
    ):
        curve = []
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
            curve.append(float(loss.detach()))
        losses[name] = curve

    assert losses["reference"][-1] < losses["reference"][0]
    assert abs(losses["compact"][-1] - losses["reference"][-1]) < 1e-2
    assert abs(losses["paged"][-1] - losses["reference"][-1]) < 1e-3
