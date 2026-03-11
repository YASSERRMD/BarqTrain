"""
Optimizer helper tests for BarqTrain.
"""

import builtins

import pytest
import torch

from barqtrain.optim import create_optimizer


def _parameters():
    layer = torch.nn.Linear(4, 4)
    return layer.parameters()


def test_create_optimizer_uses_adamw_by_default():
    optimizer = create_optimizer(_parameters(), lr=1e-4)
    assert isinstance(optimizer, torch.optim.AdamW)


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
