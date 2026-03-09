"""
Tests for model patch routing and LFM2 patch integration.
"""

from types import SimpleNamespace

import pytest
import torch

from barqtrain import patch_models


class DummyModel(torch.nn.Module):
    def __init__(self, model_type=None, architectures=None):
        super().__init__()
        self.config = SimpleNamespace(
            model_type=model_type,
            architectures=architectures or [],
        )


def test_patch_model_routes_lfm2_by_model_type(monkeypatch):
    model = DummyModel(model_type="lfm2")

    called = {"lfm2": False}

    def _patch_lfm2(m):
        called["lfm2"] = True
        return m

    monkeypatch.setattr(patch_models, "patch_lfm2", _patch_lfm2)
    patched = patch_models.patch_model(model)

    assert patched is model
    assert called["lfm2"] is True


def test_patch_model_routes_lfm2_by_architecture(monkeypatch):
    model = DummyModel(model_type="unknown", architectures=["Lfm2ForCausalLM"])

    called = {"lfm2": False}

    def _patch_lfm2(m):
        called["lfm2"] = True
        return m

    monkeypatch.setattr(patch_models, "patch_lfm2", _patch_lfm2)
    patched = patch_models.patch_model(model)

    assert patched is model
    assert called["lfm2"] is True


def test_patch_lfm2_patches_rmsnorm_forward():
    pytest.importorskip("transformers")
    pytest.importorskip("transformers.models.lfm2.modeling_lfm2")
    from transformers.models.lfm2.modeling_lfm2 import Lfm2RMSNorm

    class TinyLfm2Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="lfm2", architectures=["Lfm2ForCausalLM"])
            self.norm = Lfm2RMSNorm(16)

        def forward(self, x):
            return self.norm(x)

    model = TinyLfm2Model().eval()
    x = torch.randn(3, 16, dtype=torch.float32)

    before = model(x)
    original_forward_impl = model.norm.forward.__func__

    patch_models.patch_lfm2(model)

    assert model.norm.forward.__func__ is not original_forward_impl

    after = model(x)
    assert torch.allclose(before, after, rtol=1e-5, atol=1e-6)
