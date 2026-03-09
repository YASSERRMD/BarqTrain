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


@pytest.mark.parametrize(
    "model_type,architectures,patch_attr",
    [
        ("deepseek_v3", [], "patch_deepseek"),
        ("phi3", [], "patch_phi"),
        ("olmo2", [], "patch_olmo"),
        ("granite", [], "patch_granite"),
        ("jamba", [], "patch_jamba"),
        ("llama4_text", [], "patch_llama4"),
        ("unknown", ["DeepseekV3ForCausalLM"], "patch_deepseek"),
        ("unknown", ["Phi3ForCausalLM"], "patch_phi"),
        ("unknown", ["Olmo2ForCausalLM"], "patch_olmo"),
        ("unknown", ["GraniteForCausalLM"], "patch_granite"),
        ("unknown", ["JambaForCausalLM"], "patch_jamba"),
        ("unknown", ["Llama4ForConditionalGeneration"], "patch_llama4"),
    ],
)
def test_patch_model_routes_top_families(monkeypatch, model_type, architectures, patch_attr):
    model = DummyModel(model_type=model_type, architectures=architectures)

    called = {"hit": False}

    def _patch(m):
        called["hit"] = True
        return m

    monkeypatch.setattr(patch_models, patch_attr, _patch)
    patched = patch_models.patch_model(model)

    assert patched is model
    assert called["hit"] is True


def test_patch_model_skips_when_already_patched(monkeypatch):
    model = DummyModel(model_type="qwen2", architectures=["Qwen2ForCausalLM"])

    called = {"count": 0}

    def _patch_qwen(m):
        called["count"] += 1
        return m

    monkeypatch.setattr(patch_models, "patch_qwen", _patch_qwen)

    patch_models.patch_model(model)
    patch_models.patch_model(model)

    assert called["count"] == 1


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


def test_patch_model_routes_mistral_by_model_type(monkeypatch):
    model = DummyModel(model_type="mistral")

    called = {"mistral": False}

    def _patch_mistral(m):
        called["mistral"] = True
        return m

    monkeypatch.setattr(patch_models, "patch_mistral", _patch_mistral)
    patched = patch_models.patch_model(model)

    assert patched is model
    assert called["mistral"] is True


def test_patch_model_routes_mixtral_by_architecture(monkeypatch):
    model = DummyModel(model_type="unknown", architectures=["MixtralForCausalLM"])

    called = {"mistral": False}

    def _patch_mistral(m):
        called["mistral"] = True
        return m

    monkeypatch.setattr(patch_models, "patch_mistral", _patch_mistral)
    patched = patch_models.patch_model(model)

    assert patched is model
    assert called["mistral"] is True


def test_patch_model_routes_qwen_by_model_type(monkeypatch):
    model = DummyModel(model_type="qwen2")

    called = {"qwen": False}

    def _patch_qwen(m):
        called["qwen"] = True
        return m

    monkeypatch.setattr(patch_models, "patch_qwen", _patch_qwen)
    patched = patch_models.patch_model(model)

    assert patched is model
    assert called["qwen"] is True


def test_patch_model_routes_qwen_by_architecture(monkeypatch):
    model = DummyModel(model_type="unknown", architectures=["Qwen3ForCausalLM"])

    called = {"qwen": False}

    def _patch_qwen(m):
        called["qwen"] = True
        return m

    monkeypatch.setattr(patch_models, "patch_qwen", _patch_qwen)
    patched = patch_models.patch_model(model)

    assert patched is model
    assert called["qwen"] is True


def test_patch_model_routes_gemma_by_model_type(monkeypatch):
    model = DummyModel(model_type="gemma2")

    called = {"gemma": False}

    def _patch_gemma(m):
        called["gemma"] = True
        return m

    monkeypatch.setattr(patch_models, "patch_gemma", _patch_gemma)
    patched = patch_models.patch_model(model)

    assert patched is model
    assert called["gemma"] is True


def test_patch_model_routes_gemma_by_architecture(monkeypatch):
    model = DummyModel(model_type="unknown", architectures=["GemmaForCausalLM"])

    called = {"gemma": False}

    def _patch_gemma(m):
        called["gemma"] = True
        return m

    monkeypatch.setattr(patch_models, "patch_gemma", _patch_gemma)
    patched = patch_models.patch_model(model)

    assert patched is model
    assert called["gemma"] is True


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


def test_patch_mistral_patches_rmsnorm_forward():
    pytest.importorskip("transformers")
    pytest.importorskip("transformers.models.mistral.modeling_mistral")
    from transformers.models.mistral.modeling_mistral import MistralRMSNorm

    class TinyMistralModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="mistral",
                architectures=["MistralForCausalLM"],
            )
            self.norm = MistralRMSNorm(16)

        def forward(self, x):
            return self.norm(x)

    model = TinyMistralModel().eval()
    x = torch.randn(3, 16, dtype=torch.float32)

    before = model(x)
    original_forward_impl = model.norm.forward.__func__

    patch_models.patch_mistral(model)

    assert model.norm.forward.__func__ is not original_forward_impl

    after = model(x)
    assert torch.allclose(before, after, rtol=1e-5, atol=1e-6)


def test_patch_deepseek_patches_rmsnorm_forward():
    pytest.importorskip("transformers")
    pytest.importorskip("transformers.models.deepseek_v3.modeling_deepseek_v3")
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RMSNorm

    class TinyDeepseekModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="deepseek_v3",
                architectures=["DeepseekV3ForCausalLM"],
            )
            self.norm = DeepseekV3RMSNorm(16)

        def forward(self, x):
            return self.norm(x)

    model = TinyDeepseekModel().eval()
    x = torch.randn(3, 16, dtype=torch.float32)

    before = model(x)
    original_forward_impl = model.norm.forward.__func__

    patch_models.patch_deepseek(model)

    assert model.norm.forward.__func__ is not original_forward_impl

    after = model(x)
    assert torch.allclose(before, after, rtol=1e-5, atol=1e-6)


def test_patch_phi_patches_rmsnorm_forward():
    pytest.importorskip("transformers")
    pytest.importorskip("transformers.models.phi3.modeling_phi3")
    from transformers.models.phi3.modeling_phi3 import Phi3RMSNorm

    class TinyPhiModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="phi3", architectures=["Phi3ForCausalLM"])
            self.norm = Phi3RMSNorm(16)

        def forward(self, x):
            return self.norm(x)

    model = TinyPhiModel().eval()
    x = torch.randn(3, 16, dtype=torch.float32)

    before = model(x)
    original_forward_impl = model.norm.forward.__func__

    patch_models.patch_phi(model)

    assert model.norm.forward.__func__ is not original_forward_impl

    after = model(x)
    assert torch.allclose(before, after, rtol=1e-5, atol=1e-6)


def test_patch_llama4_patches_rmsnorm_forward():
    pytest.importorskip("transformers")
    pytest.importorskip("transformers.models.llama4.modeling_llama4")
    from transformers.models.llama4.modeling_llama4 import Llama4TextRMSNorm

    class TinyLlama4Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="llama4_text",
                architectures=["Llama4ForConditionalGeneration"],
            )
            self.norm = Llama4TextRMSNorm(16)

        def forward(self, x):
            return self.norm(x)

    model = TinyLlama4Model().eval()
    x = torch.randn(3, 16, dtype=torch.float32)

    before = model(x)
    original_forward_impl = model.norm.forward.__func__

    patch_models.patch_llama4(model)

    assert model.norm.forward.__func__ is not original_forward_impl

    after = model(x)
    assert torch.allclose(before, after, rtol=1e-5, atol=1e-6)


def test_patch_qwen_patches_rmsnorm_forward():
    pytest.importorskip("transformers")
    pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    class TinyQwenModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="qwen2", architectures=["Qwen2ForCausalLM"])
            self.norm = Qwen2RMSNorm(16)

        def forward(self, x):
            return self.norm(x)

    model = TinyQwenModel().eval()
    x = torch.randn(3, 16, dtype=torch.float32)

    before = model(x)
    original_forward_impl = model.norm.forward.__func__

    patch_models.patch_qwen(model)

    assert model.norm.forward.__func__ is not original_forward_impl

    after = model(x)
    assert torch.allclose(before, after, rtol=1e-5, atol=1e-6)


def test_patch_gemma_patches_rmsnorm_forward():
    pytest.importorskip("transformers")
    pytest.importorskip("transformers.models.gemma.modeling_gemma")
    from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

    class TinyGemmaModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="gemma", architectures=["GemmaForCausalLM"])
            self.norm = GemmaRMSNorm(16)

        def forward(self, x):
            return self.norm(x)

    model = TinyGemmaModel().eval()
    x = torch.randn(3, 16, dtype=torch.float32)

    before = model(x)
    original_forward_impl = model.norm.forward.__func__

    patch_models.patch_gemma(model)

    assert model.norm.forward.__func__ is not original_forward_impl

    after = model(x)
    assert torch.allclose(before, after, rtol=1e-5, atol=1e-6)


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
