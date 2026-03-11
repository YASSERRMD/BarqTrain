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


def test_patch_causal_lm_chunked_loss_uses_barqtrain_loss(monkeypatch):
    class TinyBackbone(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.embed = torch.nn.Embedding(32, hidden_size)

        def forward(self, input_ids=None, **kwargs):
            hidden = self.embed(input_ids)
            return SimpleNamespace(
                last_hidden_state=hidden,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )

    class TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="llama",
                architectures=["LlamaForCausalLM"],
                use_return_dict=True,
            )
            self.model = TinyBackbone(hidden_size=8)
            self.lm_head = torch.nn.Linear(8, 32, bias=False)
            self.original_forward_calls = 0

        def forward(self, input_ids=None, labels=None, **kwargs):
            self.original_forward_calls += 1
            hidden = self.model(input_ids=input_ids).last_hidden_state
            logits = self.lm_head(hidden)
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits[..., :-1, :].reshape(-1, logits.size(-1)),
                    labels[..., 1:].reshape(-1),
                )
            return SimpleNamespace(loss=loss, logits=logits)

    model = TinyCausalLM().train()
    called = {"count": 0}

    def fake_chunked_loss(hidden_states, lm_head_weight, labels):
        called["count"] += 1
        assert hidden_states.shape == (2, 3, 8)
        assert labels.shape == (2, 3)
        return hidden_states.sum() * 0 + lm_head_weight.sum() * 0

    monkeypatch.setattr(
        "barqtrain.ops.chunked_cross_entropy_loss",
        fake_chunked_loss,
    )

    patch_models.patch_llama(model)
    outputs = model(
        input_ids=torch.randint(0, 32, (2, 4)),
        labels=torch.randint(0, 32, (2, 4)),
    )

    assert called["count"] == 1
    assert outputs.loss is not None
    assert model.original_forward_calls == 0


def test_patch_causal_lm_chunked_loss_skips_eval(monkeypatch):
    class TinyBackbone(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.embed = torch.nn.Embedding(32, hidden_size)

        def forward(self, input_ids=None, **kwargs):
            hidden = self.embed(input_ids)
            return SimpleNamespace(
                last_hidden_state=hidden,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )

    class TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="llama",
                architectures=["LlamaForCausalLM"],
                use_return_dict=True,
            )
            self.model = TinyBackbone(hidden_size=8)
            self.lm_head = torch.nn.Linear(8, 32, bias=False)
            self.original_forward_calls = 0

        def forward(self, input_ids=None, labels=None, **kwargs):
            self.original_forward_calls += 1
            hidden = self.model(input_ids=input_ids).last_hidden_state
            logits = self.lm_head(hidden)
            return SimpleNamespace(loss=None, logits=logits)

    model = TinyCausalLM().eval()

    patch_models.patch_llama(model)
    _ = model(
        input_ids=torch.randint(0, 32, (2, 4)),
        labels=torch.randint(0, 32, (2, 4)),
    )

    assert model.original_forward_calls == 1


def test_preferred_attention_backend_prefers_flash_attention(monkeypatch):
    patch_models._preferred_attention_backend.cache_clear()
    monkeypatch.setattr(patch_models.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(patch_models.importlib.util, "find_spec", lambda name: object())

    assert patch_models._preferred_attention_backend() == "flash_attention_2"


def test_preferred_attention_backend_falls_back_to_sdpa(monkeypatch):
    patch_models._preferred_attention_backend.cache_clear()
    monkeypatch.setattr(patch_models.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(patch_models.importlib.util, "find_spec", lambda name: None)

    assert patch_models._preferred_attention_backend() == "sdpa"


def test_configure_attention_backend_updates_model_config(monkeypatch):
    model = DummyModel(model_type="llama")
    model.config.attn_implementation = None

    monkeypatch.setattr(
        patch_models,
        "_preferred_attention_backend",
        lambda: "flash_attention_2",
    )

    backend = patch_models._configure_attention_backend(model, "Llama")

    assert backend == "flash_attention_2"
    assert model.config._attn_implementation == "flash_attention_2"
    assert model.config.attn_implementation == "flash_attention_2"


@pytest.mark.parametrize(
    "patch_fn_name,expected_label",
    [
        ("patch_llama", "Llama"),
        ("patch_lfm2", "LFM2"),
        ("patch_mistral", "Mistral/Mixtral"),
        ("patch_deepseek", "DeepSeek"),
        ("patch_phi", "Phi"),
        ("patch_olmo", "OLMo"),
        ("patch_granite", "Granite"),
        ("patch_jamba", "Jamba"),
        ("patch_llama4", "Llama4"),
        ("patch_gemma", "Gemma"),
        ("patch_qwen", "Qwen"),
    ],
)
def test_patch_family_configures_attention_backend(monkeypatch, patch_fn_name, expected_label):
    model = DummyModel(model_type="unknown")
    called = {}

    def _configure(model_arg, label):
        called["label"] = label
        return "sdpa"

    monkeypatch.setattr(patch_models, "_configure_attention_backend", _configure)
    monkeypatch.setattr(
        patch_models,
        "_patch_rmsnorm_targets",
        lambda model_arg, target_specs: model_arg,
    )

    patched = getattr(patch_models, patch_fn_name)(model)

    assert patched is model
    assert called["label"] == expected_label


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
