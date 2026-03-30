"""Tests for BarqTrain paged KV-cache helpers."""

import types

import torch

from barqtrain.kv_cache import (
    BarqContiguousKVCache,
    BarqPagedKVCache,
    BarqPagedKVCacheLayer,
    create_contiguous_kv_cache,
    create_kv_cache,
    maybe_prepare_kv_generate_kwargs,
    create_paged_kv_cache,
    maybe_prepare_paged_kv_generate_kwargs,
)


class DummyDecoderConfig:
    num_hidden_layers = 3


class DummyConfig:
    def get_text_config(self, decoder=True):
        return DummyDecoderConfig()


def test_paged_kv_layer_update_returns_flattened_views():
    layer = BarqPagedKVCacheLayer(max_batch_size=1, max_cache_len=8, page_size=4)

    key_states = torch.arange(24, dtype=torch.float32).view(1, 2, 3, 4)
    value_states = key_states + 100.0
    keys, values = layer.update(key_states, value_states)

    assert keys.shape == (1, 2, 3, 4)
    assert values.shape == (1, 2, 3, 4)
    assert torch.equal(keys, key_states)
    assert torch.equal(values, value_states)
    assert layer.get_seq_length() == 3

    next_keys = torch.arange(16, dtype=torch.float32).view(1, 2, 2, 4) + 1000.0
    next_values = next_keys + 100.0
    keys, values = layer.update(next_keys, next_values)

    assert keys.shape == (1, 2, 5, 4)
    assert values.shape == (1, 2, 5, 4)
    assert torch.equal(keys[:, :, :3], key_states)
    assert torch.equal(keys[:, :, 3:], next_keys)
    assert torch.equal(values[:, :, :3], value_states)
    assert torch.equal(values[:, :, 3:], next_values)
    assert layer.get_seq_length() == 5


def test_create_paged_kv_cache_uses_decoder_layer_count():
    cache = create_paged_kv_cache(
        DummyConfig(),
        max_batch_size=2,
        max_cache_len=64,
        page_size=16,
    )

    assert isinstance(cache, BarqPagedKVCache)
    assert len(cache.layers) == 3
    assert cache.barqtrain_max_batch_size == 2
    assert cache.barqtrain_max_cache_len == 64
    assert cache.page_size == 16


def test_create_contiguous_kv_cache_uses_decoder_layer_count():
    cache = create_contiguous_kv_cache(
        DummyConfig(),
        max_batch_size=2,
        max_cache_len=64,
    )

    assert isinstance(cache, BarqContiguousKVCache)
    assert len(cache.layers) == 3
    assert cache.barqtrain_max_batch_size == 2
    assert cache.barqtrain_max_cache_len == 64


def test_create_kv_cache_respects_mode():
    paged = create_kv_cache(DummyConfig(), max_batch_size=1, max_cache_len=32, mode="paged")
    contiguous = create_kv_cache(DummyConfig(), max_batch_size=1, max_cache_len=32, mode="contiguous")

    assert isinstance(paged, BarqPagedKVCache)
    assert isinstance(contiguous, BarqContiguousKVCache)


def test_maybe_prepare_paged_kv_generate_kwargs_injects_cache(monkeypatch):
    monkeypatch.setattr("barqtrain.kv_cache._get_cuda_backend", lambda: object())
    monkeypatch.setenv("BARQTRAIN_PAGED_KV_MIN_CACHE_LEN", "0")

    model = types.SimpleNamespace(
        config=DummyConfig(),
        generation_config=types.SimpleNamespace(max_new_tokens=None, max_length=32),
        _barqtrain_paged_kv_supported=True,
    )
    fake_input_ids = types.SimpleNamespace(shape=(2, 8), device=torch.device("cuda"))

    updated_kwargs, used = maybe_prepare_paged_kv_generate_kwargs(
        model,
        (),
        {"input_ids": fake_input_ids, "max_new_tokens": 4},
    )

    assert used is True
    assert updated_kwargs["use_cache"] is True
    assert isinstance(updated_kwargs["past_key_values"], BarqPagedKVCache)
    assert updated_kwargs["past_key_values"].barqtrain_max_batch_size == 2
    assert updated_kwargs["past_key_values"].barqtrain_max_cache_len == 12


def test_maybe_prepare_kv_generate_kwargs_can_use_contiguous_mode(monkeypatch):
    monkeypatch.setattr("barqtrain.kv_cache._get_cuda_backend", lambda: None)
    monkeypatch.setenv("BARQTRAIN_KV_CACHE_MODE", "contiguous")
    monkeypatch.setenv("BARQTRAIN_PAGED_KV_MIN_CACHE_LEN", "0")

    model = types.SimpleNamespace(
        config=DummyConfig(),
        generation_config=types.SimpleNamespace(max_new_tokens=None, max_length=32),
        _barqtrain_paged_kv_supported=True,
    )
    fake_input_ids = types.SimpleNamespace(shape=(1, 8), device=torch.device("cuda"))

    updated_kwargs, used = maybe_prepare_kv_generate_kwargs(
        model,
        (),
        {"input_ids": fake_input_ids, "max_new_tokens": 4},
    )

    assert used is True
    assert isinstance(updated_kwargs["past_key_values"], BarqContiguousKVCache)
    assert updated_kwargs["past_key_values"].barqtrain_cache_layout == "contiguous"


def test_maybe_prepare_paged_kv_generate_kwargs_skips_short_decode(monkeypatch):
    monkeypatch.setattr("barqtrain.kv_cache._get_cuda_backend", lambda: object())
    monkeypatch.setenv("BARQTRAIN_PAGED_KV_MIN_CACHE_LEN", "256")

    model = types.SimpleNamespace(
        config=DummyConfig(),
        generation_config=types.SimpleNamespace(max_new_tokens=None, max_length=32),
        _barqtrain_paged_kv_supported=True,
    )
    fake_input_ids = types.SimpleNamespace(shape=(1, 32), device=torch.device("cuda"))

    updated_kwargs, used = maybe_prepare_paged_kv_generate_kwargs(
        model,
        (),
        {"input_ids": fake_input_ids, "max_new_tokens": 16},
    )

    assert used is False
    assert "past_key_values" not in updated_kwargs


def test_paged_kv_layer_recycles_blocks_when_cropped():
    layer = BarqPagedKVCacheLayer(max_batch_size=1, max_cache_len=8, page_size=2, total_blocks=4)

    key_states = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    value_states = key_states + 100.0
    layer.update(key_states, value_states)

    assert layer.resident_blocks() == 2
    assert layer.fragmentation_ratio() == 0.0

    layer.crop(2)

    assert layer.get_seq_length() == 2
    assert layer.resident_blocks() == 1
    assert layer.fragmentation_ratio() == 0.0


def test_paged_kv_layer_raises_when_allocator_exhausts_blocks():
    layer = BarqPagedKVCacheLayer(max_batch_size=1, max_cache_len=8, page_size=2, total_blocks=1)
    key_states = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    value_states = key_states + 100.0

    try:
        layer.update(key_states, value_states)
    except ValueError as exc:
        assert "allocator exhausted" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected allocator exhaustion")


def test_patch_generate_with_paged_kv_records_usage(monkeypatch):
    import barqtrain.patch_models as patch_models

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = DummyConfig()

        def generate(self, *args, **kwargs):
            return kwargs

    monkeypatch.setattr("barqtrain.kv_cache.maybe_prepare_paged_kv_generate_kwargs", lambda model, args, kwargs: ({**kwargs, "sentinel": True}, True))
    monkeypatch.setattr("barqtrain.kv_cache.paged_kv_supported_for_model", lambda model: True)

    model = DummyModel()
    model = patch_models._patch_generate_with_paged_kv(model, "Dummy")
    result = model.generate(input_ids="tokens")

    assert result["sentinel"] is True
    assert model._barqtrain_last_generate_used_paged_kv is True
    assert model._barqtrain_paged_kv_supported is True


def test_patch_generate_records_last_token_logits_specialization(monkeypatch):
    import barqtrain.patch_models as patch_models

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = DummyConfig()

        def forward(self, input_ids=None, logits_to_keep=None):
            return logits_to_keep

        def generate(self, *args, **kwargs):
            return kwargs

    monkeypatch.setattr(
        "barqtrain.kv_cache.maybe_prepare_paged_kv_generate_kwargs",
        lambda model, args, kwargs: (kwargs, False),
    )
    monkeypatch.setattr("barqtrain.kv_cache.paged_kv_supported_for_model", lambda model: False)
    monkeypatch.setattr(
        "barqtrain.memory.maybe_prepare_last_token_logits_generate_kwargs",
        lambda model, args, kwargs: ({**kwargs, "logits_to_keep": 1}, True),
    )

    model = patch_models._patch_generate_with_paged_kv(DummyModel(), "Dummy")
    result = model.generate(input_ids=torch.tensor([[1, 2, 3]]))

    assert result["logits_to_keep"] == 1
    assert model._barqtrain_last_generate_last_token_logits_only is True


def test_patch_generate_preserves_generation_parity_with_last_token_logits_only(monkeypatch):
    import barqtrain.patch_models as patch_models

    class ToyDecodeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(
                model_type="llama",
                architectures=["LlamaForCausalLM"],
            )

        def forward(self, input_ids=None, logits_to_keep=None):
            batch_size, seq_len = input_ids.shape
            vocab_size = 16
            decode_width = 1 if logits_to_keep == 1 else seq_len
            logits = torch.zeros(batch_size, decode_width, vocab_size, dtype=torch.float32)
            next_token = (input_ids[:, -1] + 1) % vocab_size
            logits[:, -1, :] = -1e9
            logits[torch.arange(batch_size), decode_width - 1, next_token] = 1.0
            return types.SimpleNamespace(logits=logits)

        def generate(self, input_ids=None, max_new_tokens=4, logits_to_keep=None, **kwargs):
            tokens = input_ids.clone()
            for _ in range(max_new_tokens):
                outputs = self.forward(input_ids=tokens, logits_to_keep=logits_to_keep)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tokens = torch.cat([tokens, next_token], dim=-1)
            return tokens

    monkeypatch.setattr(
        "barqtrain.kv_cache.maybe_prepare_paged_kv_generate_kwargs",
        lambda model, args, kwargs: (kwargs, False),
    )
    monkeypatch.setattr("barqtrain.kv_cache.paged_kv_supported_for_model", lambda model: False)

    baseline_model = ToyDecodeModel()
    patched_model = patch_models._patch_generate_with_paged_kv(ToyDecodeModel(), "Toy")

    input_ids = torch.tensor([[1, 2, 3]])
    baseline = baseline_model.generate(input_ids=input_ids, max_new_tokens=5)
    patched = patched_model.generate(input_ids=input_ids, max_new_tokens=5)

    assert torch.equal(patched, baseline)
    assert patched_model._barqtrain_last_generate_last_token_logits_only is True
