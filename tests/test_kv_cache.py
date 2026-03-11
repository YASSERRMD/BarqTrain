"""Tests for BarqTrain paged KV-cache helpers."""

import types

import torch

from barqtrain.kv_cache import (
    BarqPagedKVCache,
    BarqPagedKVCacheLayer,
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
