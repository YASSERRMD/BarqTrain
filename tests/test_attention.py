"""Tests for BarqTrain attention dispatch helpers."""

from types import SimpleNamespace

import torch

from barqtrain.attention import (
    dispatch_attention,
    materialize_kv_for_attention,
    select_attention_backend,
)
from barqtrain.ops import apply_rope_to_qk, padding_free_attention


class _FakeCacheLayer:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self._keys = keys
        self._values = values

    def current_tensors(self):
        return self._keys, self._values


def test_select_attention_backend_prefers_native_decode_for_last_token_cache():
    q = torch.randn(2, 4, 1, 8)
    k = torch.randn(2, 4, 1, 8)
    v = torch.randn(2, 4, 1, 8)
    cache = SimpleNamespace(barqtrain_cache_layout="paged", layers=[])

    decision = select_attention_backend(q, k, v, cache=cache, last_token_only=True)

    assert decision.backend == "barqtrain_native_decode"
    assert decision.cache_layout == "paged"
    assert decision.last_token_only is True


def test_materialize_kv_for_attention_reads_cache_layer():
    keys = torch.randn(2, 4, 3, 8)
    values = torch.randn(2, 4, 3, 8)
    cache = SimpleNamespace(layers=[_FakeCacheLayer(keys, values)])

    materialized_keys, materialized_values = materialize_kv_for_attention(cache, layer_index=0)

    assert torch.equal(materialized_keys, keys)
    assert torch.equal(materialized_values, values)


def test_dispatch_attention_matches_sdpa_for_last_token_decode():
    cached_k = torch.randn(2, 4, 3, 8)
    cached_v = torch.randn(2, 4, 3, 8)
    current_q = torch.randn(2, 4, 2, 8)
    current_k = torch.randn(2, 4, 2, 8)
    current_v = torch.randn(2, 4, 2, 8)
    cache = SimpleNamespace(
        barqtrain_cache_layout="paged_quantized",
        layers=[_FakeCacheLayer(cached_k, cached_v)],
    )

    output, decision = dispatch_attention(
        current_q,
        current_k,
        current_v,
        cache=cache,
        last_token_only=True,
        requested_backend="barqtrain_native_decode",
    )

    expected_q, expected_k = apply_rope_to_qk(current_q[:, :, -1:, :], torch.cat([cached_k, current_k], dim=-2))
    expected = torch.nn.functional.scaled_dot_product_attention(
        expected_q,
        expected_k,
        torch.cat([cached_v, current_v], dim=-2),
        is_causal=False,
    )

    assert decision.backend == "barqtrain_native_decode"
    assert decision.cache_layout == "paged_quantized"
    assert output.shape == expected.shape
    assert torch.allclose(output, expected, rtol=1e-5, atol=1e-6)


def test_dispatch_attention_uses_padding_free_path_for_packed_sequences():
    q = torch.randn(5, 2, 4)
    k = torch.randn(5, 2, 4)
    v = torch.randn(5, 2, 4)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.long)

    output, decision = dispatch_attention(
        q,
        k,
        v,
        packed_cu_seqlens=cu_seqlens,
        requested_backend="barqtrain_native_decode",
    )
    expected = padding_free_attention(q, k, v, cu_seqlens=cu_seqlens)

    assert decision.backend == "barqtrain_native_decode"
    assert decision.packed_sequences is True
    assert torch.allclose(output, expected, rtol=1e-5, atol=1e-6)
