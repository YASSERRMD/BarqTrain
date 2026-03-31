"""Attention dispatch helpers for decode-heavy BarqTrain workloads."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from barqtrain.ops import apply_rope_to_qk, flash_attention, padding_free_attention


def _normalize_attention_backend_name(name: str | None) -> str:
    backend = (name or "auto").strip().lower()
    aliases = {
        "native": "barqtrain_native_decode",
        "barqtrain_native": "barqtrain_native_decode",
        "flash": "flash_attention_2",
    }
    backend = aliases.get(backend, backend)
    if backend not in {"auto", "flash_attention_2", "barqtrain_native_decode", "sdpa"}:
        raise ValueError(f"Unsupported attention backend {backend!r}")
    return backend


def _configured_attention_backend() -> str:
    return _normalize_attention_backend_name(os.environ.get("BARQTRAIN_ATTENTION_BACKEND", "auto"))


def available_attention_backends() -> tuple[str, ...]:
    """
    Return the attention backends currently available to BarqTrain.
    """
    backends = ["barqtrain_native_decode", "sdpa"]
    if torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None:
        backends.insert(0, "flash_attention_2")
    return tuple(backends)


@dataclass(frozen=True)
class AttentionDispatchDecision:
    """
    The backend selection made for a specific attention call.
    """

    backend: str
    cache_layout: Optional[str] = None
    last_token_only: bool = False
    packed_sequences: bool = False


def materialize_kv_for_attention(
    cache_or_layer,
    *,
    layer_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Materialize the current K/V tensors from a BarqTrain cache or cache layer.
    """
    if cache_or_layer is None:
        raise ValueError("cache_or_layer must not be None")

    if hasattr(cache_or_layer, "current_tensors"):
        return cache_or_layer.current_tensors()

    layers = getattr(cache_or_layer, "layers", None)
    if layers is not None:
        if layer_index >= len(layers):
            raise IndexError(f"layer_index={layer_index} exceeds cache depth {len(layers)}")
        layer = layers[layer_index]
        if hasattr(layer, "current_tensors"):
            return layer.current_tensors()

    if isinstance(cache_or_layer, (tuple, list)) and len(cache_or_layer) >= 2:
        return cache_or_layer[0], cache_or_layer[1]

    raise TypeError("cache_or_layer does not expose BarqTrain-compatible current_tensors() data")


def select_attention_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cache=None,
    attn_mask: Optional[torch.Tensor] = None,
    last_token_only: bool = False,
    packed_cu_seqlens: Optional[torch.Tensor] = None,
    prefer_flash_attention: bool = True,
    requested_backend: Optional[str] = None,
) -> AttentionDispatchDecision:
    """
    Decide whether to use FlashAttention, BarqTrain native decode attention, or SDPA.
    """
    del q, k, v  # Shape validation happens in the execution path.

    forced_backend = _normalize_attention_backend_name(requested_backend or _configured_attention_backend())
    cache_layout = getattr(cache, "barqtrain_cache_layout", None)
    packed_sequences = packed_cu_seqlens is not None

    if packed_sequences:
        if forced_backend == "auto":
            forced_backend = "barqtrain_native_decode"
        return AttentionDispatchDecision(
            backend=forced_backend,
            cache_layout=cache_layout,
            last_token_only=False,
            packed_sequences=True,
        )

    if forced_backend != "auto":
        return AttentionDispatchDecision(
            backend=forced_backend,
            cache_layout=cache_layout,
            last_token_only=bool(last_token_only),
        )

    if cache is not None or last_token_only:
        return AttentionDispatchDecision(
            backend="barqtrain_native_decode",
            cache_layout=cache_layout,
            last_token_only=bool(last_token_only),
        )

    flash_available = (
        prefer_flash_attention
        and attn_mask is None
        and torch.cuda.is_available()
        and importlib.util.find_spec("flash_attn") is not None
    )
    if flash_available:
        return AttentionDispatchDecision(backend="flash_attention_2")

    return AttentionDispatchDecision(backend="sdpa")


def dispatch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cache=None,
    layer_index: int = 0,
    attn_mask: Optional[torch.Tensor] = None,
    apply_rope: bool = True,
    last_token_only: bool = False,
    packed_cu_seqlens: Optional[torch.Tensor] = None,
    prefer_flash_attention: bool = True,
    requested_backend: Optional[str] = None,
) -> tuple[torch.Tensor, AttentionDispatchDecision]:
    """
    Execute attention through the selected BarqTrain dispatch path.
    """
    decision = select_attention_backend(
        q,
        k,
        v,
        cache=cache,
        attn_mask=attn_mask,
        last_token_only=last_token_only,
        packed_cu_seqlens=packed_cu_seqlens,
        prefer_flash_attention=prefer_flash_attention,
        requested_backend=requested_backend,
    )

    if decision.packed_sequences:
        output = padding_free_attention(
            q,
            k,
            v,
            cu_seqlens=packed_cu_seqlens,
        )
        return output, decision

    full_k = k
    full_v = v
    if cache is not None:
        cached_k, cached_v = materialize_kv_for_attention(cache, layer_index=layer_index)
        full_k = torch.cat([cached_k, k], dim=-2)
        full_v = torch.cat([cached_v, v], dim=-2)

    query = q[..., -1:, :] if decision.last_token_only else q
    if apply_rope:
        query, full_k = apply_rope_to_qk(query, full_k)

    use_causal = attn_mask is None and cache is None and not decision.last_token_only and query.size(-2) == full_k.size(-2)
    if decision.backend == "flash_attention_2" and attn_mask is None and cache is None:
        output = flash_attention(query, full_k, full_v)
    else:
        output = F.scaled_dot_product_attention(
            query,
            full_k,
            full_v,
            attn_mask=attn_mask,
            is_causal=use_causal,
        )
    return output, decision


__all__ = [
    "AttentionDispatchDecision",
    "available_attention_backends",
    "dispatch_attention",
    "materialize_kv_for_attention",
    "select_attention_backend",
]
