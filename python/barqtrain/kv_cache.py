"""Paged KV-cache helpers for decode-time memory reduction."""

from __future__ import annotations

import math
import os
from typing import Any, Optional

import torch

from barqtrain._ffi import load_cuda_backend

try:
    from transformers.cache_utils import Cache, CacheLayerMixin
except ImportError:  # pragma: no cover - transformers is a required dependency
    Cache = object  # type: ignore[assignment,misc]
    CacheLayerMixin = object  # type: ignore[assignment,misc]


def _get_cuda_backend():
    return load_cuda_backend()


def _env_enabled(name: str, default: str = "1") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _decoder_config(config):
    return config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config


def _resolve_model_dtype(model: torch.nn.Module) -> torch.dtype:
    for parameter in model.parameters():
        return parameter.dtype
    return torch.float16


class BarqPagedKVCacheLayer(CacheLayerMixin):
    """A fixed-capacity paged KV-cache layer backed by a CUDA append kernel."""

    is_sliding = False

    def __init__(self, max_batch_size: int, max_cache_len: int, page_size: int = 16):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.page_size = page_size
        self.max_blocks = math.ceil(max_cache_len / page_size)
        self.seq_lens: Optional[torch.Tensor] = None
        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None
        self.current_batch_size = 0

    def lazy_initialization(self, key_states: torch.Tensor):
        if key_states.dim() != 4:
            raise ValueError("key_states must have shape [batch, kv_heads, seq, head_dim]")

        batch_size, num_kv_heads, _, head_dim = key_states.shape
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"batch_size={batch_size} exceeds max_batch_size={self.max_batch_size} for paged KV cache"
            )

        self.device = key_states.device
        self.dtype = key_states.dtype
        self.current_batch_size = batch_size
        self.keys = torch.zeros(
            (self.max_batch_size, num_kv_heads, self.max_blocks, self.page_size, head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros_like(self.keys)
        self.seq_lens = torch.zeros((self.max_batch_size,), dtype=torch.int32, device=self.device)
        self.is_initialized = True

    def _python_append(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        batch_size, num_kv_heads, token_count, head_dim = key_states.shape
        for batch_idx in range(batch_size):
            start = int(self.seq_lens[batch_idx].item())  # type: ignore[index]
            end = start + token_count
            if end > self.max_cache_len:
                raise ValueError(
                    f"paged KV cache capacity exceeded: requested {end}, max_cache_len={self.max_cache_len}"
                )
            for token_idx in range(token_count):
                absolute_pos = start + token_idx
                block_idx = absolute_pos // self.page_size
                page_offset = absolute_pos % self.page_size
                self.keys[batch_idx, :, block_idx, page_offset, :] = key_states[batch_idx, :, token_idx, :]
                self.values[batch_idx, :, block_idx, page_offset, :] = value_states[batch_idx, :, token_idx, :]
            self.seq_lens[batch_idx] = end  # type: ignore[index]
        self.current_batch_size = batch_size

    def append(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        if key_states.shape != value_states.shape:
            raise ValueError("key_states and value_states must have identical shapes")
        if key_states.device != self.device or value_states.device != self.device:
            raise ValueError("key_states and value_states must stay on the cache device")
        if key_states.dtype != self.dtype or value_states.dtype != self.dtype:
            raise ValueError("key_states and value_states must stay in the cache dtype")
        if key_states.size(0) > self.max_batch_size:
            raise ValueError("batch size exceeds the configured paged KV cache capacity")

        backend = _get_cuda_backend()
        if backend is not None and key_states.is_cuda:
            backend.paged_kv_append_(
                self.keys,
                self.values,
                self.seq_lens,
                key_states.contiguous(),
                value_states.contiguous(),
            )
            self.current_batch_size = key_states.size(0)
            return

        self._python_append(key_states.contiguous(), value_states.contiguous())

    def _flatten_pages(self, cache_tensor: torch.Tensor) -> torch.Tensor:
        batch_size = self.current_batch_size or cache_tensor.size(0)
        return cache_tensor[:batch_size].view(
            batch_size,
            cache_tensor.size(1),
            self.max_blocks * self.page_size,
            cache_tensor.size(-1),
        )

    def current_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = self.get_seq_length()
        keys = self._flatten_pages(self.keys)[..., :seq_len, :]
        values = self._flatten_pages(self.values)[..., :seq_len, :]
        return keys, values

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.append(key_states, value_states)
        return self.current_tensors()

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        if not self.is_initialized or self.seq_lens is None or self.current_batch_size == 0:
            return 0
        return int(self.seq_lens[: self.current_batch_size].max().item())

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def crop(self, max_length: int) -> None:
        if not self.is_initialized or self.seq_lens is None:
            return
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        max_length = max(max_length, 0)
        self.seq_lens[: self.current_batch_size].clamp_(max=max_length)

    def batch_repeat_interleave(self, repeats: int) -> None:
        if not self.is_initialized or self.current_batch_size == 0:
            return
        self.keys = self.keys[: self.current_batch_size].repeat_interleave(repeats, dim=0)
        self.values = self.values[: self.current_batch_size].repeat_interleave(repeats, dim=0)
        self.seq_lens = self.seq_lens[: self.current_batch_size].repeat_interleave(repeats, dim=0)
        self.current_batch_size = self.keys.size(0)
        self.max_batch_size = self.current_batch_size

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if not self.is_initialized or self.current_batch_size == 0:
            return
        indices = indices.to(self.keys.device)
        self.keys = self.keys[: self.current_batch_size].index_select(0, indices)
        self.values = self.values[: self.current_batch_size].index_select(0, indices)
        self.seq_lens = self.seq_lens[: self.current_batch_size].index_select(0, indices)
        self.current_batch_size = self.keys.size(0)
        self.max_batch_size = self.current_batch_size

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.batch_select_indices(beam_idx)

    def reset(self) -> None:
        if not self.is_initialized:
            return
        self.keys.zero_()
        self.values.zero_()
        self.seq_lens.zero_()
        self.current_batch_size = 0


class BarqPagedKVCache(Cache):
    """A paged KV cache that preallocates decode storage for each decoder layer."""

    def __init__(
        self,
        config,
        max_batch_size: int,
        max_cache_len: int,
        page_size: int = 16,
    ):
        decoder_config = _decoder_config(config)
        num_layers = decoder_config.num_hidden_layers
        if hasattr(decoder_config, "num_kv_shared_layers"):
            num_layers -= decoder_config.num_kv_shared_layers

        layers = [
            BarqPagedKVCacheLayer(
                max_batch_size=max_batch_size,
                max_cache_len=max_cache_len,
                page_size=page_size,
            )
            for _ in range(num_layers)
        ]
        super().__init__(layers=layers)
        self.page_size = page_size
        self.barqtrain_max_cache_len = max_cache_len
        self.barqtrain_max_batch_size = max_batch_size


def create_paged_kv_cache(
    model_or_config,
    *,
    max_batch_size: int,
    max_cache_len: int,
    page_size: int = 16,
) -> BarqPagedKVCache:
    """Create a paged KV cache for a model or decoder config."""
    config = getattr(model_or_config, "config", model_or_config)
    return BarqPagedKVCache(
        config=config,
        max_batch_size=max_batch_size,
        max_cache_len=max_cache_len,
        page_size=page_size,
    )


def maybe_prepare_paged_kv_generate_kwargs(model: torch.nn.Module, args, kwargs):
    """Inject a paged KV cache into generate() when the native CUDA path is available."""
    if kwargs.get("past_key_values") is not None:
        return kwargs, False
    if not _env_enabled("BARQTRAIN_ENABLE_PAGED_KV", "1"):
        return kwargs, False
    if not getattr(model, "_barqtrain_paged_kv_supported", False):
        return kwargs, False
    if _get_cuda_backend() is None:
        return kwargs, False

    input_ids = kwargs.get("input_ids")
    if input_ids is None and args:
        input_ids = args[0]
    inputs_embeds = kwargs.get("inputs_embeds")
    attention_mask = kwargs.get("attention_mask")

    if input_ids is not None:
        batch_size, prompt_length = input_ids.shape[:2]
        device = input_ids.device
    elif inputs_embeds is not None:
        batch_size, prompt_length = inputs_embeds.shape[:2]
        device = inputs_embeds.device
    elif attention_mask is not None:
        batch_size, prompt_length = attention_mask.shape[:2]
        device = attention_mask.device
    else:
        return kwargs, False

    if device.type != "cuda":
        return kwargs, False

    generation_config = kwargs.get("generation_config") or getattr(model, "generation_config", None)
    max_new_tokens = kwargs.get("max_new_tokens")
    if max_new_tokens is None and generation_config is not None:
        max_new_tokens = getattr(generation_config, "max_new_tokens", None)

    if max_new_tokens is not None:
        max_cache_len = prompt_length + int(max_new_tokens)
    else:
        max_length = kwargs.get("max_length")
        if max_length is None and generation_config is not None:
            max_length = getattr(generation_config, "max_length", None)
        if max_length is None:
            return kwargs, False
        max_cache_len = int(max(max_length, prompt_length))

    page_size = int(os.environ.get("BARQTRAIN_PAGED_KV_PAGE_SIZE", "16"))
    cache = create_paged_kv_cache(
        model,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        page_size=page_size,
    )
    updated_kwargs = dict(kwargs)
    updated_kwargs["past_key_values"] = cache
    updated_kwargs["use_cache"] = True
    return updated_kwargs, True


def paged_kv_supported_for_model(model: torch.nn.Module) -> bool:
    """Return True when paged KV-cache injection can run for this model."""
    return torch.cuda.is_available() and _get_cuda_backend() is not None and hasattr(model, "generate")


__all__ = [
    "BarqPagedKVCache",
    "BarqPagedKVCacheLayer",
    "create_paged_kv_cache",
    "maybe_prepare_paged_kv_generate_kwargs",
    "paged_kv_supported_for_model",
]
