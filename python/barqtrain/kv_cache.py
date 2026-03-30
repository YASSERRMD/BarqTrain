"""KV-cache helpers for decode-time memory reduction."""

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


def _min_paged_kv_cache_len() -> int:
    return int(os.environ.get("BARQTRAIN_PAGED_KV_MIN_CACHE_LEN", "256"))


def _kv_cache_mode(default: str = "auto") -> str:
    mode = os.environ.get("BARQTRAIN_KV_CACHE_MODE", default).strip().lower()
    if mode not in {"auto", "paged", "contiguous"}:
        raise ValueError(f"Unsupported BARQTRAIN_KV_CACHE_MODE={mode!r}")
    return mode


def _default_total_paged_blocks(max_batch_size: int, max_blocks_per_sequence: int) -> int:
    configured = os.environ.get("BARQTRAIN_PAGED_KV_TOTAL_BLOCKS")
    if configured is None:
        return max_batch_size * max_blocks_per_sequence
    return int(configured)


def _logical_block_count(length: int, page_size: int) -> int:
    if length <= 0:
        return 0
    return math.ceil(length / page_size)


class BarqPagedKVCacheLayer(CacheLayerMixin):
    """A fixed-capacity paged KV-cache layer backed by a physical-block allocator."""

    is_sliding = False

    def __init__(
        self,
        max_batch_size: int,
        max_cache_len: int,
        page_size: int = 16,
        total_blocks: Optional[int] = None,
    ):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.page_size = page_size
        self.max_blocks = math.ceil(max_cache_len / page_size)
        self.total_blocks = total_blocks or _default_total_paged_blocks(max_batch_size, self.max_blocks)
        self.seq_lens: Optional[torch.Tensor] = None
        self.page_table: Optional[torch.Tensor] = None
        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None
        self.current_batch_size = 0
        self._free_blocks: list[int] = []

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
            (self.total_blocks, num_kv_heads, self.page_size, head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros_like(self.keys)
        self.seq_lens = torch.zeros((self.max_batch_size,), dtype=torch.int32, device=self.device)
        self.page_table = torch.full(
            (self.max_batch_size, self.max_blocks),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self._free_blocks = list(range(self.total_blocks))
        self.is_initialized = True

    def _allocate_block(self, batch_idx: int, logical_block: int) -> None:
        block_id = int(self.page_table[batch_idx, logical_block].item())
        if block_id >= 0:
            return
        if not self._free_blocks:
            raise ValueError(
                "paged KV allocator exhausted: no free physical blocks remain "
                f"(max_batch_size={self.max_batch_size}, total_blocks={self.total_blocks})"
            )
        block_id = self._free_blocks.pop()
        self.page_table[batch_idx, logical_block] = block_id

    def _free_block(self, batch_idx: int, logical_block: int) -> None:
        block_id = int(self.page_table[batch_idx, logical_block].item())
        if block_id < 0:
            return
        self.page_table[batch_idx, logical_block] = -1
        self._free_blocks.append(block_id)

    def _ensure_page_assignments(self, batch_size: int, token_count: int) -> None:
        for batch_idx in range(batch_size):
            start = int(self.seq_lens[batch_idx].item())
            end = start + token_count
            if end > self.max_cache_len:
                raise ValueError(
                    f"paged KV cache capacity exceeded: requested {end}, max_cache_len={self.max_cache_len}"
                )
            first_block = start // self.page_size
            last_block = (end - 1) // self.page_size if token_count > 0 else first_block
            for logical_block in range(first_block, last_block + 1):
                self._allocate_block(batch_idx, logical_block)

    def _python_assign_batch(self, batch_idx: int, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        num_kv_heads, token_count, _ = key_states.shape
        del num_kv_heads
        start = int(self.seq_lens[batch_idx].item())
        end = start + token_count
        if end > self.max_cache_len:
            raise ValueError(
                f"paged KV cache capacity exceeded: requested {end}, max_cache_len={self.max_cache_len}"
            )
        for token_idx in range(token_count):
            absolute_pos = start + token_idx
            logical_block = absolute_pos // self.page_size
            page_offset = absolute_pos % self.page_size
            physical_block = int(self.page_table[batch_idx, logical_block].item())
            if physical_block < 0:
                raise ValueError("page table entry missing during KV append")
            self.keys[physical_block, :, page_offset, :] = key_states[:, token_idx, :]
            self.values[physical_block, :, page_offset, :] = value_states[:, token_idx, :]
        self.seq_lens[batch_idx] = end  # type: ignore[index]

    def _python_append(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        batch_size, _, token_count, _ = key_states.shape
        self._ensure_page_assignments(batch_size, token_count)
        for batch_idx in range(batch_size):
            self._python_assign_batch(
                batch_idx,
                key_states[batch_idx],
                value_states[batch_idx],
            )
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

        self._ensure_page_assignments(key_states.size(0), key_states.size(2))
        backend = _get_cuda_backend()
        if backend is not None and key_states.is_cuda:
            backend.paged_kv_append_(
                self.keys,
                self.values,
                self.page_table,
                self.seq_lens,
                key_states.contiguous(),
                value_states.contiguous(),
            )
            self.current_batch_size = key_states.size(0)
            return

        self._python_append(key_states.contiguous(), value_states.contiguous())

    def _python_gather(self, cache_tensor: torch.Tensor) -> torch.Tensor:
        batch_size = self.current_batch_size
        seq_len = self.get_seq_length()
        if batch_size == 0 or seq_len == 0:
            return torch.zeros(
                (batch_size, cache_tensor.size(1), seq_len, cache_tensor.size(-1)),
                dtype=cache_tensor.dtype,
                device=cache_tensor.device,
            )

        output = torch.zeros(
            (batch_size, cache_tensor.size(1), seq_len, cache_tensor.size(-1)),
            dtype=cache_tensor.dtype,
            device=cache_tensor.device,
        )
        for batch_idx in range(batch_size):
            seq_len_batch = int(self.seq_lens[batch_idx].item())
            for token_idx in range(seq_len_batch):
                logical_block = token_idx // self.page_size
                page_offset = token_idx % self.page_size
                physical_block = int(self.page_table[batch_idx, logical_block].item())
                if physical_block < 0:
                    continue
                output[batch_idx, :, token_idx, :] = cache_tensor[physical_block, :, page_offset, :]
        return output

    def current_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            raise ValueError("paged KV cache is not initialized")
        backend = _get_cuda_backend()
        if backend is not None and self.keys.is_cuda:
            keys = backend.paged_kv_gather(self.keys, self.page_table, self.seq_lens)
            values = backend.paged_kv_gather(self.values, self.page_table, self.seq_lens)
        else:
            keys = self._python_gather(self.keys)
            values = self._python_gather(self.values)
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
        for batch_idx in range(self.current_batch_size):
            old_length = int(self.seq_lens[batch_idx].item())
            new_length = min(old_length, max_length)
            old_blocks = _logical_block_count(old_length, self.page_size)
            new_blocks = _logical_block_count(new_length, self.page_size)
            for logical_block in range(new_blocks, old_blocks):
                self._free_block(batch_idx, logical_block)
            self.seq_lens[batch_idx] = new_length  # type: ignore[index]

    def batch_repeat_interleave(self, repeats: int) -> None:
        if not self.is_initialized or self.current_batch_size == 0:
            return
        keys, values = self.current_tensors()
        seq_lens = [
            int(self.seq_lens[batch_idx].item())
            for batch_idx in range(self.current_batch_size)
            for _ in range(repeats)
        ]
        repeated_keys = keys.repeat_interleave(repeats, dim=0)
        repeated_values = values.repeat_interleave(repeats, dim=0)
        self._rebuild_from_contiguous(repeated_keys, repeated_values, seq_lens)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if not self.is_initialized or self.current_batch_size == 0:
            return
        indices_cpu = [int(value) for value in indices.view(-1).tolist()]
        keys, values = self.current_tensors()
        selected_keys = keys.index_select(0, torch.tensor(indices_cpu, device=keys.device))
        selected_values = values.index_select(0, torch.tensor(indices_cpu, device=values.device))
        seq_lens = [int(self.seq_lens[idx].item()) for idx in indices_cpu]
        self._rebuild_from_contiguous(selected_keys, selected_values, seq_lens)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.batch_select_indices(beam_idx)

    def resident_blocks(self) -> int:
        return self.total_blocks - len(self._free_blocks)

    def active_blocks(self) -> int:
        active = 0
        for batch_idx in range(self.current_batch_size):
            active += _logical_block_count(int(self.seq_lens[batch_idx].item()), self.page_size)
        return active

    def fragmentation_ratio(self) -> float:
        resident = self.resident_blocks()
        if resident == 0:
            return 0.0
        return max(resident - self.active_blocks(), 0) / resident

    def _rebuild_from_contiguous(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        seq_lens: list[int],
    ) -> None:
        self.reset()
        self.current_batch_size = len(seq_lens)
        for batch_idx, seq_len in enumerate(seq_lens):
            if seq_len == 0:
                continue
            self._ensure_page_assignments(batch_idx + 1, seq_len)
            self._python_assign_batch(
                batch_idx,
                keys[batch_idx, :, :seq_len, :],
                values[batch_idx, :, :seq_len, :],
            )

    def reset(self) -> None:
        if not self.is_initialized:
            return
        self.keys.zero_()
        self.values.zero_()
        self.seq_lens.zero_()
        self.page_table.fill_(-1)
        self._free_blocks = list(range(self.total_blocks))
        self.current_batch_size = 0


class BarqPagedKVCache(Cache):
    """A paged KV cache backed by a reusable physical page allocator."""

    def __init__(
        self,
        config,
        max_batch_size: int,
        max_cache_len: int,
        page_size: int = 16,
        total_blocks: Optional[int] = None,
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
                total_blocks=total_blocks,
            )
            for _ in range(num_layers)
        ]
        super().__init__(layers=layers)
        self.page_size = page_size
        self.barqtrain_max_cache_len = max_cache_len
        self.barqtrain_max_batch_size = max_batch_size
        self.barqtrain_total_blocks = total_blocks or _default_total_paged_blocks(
            max_batch_size,
            math.ceil(max_cache_len / page_size),
        )
        self.barqtrain_cache_layout = "paged"

    def fragmentation_ratio(self) -> float:
        if not self.layers:
            return 0.0
        return sum(layer.fragmentation_ratio() for layer in self.layers) / len(self.layers)

    def resident_blocks(self) -> int:
        return sum(layer.resident_blocks() for layer in self.layers)


class BarqContiguousKVCacheLayer(CacheLayerMixin):
    """A fixed-capacity contiguous KV-cache fallback."""

    is_sliding = False

    def __init__(self, max_batch_size: int, max_cache_len: int):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
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
                f"batch_size={batch_size} exceeds max_batch_size={self.max_batch_size} for contiguous KV cache"
            )
        self.device = key_states.device
        self.dtype = key_states.dtype
        self.keys = torch.zeros(
            (self.max_batch_size, num_kv_heads, self.max_cache_len, head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros_like(self.keys)
        self.seq_lens = torch.zeros((self.max_batch_size,), dtype=torch.int32, device=self.device)
        self.current_batch_size = batch_size
        self.is_initialized = True

    def append(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        batch_size, _, token_count, _ = key_states.shape
        for batch_idx in range(batch_size):
            start = int(self.seq_lens[batch_idx].item())
            end = start + token_count
            if end > self.max_cache_len:
                raise ValueError(
                    f"contiguous KV cache capacity exceeded: requested {end}, max_cache_len={self.max_cache_len}"
                )
            self.keys[batch_idx, :, start:end, :] = key_states[batch_idx]
            self.values[batch_idx, :, start:end, :] = value_states[batch_idx]
            self.seq_lens[batch_idx] = end  # type: ignore[index]
        self.current_batch_size = batch_size

    def current_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = self.get_seq_length()
        return (
            self.keys[: self.current_batch_size, :, :seq_len, :],
            self.values[: self.current_batch_size, :, :seq_len, :],
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        self.append(key_states, value_states)
        return self.current_tensors()

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, 0

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

    def fragmentation_ratio(self) -> float:
        return 0.0

    def resident_blocks(self) -> int:
        return self.current_batch_size * self.max_cache_len

    def reset(self) -> None:
        if not self.is_initialized:
            return
        self.keys.zero_()
        self.values.zero_()
        self.seq_lens.zero_()
        self.current_batch_size = 0


class BarqContiguousKVCache(Cache):
    """A contiguous KV cache with the same public cache interface."""

    def __init__(self, config, max_batch_size: int, max_cache_len: int):
        decoder_config = _decoder_config(config)
        num_layers = decoder_config.num_hidden_layers
        if hasattr(decoder_config, "num_kv_shared_layers"):
            num_layers -= decoder_config.num_kv_shared_layers

        layers = [
            BarqContiguousKVCacheLayer(
                max_batch_size=max_batch_size,
                max_cache_len=max_cache_len,
            )
            for _ in range(num_layers)
        ]
        super().__init__(layers=layers)
        self.barqtrain_max_cache_len = max_cache_len
        self.barqtrain_max_batch_size = max_batch_size
        self.barqtrain_cache_layout = "contiguous"

    def fragmentation_ratio(self) -> float:
        return 0.0


def create_paged_kv_cache(
    model_or_config,
    *,
    max_batch_size: int,
    max_cache_len: int,
    page_size: int = 16,
    total_blocks: Optional[int] = None,
) -> BarqPagedKVCache:
    """Create a paged KV cache for a model or decoder config."""
    config = getattr(model_or_config, "config", model_or_config)
    return BarqPagedKVCache(
        config=config,
        max_batch_size=max_batch_size,
        max_cache_len=max_cache_len,
        page_size=page_size,
        total_blocks=total_blocks,
    )


def create_contiguous_kv_cache(
    model_or_config,
    *,
    max_batch_size: int,
    max_cache_len: int,
) -> BarqContiguousKVCache:
    """Create a contiguous KV cache for a model or decoder config."""
    config = getattr(model_or_config, "config", model_or_config)
    return BarqContiguousKVCache(
        config=config,
        max_batch_size=max_batch_size,
        max_cache_len=max_cache_len,
    )


def create_kv_cache(
    model_or_config,
    *,
    max_batch_size: int,
    max_cache_len: int,
    page_size: int = 16,
    total_blocks: Optional[int] = None,
    mode: str = "paged",
):
    """Create a paged or contiguous KV cache."""
    if mode == "paged":
        return create_paged_kv_cache(
            model_or_config,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            page_size=page_size,
            total_blocks=total_blocks,
        )
    if mode == "contiguous":
        return create_contiguous_kv_cache(
            model_or_config,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
        )
    raise ValueError(f"Unsupported KV cache mode: {mode}")


def maybe_prepare_kv_generate_kwargs(model: torch.nn.Module, args, kwargs):
    """Inject a BarqTrain KV cache into generate() when the native path is enabled."""
    if kwargs.get("past_key_values") is not None:
        return kwargs, False
    if not _env_enabled("BARQTRAIN_ENABLE_PAGED_KV", "1"):
        return kwargs, False
    if not getattr(model, "_barqtrain_paged_kv_supported", False):
        return kwargs, False
    cache_mode = _kv_cache_mode()
    if cache_mode == "paged" and _get_cuda_backend() is None:
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

    if max_cache_len < _min_paged_kv_cache_len():
        return kwargs, False

    page_size = int(os.environ.get("BARQTRAIN_PAGED_KV_PAGE_SIZE", "16"))
    if cache_mode == "auto":
        cache_mode = "paged" if _get_cuda_backend() is not None else "contiguous"

    cache = create_kv_cache(
        model,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        page_size=page_size,
        mode=cache_mode,
    )
    updated_kwargs = dict(kwargs)
    updated_kwargs["past_key_values"] = cache
    updated_kwargs["use_cache"] = True
    return updated_kwargs, True


def maybe_prepare_paged_kv_generate_kwargs(model: torch.nn.Module, args, kwargs):
    """Inject only the paged KV cache path for backwards-compatible callers."""
    original_mode = _kv_cache_mode()
    if original_mode != "paged":
        os.environ["BARQTRAIN_KV_CACHE_MODE"] = "paged"
    try:
        return maybe_prepare_kv_generate_kwargs(model, args, kwargs)
    finally:
        if original_mode != "paged":
            if "BARQTRAIN_KV_CACHE_MODE" in os.environ and original_mode == "auto":
                del os.environ["BARQTRAIN_KV_CACHE_MODE"]
            else:
                os.environ["BARQTRAIN_KV_CACHE_MODE"] = original_mode


def paged_kv_supported_for_model(model: torch.nn.Module) -> bool:
    """Return True when paged KV-cache injection can run for this model."""
    return hasattr(model, "generate")


__all__ = [
    "BarqContiguousKVCache",
    "BarqContiguousKVCacheLayer",
    "BarqPagedKVCache",
    "BarqPagedKVCacheLayer",
    "create_contiguous_kv_cache",
    "create_kv_cache",
    "create_paged_kv_cache",
    "maybe_prepare_kv_generate_kwargs",
    "maybe_prepare_paged_kv_generate_kwargs",
    "paged_kv_supported_for_model",
]
