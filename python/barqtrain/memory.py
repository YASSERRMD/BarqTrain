"""Memory, profiling, and generation helpers for BarqTrain benchmarks."""

from __future__ import annotations

import copy
import inspect
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch

from barqtrain._ffi import load_cuda_backend, load_rust_backend

_DETAILED_PROFILING_ENV = "BARQTRAIN_DETAILED_PROFILING"


def _get_cuda_backend():
    return load_cuda_backend()


def _get_rust_backend():
    return load_rust_backend()


def _env_enabled(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _bytes_to_mb(value: int | float) -> float:
    return float(value) / (1024**2)


@dataclass(frozen=True)
class CudaMemorySnapshot:
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    max_allocated_mb: float = 0.0
    max_reserved_mb: float = 0.0


def cuda_memory_snapshot(reset_peak: bool = False) -> CudaMemorySnapshot:
    """
    Capture the current CUDA allocator state in megabytes.
    """
    if not torch.cuda.is_available():
        return CudaMemorySnapshot()

    if reset_peak:
        torch.cuda.reset_peak_memory_stats()

    return CudaMemorySnapshot(
        allocated_mb=torch.cuda.memory_allocated() / (1024**2),
        reserved_mb=torch.cuda.memory_reserved() / (1024**2),
        max_allocated_mb=torch.cuda.max_memory_allocated() / (1024**2),
        max_reserved_mb=torch.cuda.max_memory_reserved() / (1024**2),
    )


@dataclass(frozen=True)
class BenchmarkMemoryBreakdown:
    resident_model_mb: float = 0.0
    kv_cache_mb: float = 0.0
    temporary_decode_buffers_mb: float = 0.0
    training_peak_vram_mb: float = 0.0
    inference_peak_vram_mb: float = 0.0
    detailed_profiling: bool = False


@dataclass(frozen=True)
class DecodeBenchmarkProfile:
    name: str
    prompt_length: int
    decode_length: int
    batch_size: int


@dataclass(frozen=True)
class KVCacheBenchmarkProfile:
    name: str
    prompt_length: int
    decode_length: int
    batch_size: int
    request_count: int = 1
    fixed_vram_budget_mb: int = 0


def generation_overhead_mb(
    resident_snapshot: CudaMemorySnapshot,
    peak_snapshot: CudaMemorySnapshot,
) -> float:
    """
    Compute generation overhead beyond the resident model footprint.
    """
    return max(peak_snapshot.max_allocated_mb - resident_snapshot.allocated_mb, 0.0)


def detailed_profiling_enabled() -> bool:
    """
    Return whether detailed native memory profiling is enabled.
    """
    return _env_enabled(_DETAILED_PROFILING_ENV, "0")


def set_detailed_profiling_enabled(enabled: bool) -> None:
    """
    Toggle native profiling state across Python and the CUDA extension.
    """
    os.environ[_DETAILED_PROFILING_ENV] = "1" if enabled else "0"
    backend = _get_cuda_backend()
    if backend is not None and hasattr(backend, "barqtrain_memory_set_enabled"):
        backend.barqtrain_memory_set_enabled(enabled)


def reset_native_memory_tracking(reset_peak: bool = True) -> None:
    """
    Reset native memory accounting state when the CUDA extension supports it.
    """
    backend = _get_cuda_backend()
    if backend is not None and hasattr(backend, "barqtrain_memory_reset"):
        backend.barqtrain_memory_set_enabled(detailed_profiling_enabled())
        backend.barqtrain_memory_reset(reset_peak)


def native_memory_snapshot() -> dict[str, int | bool]:
    """
    Return the current native memory tracking snapshot.
    """
    backend = _get_cuda_backend()
    if backend is not None and hasattr(backend, "barqtrain_memory_snapshot"):
        backend.barqtrain_memory_set_enabled(detailed_profiling_enabled())
        return dict(backend.barqtrain_memory_snapshot())

    return {
        "enabled": False,
        "resident_model_current_bytes": 0,
        "resident_model_peak_bytes": 0,
        "kv_cache_current_bytes": 0,
        "kv_cache_peak_bytes": 0,
        "decode_temp_current_bytes": 0,
        "decode_temp_peak_bytes": 0,
        "training_peak_bytes": 0,
        "inference_peak_bytes": 0,
    }


def _record_native_bucket_bytes(bucket: str, current_bytes: int) -> None:
    if not detailed_profiling_enabled():
        return

    backend = _get_cuda_backend()
    if backend is None or not hasattr(backend, "barqtrain_memory_set_bucket_bytes"):
        return

    backend.barqtrain_memory_set_enabled(True)
    backend.barqtrain_memory_set_bucket_bytes(bucket, int(max(current_bytes, 0)))


def _record_native_peak_bytes(mode: str, peak_bytes: int) -> None:
    if not detailed_profiling_enabled():
        return

    backend = _get_cuda_backend()
    if backend is None or not hasattr(backend, "barqtrain_memory_record_peak"):
        return

    backend.barqtrain_memory_set_enabled(True)
    backend.barqtrain_memory_record_peak(mode, int(max(peak_bytes, 0)))


def _storage_bytes_for_cuda_tensor(tensor: torch.Tensor) -> tuple[int, int]:
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
        return 0, 0

    try:
        storage = tensor.untyped_storage()
        return int(storage.data_ptr()), int(storage.nbytes())
    except Exception:
        return int(tensor.data_ptr()), int(tensor.numel() * tensor.element_size())


def _sum_unique_cuda_tensor_bytes(
    tensors: Iterable[torch.Tensor],
    *,
    seen_storages: Optional[set[int]] = None,
) -> int:
    if seen_storages is None:
        seen_storages = set()
    total_bytes = 0
    for tensor in tensors:
        storage_ptr, nbytes = _storage_bytes_for_cuda_tensor(tensor)
        if storage_ptr == 0 or storage_ptr in seen_storages:
            continue
        seen_storages.add(storage_ptr)
        total_bytes += nbytes
    return total_bytes


def model_resident_cuda_bytes(model: torch.nn.Module) -> int:
    """
    Measure resident CUDA model memory without double-counting shared storages.
    """
    rust_backend = _get_rust_backend()
    if rust_backend is not None and hasattr(rust_backend, "model_cuda_bytes"):
        return int(rust_backend.model_cuda_bytes(model))

    parameters = model.parameters() if hasattr(model, "parameters") else ()
    buffers = model.buffers() if hasattr(model, "buffers") else ()
    seen_storages: set[int] = set()
    return _sum_unique_cuda_tensor_bytes(
        parameters,
        seen_storages=seen_storages,
    ) + _sum_unique_cuda_tensor_bytes(
        buffers,
        seen_storages=seen_storages,
    )


def paged_kv_cache_bytes(cache) -> int:
    """
    Measure the active paged KV-cache allocation in bytes.
    """
    total_bytes = 0
    seen_storages: set[int] = set()
    for layer in getattr(cache, "layers", []):
        for tensor_name in (
            "keys",
            "values",
            "quantized_keys",
            "quantized_values",
            "residual_keys",
            "residual_values",
            "key_scales",
            "value_scales",
            "seq_lens",
            "page_table",
            "residual_page_table",
            "physical_block_to_residual_slot",
        ):
            tensor = getattr(layer, tensor_name, None)
            if tensor is None:
                continue
            storage_ptr, nbytes = _storage_bytes_for_cuda_tensor(tensor)
            if storage_ptr == 0 or storage_ptr in seen_storages:
                continue
            seen_storages.add(storage_ptr)
            total_bytes += nbytes
    return total_bytes


def capture_cuda_peak_bytes(reset_peak: bool = False) -> int:
    """
    Read the CUDA allocator peak in bytes.
    """
    if not torch.cuda.is_available():
        return 0
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    return int(torch.cuda.max_memory_allocated())


def track_resident_model_memory(model: torch.nn.Module) -> int:
    """
    Measure and record resident model memory.
    """
    resident_bytes = model_resident_cuda_bytes(model)
    _record_native_bucket_bytes("resident_model", resident_bytes)
    return resident_bytes


def track_kv_cache_memory(cache) -> int:
    """
    Measure and record KV-cache residency.
    """
    kv_cache_allocation_bytes = paged_kv_cache_bytes(cache)
    _record_native_bucket_bytes("kv_cache", kv_cache_allocation_bytes)
    return kv_cache_allocation_bytes


def track_decode_temp_memory(
    *,
    resident_model_bytes: int,
    kv_cache_bytes: int,
    inference_peak_bytes: int,
) -> int:
    """
    Attribute decode scratch memory after model and KV residency are accounted for.
    """
    decode_temp_bytes = max(int(inference_peak_bytes) - int(resident_model_bytes) - int(kv_cache_bytes), 0)
    _record_native_bucket_bytes("decode_temp", decode_temp_bytes)
    return decode_temp_bytes


def record_training_peak_bytes(training_peak_bytes: int) -> None:
    """
    Record the benchmark training peak in the native tracker.
    """
    _record_native_peak_bytes("training", int(training_peak_bytes))


def record_inference_peak_bytes(inference_peak_bytes: int) -> None:
    """
    Record the benchmark inference peak in the native tracker.
    """
    _record_native_peak_bytes("inference", int(inference_peak_bytes))


def build_memory_breakdown(
    *,
    resident_model_bytes: int,
    kv_cache_bytes: int,
    temporary_decode_buffer_bytes: int,
    training_peak_bytes: int,
    inference_peak_bytes: int,
    detailed_profiling: Optional[bool] = None,
) -> BenchmarkMemoryBreakdown:
    """
    Build the canonical BarqTrain memory report, preferring the native Rust formatter.
    """
    profiling_state = detailed_profiling_enabled() if detailed_profiling is None else detailed_profiling
    rust_backend = _get_rust_backend()
    if rust_backend is not None and hasattr(rust_backend, "build_memory_breakdown"):
        native_report = rust_backend.build_memory_breakdown(
            int(resident_model_bytes),
            int(kv_cache_bytes),
            int(temporary_decode_buffer_bytes),
            int(training_peak_bytes),
            int(inference_peak_bytes),
            bool(profiling_state),
        )
        return BenchmarkMemoryBreakdown(
            resident_model_mb=float(native_report.resident_model_mb),
            kv_cache_mb=float(native_report.kv_cache_mb),
            temporary_decode_buffers_mb=float(native_report.temporary_decode_buffers_mb),
            training_peak_vram_mb=float(native_report.training_peak_vram_mb),
            inference_peak_vram_mb=float(native_report.inference_peak_vram_mb),
            detailed_profiling=bool(native_report.detailed_profiling),
        )

    return BenchmarkMemoryBreakdown(
        resident_model_mb=_bytes_to_mb(resident_model_bytes),
        kv_cache_mb=_bytes_to_mb(kv_cache_bytes),
        temporary_decode_buffers_mb=_bytes_to_mb(temporary_decode_buffer_bytes),
        training_peak_vram_mb=_bytes_to_mb(training_peak_bytes),
        inference_peak_vram_mb=_bytes_to_mb(inference_peak_bytes),
        detailed_profiling=bool(profiling_state),
    )


def phase1_inference_profiles(
    batch_sizes: Sequence[int] = (1, 4, 8),
    *,
    short_prompt_length: int = 64,
    long_prompt_length: int = 1024,
    short_decode_length: int = 32,
    long_decode_length: int = 256,
) -> list[DecodeBenchmarkProfile]:
    """
    Return the required Phase 1 inference benchmark matrix.
    """
    rust_backend = _get_rust_backend()
    if rust_backend is not None and hasattr(rust_backend, "phase1_decode_profiles"):
        native_profiles = rust_backend.phase1_decode_profiles(
            list(batch_sizes),
            int(short_prompt_length),
            int(long_prompt_length),
            int(short_decode_length),
            int(long_decode_length),
        )
        return [
            DecodeBenchmarkProfile(
                name=str(profile.name),
                prompt_length=int(profile.prompt_length),
                decode_length=int(profile.decode_length),
                batch_size=int(profile.batch_size),
            )
            for profile in native_profiles
        ]

    profiles: list[DecodeBenchmarkProfile] = []
    for batch_size in batch_sizes:
        profiles.append(
            DecodeBenchmarkProfile(
                name="short_prompt_long_decode",
                prompt_length=short_prompt_length,
                decode_length=long_decode_length,
                batch_size=int(batch_size),
            )
        )
        profiles.append(
            DecodeBenchmarkProfile(
                name="long_prompt_short_decode",
                prompt_length=long_prompt_length,
                decode_length=short_decode_length,
                batch_size=int(batch_size),
            )
        )
    return profiles


def phase2_kv_cache_profiles(
    batch_sizes: Sequence[int] = (1, 4, 8),
    *,
    short_prompt_length: int = 64,
    long_prompt_length: int = 1024,
    short_decode_length: int = 32,
    long_decode_length: int = 256,
    serving_request_count: int = 8,
    fixed_vram_budget_mb: int = 2048,
) -> list[KVCacheBenchmarkProfile]:
    """
    Return the required Phase 2 contiguous-vs-paged KV benchmark matrix.
    """
    rust_backend = _get_rust_backend()
    if rust_backend is not None and hasattr(rust_backend, "phase2_kv_cache_profiles"):
        native_profiles = rust_backend.phase2_kv_cache_profiles(
            list(batch_sizes),
            int(short_prompt_length),
            int(long_prompt_length),
            int(short_decode_length),
            int(long_decode_length),
            int(serving_request_count),
            int(fixed_vram_budget_mb),
        )
        return [
            KVCacheBenchmarkProfile(
                name=str(profile.name),
                prompt_length=int(profile.prompt_length),
                decode_length=int(profile.decode_length),
                batch_size=int(profile.batch_size),
                request_count=int(profile.request_count),
                fixed_vram_budget_mb=int(profile.fixed_vram_budget_mb),
            )
            for profile in native_profiles
        ]

    profiles: list[KVCacheBenchmarkProfile] = []
    for batch_size in batch_sizes:
        profiles.append(
            KVCacheBenchmarkProfile(
                name="long_prompt_generation",
                prompt_length=long_prompt_length,
                decode_length=long_decode_length,
                batch_size=int(batch_size),
            )
        )
        profiles.append(
            KVCacheBenchmarkProfile(
                name="multi_request_serving",
                prompt_length=short_prompt_length,
                decode_length=long_decode_length,
                batch_size=int(batch_size),
                request_count=int(max(serving_request_count, 1)),
            )
        )
        profiles.append(
            KVCacheBenchmarkProfile(
                name="fixed_vram_batch_growth",
                prompt_length=long_prompt_length,
                decode_length=short_decode_length,
                batch_size=int(batch_size),
                fixed_vram_budget_mb=int(fixed_vram_budget_mb),
            )
        )
    return profiles


def phase3_quantized_kv_profiles(
    batch_sizes: Sequence[int] = (1, 4, 8),
    *,
    short_prompt_length: int = 64,
    long_prompt_length: int = 1024,
    quality_decode_length: int = 64,
    long_decode_length: int = 256,
) -> list[KVCacheBenchmarkProfile]:
    """
    Return the required Phase 3 quantized KV benchmark matrix.
    """
    rust_backend = _get_rust_backend()
    if rust_backend is not None and hasattr(rust_backend, "phase3_quantized_kv_profiles"):
        native_profiles = rust_backend.phase3_quantized_kv_profiles(
            list(batch_sizes),
            int(short_prompt_length),
            int(long_prompt_length),
            int(quality_decode_length),
            int(long_decode_length),
        )
        return [
            KVCacheBenchmarkProfile(
                name=str(profile.name),
                prompt_length=int(profile.prompt_length),
                decode_length=int(profile.decode_length),
                batch_size=int(profile.batch_size),
                request_count=int(profile.request_count),
                fixed_vram_budget_mb=int(profile.fixed_vram_budget_mb),
            )
            for profile in native_profiles
        ]

    profiles: list[KVCacheBenchmarkProfile] = []
    for batch_size in batch_sizes:
        profiles.append(
            KVCacheBenchmarkProfile(
                name="memory_savings_vs_latency",
                prompt_length=long_prompt_length,
                decode_length=long_decode_length,
                batch_size=int(batch_size),
            )
        )
        profiles.append(
            KVCacheBenchmarkProfile(
                name="long_context_generation_quality",
                prompt_length=long_prompt_length,
                decode_length=quality_decode_length,
                batch_size=int(batch_size),
            )
        )
        profiles.append(
            KVCacheBenchmarkProfile(
                name="throughput_per_gb",
                prompt_length=short_prompt_length,
                decode_length=long_decode_length,
                batch_size=int(batch_size),
            )
        )
    return profiles


def _model_forward_parameter_name(
    model: torch.nn.Module,
    candidates: tuple[str, ...],
) -> Optional[str]:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return None

    for candidate in candidates:
        if candidate in signature.parameters:
            return candidate
    return None


def preferred_last_token_logits_kwarg(model: torch.nn.Module) -> Optional[str]:
    """
    Detect a forward kwarg that limits logits materialization to the decode token.
    """
    supports_logits_to_keep = getattr(model, "_supports_logits_to_keep", None)
    if callable(supports_logits_to_keep):
        try:
            if supports_logits_to_keep():
                return "logits_to_keep"
        except Exception:
            pass
    return _model_forward_parameter_name(model, ("logits_to_keep", "num_logits_to_keep"))


def build_generation_kwargs(
    model: torch.nn.Module,
    max_new_tokens: int,
    *,
    prefer_last_token_logits: bool = True,
) -> dict:
    """
    Build deterministic generation kwargs and, when supported, request last-token logits only.
    """
    kwargs = {"max_new_tokens": max_new_tokens}
    generation_config = copy.deepcopy(getattr(model, "generation_config", None))

    if generation_config is not None:
        generation_config.do_sample = False
        for attr in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, attr):
                setattr(generation_config, attr, None)
        kwargs["generation_config"] = generation_config
    else:
        kwargs.update(
            {
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "top_k": None,
            }
        )

    if prefer_last_token_logits:
        kwarg_name = preferred_last_token_logits_kwarg(model)
        if kwarg_name is not None:
            kwargs[kwarg_name] = 1

    return kwargs


def last_token_logits_only_enabled() -> bool:
    """
    Return whether decode-time last-token logits specialization is enabled.
    """
    return _env_enabled("BARQTRAIN_LAST_TOKEN_LOGITS_ONLY", "1")


def maybe_prepare_last_token_logits_generate_kwargs(
    model: torch.nn.Module,
    args,
    kwargs: dict,
) -> tuple[dict, bool]:
    """
    Request last-token logits only for generate() when the model supports it.
    """
    del args  # The specialization only depends on model support and generate kwargs.

    if not last_token_logits_only_enabled():
        return kwargs, False

    kwarg_name = preferred_last_token_logits_kwarg(model)
    if kwarg_name is None:
        return kwargs, False

    if kwargs.get("output_logits"):
        return kwargs, False

    generation_config = kwargs.get("generation_config") or getattr(model, "generation_config", None)
    if generation_config is not None and getattr(generation_config, "output_logits", False):
        return kwargs, False

    if kwarg_name in kwargs:
        return kwargs, int(kwargs[kwarg_name]) == 1

    updated_kwargs = dict(kwargs)
    updated_kwargs[kwarg_name] = 1
    return updated_kwargs, True


__all__ = [
    "BenchmarkMemoryBreakdown",
    "CudaMemorySnapshot",
    "DecodeBenchmarkProfile",
    "KVCacheBenchmarkProfile",
    "build_generation_kwargs",
    "build_memory_breakdown",
    "capture_cuda_peak_bytes",
    "cuda_memory_snapshot",
    "detailed_profiling_enabled",
    "generation_overhead_mb",
    "last_token_logits_only_enabled",
    "maybe_prepare_last_token_logits_generate_kwargs",
    "model_resident_cuda_bytes",
    "native_memory_snapshot",
    "paged_kv_cache_bytes",
    "phase1_inference_profiles",
    "phase2_kv_cache_profiles",
    "phase3_quantized_kv_profiles",
    "preferred_last_token_logits_kwarg",
    "record_inference_peak_bytes",
    "record_training_peak_bytes",
    "reset_native_memory_tracking",
    "set_detailed_profiling_enabled",
    "track_decode_temp_memory",
    "track_kv_cache_memory",
    "track_resident_model_memory",
]
