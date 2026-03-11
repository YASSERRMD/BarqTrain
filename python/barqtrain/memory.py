"""Memory and generation helpers for BarqTrain benchmarks."""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from typing import Optional

import torch


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


def generation_overhead_mb(
    resident_snapshot: CudaMemorySnapshot,
    peak_snapshot: CudaMemorySnapshot,
) -> float:
    """
    Compute generation overhead beyond the resident model footprint.
    """
    return max(peak_snapshot.max_allocated_mb - resident_snapshot.allocated_mb, 0.0)


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
