"""
Runtime native loading helpers for BarqTrain.

This module centralizes Python-side FFI loading for optional native backends:
- CUDA kernels via Torch's C++/CUDA extension loader
- Rust data pipeline bindings via Python extension import
"""

from __future__ import annotations

import importlib
import os
import warnings
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CUDA_SOURCES = (
    _PROJECT_ROOT / "csrc" / "src" / "bindings.cpp",
    _PROJECT_ROOT / "csrc" / "kernels" / "rmsnorm.cu",
    _PROJECT_ROOT / "csrc" / "kernels" / "flash_attention.cu",
    _PROJECT_ROOT / "csrc" / "kernels" / "chunked_cross_entropy.cu",
    _PROJECT_ROOT / "csrc" / "kernels" / "lora.cu",
)


def _env_enabled(name: str, default: str = "1") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}


@lru_cache(maxsize=1)
def load_cuda_backend() -> Optional[ModuleType]:
    """
    Load the CUDA backend module.

    Order:
    1) Import pre-built `barqtrain_cuda` if available.
    2) If not available and auto-build is enabled, JIT-build with Torch extension loader.
    3) Fall back to None.
    """
    try:
        return importlib.import_module("barqtrain_cuda")
    except ImportError:
        pass

    if not _env_enabled("BARQTRAIN_AUTO_BUILD", "1"):
        return None

    missing_sources = [src for src in _CUDA_SOURCES if not src.exists()]
    if missing_sources:
        warnings.warn(
            "BarqTrain CUDA sources missing, skipping CUDA backend: "
            + ", ".join(str(path) for path in missing_sources)
        )
        return None

    try:
        from torch.utils.cpp_extension import CUDA_HOME, load
    except Exception as exc:
        warnings.warn(
            f"BarqTrain CUDA auto-build skipped because Torch build helpers are unavailable: {exc}"
        )
        return None

    if CUDA_HOME is None:
        warnings.warn(
            "BarqTrain CUDA auto-build skipped because CUDA_HOME is not set. "
            "Set CUDA_HOME or pre-build the extension."
        )
        return None

    build_dir = Path.home() / ".cache" / "barqtrain" / "torch_extensions"
    build_dir.mkdir(parents=True, exist_ok=True)

    try:
        return load(
            name="barqtrain_cuda",
            sources=[str(src) for src in _CUDA_SOURCES],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "--use_fast_math",
                "-Xptxas=-v",
                "--ptxas-options=-v",
            ],
            with_cuda=True,
            build_directory=str(build_dir),
            verbose=_env_enabled("BARQTRAIN_VERBOSE_BUILD", "0"),
        )
    except Exception as exc:
        warnings.warn(f"BarqTrain CUDA auto-build failed: {exc}")
        return None


@lru_cache(maxsize=1)
def load_rust_backend() -> Optional[ModuleType]:
    """
    Load the Rust backend Python extension if available.
    """
    try:
        return importlib.import_module("barqtrain_rs")
    except ImportError:
        return None
