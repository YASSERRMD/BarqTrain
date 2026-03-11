"""
Runtime native loading helpers for BarqTrain.

This module centralizes Python-side FFI loading for optional native backends:
- CUDA kernels via Torch's C++/CUDA extension loader
- Rust data pipeline bindings via Python extension import
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import sys
import warnings
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
    _PROJECT_ROOT / "csrc" / "kernels" / "paged_kv_cache.cu",
)
_CUDA_BACKEND: Optional[ModuleType] = None
_RUST_BACKEND: Optional[ModuleType] = None
_CUDA_BUILD_FAILED = False
_RUST_BUILD_FAILED = False


def _env_enabled(name: str, default: str = "1") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _detect_cuda_home() -> Optional[str]:
    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        cuda_home = os.environ.get(env_name)
        if cuda_home and Path(cuda_home).exists():
            return cuda_home

    try:
        import torch
        import torch.utils.cpp_extension as cpp_extension

        cuda_home = cpp_extension.CUDA_HOME
        if cuda_home:
            return cuda_home

        find_cuda_home = getattr(cpp_extension, "_find_cuda_home", None)
        if callable(find_cuda_home):
            cuda_home = find_cuda_home()
            if cuda_home:
                return cuda_home

        if torch.version.cuda:
            version = torch.version.cuda
            candidates = [
                f"/usr/local/cuda-{version}",
                f"/usr/local/cuda-{version.split('.')[0]}",
            ]
            for candidate in candidates:
                if Path(candidate).exists():
                    return candidate
    except Exception:
        pass

    nvcc = shutil.which("nvcc")
    if nvcc:
        return str(Path(nvcc).resolve().parents[1])

    for candidate in ("/usr/local/cuda", "/opt/cuda"):
        if Path(candidate).exists():
            return candidate

    return None


def _ensure_torch_cuda_arch_list() -> None:
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return

    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
            return
    except Exception:
        pass

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get(
        "BARQTRAIN_CUDA_ARCH_LIST",
        "7.5",
    )


def load_cuda_backend() -> Optional[ModuleType]:
    """
    Load the CUDA backend module.

    Order:
    1) Import pre-built `barqtrain_cuda` if available.
    2) If not available and auto-build is enabled, JIT-build with Torch extension loader.
    3) Fall back to None.
    """
    global _CUDA_BACKEND, _CUDA_BUILD_FAILED
    if _CUDA_BACKEND is not None:
        return _CUDA_BACKEND

    try:
        _CUDA_BACKEND = importlib.import_module("barqtrain_cuda")
        return _CUDA_BACKEND
    except ImportError:
        pass

    if _CUDA_BUILD_FAILED:
        return None

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
        import torch.utils.cpp_extension as cpp_extension
    except Exception as exc:
        warnings.warn(
            f"BarqTrain CUDA auto-build skipped because Torch build helpers are unavailable: {exc}"
        )
        return None

    cuda_home = _detect_cuda_home()
    if cuda_home:
        os.environ.setdefault("CUDA_HOME", cuda_home)
        cpp_extension.CUDA_HOME = cuda_home
        _ensure_torch_cuda_arch_list()
    elif cpp_extension.CUDA_HOME is not None:
        cuda_home = cpp_extension.CUDA_HOME

    if cuda_home is None:
        warnings.warn(
            "BarqTrain CUDA auto-build skipped because a CUDA toolkit was not found. "
            "Set CUDA_HOME or install the extension with BARQTRAIN_BUILD_CUDA=1."
        )
        return None

    build_dir = Path.home() / ".cache" / "barqtrain" / "torch_extensions"
    build_dir.mkdir(parents=True, exist_ok=True)

    try:
        _CUDA_BACKEND = cpp_extension.load(
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
        return _CUDA_BACKEND
    except Exception as exc:
        _CUDA_BUILD_FAILED = True
        warnings.warn(f"BarqTrain CUDA auto-build failed: {exc}")
        return None


def load_rust_backend() -> Optional[ModuleType]:
    """
    Load the Rust backend Python extension if available.
    """
    global _RUST_BACKEND, _RUST_BUILD_FAILED
    if _RUST_BACKEND is not None:
        return _RUST_BACKEND

    try:
        _RUST_BACKEND = importlib.import_module("barqtrain_rs")
        return _RUST_BACKEND
    except ImportError:
        if _RUST_BUILD_FAILED:
            return None
        if not _env_enabled("BARQTRAIN_AUTO_BUILD", "1"):
            return None

        if shutil.which("cargo") and importlib.util.find_spec("maturin") is not None:
            try:
                import subprocess

                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "maturin",
                        "develop",
                        "--release",
                        "--manifest-path",
                        str(_PROJECT_ROOT / "rust" / "Cargo.toml"),
                    ],
                    cwd=str(_PROJECT_ROOT),
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                _RUST_BACKEND = importlib.import_module("barqtrain_rs")
                return _RUST_BACKEND
            except Exception:
                _RUST_BUILD_FAILED = True
                return None
        return None
