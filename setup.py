"""
BarqTrain setup.py

All Python package metadata is in pyproject.toml.
This file adds the optional CUDA extension when BARQTRAIN_BUILD_CUDA=1 and
builds the Rust extension when setuptools-rust and a Rust toolchain are available.
setup() is ALWAYS called so pip can gather editable install metadata.
"""

import os
import shutil
from pathlib import Path
from setuptools import setup

_cuda_ext = None
_build_ext_cls = {}
_rust_extensions = []
_rust_optional = os.environ.get("BARQTRAIN_OPTIONAL_RUST", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _detect_cuda_home(initial_cuda_home=None):
    if initial_cuda_home and Path(initial_cuda_home).exists():
        return initial_cuda_home

    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        cuda_home = os.environ.get(env_name)
        if cuda_home and Path(cuda_home).exists():
            return cuda_home

    nvcc = shutil.which("nvcc")
    if nvcc:
        return str(Path(nvcc).resolve().parents[1])

    try:
        import torch

        if torch.version.cuda:
            version = torch.version.cuda
            for candidate in (
                f"/usr/local/cuda-{version}",
                f"/usr/local/cuda-{version.split('.')[0]}",
            ):
                if Path(candidate).exists():
                    return candidate
    except Exception:
        pass

    for candidate in ("/usr/local/cuda", "/opt/cuda"):
        if Path(candidate).exists():
            return candidate

    return None

try:
    from setuptools_rust import Binding, RustExtension

    if Path("rust/Cargo.toml").exists():
        _rust_extensions.append(
            RustExtension(
                "barqtrain_rs",
                path="rust/Cargo.toml",
                binding=Binding.PyO3,
                debug=False,
                optional=_rust_optional,
            )
        )
except Exception as e:
    print(f"Warning: Rust extension skipped: {e}")

if os.environ.get("BARQTRAIN_BUILD_CUDA"):
    try:
        import torch.utils.cpp_extension as cpp_extension
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

        cuda_home = _detect_cuda_home(CUDA_HOME)
        if cuda_home:
            os.environ.setdefault("CUDA_HOME", cuda_home)
            cpp_extension.CUDA_HOME = cuda_home

        if cuda_home and Path("csrc").exists():
            _sources = [s for s in [
                "csrc/src/bindings.cpp",
                "csrc/kernels/rmsnorm.cu",
                "csrc/kernels/flash_attention.cu",
                "csrc/kernels/chunked_cross_entropy.cu",
                "csrc/kernels/lora.cu",
            ] if Path(s).exists()]

            if _sources:
                _cuda_ext = CUDAExtension(
                    name="barqtrain_cuda",
                    sources=_sources,
                    extra_compile_args={
                        "cxx":  ["-O3", "-std=c++17"],
                        "nvcc": ["-O3", "-std=c++17", "--use_fast_math"],
                    },
                )
                _build_ext_cls = {"build_ext": BuildExtension}
        else:
            print("Warning: CUDA extension skipped: CUDA toolkit not found")
    except Exception as e:
        print(
            "Warning: CUDA extension skipped: "
            f"{e}. If torch is already installed, retry with "
            "`BARQTRAIN_BUILD_CUDA=1 pip install -e . --no-build-isolation`."
        )

# Always call setup() — pip needs it for editable install metadata.
# pyproject.toml owns all package/name/version metadata.
setup(
    ext_modules=[_cuda_ext] if _cuda_ext else [],
    rust_extensions=_rust_extensions,
    cmdclass=_build_ext_cls,
    zip_safe=False,
)
