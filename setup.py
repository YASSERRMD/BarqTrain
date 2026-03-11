"""
BarqTrain setup.py

All Python package metadata is in pyproject.toml.
This file adds the optional CUDA extension when BARQTRAIN_BUILD_CUDA=1 and
builds the Rust extension when setuptools-rust and a Rust toolchain are available.
setup() is ALWAYS called so pip can gather editable install metadata.
"""

import os
from pathlib import Path
from setuptools import setup

_cuda_ext = None
_build_ext_cls = {}
_rust_extensions = []

try:
    from setuptools_rust import Binding, RustExtension

    if Path("rust/Cargo.toml").exists():
        _rust_extensions.append(
            RustExtension(
                "barqtrain_rs",
                path="rust/Cargo.toml",
                binding=Binding.PyO3,
                debug=False,
                optional=True,
            )
        )
except Exception as e:
    print(f"Warning: Rust extension skipped: {e}")

if os.environ.get("BARQTRAIN_BUILD_CUDA"):
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

        if CUDA_HOME and Path("csrc").exists():
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
    except Exception as e:
        print(f"Warning: CUDA extension skipped: {e}")

# Always call setup() — pip needs it for editable install metadata.
# pyproject.toml owns all package/name/version metadata.
setup(
    ext_modules=[_cuda_ext] if _cuda_ext else [],
    rust_extensions=_rust_extensions,
    cmdclass=_build_ext_cls,
    zip_safe=False,
)
