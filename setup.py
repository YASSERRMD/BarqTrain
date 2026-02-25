"""
BarqTrain setup.py — CUDA extension only.

All Python package metadata lives in pyproject.toml.
This file exists solely to build the optional CUDA C++ extension:

    BARQTRAIN_BUILD_CUDA=1 pip install -e .
"""

import os
from pathlib import Path

from setuptools import setup

_cuda_ext = None

if os.environ.get("BARQTRAIN_BUILD_CUDA"):
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME  # noqa

        if CUDA_HOME and Path("csrc").exists():
            _sources = [
                "csrc/src/bindings.cpp",
                "csrc/kernels/rmsnorm.cu",
                "csrc/kernels/flash_attention.cu",
                "csrc/kernels/chunked_cross_entropy.cu",
                "csrc/kernels/lora.cu",
            ]
            _existing = [s for s in _sources if Path(s).exists()]
            if _existing:
                _cuda_ext = CUDAExtension(
                    name="barqtrain_cuda",
                    sources=_existing,
                    extra_compile_args={
                        "cxx":  ["-O3", "-std=c++17"],
                        "nvcc": ["-O3", "-std=c++17", "--use_fast_math"],
                    },
                )
    except Exception as e:
        print(f"Warning: CUDA extension skipped: {e}")

# Only call setup() when building the CUDA extension.
# pyproject.toml owns all package/metadata configuration.
if _cuda_ext is not None:
    from torch.utils.cpp_extension import BuildExtension  # noqa
    setup(
        ext_modules=[_cuda_ext],
        cmdclass={"build_ext": BuildExtension},
    )
