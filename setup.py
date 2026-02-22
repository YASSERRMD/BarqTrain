"""
Setup script for BarqTrain

This script handles building both the CUDA C++ extension and the
Rust PyO3 module.
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    _TORCH_AVAILABLE = True
except ImportError:
    BuildExtension = None
    CUDAExtension = None
    _TORCH_AVAILABLE = False

# Project metadata
PROJECT_NAME = "barqtrain"
VERSION = "0.1.0"
DESCRIPTION = "High-performance LLM fine-tuning accelerator with CUDA kernels and Rust pipelines"
LONG_DESCRIPTION = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")


def build_cuda_extension():
    """Build the CUDA C++ extension.

    Only attempted when BARQTRAIN_BUILD_CUDA=1 is set in the environment.
    This prevents accidental compilation during plain `pip install -e .`
    on machines that happen to have CUDA installed (e.g. Google Colab).

    To build::

        BARQTRAIN_BUILD_CUDA=1 pip install -e .
    """
    # Opt-in only: require explicit env var
    if not os.environ.get("BARQTRAIN_BUILD_CUDA", ""):
        return None
    if not _TORCH_AVAILABLE:
        print("Warning: torch not available, skipping CUDA extension build")
        return None
    try:
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is None:
            print("Warning: CUDA_HOME not set, skipping CUDA extension build")
            return None

        if not os.path.exists("csrc"):
            print("Warning: csrc/ directory not found, skipping CUDA extension build")
            return None

        # Source files for CUDA extension
        sources = [
            "csrc/src/bindings.cpp",
            "csrc/kernels/rmsnorm.cu",
            "csrc/kernels/flash_attention.cu",
            "csrc/kernels/chunked_cross_entropy.cu",
            "csrc/kernels/lora.cu",
        ]

        # Filter existing sources
        existing_sources = [s for s in sources if Path(s).exists()]

        if not existing_sources:
            return None

        ext = CUDAExtension(
            name="barqtrain_cuda",
            sources=existing_sources,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "-Xptxas=-v",
                    "--ptxas-options=-v",
                ],
            },
        )
        return ext
    except Exception as e:
        print(f"Warning: Could not build CUDA extension: {e}")
        return None


def build_rust_extension():
    """Build the Rust PyO3 extension if Rust is available."""
    try:
        import torch

        # Check if Rust and Cargo are available
        result = subprocess.run(
            ["cargo", "--version"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print("Warning: Rust/Cargo not found, skipping Rust extension")
            return None

        # Check if maturin is available
        result = subprocess.run(
            ["maturin", "--version"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print("Warning: maturin not found. Install with: pip install maturin")
            return None

        # Build Rust extension with maturin
        rust_dir = Path("rust")
        if not rust_dir.exists():
            return None

        print("Building Rust extension...")
        result = subprocess.run(
            ["maturin", "develop", "--release"],
            cwd=rust_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Warning: Rust build failed: {result.stderr}")
            return None

        print("Rust extension built successfully")
        return None  # Maturin handles installation directly

    except Exception as e:
        print(f"Warning: Could not build Rust extension: {e}")
        return None


_cuda_ext = build_cuda_extension()

# Setup configuration
setup(
    name=PROJECT_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="BarqTrain Contributors",
    python_requires=">=3.8",
    packages=["barqtrain", "barqtrain.benchmarks"],
    package_dir={"": "python"},
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "peft>=0.6.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "benchmark": [
            "matplotlib>=3.7.0",
            "pandas>=2.0.0",
            "tqdm>=4.66.0",
            "psutil>=5.9.0",
            "wandb>=0.15.0",
        ],
    },
    ext_modules=[_cuda_ext] if _cuda_ext is not None else [],
    cmdclass={"build_ext": BuildExtension} if (_cuda_ext is not None and BuildExtension) else {},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Rust",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="LLM fine-tuning CUDA Rust optimization transformers",
    project_urls={
        "Homepage": "https://github.com/YASSERRMD/BarqTrain",
        "Repository": "https://github.com/YASSERRMD/BarqTrain",
        "Issues": "https://github.com/YASSERRMD/BarqTrain/issues",
    },
)

# Build Rust extension separately if available
if "--build-rust" in sys.argv:
    sys.argv.remove("--build-rust")
    build_rust_extension()
