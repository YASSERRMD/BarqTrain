"""
Setup script for BarqTrain.

This setup builds:
- CUDA C++ extension (when CUDA and Torch build helpers are available)
- Rust extension via setuptools-rust (when Rust toolchain is available)
"""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

PROJECT_ROOT = Path(__file__).parent.resolve()


def _get_cuda_build_config():
    """Return (ext_modules, cmdclass) for optional CUDA extension."""
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    except Exception as exc:
        print(f"Warning: Torch CUDA build helpers unavailable, skipping CUDA extension: {exc}")
        return [], {}

    if CUDA_HOME is None:
        print("Warning: CUDA_HOME is not set; skipping CUDA extension build.")
        return [], {}

    sources = [
        PROJECT_ROOT / "csrc" / "src" / "bindings.cpp",
        PROJECT_ROOT / "csrc" / "kernels" / "rmsnorm.cu",
        PROJECT_ROOT / "csrc" / "kernels" / "flash_attention.cu",
        PROJECT_ROOT / "csrc" / "kernels" / "chunked_cross_entropy.cu",
        PROJECT_ROOT / "csrc" / "kernels" / "lora.cu",
    ]
    existing_sources = [str(src) for src in sources if src.exists()]

    if not existing_sources:
        print("Warning: CUDA sources not found, skipping CUDA extension.")
        return [], {}

    ext_modules = [
        CUDAExtension(
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
    ]

    return ext_modules, {"build_ext": BuildExtension}


def _get_rust_extensions():
    """Return rust_extensions list, or None when setuptools-rust is unavailable."""
    cargo_toml = PROJECT_ROOT / "rust" / "Cargo.toml"
    if not cargo_toml.exists():
        return []

    try:
        from setuptools_rust import Binding, RustExtension
    except Exception as exc:
        print(
            "Warning: setuptools-rust unavailable, skipping Rust extension build: "
            f"{exc}"
        )
        return None

    return [
        RustExtension(
            "barqtrain_rs",
            path=str(cargo_toml),
            binding=Binding.PyO3,
            optional=True,
        )
    ]


cuda_ext_modules, cuda_cmdclass = _get_cuda_build_config()
rust_extensions = _get_rust_extensions()

setup_kwargs = dict(
    name="barqtrain",
    version="0.1.0",
    description="High-performance LLM fine-tuning accelerator with CUDA kernels and Rust pipelines",
    long_description=(PROJECT_ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="BarqTrain Contributors",
    python_requires=">=3.8",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    include_package_data=True,
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
    ext_modules=cuda_ext_modules,
    cmdclass=cuda_cmdclass,
    zip_safe=False,
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

if rust_extensions is not None:
    setup_kwargs["rust_extensions"] = rust_extensions

setup(**setup_kwargs)
