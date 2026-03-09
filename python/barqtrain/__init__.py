"""
BarqTrain: High-Performance LLM Fine-Tuning Accelerator

BarqTrain combines Rust for GIL-free data pipelines, CUDA C++ for custom
fused kernels, and Python for the user-facing API to achieve massive
speedups and memory reductions.

Basic usage:
    >>> from barqtrain import patch_model
    >>> from transformers import AutoModelForCausalLM
    >>>
    >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> patch_model(model)  # Replaces HF layers with BarqTrain optimized kernels
"""

__version__ = "0.1.0"

# Ensure the package is importable even when running directly from the repo
# (e.g. on Colab after git clone without pip install -e .)
import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # python/
if _here not in _sys.path:
    _sys.path.insert(0, _here)

from barqtrain.patch_models import patch_model, patch_llama, patch_qwen

__all__ = [
    "patch_model",
    "patch_llama",
    "patch_qwen",
    "__version__",
]
