"""
Model patching utilities for BarqTrain

This module provides functions to monkey-patch Hugging Face models
with BarqTrain's optimized CUDA kernels and Rust operations.
"""

from typing import Union

import torch


def patch_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Patch a Hugging Face model with BarqTrain optimizations.

    This function replaces standard Hugging Face components with
    BarqTrain's fused CUDA kernels for improved performance and
    reduced memory usage.

    Args:
        model: A Hugging Face model (e.g., LlamaForCausalLM, Qwen2ForCausalLM)

    Returns:
        The same model with BarqTrain optimizations applied

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> from barqtrain import patch_model
        >>>
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> patch_model(model)
    """
    # Detect model type and apply appropriate patches
    model_type = model.config.model_type if hasattr(model, "config") else None

    if model_type == "llama":
        return patch_llama(model)
    elif model_type == "qwen2":
        return patch_qwen(model)
    else:
        # Generic patching for other models
        return _patch_generic(model)


def patch_llama(model: torch.nn.Module) -> torch.nn.Module:
    """
    Patch Llama/Llama2/Llama3 models with BarqTrain optimizations.

    Replaces:
    - RMSNorm with fused RMSNorm kernel
    - Attention with FlashAttention + fused RoPE (Phase 5)
    - Loss with chunked cross-entropy (Phase 4)

    Args:
        model: A LlamaForCausalLM or compatible model

    Returns:
        The patched model
    """
    from barqtrain.ops import fused_rms_norm

    try:
        import transformers.models.llama.modeling_llama as llama_model
    except ImportError:
        return model

    # Patch all RMSNorm layers
    patched_count = 0
    for name, module in model.named_modules():
        if isinstance(module, llama_model.LlamaRMSNorm):
            # Store original weight
            original_weight = module.weight.data.clone()
            original_eps = module.variance_epsilon

            # Replace forward with fused version
            def make_forward(eps=original_eps):
                def forward(self, x):
                    return fused_rms_norm(x, self.weight, eps)
                return forward

            # Monkey-patch the forward method
            import types
            module.forward = types.MethodType(make_forward(), module)

            # Ensure weight is on correct device
            module.weight.data = original_weight

            patched_count += 1

    if patched_count > 0:
        print(f"BarqTrain: Patched {patched_count} RMSNorm layer(s) with fused kernel")

    # TODO: Add attention patching in Phase 5
    # TODO: Add loss patching in Phase 4

    return model


def patch_qwen(model: torch.nn.Module) -> torch.nn.Module:
    """
    Patch Qwen/Qwen2 models with BarqTrain optimizations.

    Args:
        model: A Qwen2ForCausalLM or compatible model

    Returns:
        The patched model
    """
    # TODO: Implement actual patching once kernels are ready
    # This is a placeholder for Phase 3-5
    return model


def _patch_generic(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply generic patches to any compatible model.

    Args:
        model: Any PyTorch model

    Returns:
        The patched model
    """
    # TODO: Implement generic patching logic
    return model


def unpatch_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Restore original Hugging Face implementations.

    Args:
        model: A patched BarqTrain model

    Returns:
        The model with original implementations restored
    """
    # TODO: Implement unpatching logic
    return model
