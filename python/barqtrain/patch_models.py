"""
Model patching utilities for BarqTrain

This module provides functions to monkey-patch Hugging Face models
with BarqTrain's optimized CUDA kernels and Rust operations.
"""

import types

import torch


def patch_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Patch a Hugging Face model with BarqTrain optimizations.

    This function replaces standard Hugging Face components with
    BarqTrain's fused CUDA kernels for improved performance and
    reduced memory usage.

    Args:
        model: A Hugging Face model (e.g., LlamaForCausalLM, Qwen2ForCausalLM, Lfm2ForCausalLM)

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
    model_config = getattr(model, "config", None)
    model_type = getattr(model_config, "model_type", None)
    architectures = [arch.lower() for arch in getattr(model_config, "architectures", [])]

    if model_type == "llama":
        return patch_llama(model)
    if model_type == "qwen2":
        return patch_qwen(model)
    if model_type == "lfm2" or any(arch.startswith("lfm2") for arch in architectures):
        return patch_lfm2(model)

    # Generic patching for other models
    return _patch_generic(model)


def _patch_rmsnorm_layers(
    model: torch.nn.Module,
    rmsnorm_class: type,
    model_label: str,
) -> torch.nn.Module:
    """
    Patch all matching RMSNorm layers in a model with BarqTrain fused RMSNorm.
    """
    from barqtrain.ops import fused_rms_norm

    patched_count = 0

    for _, module in model.named_modules():
        if isinstance(module, rmsnorm_class):
            original_weight = module.weight.data.clone()
            eps = getattr(module, "variance_epsilon", 1e-6)

            def make_forward(eps_value=eps):
                def forward(self, x):
                    return fused_rms_norm(x, self.weight, eps_value)

                return forward

            module.forward = types.MethodType(make_forward(), module)
            module.weight.data = original_weight
            patched_count += 1

    if patched_count > 0:
        print(
            f"BarqTrain: Patched {patched_count} {model_label} RMSNorm layer(s) with fused kernel"
        )

    return model


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
    try:
        import transformers.models.llama.modeling_llama as llama_model
    except ImportError:
        return model

    return _patch_rmsnorm_layers(model, llama_model.LlamaRMSNorm, "Llama")


def patch_lfm2(model: torch.nn.Module) -> torch.nn.Module:
    """
    Patch Liquid LFM2/LFM2.5 models with BarqTrain optimizations.

    Replaces:
    - RMSNorm with fused RMSNorm kernel

    Args:
        model: A Lfm2ForCausalLM or compatible model

    Returns:
        The patched model
    """
    try:
        import transformers.models.lfm2.modeling_lfm2 as lfm2_model
    except ImportError:
        return model

    return _patch_rmsnorm_layers(model, lfm2_model.Lfm2RMSNorm, "LFM2")


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
