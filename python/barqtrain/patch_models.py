"""
Model patching utilities for BarqTrain

This module provides functions to monkey-patch Hugging Face models
with BarqTrain's optimized CUDA kernels and Rust operations.
"""

import importlib
import types

import torch


def patch_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Patch a Hugging Face model with BarqTrain optimizations.

    This function replaces standard Hugging Face components with
    BarqTrain's fused CUDA kernels for improved performance and
    reduced memory usage.

    Args:
        model: A Hugging Face model (e.g., LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM, Lfm2ForCausalLM, GemmaForCausalLM)

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
    if model_type in {"mistral", "mixtral"} or any(
        "mistral" in arch or "mixtral" in arch for arch in architectures
    ):
        return patch_mistral(model)
    if model_type == "qwen2":
        return patch_qwen(model)
    if (model_type and "gemma" in model_type) or any("gemma" in arch for arch in architectures):
        return patch_gemma(model)
    if model_type == "lfm2" or any(arch.startswith("lfm2") for arch in architectures):
        return patch_lfm2(model)

    # Generic patching for other models
    return _patch_generic(model)


def _patch_rmsnorm_layers(
    model: torch.nn.Module,
    rmsnorm_class: type,
    model_label: str,
    eps_attributes: tuple = ("variance_epsilon", "eps"),
    weight_transform=None,
    cast_input_to_float32: bool = False,
    cast_output_to_input_dtype: bool = False,
) -> torch.nn.Module:
    """
    Patch all matching RMSNorm layers in a model with BarqTrain fused RMSNorm.
    """
    from barqtrain.ops import fused_rms_norm

    if weight_transform is None:
        weight_transform = lambda w: w

    patched_count = 0

    for _, module in model.named_modules():
        if isinstance(module, rmsnorm_class):
            eps = 1e-6
            for attr_name in eps_attributes:
                if hasattr(module, attr_name):
                    eps = getattr(module, attr_name)
                    break

            def make_forward(
                eps_value=eps,
                transform=weight_transform,
                cast_input=cast_input_to_float32,
                cast_output=cast_output_to_input_dtype,
            ):
                def forward(self, x):
                    input_x = x.float() if cast_input else x
                    weight = transform(self.weight)
                    output = fused_rms_norm(input_x, weight, eps_value)
                    return output.type_as(x) if cast_output else output

                return forward

            module.forward = types.MethodType(make_forward(), module)
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


def patch_mistral(model: torch.nn.Module) -> torch.nn.Module:
    """
    Patch Mistral/Mixtral family models with BarqTrain optimizations.

    Replaces:
    - RMSNorm with fused RMSNorm kernel

    Args:
        model: A Mistral or Mixtral compatible model

    Returns:
        The patched model
    """
    rmsnorm_targets = [
        ("transformers.models.mistral.modeling_mistral", "MistralRMSNorm", "Mistral"),
        ("transformers.models.mixtral.modeling_mixtral", "MixtralRMSNorm", "Mixtral"),
    ]

    for module_name, class_name, label in rmsnorm_targets:
        try:
            module = importlib.import_module(module_name)
            rmsnorm_cls = getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
        model = _patch_rmsnorm_layers(model, rmsnorm_cls, label)

    return model


def patch_gemma(model: torch.nn.Module) -> torch.nn.Module:
    """
    Patch Google Gemma family models with BarqTrain optimizations.

    Replaces:
    - RMSNorm with fused RMSNorm kernel

    Supported classes include Gemma/Gemma2/Gemma3 and RecurrentGemma RMSNorm.

    Args:
        model: A Gemma-family model

    Returns:
        The patched model
    """
    rmsnorm_targets = [
        ("transformers.models.gemma.modeling_gemma", "GemmaRMSNorm", "Gemma"),
        ("transformers.models.gemma2.modeling_gemma2", "Gemma2RMSNorm", "Gemma2"),
        ("transformers.models.gemma3.modeling_gemma3", "Gemma3RMSNorm", "Gemma3"),
        (
            "transformers.models.recurrent_gemma.modeling_recurrent_gemma",
            "RecurrentGemmaRMSNorm",
            "RecurrentGemma",
        ),
    ]

    for module_name, class_name, label in rmsnorm_targets:
        try:
            module = importlib.import_module(module_name)
            rmsnorm_cls = getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
        model = _patch_rmsnorm_layers(
            model,
            rmsnorm_cls,
            label,
            eps_attributes=("eps", "variance_epsilon"),
            weight_transform=lambda w: 1.0 + w.float(),
            cast_input_to_float32=True,
            cast_output_to_input_dtype=True,
        )

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
