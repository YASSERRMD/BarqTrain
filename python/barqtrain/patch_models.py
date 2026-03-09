"""
Model patching utilities for BarqTrain

This module provides functions to monkey-patch Hugging Face models
with BarqTrain's optimized CUDA kernels and Rust operations.
"""

import importlib
import types
from functools import lru_cache

import torch


def _apply_patch_once(model: torch.nn.Module, patch_fn) -> torch.nn.Module:
    """
    Apply a patch function only once per model instance.
    """
    if getattr(model, "_barqtrain_model_patched", False):
        return model
    patched_model = patch_fn(model)
    setattr(patched_model, "_barqtrain_model_patched", True)
    return patched_model


@lru_cache(maxsize=256)
def _resolve_rmsnorm_class(module_name: str, class_name: str):
    """
    Resolve and cache RMSNorm class lookups from transformers modules.
    """
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        return None


def _patch_rmsnorm_resolved_targets(
    model: torch.nn.Module,
    resolved_targets: list,
) -> torch.nn.Module:
    """
    Patch RMSNorm layers in one pass over model.modules().
    """
    from barqtrain.ops import fused_rms_norm

    if not resolved_targets:
        return model

    patched_counts = {label: 0 for _, label, *_ in resolved_targets}

    for module in model.modules():
        if getattr(module, "_barqtrain_rmsnorm_patched", False):
            continue

        for (
            rmsnorm_class,
            model_label,
            eps_attributes,
            weight_transform,
            cast_input_to_float32,
            cast_output_to_input_dtype,
        ) in resolved_targets:
            if not isinstance(module, rmsnorm_class):
                continue

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
            setattr(module, "_barqtrain_rmsnorm_patched", True)
            patched_counts[model_label] += 1
            break

    for label, count in patched_counts.items():
        if count > 0:
            print(f"BarqTrain: Patched {count} {label} RMSNorm layer(s) with fused kernel")

    return model


def patch_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Patch a Hugging Face model with BarqTrain optimizations.

    This function replaces standard Hugging Face components with
    BarqTrain's fused CUDA kernels for improved performance and
    reduced memory usage.

    Args:
        model: A Hugging Face model (e.g., LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM, Lfm2ForCausalLM, GemmaForCausalLM, DeepseekV3ForCausalLM, Phi3ForCausalLM)

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

    if model_type in {"llama4", "llama4_text"} or any("llama4" in arch for arch in architectures):
        return _apply_patch_once(model, patch_llama4)
    if model_type == "llama":
        return _apply_patch_once(model, patch_llama)
    if model_type in {"mistral", "mixtral"} or any(
        "mistral" in arch or "mixtral" in arch for arch in architectures
    ):
        return _apply_patch_once(model, patch_mistral)
    if model_type in {"deepseek_v2", "deepseek_v3"} or any("deepseek" in arch for arch in architectures):
        return _apply_patch_once(model, patch_deepseek)
    if model_type in {"phi3", "phi4_multimodal"} or any("phi3" in arch or "phi4" in arch for arch in architectures):
        return _apply_patch_once(model, patch_phi)
    if model_type in {"olmo2", "olmoe"} or any("olmo" in arch for arch in architectures):
        return _apply_patch_once(model, patch_olmo)
    if model_type == "granite" or any("granite" in arch for arch in architectures):
        return _apply_patch_once(model, patch_granite)
    if model_type == "jamba" or any("jamba" in arch for arch in architectures):
        return _apply_patch_once(model, patch_jamba)
    if (model_type and "qwen" in model_type) or any("qwen" in arch for arch in architectures):
        return _apply_patch_once(model, patch_qwen)
    if (model_type and "gemma" in model_type) or any("gemma" in arch for arch in architectures):
        return _apply_patch_once(model, patch_gemma)
    if model_type == "lfm2" or any(arch.startswith("lfm2") for arch in architectures):
        return _apply_patch_once(model, patch_lfm2)

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
    Patch all matching RMSNorm layers for a single class target.
    """
    if weight_transform is None:
        weight_transform = lambda w: w

    return _patch_rmsnorm_resolved_targets(
        model,
        [
            (
                rmsnorm_class,
                model_label,
                eps_attributes,
                weight_transform,
                cast_input_to_float32,
                cast_output_to_input_dtype,
            )
        ],
    )


def _patch_rmsnorm_targets(
    model: torch.nn.Module,
    target_specs: list,
) -> torch.nn.Module:
    """
    Patch RMSNorm layers using a list of lazy class specs in one model scan.
    """
    resolved_targets = []
    for (
        module_name,
        class_name,
        model_label,
        eps_attributes,
        weight_transform,
        cast_input_to_float32,
        cast_output_to_input_dtype,
    ) in target_specs:
        rmsnorm_class = _resolve_rmsnorm_class(module_name, class_name)
        if rmsnorm_class is None:
            continue
        resolved_targets.append(
            (
                rmsnorm_class,
                model_label,
                eps_attributes,
                weight_transform,
                cast_input_to_float32,
                cast_output_to_input_dtype,
            )
        )

    return _patch_rmsnorm_resolved_targets(model, resolved_targets)


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
    return _patch_rmsnorm_targets(
        model,
        [
            (
                "transformers.models.llama.modeling_llama",
                "LlamaRMSNorm",
                "Llama",
                ("variance_epsilon", "eps"),
                lambda w: w,
                False,
                False,
            )
        ],
    )


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
    return _patch_rmsnorm_targets(
        model,
        [
            (
                "transformers.models.lfm2.modeling_lfm2",
                "Lfm2RMSNorm",
                "LFM2",
                ("variance_epsilon", "eps"),
                lambda w: w,
                False,
                False,
            )
        ],
    )


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
    return _patch_rmsnorm_targets(
        model,
        [
            (
                "transformers.models.mistral.modeling_mistral",
                "MistralRMSNorm",
                "Mistral",
                ("variance_epsilon", "eps"),
                lambda w: w,
                False,
                False,
            ),
            (
                "transformers.models.mixtral.modeling_mixtral",
                "MixtralRMSNorm",
                "Mixtral",
                ("variance_epsilon", "eps"),
                lambda w: w,
                False,
                False,
            ),
        ],
    )


def patch_deepseek(model: torch.nn.Module) -> torch.nn.Module:
    """
    Patch DeepSeek family models with BarqTrain optimizations.

    Replaces:
    - RMSNorm with fused RMSNorm kernel
    """
    return _patch_rmsnorm_targets(
        model,
        [
            (
                "transformers.models.deepseek_v2.modeling_deepseek_v2",
                "DeepseekV2RMSNorm",
                "DeepSeekV2",
                ("variance_epsilon", "eps"),
                lambda w: w,
                True,
                True,
            ),
            (
                "transformers.models.deepseek_v3.modeling_deepseek_v3",
                "DeepseekV3RMSNorm",
                "DeepSeekV3",
                ("variance_epsilon", "eps"),
                lambda w: w,
                True,
                True,
            ),
        ],
    )


def patch_phi(model: torch.nn.Module) -> torch.nn.Module:
    """
    Patch Microsoft Phi family models with BarqTrain optimizations.

    Replaces:
    - RMSNorm with fused RMSNorm kernel
    """
    return _patch_rmsnorm_targets(
        model,
        [
            (
                "transformers.models.phi3.modeling_phi3",
                "Phi3RMSNorm",
                "Phi3",
                ("variance_epsilon", "eps"),
                lambda w: w,
                True,
                True,
            ),
            (
                "transformers.models.phi4_multimodal.modeling_phi4_multimodal",
                "Phi4MultimodalRMSNorm",
                "Phi4Multimodal",
                ("variance_epsilon", "eps"),
                lambda w: w,
                True,
                True,
            ),
        ],
    )


def patch_olmo(model: torch.nn.Module) -> torch.nn.Module:
    """
    Patch OLMo family models with BarqTrain optimizations.

    Replaces:
    - RMSNorm with fused RMSNorm kernel
    """
    return _patch_rmsnorm_targets(
        model,
        [
            (
                "transformers.models.olmo2.modeling_olmo2",
                "Olmo2RMSNorm",
                "OLMo2",
                ("variance_epsilon", "eps"),
                lambda w: w,
                True,
                True,
            ),
            (
                "transformers.models.olmoe.modeling_olmoe",
                "OlmoeRMSNorm",
                "OLMoE",
                ("variance_epsilon", "eps"),
                lambda w: w,
                True,
                True,
            ),
        ],
    )


def patch_granite(model: torch.nn.Module) -> torch.nn.Module:
    """
    Patch IBM Granite models with BarqTrain optimizations.

    Replaces:
    - RMSNorm with fused RMSNorm kernel
    """
    return _patch_rmsnorm_targets(
        model,
        [
            (
                "transformers.models.granite.modeling_granite",
                "GraniteRMSNorm",
                "Granite",
                ("variance_epsilon", "eps"),
                lambda w: w,
                True,
                True,
            )
        ],
    )


def patch_jamba(model: torch.nn.Module) -> torch.nn.Module:
    """
    Patch AI21 Jamba models with BarqTrain optimizations.

    Replaces:
    - RMSNorm with fused RMSNorm kernel
    """
    return _patch_rmsnorm_targets(
        model,
        [
            (
                "transformers.models.jamba.modeling_jamba",
                "JambaRMSNorm",
                "Jamba",
                ("variance_epsilon", "eps"),
                lambda w: w,
                True,
                True,
            )
        ],
    )


def patch_llama4(model: torch.nn.Module) -> torch.nn.Module:
    """
    Patch Llama4 models with BarqTrain optimizations.

    Replaces:
    - RMSNorm with fused RMSNorm kernel
    """
    return _patch_rmsnorm_targets(
        model,
        [
            (
                "transformers.models.llama4.modeling_llama4",
                "Llama4TextRMSNorm",
                "Llama4",
                ("eps", "variance_epsilon"),
                lambda w: w,
                False,
                False,
            )
        ],
    )


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
    return _patch_rmsnorm_targets(
        model,
        [
            (
                "transformers.models.gemma.modeling_gemma",
                "GemmaRMSNorm",
                "Gemma",
                ("eps", "variance_epsilon"),
                lambda w: 1.0 + w.float(),
                True,
                True,
            ),
            (
                "transformers.models.gemma2.modeling_gemma2",
                "Gemma2RMSNorm",
                "Gemma2",
                ("eps", "variance_epsilon"),
                lambda w: 1.0 + w.float(),
                True,
                True,
            ),
            (
                "transformers.models.gemma3.modeling_gemma3",
                "Gemma3RMSNorm",
                "Gemma3",
                ("eps", "variance_epsilon"),
                lambda w: 1.0 + w.float(),
                True,
                True,
            ),
            (
                "transformers.models.recurrent_gemma.modeling_recurrent_gemma",
                "RecurrentGemmaRMSNorm",
                "RecurrentGemma",
                ("eps", "variance_epsilon"),
                lambda w: 1.0 + w.float(),
                True,
                True,
            ),
        ],
    )


def patch_qwen(model: torch.nn.Module) -> torch.nn.Module:
    """
    Patch Qwen family models with BarqTrain optimizations.

    Args:
        model: A Qwen family compatible model

    Returns:
        The patched model
    """
    return _patch_rmsnorm_targets(
        model,
        [
            (
                "transformers.models.qwen2.modeling_qwen2",
                "Qwen2RMSNorm",
                "Qwen2",
                ("variance_epsilon", "eps"),
                lambda w: w,
                True,
                True,
            ),
            (
                "transformers.models.qwen2_moe.modeling_qwen2_moe",
                "Qwen2MoeRMSNorm",
                "Qwen2Moe",
                ("variance_epsilon", "eps"),
                lambda w: w,
                True,
                True,
            ),
            (
                "transformers.models.qwen3.modeling_qwen3",
                "Qwen3RMSNorm",
                "Qwen3",
                ("variance_epsilon", "eps"),
                lambda w: w,
                True,
                True,
            ),
        ],
    )


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
