"""
Model patching utilities for BarqTrain

This module provides functions to monkey-patch Hugging Face models
with BarqTrain's optimized CUDA kernels and Rust operations.
"""

import importlib
import importlib.util
import inspect
import types
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch


def _apply_patch_once(model: torch.nn.Module, patch_fn) -> torch.nn.Module:
    """
    Apply a patch function only once per model instance.
    """
    if getattr(model, "_barqtrain_model_patched", False):
        return model
    patched_model = patch_fn(model)
    refresh_rmsnorm_block_fusion_sites(patched_model)
    setattr(patched_model, "_barqtrain_model_patched", True)
    return patched_model


@lru_cache(maxsize=1)
def _preferred_attention_backend() -> Optional[str]:
    """
    Pick the fastest attention backend available in the current environment.
    """
    if torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return "sdpa"
    return None


def _configure_attention_backend(model: torch.nn.Module, model_label: str) -> Optional[str]:
    """
    Configure the model to use FlashAttention or SDPA when available.
    """
    config = getattr(model, "config", None)
    backend = _preferred_attention_backend()
    if config is None or backend is None:
        return None

    current_backend = getattr(config, "_attn_implementation", None)
    if current_backend == backend:
        return backend

    setattr(config, "_attn_implementation", backend)
    if hasattr(config, "attn_implementation"):
        setattr(config, "attn_implementation", backend)

    print(f"BarqTrain: Enabled {backend} attention backend for {model_label}")
    return backend


@dataclass(frozen=True)
class RMSNormFusionSite:
    """
    A common transformer call site where BarqTrain can apply deeper RMSNorm fusion.
    """

    pattern: str
    module_path: str
    norm_path: str
    consumer_path: str


def _named_child_module(parent: torch.nn.Module, attr_name: str) -> Optional[torch.nn.Module]:
    child = getattr(parent, attr_name, None)
    return child if isinstance(child, torch.nn.Module) else None


def _join_module_path(parent_path: str, child_name: str) -> str:
    return f"{parent_path}.{child_name}" if parent_path else child_name


def identify_rmsnorm_block_fusion_sites(model: torch.nn.Module) -> list[RMSNormFusionSite]:
    """
    Inspect a model for common pre-attention and pre-MLP RMSNorm fusion opportunities.
    """
    pre_attention_norm_names = (
        "input_layernorm",
        "pre_attention_layernorm",
        "attention_norm",
        "attn_norm",
    )
    pre_mlp_norm_names = (
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "ffn_norm",
        "mlp_norm",
    )
    attention_consumer_names = ("self_attn", "attention", "attn")
    mlp_consumer_names = ("mlp", "feed_forward", "ffn", "experts")

    sites: list[RMSNormFusionSite] = []
    seen: set[tuple[str, str, str, str]] = set()

    for module_path, module in model.named_modules():
        attention_consumer_path = None
        mlp_consumer_path = None

        for attr_name in attention_consumer_names:
            if _named_child_module(module, attr_name) is not None:
                attention_consumer_path = _join_module_path(module_path, attr_name)
                break

        for attr_name in mlp_consumer_names:
            if _named_child_module(module, attr_name) is not None:
                mlp_consumer_path = _join_module_path(module_path, attr_name)
                break

        for attr_name in pre_attention_norm_names:
            if _named_child_module(module, attr_name) is None:
                continue
            norm_path = _join_module_path(module_path, attr_name)
            for pattern, consumer_path in (
                ("residual_add_rmsnorm", norm_path),
                ("rmsnorm_attention_projection", attention_consumer_path),
            ):
                if consumer_path is None:
                    continue
                key = (pattern, module_path, norm_path, consumer_path)
                if key in seen:
                    continue
                seen.add(key)
                sites.append(
                    RMSNormFusionSite(
                        pattern=pattern,
                        module_path=module_path or "<root>",
                        norm_path=norm_path,
                        consumer_path=consumer_path,
                    )
                )

        for attr_name in pre_mlp_norm_names:
            if _named_child_module(module, attr_name) is None:
                continue
            norm_path = _join_module_path(module_path, attr_name)
            for pattern, consumer_path in (
                ("residual_add_rmsnorm", norm_path),
                ("rmsnorm_mlp_projection", mlp_consumer_path),
            ):
                if consumer_path is None:
                    continue
                key = (pattern, module_path, norm_path, consumer_path)
                if key in seen:
                    continue
                seen.add(key)
                sites.append(
                    RMSNormFusionSite(
                        pattern=pattern,
                        module_path=module_path or "<root>",
                        norm_path=norm_path,
                        consumer_path=consumer_path,
                    )
                )

    return sites


def refresh_rmsnorm_block_fusion_sites(model: torch.nn.Module) -> tuple[RMSNormFusionSite, ...]:
    """
    Re-scan a model and store the currently visible RMSNorm block-fusion call sites.
    """
    sites = tuple(identify_rmsnorm_block_fusion_sites(model))
    setattr(model, "_barqtrain_rmsnorm_block_fusion_sites", sites)
    if sites:
        pattern_counts: dict[str, int] = {}
        for site in sites:
            pattern_counts[site.pattern] = pattern_counts.get(site.pattern, 0) + 1
        summary = ", ".join(f"{pattern}={count}" for pattern, count in sorted(pattern_counts.items()))
        print(f"BarqTrain: Identified RMSNorm block-fusion sites ({summary})")
    return sites


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


def _patch_causal_lm_chunked_loss(
    model: torch.nn.Module,
    model_label: str,
) -> torch.nn.Module:
    """
    Patch compatible decoder-only CausalLM models to use chunked loss during training.
    """
    if getattr(model, "_barqtrain_chunked_loss_patched", False):
        return model

    if not hasattr(model, "model") or not hasattr(model, "lm_head"):
        return model

    try:
        from transformers.modeling_outputs import CausalLMOutputWithPast
    except ImportError:
        return model

    original_forward = model.forward
    try:
        original_forward_signature = inspect.signature(original_forward)
        original_forward_parameters = set(original_forward_signature.parameters)
    except (TypeError, ValueError):
        original_forward_parameters = set()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        logits_to_keep=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        use_return_dict = (
            return_dict if return_dict is not None else getattr(self.config, "use_return_dict", True)
        )
        requested_logits_to_keep = (
            logits_to_keep if logits_to_keep is not None else num_logits_to_keep
        )
        use_chunked_loss = (
            labels is not None
            and self.training
            and use_return_dict
            and getattr(self, "_barqtrain_chunked_loss_enabled", True)
            and past_key_values is None
        )
        use_last_token_projection = (
            labels is None
            and use_return_dict
            and getattr(self, "_barqtrain_last_token_projection_enabled", True)
            and requested_logits_to_keep is not None
            and int(requested_logits_to_keep) == 1
            and not bool(kwargs.get("output_logits", False))
        )

        setattr(self, "_barqtrain_last_forward_used_fused_lm_head_loss", use_chunked_loss)
        setattr(self, "_barqtrain_last_forward_used_last_token_projection", use_last_token_projection)

        if not use_chunked_loss and not use_last_token_projection:
            forward_kwargs = dict(kwargs)
            if "logits_to_keep" in original_forward_parameters and logits_to_keep is not None:
                forward_kwargs["logits_to_keep"] = logits_to_keep
            if "num_logits_to_keep" in original_forward_parameters and num_logits_to_keep is not None:
                forward_kwargs["num_logits_to_keep"] = num_logits_to_keep
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **forward_kwargs,
            )

        from barqtrain.ops import (
            fused_lm_head_cross_entropy_loss,
            fused_lm_head_projection,
        )

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        hidden_states = getattr(model_outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = model_outputs[0]

        loss = None
        logits = None

        if use_chunked_loss:
            loss = fused_lm_head_cross_entropy_loss(
                hidden_states,
                self.lm_head.weight,
                labels,
                shift=True,
            )

        if use_last_token_projection:
            logits = fused_lm_head_projection(
                hidden_states,
                self.lm_head.weight,
                last_token_only=True,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=getattr(model_outputs, "past_key_values", None),
            hidden_states=getattr(model_outputs, "hidden_states", None),
            attentions=getattr(model_outputs, "attentions", None),
        )

    model.forward = types.MethodType(forward, model)
    setattr(model, "_barqtrain_chunked_loss_patched", True)
    print(f"BarqTrain: Enabled chunked loss patch for {model_label}")
    return model


def _patch_generate_with_paged_kv(
    model: torch.nn.Module,
    model_label: str,
) -> torch.nn.Module:
    """
    Wrap model.generate() to inject decode-time cache and logits optimizations.
    """
    if getattr(model, "_barqtrain_generate_paged_kv_patched", False):
        return model
    if not hasattr(model, "generate"):
        return model

    from barqtrain.kv_cache import maybe_prepare_kv_generate_kwargs, paged_kv_supported_for_model
    from barqtrain.memory import (
        capture_cuda_peak_bytes,
        detailed_profiling_enabled,
        maybe_prepare_last_token_logits_generate_kwargs,
        record_inference_peak_bytes,
        reset_native_memory_tracking,
        track_decode_temp_memory,
        track_kv_cache_memory,
        track_resident_model_memory,
    )

    original_generate = model.generate

    def generate(self, *args, **kwargs):
        resident_model_bytes = 0
        if detailed_profiling_enabled():
            reset_native_memory_tracking(reset_peak=False)
            resident_model_bytes = track_resident_model_memory(self)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        updated_kwargs, used_native_kv = maybe_prepare_kv_generate_kwargs(self, args, kwargs)
        updated_kwargs, used_last_token_logits = maybe_prepare_last_token_logits_generate_kwargs(
            self,
            args,
            updated_kwargs,
        )
        cache = updated_kwargs.get("past_key_values")
        kv_cache_bytes = 0
        if detailed_profiling_enabled() and cache is not None:
            kv_cache_bytes = track_kv_cache_memory(cache)

        cache_layout = getattr(cache, "barqtrain_cache_layout", None)
        setattr(self, "_barqtrain_last_generate_used_paged_kv", used_native_kv and cache_layout == "paged")
        setattr(self, "_barqtrain_last_generate_used_contiguous_kv", used_native_kv and cache_layout == "contiguous")
        setattr(self, "_barqtrain_last_generate_used_quantized_kv", used_native_kv and cache_layout == "paged_quantized")
        setattr(self, "_barqtrain_last_generate_kv_cache_layout", cache_layout)
        setattr(self, "_barqtrain_last_generate_last_token_logits_only", used_last_token_logits)
        setattr(self, "_barqtrain_last_generate_cache", cache)

        result = original_generate(*args, **updated_kwargs)

        if detailed_profiling_enabled():
            inference_peak_bytes = capture_cuda_peak_bytes()
            record_inference_peak_bytes(inference_peak_bytes)
            if cache is not None:
                kv_cache_bytes = track_kv_cache_memory(cache)
            decode_temp_bytes = track_decode_temp_memory(
                resident_model_bytes=resident_model_bytes,
                kv_cache_bytes=kv_cache_bytes,
                inference_peak_bytes=inference_peak_bytes,
            )
            setattr(self, "_barqtrain_last_generate_resident_model_bytes", resident_model_bytes)
            setattr(self, "_barqtrain_last_generate_kv_cache_bytes", kv_cache_bytes)
            setattr(self, "_barqtrain_last_generate_decode_temp_bytes", decode_temp_bytes)
            setattr(self, "_barqtrain_last_generate_inference_peak_bytes", inference_peak_bytes)

        return result

    model.generate = types.MethodType(generate, model)
    setattr(model, "_barqtrain_generate_paged_kv_patched", True)
    setattr(model, "_barqtrain_paged_kv_supported", paged_kv_supported_for_model(model))
    setattr(model, "_barqtrain_supported_kv_cache_layouts", ("paged", "contiguous", "paged_quantized"))
    if getattr(model, "_barqtrain_paged_kv_supported", False):
        print(f"BarqTrain: Enabled native KV-cache injection for {model_label}")
    print(f"BarqTrain: Enabled last-token decode logits specialization for {model_label}")
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


def patch_inference(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply BarqTrain's inference-oriented patches without forcing RMSNorm rewrites.

    This is intended for decode benchmarking and low-memory generation where the
    main BarqTrain value comes from cache handling and backend selection.
    """
    model_config = getattr(model, "config", None)
    model_type = getattr(model_config, "model_type", None) or "model"
    _configure_attention_backend(model, model_type)
    model = _patch_causal_lm_chunked_loss(model, model_type)
    model = _patch_generate_with_paged_kv(model, model_type)
    refresh_rmsnorm_block_fusion_sites(model)
    return model


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
    _configure_attention_backend(model, "Llama")
    model = _patch_causal_lm_chunked_loss(model, "Llama")
    model = _patch_generate_with_paged_kv(model, "Llama")
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
    _configure_attention_backend(model, "LFM2")
    model = _patch_causal_lm_chunked_loss(model, "LFM2")
    model = _patch_generate_with_paged_kv(model, "LFM2")
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
    _configure_attention_backend(model, "Mistral/Mixtral")
    model = _patch_causal_lm_chunked_loss(model, "Mistral/Mixtral")
    model = _patch_generate_with_paged_kv(model, "Mistral/Mixtral")
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
    _configure_attention_backend(model, "DeepSeek")
    model = _patch_causal_lm_chunked_loss(model, "DeepSeek")
    model = _patch_generate_with_paged_kv(model, "DeepSeek")
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
    _configure_attention_backend(model, "Phi")
    model = _patch_causal_lm_chunked_loss(model, "Phi")
    model = _patch_generate_with_paged_kv(model, "Phi")
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
    _configure_attention_backend(model, "OLMo")
    model = _patch_causal_lm_chunked_loss(model, "OLMo")
    model = _patch_generate_with_paged_kv(model, "OLMo")
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
    _configure_attention_backend(model, "Granite")
    model = _patch_causal_lm_chunked_loss(model, "Granite")
    model = _patch_generate_with_paged_kv(model, "Granite")
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
    _configure_attention_backend(model, "Jamba")
    model = _patch_causal_lm_chunked_loss(model, "Jamba")
    model = _patch_generate_with_paged_kv(model, "Jamba")
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
    _configure_attention_backend(model, "Llama4")
    model = _patch_causal_lm_chunked_loss(model, "Llama4")
    model = _patch_generate_with_paged_kv(model, "Llama4")
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
    _configure_attention_backend(model, "Gemma")
    model = _patch_causal_lm_chunked_loss(model, "Gemma")
    model = _patch_generate_with_paged_kv(model, "Gemma")
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
    _configure_attention_backend(model, "Qwen")
    model = _patch_causal_lm_chunked_loss(model, "Qwen")
    model = _patch_generate_with_paged_kv(model, "Qwen")
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
    return _patch_generate_with_paged_kv(model, "generic model")


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
