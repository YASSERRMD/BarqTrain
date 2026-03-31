"""Activation checkpointing helpers for patched BarqTrain training paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActivationCheckpointPreset:
    name: str
    checkpoint_attention: bool
    checkpoint_mlp: bool


_PRESETS = {
    "max_throughput": ActivationCheckpointPreset(
        name="max_throughput",
        checkpoint_attention=False,
        checkpoint_mlp=False,
    ),
    "balanced": ActivationCheckpointPreset(
        name="balanced",
        checkpoint_attention=True,
        checkpoint_mlp=False,
    ),
    "max_memory_saving": ActivationCheckpointPreset(
        name="max_memory_saving",
        checkpoint_attention=True,
        checkpoint_mlp=True,
    ),
}


def activation_checkpoint_presets() -> dict[str, ActivationCheckpointPreset]:
    """Return the supported activation-checkpointing presets."""
    return dict(_PRESETS)


def get_activation_checkpoint_preset(name: str) -> ActivationCheckpointPreset:
    """Resolve a named activation-checkpointing preset."""
    preset_name = name.lower()
    if preset_name not in _PRESETS:
        raise ValueError(f"Unsupported activation checkpoint preset: {name}")
    return _PRESETS[preset_name]


def _module_matches_attention(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("attn", "attention"))


def _module_matches_mlp(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("mlp", "ffn", "feed_forward"))


def _wrap_module_with_checkpoint(module: torch.nn.Module) -> None:
    if getattr(module, "_barqtrain_checkpoint_wrapped", False):
        return

    original_forward = module.forward

    def forward(self, *args, **kwargs):
        if (
            not getattr(self, "_barqtrain_checkpoint_enabled", True)
            or not self.training
            or not torch.is_grad_enabled()
            or kwargs.get("use_cache") is True
            or any(arg is not None and not isinstance(arg, torch.Tensor) for arg in args)
        ):
            return original_forward(*args, **kwargs)

        def run_function(*tensor_args):
            return original_forward(*tensor_args, **kwargs)

        return torch.utils.checkpoint.checkpoint(
            run_function,
            *args,
            use_reentrant=False,
        )

    module._barqtrain_original_forward = original_forward
    module.forward = forward.__get__(module, module.__class__)
    module._barqtrain_checkpoint_wrapped = True
    module._barqtrain_checkpoint_enabled = True


def reset_activation_checkpointing(model: torch.nn.Module) -> torch.nn.Module:
    """Restore original forwards for any BarqTrain checkpoint-wrapped modules."""
    for module in model.modules():
        original_forward = getattr(module, "_barqtrain_original_forward", None)
        if original_forward is not None:
            module.forward = original_forward
            delattr(module, "_barqtrain_original_forward")
        if hasattr(module, "_barqtrain_checkpoint_wrapped"):
            delattr(module, "_barqtrain_checkpoint_wrapped")
        if hasattr(module, "_barqtrain_checkpoint_enabled"):
            delattr(module, "_barqtrain_checkpoint_enabled")
    setattr(model, "_barqtrain_activation_checkpoint_preset", "max_throughput")
    return model


def apply_activation_checkpointing(
    model: torch.nn.Module,
    preset: str = "balanced",
) -> torch.nn.Module:
    """
    Apply a named activation-checkpointing preset to a patched training model.

    The presets target common attention/MLP hot paths while keeping a padded
    fallback path when checkpointing is disabled.
    """
    resolved_preset = get_activation_checkpoint_preset(preset)
    reset_activation_checkpointing(model)

    if resolved_preset.name == "max_throughput":
        setattr(model, "_barqtrain_activation_checkpoint_preset", resolved_preset.name)
        return model

    wrapped = 0
    for name, module in model.named_modules():
        if not name:
            continue
        should_wrap = (
            resolved_preset.checkpoint_attention and _module_matches_attention(name)
        ) or (
            resolved_preset.checkpoint_mlp and _module_matches_mlp(name)
        )
        if not should_wrap:
            continue
        _wrap_module_with_checkpoint(module)
        wrapped += 1

    setattr(model, "_barqtrain_activation_checkpoint_preset", resolved_preset.name)
    setattr(model, "_barqtrain_activation_checkpoint_wrapped_modules", wrapped)
    return model


__all__ = [
    "ActivationCheckpointPreset",
    "activation_checkpoint_presets",
    "apply_activation_checkpointing",
    "get_activation_checkpoint_preset",
    "reset_activation_checkpointing",
]
