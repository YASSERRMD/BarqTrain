"""
Optimizer helpers for memory-efficient BarqTrain training.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import torch


def create_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    lr: float = 1e-5,
    weight_decay: float = 0.0,
    optimizer_name: str = "adamw",
    **kwargs,
):
    """
    Create a training optimizer with optional paged bitsandbytes variants.

    Supported names:
    - `adamw`
    - `paged_adamw_32bit`
    - `paged_adamw_8bit`
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, **kwargs)

    if optimizer_name not in {"paged_adamw_32bit", "paged_adamw_8bit"}:
        raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")

    try:
        import bitsandbytes as bnb
    except ImportError:
        warnings.warn(
            "bitsandbytes is not installed. Falling back to torch.optim.AdamW "
            f"instead of {optimizer_name}."
        )
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, **kwargs)

    if optimizer_name == "paged_adamw_32bit":
        optimizer_cls = bnb.optim.PagedAdamW32bit
    else:
        optimizer_cls = bnb.optim.PagedAdamW8bit

    return optimizer_cls(parameters, lr=lr, weight_decay=weight_decay, **kwargs)


__all__ = ["create_optimizer"]
