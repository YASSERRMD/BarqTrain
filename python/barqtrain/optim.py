"""
Optimizer helpers for memory-efficient BarqTrain training.
"""

from __future__ import annotations

import math
import warnings
from typing import Iterable

import torch


def _state_dtype_for_param(param: torch.nn.Parameter, state_mode: str) -> torch.dtype:
    if state_mode == "compact":
        if param.dtype == torch.bfloat16:
            return torch.bfloat16
        if param.dtype in {torch.float16, torch.float32}:
            return torch.float16
    return torch.float32


class BarqTrainAdamW(torch.optim.Optimizer):
    """
    Native AdamW-compatible optimizer with explicit optimizer-state layouts.

    Supported state modes:
    - `full`: fp32 moment buffers
    - `compact`: fp16/bf16 moment buffers when possible
    - `paged`: fp32 moment buffers split into fixed-size pages
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-5,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        state_mode: str = "full",
        page_size: int = 4096,
    ):
        if lr < 0.0:
            raise ValueError("lr must be >= 0")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas must be in [0, 1)")
        if page_size <= 0:
            raise ValueError("page_size must be > 0")
        if state_mode not in {"full", "compact", "paged"}:
            raise ValueError(f"Unsupported optimizer state_mode: {state_mode}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.state_mode = state_mode
        self.page_size = int(page_size)
        self.barqtrain_state_mode = state_mode

    def _init_state(self, param: torch.nn.Parameter, state: dict) -> None:
        state["step"] = 0
        state_dtype = _state_dtype_for_param(param, self.state_mode)
        flat_numel = param.numel()
        if self.state_mode == "paged":
            exp_avg_pages = []
            exp_avg_sq_pages = []
            remaining = flat_numel
            while remaining > 0:
                page_len = min(remaining, self.page_size)
                exp_avg_pages.append(torch.zeros(page_len, device=param.device, dtype=state_dtype))
                exp_avg_sq_pages.append(torch.zeros(page_len, device=param.device, dtype=state_dtype))
                remaining -= page_len
            state["exp_avg_pages"] = exp_avg_pages
            state["exp_avg_sq_pages"] = exp_avg_sq_pages
        else:
            state["exp_avg"] = torch.zeros_like(param, dtype=state_dtype)
            state["exp_avg_sq"] = torch.zeros_like(param, dtype=state_dtype)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D401
        """Perform a single AdamW update step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("BarqTrainAdamW does not support sparse gradients")

                grad = param.grad.detach()
                state = self.state[param]
                if len(state) == 0:
                    self._init_state(param, state)

                state["step"] += 1
                step = state["step"]
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)

                if self.state_mode == "paged":
                    flat_param = param.view(-1)
                    flat_grad = grad.view(-1).float()
                    exp_avg_pages = state["exp_avg_pages"]
                    exp_avg_sq_pages = state["exp_avg_sq_pages"]
                    offset = 0
                    for exp_avg_page, exp_avg_sq_page in zip(exp_avg_pages, exp_avg_sq_pages):
                        page_len = exp_avg_page.numel()
                        grad_page = flat_grad[offset : offset + page_len].to(exp_avg_page.dtype)
                        exp_avg_page.mul_(beta1).add_(grad_page, alpha=1.0 - beta1)
                        exp_avg_sq_page.mul_(beta2).addcmul_(
                            grad_page,
                            grad_page,
                            value=1.0 - beta2,
                        )
                        denom = exp_avg_sq_page.float().sqrt().add_(eps)
                        update = exp_avg_page.float() / denom
                        flat_param[offset : offset + page_len].add_(
                            update.to(flat_param.dtype),
                            alpha=-step_size,
                        )
                        offset += page_len
                    continue

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad_state = grad.to(exp_avg.dtype)
                exp_avg.mul_(beta1).add_(grad_state, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_state, grad_state, value=1.0 - beta2)
                denom = exp_avg_sq.float().sqrt().add_(eps)
                update = exp_avg.float() / denom
                param.add_(update.to(param.dtype), alpha=-step_size)

        return loss


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """
    Estimate the resident bytes used by optimizer state tensors.
    """
    total_bytes = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total_bytes += int(value.numel() * value.element_size())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, torch.Tensor):
                        total_bytes += int(item.numel() * item.element_size())
    return total_bytes


def create_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    lr: float = 1e-5,
    weight_decay: float = 0.0,
    optimizer_name: str = "adamw",
    **kwargs,
):
    """
    Create a training optimizer with optional native and paged variants.

    Supported names:
    - `adamw`
    - `barqtrain_adamw`
    - `barqtrain_adamw_compact`
    - `barqtrain_adamw_paged`
    - `paged_adamw_32bit`
    - `paged_adamw_8bit`
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, **kwargs)

    native_modes = {
        "barqtrain_adamw": "full",
        "barqtrain_adamw_compact": "compact",
        "barqtrain_adamw_paged": "paged",
    }
    if optimizer_name in native_modes:
        return BarqTrainAdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            state_mode=native_modes[optimizer_name],
            **kwargs,
        )

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


__all__ = ["BarqTrainAdamW", "create_optimizer", "optimizer_state_bytes"]
