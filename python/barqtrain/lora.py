"""LoRA (Low-Rank Adaptation) utilities for BarqTrain."""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from barqtrain._ffi import load_cuda_backend

_CUDA_FALLBACK_WARNED = False


def _get_cuda_backend():
    return load_cuda_backend()


def _warn_cuda_fallback_once() -> None:
    global _CUDA_FALLBACK_WARNED
    if _CUDA_FALLBACK_WARNED:
        return
    warnings.warn(
        "BarqTrain CUDA backend unavailable. Falling back to PyTorch LoRA implementation. "
        "Install with: pip install -e ."
    )
    _CUDA_FALLBACK_WARNED = True


def _reshape_lora_input(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    original_shape = tuple(x.shape)
    if x.dim() <= 2:
        return x, original_shape
    return x.reshape(-1, x.size(-1)), original_shape


def _cast_lora_weight(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return weight.to(device=x.device, dtype=x.dtype)


class FusedLoRAFunction(torch.autograd.Function):
    """
    Fused LoRA forward pass combining base and adapter weights.

    Computes: output = x @ W_base + scaling * (x @ A @ B)

    This fusion reduces memory reads/writes by computing both
    the base weight multiplication and LoRA adapter in a single
    kernel launch.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        W_base: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        scaling: float,
    ) -> torch.Tensor:
        """
        Forward pass for fused LoRA computation.

        Args:
            x: Input tensor [batch_size, in_features]
            W_base: Base weight [out_features, in_features]
            A: LoRA adapter A matrix [rank, in_features]
            B: LoRA adapter B matrix [out_features, rank]
            scaling: LoRA scaling factor (typically alpha / rank)

        Returns:
            Output tensor [batch_size, out_features]
        """
        x_2d, original_shape = _reshape_lora_input(x)
        W_base_cast = _cast_lora_weight(W_base, x_2d)
        A_cast = _cast_lora_weight(A, x_2d)
        B_cast = _cast_lora_weight(B, x_2d)

        cuda_backend = _get_cuda_backend()
        if cuda_backend is not None and x_2d.is_cuda:
            # Use CUDA kernel
            output = cuda_backend.fused_lora_forward(
                x_2d.contiguous(),
                W_base_cast.contiguous(),
                A_cast.contiguous(),
                B_cast.contiguous(),
                scaling,
            )
        else:
            if x_2d.is_cuda:
                _warn_cuda_fallback_once()
            # Fallback to PyTorch implementation
            lora_output = x_2d @ A_cast.T @ B_cast.T
            output = x_2d @ W_base_cast.T + lora_output * scaling

        ctx.save_for_backward(x_2d, W_base_cast, A_cast, B_cast)
        ctx.scaling = scaling
        ctx.original_shape = original_shape
        if len(original_shape) > 2:
            output = output.reshape(*original_shape[:-1], W_base.shape[0])
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """
        Backward pass for fused LoRA.

        Args:
            grad_output: Gradient from next layer

        Returns:
            Tuple of gradients (grad_x, grad_W_base, grad_A, grad_B, None)
        """
        x, W_base, A, B = ctx.saved_tensors
        scaling = ctx.scaling

        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1]) if grad_output.dim() > 2 else grad_output
        grad_x = grad_output_2d @ W_base + scaling * (grad_output_2d @ B @ A)
        grad_W_base = grad_output_2d.T @ x
        grad_A = scaling * (grad_output_2d @ B).T @ x
        grad_B = scaling * grad_output_2d.T @ (x @ A.T)

        if len(ctx.original_shape) > 2:
            grad_x = grad_x.reshape(*ctx.original_shape)

        return grad_x, grad_W_base, grad_A, grad_B, None


def fused_lora_linear(
    x: torch.Tensor,
    W_base: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scaling: float = 1.0,
) -> torch.Tensor:
    """
    Apply fused LoRA linear transformation.

    This function combines the base weight multiplication with
    the LoRA adapter computation for improved efficiency.

    Args:
        x: Input tensor [batch_size, in_features]
        W_base: Base weight [out_features, in_features]
        A: LoRA A matrix [rank, in_features]
        B: LoRA B matrix [out_features, rank]
        scaling: LoRA scaling factor

    Returns:
        Output tensor [batch_size, out_features]

    Example:
        >>> import torch
        >>> from barqtrain.lora import fused_lora_linear
        >>>
        >>> x = torch.randn(32, 768, device='cuda')
        >>> W = torch.randn(768, 768, device='cuda')
        >>> A = torch.randn(8, 768, device='cuda')  # rank=8
        >>> B = torch.randn(768, 8, device='cuda')
        >>>
        >>> output = fused_lora_linear(x, W, A, B, scaling=0.01)
    """
    return FusedLoRAFunction.apply(x, W_base, A, B, scaling)


class FusedLoRALinear(nn.Module):
    """
    A linear layer with fused LoRA adapter.

    This module replaces a standard nn.Linear layer with a version
    that has an efficient LoRA adapter fused into the computation.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        rank: LoRA rank (typically much smaller than in/out features)
        alpha: LoRA alpha parameter (scaling = alpha / rank)
        dropout: Dropout probability for LoRA layers
        bias: Whether to include bias in the base layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank if rank > 0 else 1.0

        # Base weight (frozen during LoRA fine-tuning)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.base_bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # LoRA adapter weights
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        # Dropout for LoRA
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters following LoRA best practices."""
        # Base weight initialization
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5.0))
        if self.base_bias is not None:
            fan_in = self.base_weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.base_bias, -bound, bound)

        # LoRA initialization (A: Kaiming, B: zeros)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5.0))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fused LoRA computation.

        Args:
            x: Input tensor

        Returns:
            Output tensor with base + LoRA contribution
        """
        # Apply dropout to input before LoRA
        x_dropped = self.lora_dropout(x)

        # Fused computation: x @ W_base + scaling * (x @ A @ B)
        output = fused_lora_linear(
            x_dropped, self.base_weight, self.lora_A, self.lora_B, self.scaling
        )

        # Add bias if present
        if self.base_bias is not None:
            output = output + self.base_bias

        return output

    def merge_weights(self) -> None:
        """
        Merge LoRA weights into base weights.

        After merging, the LoRA adapters can be removed for inference.
        This modifies the base weights in-place.
        """
        with torch.no_grad():
            # Compute delta = scaling * B @ A
            delta = self.scaling * (self.lora_B @ self.lora_A)
            # Add to base weight
            self.base_weight.add_(delta)

    @classmethod
    def from_linear(
        cls,
        linear_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> "FusedLoRALinear":
        """
        Convert a standard nn.Linear layer to FusedLoRALinear.

        Args:
            linear_layer: Original linear layer
            rank: LoRA rank
            alpha: LoRA alpha
            dropout: LoRA dropout

        Returns:
            FusedLoRALinear layer with weights from original layer
        """
        lora_layer = cls(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=linear_layer.bias is not None,
        )

        # Copy weights from original layer
        with torch.no_grad():
            lora_layer.base_weight.copy_(linear_layer.weight.data)
            if linear_layer.bias is not None:
                lora_layer.base_bias.copy_(linear_layer.bias.data)

        return lora_layer


def patch_lora_modules(
    model: nn.Module,
    *,
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    freeze_base: bool = True,
) -> nn.Module:
    """
    Replace matching Linear modules with FusedLoRALinear modules in-place.
    """
    replaced_modules: list[str] = []
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(module_name == target or module_name.endswith(f".{target}") for target in target_modules):
            continue

        parent_path, _, child_name = module_name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        fused_module = FusedLoRALinear.from_linear(
            module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        if freeze_base:
            fused_module.base_weight.requires_grad_(False)
            if fused_module.base_bias is not None:
                fused_module.base_bias.requires_grad_(False)
        setattr(parent, child_name, fused_module)
        replaced_modules.append(module_name)

    setattr(model, "_barqtrain_fused_lora_patched", bool(replaced_modules))
    setattr(model, "_barqtrain_fused_lora_target_modules", tuple(target_modules))
    setattr(model, "_barqtrain_fused_lora_modules", tuple(replaced_modules))
    return model


__all__ = [
    "FusedLoRAFunction",
    "fused_lora_linear",
    "FusedLoRALinear",
    "patch_lora_modules",
]
