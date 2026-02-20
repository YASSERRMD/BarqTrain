"""
LoRA (Low-Rank Adaptation) utilities for BarqTrain

This module provides fused LoRA implementations for efficient
fine-tuning with reduced memory and compute overhead.
"""

from typing import Optional

import torch
import torch.nn as nn

try:
    import barqtrain_cuda as _C
except ImportError:
    _C = None


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
        if _C is not None:
            # Use CUDA kernel
            output = _C.fused_lora_forward(x, W_base, A, B, scaling)
            ctx.save_for_backward(x, W_base, A, B)
            ctx.scaling = scaling
            return output
        else:
            # Fallback to PyTorch implementation
            lora_output = x @ A.T @ B.T
            output = x @ W_base.T + lora_output * scaling
            ctx.save_for_backward(x, W_base, A, B)
            ctx.scaling = scaling
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

        # Compute gradients
        # For a production implementation, these would also be fused
        # For now, use standard PyTorch autograd

        # grad_x = grad_output @ W_base + scaling * grad_output @ B @ A
        grad_W_base_part = grad_output.T @ x
        lora_grad = (grad_output @ B.T * scaling).T @ x

        grad_x = grad_output @ W_base + scaling * (grad_output @ B.T @ A.T)
        grad_W_base = grad_output.T @ x
        grad_A = scaling * (grad_output @ B.T).T @ x
        grad_B = scaling * grad_output.T @ (x @ A.T)

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
        nn.init.kaiming_uniform_(self.base_weight, a=torch.sqrt(torch.tensor(5.0)))
        if self.base_bias is not None:
            fan_in = self.base_weight.shape[1]
            bound = 1 / torch.sqrt(torch.tensor(fan_in))
            nn.init.uniform_(self.base_bias, -bound, bound)

        # LoRA initialization (A: Kaiming, B: zeros)
        nn.init.kaiming_uniform_(self.lora_A, a=torch.sqrt(torch.tensor(5.0)))
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
            self.base_weight.add_(delta.T)

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


__all__ = [
    "FusedLoRAFunction",
    "fused_lora_linear",
    "FusedLoRALinear",
]
