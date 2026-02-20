"""
Custom PyTorch operations for BarqTrain

This module provides PyTorch autograd wrappers for BarqTrain's
CUDA kernels, enabling seamless integration with the PyTorch ecosystem.
"""

import torch
import torch.nn.functional as F

try:
    import barqtrain_cuda as _C
except ImportError:
    _C = None
    import warnings

    warnings.warn(
        "BarqTrain CUDA extension not available. "
        "Falling back to PyTorch implementations. "
        "Build with: python setup.py install"
    )


class FusedRMSNormFunction(torch.autograd.Function):
    """
    Fused RMSNorm with forward and backward passes.

    This combines the RMS normalization and weight multiplication
    into a single kernel launch for better performance.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Forward pass for fused RMSNorm.

        Args:
            x: Input tensor of shape (batch_size, hidden_size)
            weight: Weight tensor of shape (hidden_size,)
            eps: Small constant for numerical stability

        Returns:
            Normalized output tensor
        """
        if _C is not None:
            # Use CUDA kernel
            rms_cache = torch.empty(x.size(0), dtype=torch.float32, device=x.device)
            output = _C.fused_rmsnorm(x, weight, eps)
            ctx.save_for_backward(x, weight, rms_cache)
            ctx.eps = eps
            return output
        else:
            # Fallback to PyTorch implementation
            variance = x.pow(2).mean(dim=-1, keepdim=True) + eps
            x = x * torch.rsqrt(variance)
            ctx.save_for_backward(x, weight)
            ctx.eps = eps
            return x * weight

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """
        Backward pass for fused RMSNorm.

        Args:
            grad_output: Gradient from the next layer

        Returns:
            Tuple of (grad_input, grad_weight, None)
        """
        if _C is not None:
            # Use CUDA kernel
            x, weight, rms_cache = ctx.saved_tensors
            grad_input = _C.fused_rmsnorm_backward(grad_output, x, weight, rms_cache)
            # Compute grad_weight (simplified - should use atomic adds in kernel)
            y = x / rms_cache.unsqueeze(1)
            grad_weight = (grad_output * y).sum(dim=0)
            return grad_input, grad_weight, None
        else:
            # Fallback to PyTorch implementation
            x, weight = ctx.saved_tensors
            eps = ctx.eps

            # Recompute forward pass values
            variance = x.pow(2).mean(dim=-1, keepdim=True) + eps
            x_normalized = x * torch.rsqrt(variance)
            y = x_normalized * weight

            # Compute gradients
            # This is a simplified backward pass
            grad_weight = (grad_output * x_normalized).sum(dim=0)

            # grad_x computation
            mean = (x * grad_output * weight).sum(dim=-1, keepdim=True) / x.size(-1)
            grad_x = (grad_output * weight - mean * x_normalized) / torch.sqrt(variance)

            return grad_x, grad_weight, None


def fused_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Apply fused RMSNorm to input tensor.

    This function provides a convenient interface to the fused RMSNorm
    operation, combining forward and backward passes.

    Args:
        x: Input tensor of shape (batch_size, seq_len, hidden_size) or (batch_size, hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Small constant for numerical stability (default: 1e-6)

    Returns:
        Normalized tensor with same shape as input

    Example:
        >>> import torch
        >>> from barqtrain.ops import fused_rms_norm
        >>>
        >>> x = torch.randn(2, 128, 768, device='cuda')
        >>> weight = torch.ones(768, device='cuda')
        >>> output = fused_rms_norm(x, weight)
    """
    # Handle 3D input (batch, seq, hidden)
    if x.dim() == 3:
        original_shape = x.shape
        x = x.view(-1, x.size(-1))  # (batch * seq, hidden)
        output = FusedRMSNormFunction.apply(x, weight, eps)
        return output.view(original_shape)
    else:
        return FusedRMSNormFunction.apply(x, weight, eps)


class FusedRMSNorm(torch.nn.Module):
    """
    Fused RMSNorm layer for use in neural networks.

    This is a drop-in replacement for torch.nn.RMSNorm with
    improved performance due to kernel fusion.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Initialize the fused RMSNorm layer.

        Args:
            hidden_size: Size of the hidden dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fused RMSNorm to input.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        return fused_rms_norm(x, self.weight, self.eps)


# Additional ops will be added here as they are implemented
__all__ = ["FusedRMSNormFunction", "fused_rms_norm", "FusedRMSNorm"]
