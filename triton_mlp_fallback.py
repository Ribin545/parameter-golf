"""
Pure-PyTorch fallback for triton_mlp.fused_relu2 (x @ W + LeakyReLU(0.1)^2).
Used on Windows where Triton is not available.
"""

import torch
from torch import Tensor


def fused_relu2(x: Tensor, w: Tensor) -> Tensor:
    """x @ w followed by LeakyReLU(0.1)^2 activation — pure PyTorch."""
    y = x @ w
    leaky = torch.where(y > 0, y, 0.1 * y)
    return leaky * leaky