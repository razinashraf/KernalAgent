"""Bias add + GELU + residual add, fused.

This is the tail end of a transformer MLP block. Eager PyTorch runs three kernels and
writes two intermediate tensors to memory; a fused kernel reads x, bias, residual once
and writes y once.
"""

import torch
import torch.nn.functional as F

NAME = "bias_gelu_residual"
CATEGORY = "fused"
DESCRIPTION = (
    "Fused y = gelu(x + bias) + residual, where gelu uses the tanh approximation "
    "(0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))). "
    "x and residual have shape (M, N); bias has shape (N,) and is broadcast over rows. "
    "All inputs are contiguous. Return a new tensor of shape (M, N) with x's dtype."
)
DTYPES = [torch.float16, torch.float32]
TOLERANCES = {
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
    torch.float16: {"atol": 1e-2, "rtol": 1e-2},
}
SHAPES = {
    "small": {"M": 1024, "N": 1024},
    "medium": {"M": 4096, "N": 4096},
    "large": {"M": 8192, "N": 8192},
}
EDGE_SHAPES = [
    {"M": 1, "N": 1},
    {"M": 1000, "N": 1537},
    {"M": 5, "N": 50001},
]


def random_shape(rng):
    return {"M": rng.randint(1, 2048), "N": rng.randint(1, 4096)}


def make_inputs(shape, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    M, N = shape["M"], shape["N"]
    x = torch.randn(M, N, device="cuda", dtype=dtype, generator=g)
    bias = torch.randn(N, device="cuda", dtype=dtype, generator=g)
    residual = torch.randn(M, N, device="cuda", dtype=dtype, generator=g)
    return [x, bias, residual]


def reference(x, bias, residual):
    return F.gelu(x + bias, approximate="tanh") + residual
