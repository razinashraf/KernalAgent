"""Residual add followed by LayerNorm, fused — the "Add & Norm" of a transformer layer.

A reduction fused with an elementwise op. Eager writes (x + residual) to memory and then
reads it back for the LayerNorm; the fused kernel keeps the sum in registers.
"""

import torch
import torch.nn.functional as F

NAME = "add_layernorm"
CATEGORY = "fused"
EPS = 1e-5
DESCRIPTION = (
    "Fused residual add + LayerNorm: h = x + residual; "
    "y = (h - mean(h)) / sqrt(var(h) + eps) * weight + bias over the last dimension, "
    f"biased variance, eps = {EPS}. x and residual have shape (M, N); weight and bias have "
    "shape (N,). All inputs are contiguous. Return only y: a new (M, N) tensor with x's dtype. "
    "Accumulate statistics in float32."
)
DTYPES = [torch.float16, torch.float32]
TOLERANCES = {
    torch.float32: {"atol": 1e-4, "rtol": 1e-4},
    torch.float16: {"atol": 1e-2, "rtol": 1e-2},
}
SHAPES = {
    "small": {"M": 1024, "N": 1024},
    "medium": {"M": 4096, "N": 4096},
    "large": {"M": 16384, "N": 8192},
}
EDGE_SHAPES = [
    {"M": 1, "N": 1},
    {"M": 1000, "N": 1537},
    {"M": 33, "N": 12345},
]


def random_shape(rng):
    return {"M": rng.randint(1, 4096), "N": rng.randint(1, 8192)}


def make_inputs(shape, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    M, N = shape["M"], shape["N"]
    x = torch.randn(M, N, device="cuda", dtype=dtype, generator=g) + 0.5
    residual = torch.randn(M, N, device="cuda", dtype=dtype, generator=g)
    weight = 1 + 0.1 * torch.randn(N, device="cuda", dtype=dtype, generator=g)
    bias = 0.1 * torch.randn(N, device="cuda", dtype=dtype, generator=g)
    return [x, residual, weight, bias]


def reference(x, residual, weight, bias):
    h = x + residual
    return F.layer_norm(h, (h.shape[-1],), weight, bias, eps=EPS)
