"""Scaled dot-product attention (non-causal, no masking, no dropout).

The hardest problem. Naively it materialises an (S, S) score matrix per head. The
FlashAttention idea is to stream over K/V in blocks and keep a running ("online")
softmax, so the score matrix never touches main memory.

The reference uses PyTorch's scaled_dot_product_attention, which already dispatches to
an optimised fused kernel. That makes eager a strong baseline here.
"""

import torch
import torch.nn.functional as F

NAME = "attention"
CATEGORY = "attention"
DESCRIPTION = (
    "Non-causal scaled dot-product attention: y = softmax(q @ k^T / sqrt(D)) @ v, softmax "
    "over the last dimension. q, k, v have shape (B, H, S, D) and are contiguous; D is 32 or 64. "
    "Return a new tensor of shape (B, H, S, D) with q's dtype. Accumulate in float32."
)
DTYPES = [torch.float16, torch.float32]
TOLERANCES = {
    torch.float32: {"atol": 1e-3, "rtol": 1e-3},
    torch.float16: {"atol": 2e-2, "rtol": 2e-2},
}
SHAPES = {
    "small": {"B": 1, "H": 4, "S": 256, "D": 64},
    "medium": {"B": 4, "H": 8, "S": 1024, "D": 64},
    "large": {"B": 8, "H": 16, "S": 2048, "D": 64},
}
EDGE_SHAPES = [
    {"B": 1, "H": 1, "S": 1, "D": 32},
    {"B": 2, "H": 1, "S": 17, "D": 32},
    {"B": 1, "H": 3, "S": 1000, "D": 64},
]


def random_shape(rng):
    return {
        "B": rng.randint(1, 2),
        "H": rng.randint(1, 4),
        "S": rng.randint(1, 1024),
        "D": rng.choice([32, 64]),
    }


def flops(shape):
    # Two matmuls of (S x D) @ (D x S) and (S x S) @ (S x D), per batch and head.
    return 4 * shape["B"] * shape["H"] * shape["S"] * shape["S"] * shape["D"]


def make_inputs(shape, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    size = (shape["B"], shape["H"], shape["S"], shape["D"])
    q = torch.randn(size, device="cuda", dtype=dtype, generator=g)
    k = torch.randn(size, device="cuda", dtype=dtype, generator=g)
    v = torch.randn(size, device="cuda", dtype=dtype, generator=g)
    return [q, k, v]


def reference(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)
