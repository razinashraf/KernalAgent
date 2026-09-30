"""LayerNorm over the last dimension (with learnable weight and bias).

A reduction that needs two statistics per row (mean and variance), followed by an
elementwise transform. Same "one program per row" pattern as softmax.
"""

import torch
import torch.nn.functional as F

NAME = "layernorm"
CATEGORY = "reduction"
EPS = 1e-5
DESCRIPTION = (
    "LayerNorm over the last dimension: y = (x - mean) / sqrt(var + eps) * weight + bias, "
    f"with biased variance and eps = {EPS}. x has shape (M, N); weight and bias have shape (N,). "
    "All inputs are contiguous. Return a new tensor of shape (M, N) with x's dtype. "
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
    # Non-zero mean so a kernel that forgets to subtract the mean fails loudly.
    x = torch.randn(M, N, device="cuda", dtype=dtype, generator=g) * 2 + 0.5
    weight = 1 + 0.1 * torch.randn(N, device="cuda", dtype=dtype, generator=g)
    bias = 0.1 * torch.randn(N, device="cuda", dtype=dtype, generator=g)
    return [x, weight, bias]


def reference(x, weight, bias):
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps=EPS)
