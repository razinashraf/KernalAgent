"""RMSNorm — LayerNorm's simpler cousin used in LLaMA. Only one statistic per row."""

import torch

NAME = "rmsnorm"
CATEGORY = "reduction"
EPS = 1e-6
DESCRIPTION = (
    "RMSNorm over the last dimension: y = x / sqrt(mean(x^2) + eps) * weight, "
    f"with eps = {EPS}. x has shape (M, N); weight has shape (N,). All inputs are contiguous. "
    "Return a new tensor of shape (M, N) with x's dtype. Accumulate in float32."
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
    x = torch.randn(M, N, device="cuda", dtype=dtype, generator=g) * 2
    weight = 1 + 0.1 * torch.randn(N, device="cuda", dtype=dtype, generator=g)
    return [x, weight]


def reference(x, weight):
    # Written out by hand (instead of F.rms_norm) so the math is visible.
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + EPS)
    return x / rms * weight
