"""SiLU(a) * b — the "SwiGLU" gate used in LLaMA-style MLPs.

Two inputs, one output, all elementwise. In eager PyTorch this is two kernels
(silu, then mul) with an intermediate tensor written to and read back from memory.
A fused kernel does it in one pass: read a and b, write y. That is 3 memory
transfers instead of 5, so the ideal speedup over eager is roughly 5/3.
"""

import torch
import torch.nn.functional as F

NAME = "silu_mul"
CATEGORY = "elementwise"
DESCRIPTION = (
    "Fused SwiGLU gate: y = silu(a) * b where silu(a) = a * sigmoid(a). "
    "Inputs a and b both have shape (M, N) and are contiguous. "
    "Return a new tensor of shape (M, N) with the same dtype."
)
DTYPES = [torch.float16, torch.float32]
TOLERANCES = {
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
    torch.float16: {"atol": 1e-2, "rtol": 1e-2},
}
SHAPES = {
    "small": {"M": 1024, "N": 2048},
    "medium": {"M": 4096, "N": 11008},   # LLaMA-7B MLP width
    "large": {"M": 8192, "N": 14336},    # LLaMA-3-8B MLP width
}
EDGE_SHAPES = [
    {"M": 1, "N": 1},
    {"M": 1000, "N": 1537},
    {"M": 7, "N": 11007},
]


def random_shape(rng):
    return {"M": rng.randint(1, 2048), "N": rng.randint(1, 4096)}


def make_inputs(shape, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(shape["M"], shape["N"], device="cuda", dtype=dtype, generator=g) * 2
    b = torch.randn(shape["M"], shape["N"], device="cuda", dtype=dtype, generator=g)
    return [a, b]


def reference(a, b):
    return F.silu(a) * b
