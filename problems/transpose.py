"""2-D matrix transpose — no math at all, pure memory movement.

This is THE problem for learning memory coalescing. A GPU reads memory in chunks
(e.g. 32 consecutive 4-byte words per warp). Reading x row by row is coalesced, but
then writing y column by column is not: neighbouring threads write addresses N elements
apart, so each write touches a different memory chunk. Fast transposes load a tile,
flip it on-chip, and write it out row by row so BOTH sides are coalesced.
"""

import torch

NAME = "transpose"
CATEGORY = "memory"
DESCRIPTION = (
    "Matrix transpose: y[j, i] = x[i, j]. x has shape (M, N) and is contiguous. "
    "Return a NEW contiguous tensor of shape (N, M) with x's dtype (a view is not accepted)."
)
DTYPES = [torch.float16, torch.float32]
# Transpose moves values without changing them, so the result must be bit-exact.
TOLERANCES = {
    torch.float32: {"atol": 0.0, "rtol": 0.0},
    torch.float16: {"atol": 0.0, "rtol": 0.0},
}
SHAPES = {
    "small": {"M": 1024, "N": 1024},
    "medium": {"M": 4096, "N": 4096},
    "large": {"M": 8192, "N": 8192},
}
EDGE_SHAPES = [
    {"M": 1, "N": 1},
    {"M": 1000, "N": 1537},
    {"M": 1, "N": 4099},
    {"M": 4099, "N": 3},
]


def random_shape(rng):
    return {"M": rng.randint(1, 4096), "N": rng.randint(1, 4096)}


def make_inputs(shape, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(shape["M"], shape["N"], device="cuda", dtype=dtype, generator=g)
    return [x]


def reference(x):
    return x.t().contiguous()
