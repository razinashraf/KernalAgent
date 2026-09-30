"""Row-wise logsumexp — a reduction whose output is much smaller than its input.

Output is one number per row, so almost all the memory traffic is reading x.
"""

import torch

NAME = "logsumexp"
CATEGORY = "reduction"
DESCRIPTION = (
    "Row-wise logsumexp: y[i] = log(sum_j exp(x[i, j])), computed stably as "
    "m + log(sum_j exp(x[i, j] - m)) with m = max_j x[i, j]. x has shape (M, N) and is "
    "contiguous. Return a new tensor of shape (M,) with x's dtype. Accumulate in float32."
)
DTYPES = [torch.float16, torch.float32]
TOLERANCES = {
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
    torch.float16: {"atol": 1e-2, "rtol": 1e-2},
}
SHAPES = {
    "small": {"M": 1024, "N": 1024},
    "medium": {"M": 4096, "N": 8192},
    "large": {"M": 16384, "N": 16384},
}
EDGE_SHAPES = [
    {"M": 1, "N": 1},
    {"M": 1000, "N": 1537},
    {"M": 3, "N": 100003},   # very long rows: a single block per row won't fit
]


def random_shape(rng):
    return {"M": rng.randint(1, 4096), "N": rng.randint(1, 8192)}


def make_inputs(shape, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # *3 + a large offset: a kernel that skips the "subtract the max" trick overflows
    # exp() in fp16 (max ~65504, exp(11.1) already exceeds it).
    x = torch.randn(shape["M"], shape["N"], device="cuda", dtype=dtype, generator=g) * 3 + 20
    return [x]


def reference(x):
    return torch.logsumexp(x, dim=-1)
