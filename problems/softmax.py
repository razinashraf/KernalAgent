"""Row-wise softmax — the classic reduction problem.

Each output row needs two facts about its whole input row: the max (for numerical
stability) and the sum of exponentials. So threads working on one row must cooperate:
this is a "reduction". The standard trick is one program per row, with the whole row
loaded into registers so it is read from memory only once.
"""

import torch

NAME = "softmax"
CATEGORY = "reduction"
DESCRIPTION = (
    "Row-wise softmax over the last dimension: y[i, :] = exp(x[i, :] - max(x[i, :])) / "
    "sum(exp(x[i, :] - max(x[i, :]))). Input x has shape (M, N) and is contiguous. "
    "Return a new tensor of shape (M, N) with the same dtype. "
    "Accumulate in float32 even for fp16 inputs."
)
DTYPES = [torch.float16, torch.float32]
TOLERANCES = {
    torch.float32: {"atol": 1e-5, "rtol": 1e-4},
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
    {"M": 37, "N": 20000},   # long rows: one-block-per-row kernels get register pressure
]


def random_shape(rng):
    return {"M": rng.randint(1, 4096), "N": rng.randint(1, 8192)}


def make_inputs(shape, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(shape["M"], shape["N"], device="cuda", dtype=dtype, generator=g) * 4
    return [x]


def reference(x):
    return torch.softmax(x, dim=-1)
