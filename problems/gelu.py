"""GELU activation (tanh approximation) — the simplest elementwise problem.

Elementwise = every output element depends on exactly one input element. There is no
data sharing between elements, so the kernel's speed is limited only by how fast the GPU
can read x from memory and write y back ("memory-bound"). A perfect kernel reads each
byte once and writes each byte once.
"""

import torch
import torch.nn.functional as F

NAME = "gelu"
CATEGORY = "elementwise"
DESCRIPTION = (
    "Elementwise GELU with the tanh approximation: "
    "y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))). "
    "Input x has shape (M, N) and is contiguous. Return a new tensor of the same shape and dtype."
)

# DTYPES[0] is the benchmarked dtype. fp16 halves the bytes moved vs fp32, so it is the
# realistic choice for a memory-bound op. fp32 is also tested for correctness.
DTYPES = [torch.float16, torch.float32]

# The candidate is compared against the reference computed in float32.
# fp16 only has ~3 decimal digits of precision, so its tolerance is much looser.
TOLERANCES = {
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
    torch.float16: {"atol": 1e-2, "rtol": 1e-2},
}

SHAPES = {
    "small": {"M": 1024, "N": 1024},    # 1M elements: launch overhead matters here
    "medium": {"M": 4096, "N": 4096},   # 16M elements
    "large": {"M": 8192, "N": 8192},    # 64M elements: pure bandwidth test
}

# Shapes that break kernels which assume "size is a multiple of my block size".
EDGE_SHAPES = [
    {"M": 1, "N": 1},
    {"M": 1000, "N": 1537},
    {"M": 3, "N": 100003},
]


def random_shape(rng):
    return {"M": rng.randint(1, 2048), "N": rng.randint(1, 4096)}


def make_inputs(shape, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # *3 spreads values over the interesting part of GELU (negative tail, bend, linear part)
    x = torch.randn(shape["M"], shape["N"], device="cuda", dtype=dtype, generator=g) * 3
    return [x]


def reference(x):
    return F.gelu(x, approximate="tanh")
