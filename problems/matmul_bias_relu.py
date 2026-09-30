"""Matrix multiply with a fused bias + ReLU epilogue — the first compute-bound problem.

A matmul does 2*M*N*K floating point operations on only (M*K + K*N + M*N) elements of
data, so for large sizes the limit is arithmetic throughput, not memory. The "epilogue"
(bias add + ReLU) is applied to the result tile while it is still in registers, which
saves writing the matmul result to memory and reading it back.

On the T4, fp16 matmuls can use Tensor Cores (tl.dot does this automatically).
"""

import torch

NAME = "matmul_bias_relu"
CATEGORY = "matmul"
DESCRIPTION = (
    "Matmul with fused epilogue: y = relu(a @ b + bias). a has shape (M, K), b has shape "
    "(K, N), bias has shape (N,) broadcast over rows. All inputs are contiguous (row-major). "
    "Return a new tensor of shape (M, N) with a's dtype. Accumulate in float32."
)
DTYPES = [torch.float16, torch.float32]
# Looser than elementwise ops: a length-K dot product summed in a different order gives
# slightly different rounding, and that is fine.
TOLERANCES = {
    torch.float32: {"atol": 1e-3, "rtol": 1e-3},
    torch.float16: {"atol": 2e-2, "rtol": 2e-2},
}
SHAPES = {
    "small": {"M": 512, "N": 512, "K": 512},
    "medium": {"M": 2048, "N": 2048, "K": 2048},
    "large": {"M": 4096, "N": 4096, "K": 4096},
}
EDGE_SHAPES = [
    {"M": 1, "N": 1, "K": 1},
    {"M": 1000, "N": 999, "K": 1537},   # K not a multiple of the block: needs masking
    {"M": 1, "N": 4096, "K": 4096},     # a single row (matrix-vector like)
]


def random_shape(rng):
    return {"M": rng.randint(1, 2048), "N": rng.randint(1, 2048), "K": rng.randint(1, 2048)}


def flops(shape):
    return 2 * shape["M"] * shape["N"] * shape["K"]


def make_inputs(shape, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    M, N, K = shape["M"], shape["N"], shape["K"]
    a = torch.randn(M, K, device="cuda", dtype=dtype, generator=g)
    # Divide by sqrt(K) so outputs stay around magnitude 1 regardless of K.
    b = torch.randn(K, N, device="cuda", dtype=dtype, generator=g) / (K ** 0.5)
    bias = 0.1 * torch.randn(N, device="cuda", dtype=dtype, generator=g)
    return [a, b, bias]


def reference(a, b, bias):
    return torch.relu(a @ b + bias)
