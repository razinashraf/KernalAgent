"""Hand-written Triton kernel for problems/rmsnorm.py — a known-good example.

Same one-program-per-row pattern as softmax, plus a second input (weight) that is shared
by every row. All rows read the same weight vector, so after the first few programs it
is served from the L2 cache; it costs almost nothing extra.
"""

import torch
import triton
import triton.language as tl

EPS = 1e-6   # must match problems/rmsnorm.py


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, row_stride, eps, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    # other=0.0 for masked columns: they add 0 to the sum of squares below.
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Divide by n_cols (the real length), NOT by BLOCK (which includes padding).
    mean_square = tl.sum(x * x, axis=0) / n_cols
    inv_rms = 1.0 / tl.sqrt(mean_square + eps)
    y = x * inv_rms * w

    tl.store(out_ptr + row * row_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def kernel(x, weight):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(n_cols)
    num_warps = 4 if BLOCK <= 1024 else (8 if BLOCK <= 8192 else 16)
    _rmsnorm_kernel[(n_rows,)](x, weight, out, n_cols, x.stride(0), EPS,
                               BLOCK=BLOCK, num_warps=num_warps)
    return out
