"""Hand-written Triton kernel for problems/softmax.py — a known-good example.

STRATEGY: one program per row. Each program loads its entire row into registers, does
max / exp / sum / divide there, and writes the row back. Every input byte is read from
DRAM exactly once and every output byte written once, the minimum possible.

Eager PyTorch softmax is already a single fused kernel, so don't expect a big win over
eager; this example is here to show the reduction pattern clearly.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_cols, x_row_stride, out_row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)

    # tl.arange needs a power-of-2 length, so BLOCK is the next power of 2 >= n_cols and
    # the mask switches off the extra columns.
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    # The stride is how far apart (in elements) consecutive rows are in memory.
    row_ptr = x_ptr + row * x_row_stride
    # Masked-off columns are filled with -inf: exp(-inf) = 0, so they don't change the
    # sum, and they can never be the max.
    x = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)

    # Subtracting the row max doesn't change the answer mathematically
    # (the exp(-max) factor cancels) but keeps exp() from overflowing.
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator

    tl.store(out_ptr + row * out_row_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def kernel(x):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(n_cols)

    # More warps = more threads sharing the row, so fewer values per thread. Too few
    # warps for a long row means each thread holds hundreds of values, runs out of
    # registers and "spills" to slow local memory.
    if BLOCK <= 1024:
        num_warps = 4
    elif BLOCK <= 8192:
        num_warps = 8
    else:
        num_warps = 16

    _softmax_kernel[(n_rows,)](x, out, n_cols, x.stride(0), out.stride(0),
                               BLOCK=BLOCK, num_warps=num_warps)
    return out
